"""Read-only master FEW phase service for an isolated response process."""

# 2026-09-11 17:07 CST (linux): Local stdin/stdout only, no network or files
# written. Preserve model/data provenance in the handshake. Protocol transfer
# overhead is included equally in all full-chain timing paths.
import argparse
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

import few
from few.waveform import GenerateEMRIWaveform

from single_pass_basis import build_ssb_basis, count_intrinsic


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("prograde", "retrograde"), required=True)
    args = parser.parse_args()
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    params = [1e6, 10.0, 0.6, 8.0, 0.3, 1.0, 1.0, 1.1, 0.7, 0.8, 1.5, 0.0, 0.4, 0.5]
    if args.case == "retrograde":
        params[3], params[4], params[5] = 12.0, 0.4, -1.0
    settings = dict(T=0.003, dt=15.0)
    # 2026-09-11 17:30 CST (linux): Batch both algorithms in ONE request and
    # transfer the same number of complex samples. This isolates intrinsic reuse
    # from a reduction in pipe round trips. All state remains process-local.
    start = perf_counter()
    model = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        force_backend="cpu",
        frame="detector",
        return_list=False,
    )
    wave = np.asarray(model(*params, **settings))
    modes = np.stack(
        [getattr(model.waveform_generator, k) for k in ("ls", "ms", "ks", "ns")]
    )
    setup_seconds = perf_counter() - start
    source_root = Path(few.__file__).resolve().parents[2]
    sources = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (name == "few" or name.startswith("few.")) and path and path.endswith(".py"):
            relative = Path(path).resolve().relative_to(source_root)
            entry = identity(path)
            if str(relative) != "src/few/_version.py":
                original = subprocess.check_output(
                    [
                        "git",
                        "show",
                        "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191:" + str(relative),
                    ],
                    cwd=source_root,
                )
                if hashlib.sha256(original).hexdigest() != entry["sha256"]:
                    raise AssertionError("Not pinned master: " + str(relative))
            sources[str(relative)] = entry
    native_dir = Path(
        importlib.import_module("few_backend_cpu.pyinterp").__file__
    ).parent
    handshake = dict(
        case=args.case,
        parameters=params.copy(),
        settings=settings,
        modes=modes.tolist(),
        m=int(np.max(np.abs(modes[1]))),
        samples=wave.size,
        setup_seconds=setup_seconds,
        source_root=str(source_root),
        sources=sources,
        producer=identity(__file__),
        basis_implementation=identity(Path(__file__).with_name("single_pass_basis.py")),
        native=[identity(p) for p in sorted(native_dir.glob("*.so"))],
        data=[
            identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        ],
    )
    print(json.dumps(handshake), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("quit"):
            break
        operation = request.get("operation", "wave")
        phase = float(request.get("phase", 0.0))
        if not np.isfinite(phase):
            raise ValueError("Nonfinite phase")
        params[11] = phase
        start = perf_counter()
        if operation == "blackbox":
            grid = phase + 2 * np.pi * np.arange(2 * handshake["m"] + 1) / (
                2 * handshake["m"] + 1
            )
            values = []
            with count_intrinsic(model.waveform_generator) as counts:
                for angle in grid:
                    params[11] = angle
                    values.append(np.asarray(model(*params, **settings)).copy())
                    if not np.array_equal(
                        np.stack(
                            [
                                getattr(model.waveform_generator, k)
                                for k in ("ls", "ms", "ks", "ns")
                            ]
                        ),
                        modes,
                    ):
                        raise AssertionError("Blackbox changed mode selection")
            value = np.stack(values)
            operation_meta = dict(
                **counts,
                outer_calls=len(grid),
                native_group_calls=len(grid),
            )
            if counts["trajectory_calls"] != len(grid) or counts[
                "amplitude_calls"
            ] != len(grid):
                raise AssertionError("Blackbox observed counts differ")
        elif operation == "grouped":
            value, operation_meta = build_ssb_basis(model, params, settings)
        elif operation == "diagnostics":
            # 2026-09-11 17:36 CST (linux): Test ownership IN THIS PROCESS and
            # native-exception restoration outside performance measurements.
            inner = model.waveform_generator
            sum_owner, traj_owner = inner.create_waveform, inner.inspiral_generator
            amp = inner.amplitude_generator
            before_override = "get_amplitudes" in vars(amp)
            amp_method = amp.get_amplitudes
            value, operation_meta = build_ssb_basis(
                model, params, settings, audit_phases=True
            )
            saved = value.copy()
            normal = np.asarray(model(*params, **settings)).copy()
            operation_meta["worker_basis_ownership_passes"] = (
                value.tobytes() == saved.tobytes()
            )
            native_overridden = "sum" in vars(sum_owner)
            native_previous = vars(sum_owner).get("sum")

            def intentional_failure(*pos, **kw):
                raise RuntimeError("exp15 intentional native failure")

            sum_owner.sum = intentional_failure
            caught = False
            try:
                build_ssb_basis(model, params, settings)
            except RuntimeError as error:
                if str(error) != "exp15 intentional native failure":
                    raise
                caught = True
            finally:
                if native_overridden:
                    sum_owner.sum = native_previous
                else:
                    delattr(sum_owner, "sum")
            operation_meta["exception_caught"] = caught
            operation_meta["callable_restoration_passes"] = (
                inner.create_waveform is sum_owner
                and inner.inspiral_generator is traj_owner
                and ("get_amplitudes" in vars(amp)) == before_override
                and amp.get_amplitudes == amp_method
            )
            resumed = np.asarray(model(*params, **settings))
            operation_meta["post_failure_wave_byte_equal"] = (
                normal.tobytes() == resumed.tobytes()
            )
            if not all(
                operation_meta[key]
                for key in (
                    "worker_basis_ownership_passes",
                    "exception_caught",
                    "callable_restoration_passes",
                    "post_failure_wave_byte_equal",
                )
            ):
                raise AssertionError("Worker state/ownership diagnostic failed")
        elif operation == "wave":
            value = np.asarray(model(*params, **settings))
            operation_meta = dict(
                trajectory_calls=1,
                amplitude_calls=1,
                outer_calls=1,
                native_group_calls=1,
            )
        else:
            raise ValueError("Unknown operation")
        operation_seconds = perf_counter() - start
        actual_modes = np.stack(
            [getattr(model.waveform_generator, k) for k in ("ls", "ms", "ks", "ns")]
        )
        if (
            not np.array_equal(actual_modes, modes)
            or value.dtype != np.complex128
            or value.shape
            != (
                wave.shape
                if operation == "wave"
                else (2 * handshake["m"] + 1, wave.size)
            )
            or not np.all(np.isfinite(value))
        ):
            raise AssertionError("Phase changes fixed mode set or output contract")
        print(
            json.dumps(
                dict(
                    wave=base64.b64encode(value.tobytes()).decode("ascii"),
                    shape=value.shape,
                    operation_seconds=operation_seconds,
                    operation=operation,
                    metadata=operation_meta,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
