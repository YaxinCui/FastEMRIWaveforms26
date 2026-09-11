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

from single_pass_basis import build_ssb_basis


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
        if name.startswith("few.") and path and path.endswith(".py"):
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
                trajectory_calls=len(grid),
                amplitude_calls=len(grid),
                outer_calls=len(grid),
                native_group_calls=len(grid),
            )
        elif operation == "grouped":
            value, operation_meta = build_ssb_basis(model, params, settings)
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
