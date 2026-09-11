"""Actual-table first-result, residency and regression worker (one process)."""

# 2026-09-11 15:54 CST (linux): One model per process. Preserve ONCE file
# integrity checks; final explicit hashes are computed AFTER all timed calls.
import argparse
import gc
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.geodesic import get_separatrix
from few.utils.mappings.kerrecceq import kerrecceq_forward_map
from few.waveform import FastKerrEccentricEquatorialFlux
from few.waveform.base import SphericalHarmonicWaveformBase


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda12x"), required=True)
    parser.add_argument("--policy", choices=("master", "eager", "lazy"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--verified-first", action="store_true")
    args = parser.parse_args()
    archive = args.output.with_suffix(".npz")
    if args.output.exists() or archive.exists():
        raise FileExistsError("Reports and archives must be new")
    if not args.benchmark and args.policy != "master" and args.reference is None:
        raise ValueError(
            "Candidate validation requires its own backend's master reference"
        )
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    backend = few.get_backend(args.backend)
    xp = backend.xp

    def sync():
        if backend.uses_cuda:
            xp.cuda.runtime.deviceSynchronize()

    def host(value):
        return value.get() if hasattr(value, "get") else np.asarray(value)

    def telemetry():
        if not backend.uses_cuda:
            return None
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        return dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    def memory():
        sync()
        result = {}
        for name in ("smaps_rollup", "io", "status"):
            text = (Path("/proc/self") / name).read_text()
            keys = {
                "Rss",
                "Pss",
                "Private_Dirty",
                "VmRSS",
                "VmHWM",
                "rchar",
                "read_bytes",
            }
            result[name] = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in text.splitlines()
                if ":" in line and line.split(":", 1)[0] in keys
            }
        if backend.uses_cuda:
            pool = xp.get_default_memory_pool()
            result["cupy_pool"] = {
                "used_bytes": pool.used_bytes(),
                "total_bytes": pool.total_bytes(),
            }
            result["cuda_mem_info"] = list(xp.cuda.runtime.memGetInfo())
        return result

    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        policy=args.policy,
        backend=args.backend,
        benchmark=args.benchmark,
        verified_first=args.verified_first,
        passes=False,
        terminal=False,
        base_commit="47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        host_derivative_adapter=backend.uses_cuda,
        scope="Actual source waveform and amplitude, not TDI/inference or independent Teukolsky truth",
        timing_scope="Synchronized wall-clock; construction+first includes adapter setup and ONCE integrity check unless explicit verified-first phase; no memory snapshots within joint interval",
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        xp=xp.__name__,
        calls=[],
        amplitudes=[],
        sources=[
            identity(__file__),
            identity(inspect.getsourcefile(AmpInterpKerrEccEq)),
            identity(inspect.getsourcefile(SphericalHarmonicWaveformBase)),
        ],
    )
    arrays = {}
    ref = None

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def record(key, value):
        array = host(value).copy()
        if (
            not array.size
            or array.dtype.kind not in "iufc"
            or not np.all(np.isfinite(array))
        ):
            raise AssertionError("Invalid numeric output " + key)
        if ref is not None and (
            key not in ref.files
            or array.shape != ref[key].shape
            or array.dtype != ref[key].dtype
            or array.tobytes() != ref[key].tobytes()
        ):
            raise AssertionError("Same-backend master output changed: " + key)
        arrays[key] = array
        return dict(
            key=key,
            shape=array.shape,
            dtype=str(array.dtype),
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            master_byte_equal=ref is not None,
        )

    try:
        helper = importlib.util.find_spec("few.amplitude._spincache")
        if (helper is None) != (args.policy == "master"):
            raise AssertionError("Policy/source overlay mismatch")
        if helper:
            report["sources"].append(identity(helper.origin))
        if args.reference:
            ref = np.load(args.reference, allow_pickle=False)
            baseline = json.loads(args.reference.with_suffix(".json").read_text())
            if (
                not baseline["passes"]
                or not baseline["terminal"]
                or baseline["backend"] != args.backend
                or baseline["policy"] != "master"
                or baseline["benchmark"] != args.benchmark
            ):
                raise AssertionError("Invalid same-backend baseline")
            report["reference"] = identity(args.reference)
            if report["reference"] != baseline["archive"]:
                raise AssertionError("Baseline archive changed")
        report["gpu_before"] = telemetry()
        report["memory_before"] = memory()
        # A separate optional phase distinguishes validated-table reuse from
        # fresh-file-manager service. Its cost is recorded, never hidden.
        verified_start = time.perf_counter()
        if args.verified_first:
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5"):
                few.get_file_manager().get_file(name)
        report["verification_seconds"] = time.perf_counter() - verified_start
        settings = dict(T=0.003, dt=15.0, Phi_phi0=0.3, Phi_theta0=0.4, Phi_r0=0.5)
        report["settings"] = settings
        sync()
        joint_start = time.perf_counter()
        kwargs = (
            {}
            if args.policy == "master"
            else dict(spin_cache_size=2 if args.policy == "lazy" else None)
        )
        model = FastKerrEccentricEquatorialFlux(
            force_backend=args.backend, amplitude_kwargs=kwargs
        )
        sync()
        report["construction_seconds"] = time.perf_counter() - joint_start
        if backend.uses_cuda:
            trajectory = model.inspiral_generator.inspiral_generator
            original = trajectory.eval_integrator_derivative_spline

            def derivative(times, *pos, **kw):
                return original(
                    times.get() if hasattr(times, "get") else times, *pos, **kw
                )

            trajectory.eval_integrator_derivative_spline = derivative
        amp = model.amplitude_generator

        def waveform(spin=0.6, p0=12.0, xi=1.0, selection=None):
            return model(
                1e6,
                10.0,
                spin,
                p0,
                0.3,
                xi,
                1.1,
                0.7,
                dist=1.0,
                mode_selection=selection,
                **settings,
            )

        def cache_info():
            method = getattr(amp, "spin_cache_info", None)
            return method() if method else None

        report["cache_at_construction"] = cache_info()
        first_start = time.perf_counter()
        value = waveform()
        sync()
        first_end = time.perf_counter()
        report["first_seconds"] = first_end - first_start
        report["construction_plus_first_seconds"] = first_end - joint_start
        report["verified_setup_plus_first_seconds"] = first_end - verified_start
        if np.max(np.abs(host(value))) == 0:
            raise AssertionError("Zero first waveform")
        report["calls"].append(
            dict(
                label="first",
                seconds=report["first_seconds"],
                wave=record("first", value),
                modes=record(
                    "first_modes", xp.stack([model.ls, model.ms, model.ks, model.ns])
                ),
                cache=cache_info(),
                memory=memory(),
            )
        )
        del value
        # Changed brackets and return-to-original reveal reload costs rather
        # than measuring an endlessly warm single-spin cache only.
        requests = [
            ("repeat1", 0.6, 12.0, 1.0, None),
            ("repeat2", 0.6, 12.0, 1.0, None),
            ("near_spin", 0.61, 12.0, 1.0, None),
            ("far_spin", 0.9, 12.0, 1.0, None),
            ("retrograde", 0.6, 12.0, -1.0, None),
            ("return", 0.6, 12.0, 1.0, None),
        ]
        if not args.benchmark:
            requests += [
                ("outer_region", 0.6, 40.0, 1.0, None),
                ("few_modes", 0.6, 12.0, 1.0, [(2, 2, 0, 0), (3, 1, 0, 2)]),
            ]
        for label, spin, p0, xi, selection in requests:
            sync()
            start = time.perf_counter()
            value = waveform(spin, p0, xi, selection)
            sync()
            duration = time.perf_counter() - start
            report["calls"].append(
                dict(
                    label=label,
                    parameters=[spin, p0, xi, selection],
                    seconds=duration,
                    wave=record(label, value),
                    modes=record(
                        label + "_modes",
                        xp.stack([model.ls, model.ms, model.ks, model.ns]),
                    ),
                    cache=cache_info(),
                    memory=memory(),
                )
            )
            del value
            save()
        if not args.benchmark:
            modes = [
                (2, 2, 0, 0),
                (3, 1, 0, 2),
                (3, -1, 0, -2),
                (2, 0, 0, -1),
                (10, 10, 0, 55),
            ]
            for label, a, xi in (
                ("prograde", 0.6, 1.0),
                ("canonical_retrograde", -0.6, 1.0),
                ("direct_retrograde", 0.6, -1.0),
                ("spin_edge", 0.999, 1.0),
                ("negative_edge", -0.999, 1.0),
                ("zero_spin", 0.0, 1.0),
                ("region_boundary", 0.6, 1.0),
            ):
                # 2026-09-11 15:57 CST (linux): p=8 is inside the retrograde
                # separatrix for this fixture; preserve failed pilot separately.
                p = xp.asarray([12.0, 14.0, 40.0])
                e = xp.asarray([0.2, 0.3, 0.4])
                if label == "region_boundary":
                    edge = (
                        float(np.asarray(get_separatrix(0.6, 0.3, 1.0)).item()) + 9.001
                    )
                    p = xp.asarray(
                        [np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)]
                    )
                    e = xp.full(3, 0.3)
                inputs = (a, p, e, xi)
                for kind in ("all", "tuples", "raw"):
                    if kind == "all":
                        value = amp(*inputs)
                    elif kind == "tuples":
                        result = amp(*inputs, specific_modes=modes)
                        value = xp.stack([result[mode] for mode in modes], axis=1)
                    else:
                        ids = xp.asarray(
                            [int(amp.index_map_arr[mode]) for mode in modes]
                        )
                        value = amp(*inputs, specific_modes=ids)
                    report["amplitudes"].append(record(label + "_" + kind, value))
                u, w, y, z, mask = kerrecceq_forward_map(
                    np.full(3, a * xi),
                    host(p),
                    host(e),
                    np.ones(3),
                    return_mask=True,
                    kind="amplitude",
                )
                report.setdefault("mapping_cases", []).append(
                    dict(
                        label=label,
                        z=z.tolist(),
                        region_A=mask.tolist(),
                        cache=cache_info(),
                    )
                )
        if ref is not None and set(arrays) != set(ref.files):
            raise AssertionError("Missing/extra reference arrays")
        report["memory_final_live"] = memory()
        report["cache_final_live"] = cache_info()
        # Explicitly outside ALL timings, after reading the actual output.
        report["data"] = [
            identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        ]
        native_dir = Path(
            importlib.import_module(f"few_backend_{args.backend}.pyinterp").__file__
        ).parent
        report["native"] = [identity(path) for path in sorted(native_dir.glob("*.so"))]
        if args.reference and (
            report["data"] != baseline["data"] or report["native"] != baseline["native"]
        ):
            raise AssertionError("Data/native changed from master reference")
        np.savez_compressed(archive, **arrays)
        report["archive"] = identity(archive)
        report["gpu_after"] = telemetry()
        # Release diagnostic is separate from active-memory/timing evidence.
        amp = model = None
        gc.collect()
        sync()
        report["memory_after_gc"] = memory()
        if backend.uses_cuda:
            xp.get_default_memory_pool().free_all_blocks()
            report["memory_after_pool_trim"] = memory()
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if ref is not None:
            ref.close()
        report["terminal"] = True
        save()
        print(
            json.dumps(
                dict(
                    passes=report["passes"],
                    arrays=len(arrays),
                    policy=args.policy,
                    backend=args.backend,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
