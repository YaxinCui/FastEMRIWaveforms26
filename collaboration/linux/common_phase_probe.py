"""Warm complete-waveform A/B; run with either a clean-master or exp01 package."""

# 2026-09-11 12:10 CST (linux): Generated JSON/NPY are experiment outputs.
# No reference values are silently updated and initialization is timed separately.
import argparse
import hashlib
import importlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy
import few
from few.waveform import FastKerrEccentricEquatorialFlux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays-dir", type=Path)
    parser.add_argument("--backend", default="cuda12x")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--host-derivative-adapter", action="store_true")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--durations", type=float, nargs="+", default=[.001, 1.])
    parser.add_argument("--cases", nargs="+", default=["nominal", "high_e", "retrograde"])
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Require at least one measured repetition")
    # master setter's positional extra-path tuple triggers list+tuple during
    # configuration merging; use its documented environment-string interface.
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir)
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    backend = few.get_backend(args.backend)
    xp = backend.xp
    def sync():
        if backend.uses_cuda:
            xp.cuda.runtime.deviceSynchronize()
    def host(x):
        return x.get() if backend.uses_cupy else np.asarray(x)
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_commit": "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        "few_path": few.__file__, "few_version": few.__version__,
        "backend": args.backend, "platform": platform.platform(),
        "protocol": "complete waveforms; shared model/data; 2 warmups each; alternating AB/BA synchronized wall time; no TDI/likelihood",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host_derivative_adapter": args.host_derivative_adapter,
        "limits": {"relative_l2":2e-12, "normalized_max":2e-12},
        "numpy":np.__version__, "scipy":scipy.__version__,
        "passes":False,
        "cases": [],
    }
    if backend.uses_cuda:
        report["gpu"] = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        report["cupy"] = xp.__version__
        report["cuda_runtime"] = xp.cuda.runtime.runtimeGetVersion()
        report["cuda_driver"] = xp.cuda.runtime.driverGetVersion()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.arrays_dir:
        args.arrays_dir.mkdir(parents=True, exist_ok=True)
    try:
        start = time.perf_counter()
        model = FastKerrEccentricEquatorialFlux(force_backend=args.backend)
        if args.host_derivative_adapter:
            # Explicit, identical harness adaptation for both variants; production
            # fix belongs to exp04, not this master-based common-phase branch.
            trajectory = model.inspiral_generator.inspiral_generator
            original = trajectory.eval_integrator_derivative_spline
            def host_derivative(times, *pos, **kw):
                return original(times.get() if hasattr(times, "get") else times, *pos, **kw)
            trajectory.eval_integrator_derivative_spline = host_derivative
        sync()
        report["construction_seconds"] = time.perf_counter() - start
        # 2026-09-11 13:10 CST (linux): Record actual source/native/data identities
        # outside timing; historical pilots did not record GPU clock telemetry.
        paths = [Path(few.__file__).parent / x for x in (
            "waveform/base.py", "summation/interpolatedmodesum.py")]
        common_source = Path(few.__file__).parent / "summation/commonphase.py"
        if common_source.exists():
            paths.append(common_source)
        paths.append(Path(importlib.import_module(f"few_backend_{args.backend}.pyinterp").__file__))
        paths.append(Path(few.get_file_manager().get_file("ZNAmps_l10_m10_n55_DS2Outer.h5")))
        report["identities"] = []
        for path in paths:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream,"sha256").hexdigest()
            report["identities"].append({"path":str(path.resolve()),"sha256":digest})
        report["gpu_clock_telemetry"] = "not recorded; timing is pilot evidence, not controlled-clock certification"
        cases = {
            "nominal": (1e6, 10., .6, 8., .3, 1., np.pi/3, np.pi/4),
            "high_e": (1e6, 10., .6, 12., .8, 1., np.pi/3, np.pi/4),
            "retrograde": (1e6, 10., .6, 12., .3, -1., np.pi/3, np.pi/4),
        }
        variants = [False] if args.baseline_only else [False, True]
        for case in args.cases:
            for duration in args.durations:
                row = {"case": case, "parameters": list(cases[case]), "T_years": duration, "dt_seconds":15., "variants":{}}
                samples = {v: [] for v in variants}
                outputs = {}
                mode_lists = {}
                for v in variants:
                    model.create_waveform.common_phase = v
                    for _ in range(2):
                        model(*cases[case], T=duration, dt=15., dist=1.)
                        sync()
                for repeat in range(args.repetitions):
                    for v in (variants if repeat % 2 == 0 else variants[::-1]):
                        model.create_waveform.common_phase = v
                        sync()
                        start = time.perf_counter()
                        outputs[v] = model(*cases[case], T=duration, dt=15., dist=1.)
                        sync()
                        samples[v].append(1e3*(time.perf_counter()-start))
                        mode_lists[v] = np.stack([host(getattr(model, name)) for name in ("ls","ms","ks","ns")])
                for v in variants:
                    label = "grouped" if v else "legacy"
                    values = host(outputs[v])
                    if values.ndim != 1 or not values.size or values.dtype != np.complex128:
                        raise ValueError("Require a nonempty complex128 waveform")
                    if not np.all(np.isfinite(values)):
                        raise FloatingPointError("Non-finite waveform output")
                    row["variants"][label] = {"samples_ms": samples[v], "median_ms":float(np.median(samples[v])),
                        "finite":True,"dtype":str(values.dtype),"samples":int(values.size)}
                    if args.arrays_dir:
                        np.save(args.arrays_dir / f"{case}_{duration}_{label}.npy", host(outputs[v]))
                row["samples"] = int(outputs[False].size)
                row["selected_modes"] = int(mode_lists[False].shape[1])
                if not args.baseline_only:
                    ref, got = host(outputs[False]), host(outputs[True])
                    if ref.shape != got.shape or np.max(np.abs(ref)) == 0.0:
                        raise ValueError("Incompatible or zero reference waveform")
                    delta = got-ref
                    row.update({"groups": int(model.create_waveform.common_phase_counts[1]),
                        "speedup": float(np.median(samples[False])/np.median(samples[True])),
                        "relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(ref)),
                        "normalized_max": float(np.max(np.abs(delta))/np.max(np.abs(ref))),
                        "modes_equal": bool(np.array_equal(mode_lists[False], mode_lists[True])),
                        "finite": bool(np.all(np.isfinite(got)))})
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
                if not args.baseline_only and not (
                    row["modes_equal"] and all(np.isfinite(row[key]) and row[key] <= limit
                    for key,limit in report["limits"].items())):
                    raise AssertionError("Common-phase numerical gate failed")
                print(json.dumps(row), flush=True)
    except Exception as exc:
        report["error"] = {"type":type(exc).__name__, "message":str(exc)}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        raise
    report["passes"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
