"""Standalone CPU/CUDA regression for the master trajectory boundary fix."""

# 2026-09-11 13:04 CST (linux): No exp01 dependency or derivative adapter. Keep
# the real default threshold selector, source/native/data hashes, strict gates.
import argparse
import gc
import hashlib
import importlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy

import few
from few.waveform import FastKerrEccentricEquatorialFlux
from few.waveform.base import SphericalHarmonicWaveformBase


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_commit": "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        "script": identity(__file__),
        "waveform_source": identity(inspect.getsourcefile(SphericalHarmonicWaveformBase)),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "adapter": False, "scope": "Three short waveforms; no performance or astrophysical-truth claim",
        "limits": {"relative_l2": 2e-11, "normalized_max": 2e-11},
        "backends": {}, "comparisons": [], "passes": False,
    }
    parameters = {
        "nominal": (1e6, 10.0, 0.6, 8.0, 0.3, 1.0, np.pi/3, np.pi/4),
        "high_e": (1e6, 10.0, 0.6, 12.0, 0.8, 1.0, np.pi/3, np.pi/4),
        "retrograde": (1e6, 10.0, 0.6, 12.0, 0.3, -1.0, np.pi/3, np.pi/4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        data = few.get_file_manager().get_file("ZNAmps_l10_m10_n55_DS2Outer.h5")
        report["amplitude_data"] = identity(data)
        outputs, modes = {}, {}
        for name in ("cuda12x", "cpu"):
            backend = few.get_backend(name)
            model = FastKerrEccentricEquatorialFlux(force_backend=name)
            xp = backend.xp
            metadata = {"native_interpolation": identity(importlib.import_module(f"few_backend_{name}.pyinterp").__file__),
                        "mode_selection": model.mode_selector.mode_selection,
                        "cases": []}
            report["backends"][name] = metadata
            if name == "cuda12x":
                assert backend.uses_cuda
                metadata.update({"cupy": xp.__version__,
                    "gpu": xp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                    "runtime": xp.cuda.runtime.runtimeGetVersion(),
                    "driver": xp.cuda.runtime.driverGetVersion()})
            for case in [*parameters, "nominal"]:
                wave = model(*parameters[case], T=0.001, dt=15.0, dist=1.0)
                if backend.uses_cuda:
                    xp.cuda.runtime.deviceSynchronize()
                assert isinstance(wave, xp.ndarray)
                assert wave.dtype == xp.complex128 and wave.ndim == 1 and wave.size > 0
                host = wave.get() if backend.uses_cupy else np.asarray(wave)
                assert np.all(np.isfinite(host))
                assert np.max(np.abs(host)) > 0.0
                indices = []
                for field in ("ls", "ms", "ks", "ns"):
                    value = getattr(model, field)
                    indices.append(value.get() if backend.uses_cupy else np.asarray(value))
                indices = np.stack(indices)
                assert indices.shape[0] == 4 and indices.shape[1] > 0
                key = (name, case)
                if key in outputs:
                    np.testing.assert_array_equal(outputs[key], host)
                    np.testing.assert_array_equal(modes[key], indices)
                outputs[key], modes[key] = host.copy(), indices.copy()
                metadata["cases"].append({"case": case, "parameters": parameters[case],
                    "T_years": 0.001, "dt_seconds": 15.0, "distance_gpc": 1.0,
                    "samples": host.size, "dtype": str(host.dtype), "finite": True,
                    "waveform_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
                    "mode_indices_lmkn": indices.T.tolist()})
                save()
            del model, wave
            gc.collect()
        for case in parameters:
            ref, got = outputs["cpu", case], outputs["cuda12x", case]
            assert ref.shape == got.shape
            np.testing.assert_array_equal(modes["cpu", case], modes["cuda12x", case])
            scale = np.max(np.abs(ref))
            delta = (got - ref) / scale
            row = {"case": case, "relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(ref/scale)),
                   "normalized_max": float(np.max(np.abs(delta))), "modes_equal": True}
            row["passes"] = all(np.isfinite(row[k]) and row[k] <= report["limits"][k]
                                for k in ("relative_l2", "normalized_max"))
            report["comparisons"].append(row)
            save()
            if not row["passes"]:
                raise AssertionError(f"CPU/CUDA comparison failed: {case}")
        report["passes"] = True
        save()
        print(json.dumps(report["comparisons"], indent=2), flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
