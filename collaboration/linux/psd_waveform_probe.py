"""Real CPU/CUDA PSD-selection comparison and portable FP64 references."""

# 2026-09-11 14:02 CST (linux): Validate the production boundary repair with
# no adapter. Small NPZ waveforms may be read by another independent experiment.
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

import few
from few.utils.modeselector import ModeSelector
from few.waveform import FastKerrEccentricEquatorialFlux
from few.waveform.base import SphericalHarmonicWaveformBase

CASES = {
    "nominal": (1e6, 10.0, 0.6, 8.0, 0.3, 1.0),
    "high_e": (1e6, 10.0, 0.6, 12.0, 0.8, 1.0),
    "retrograde": (1e6, 10.0, 0.6, 12.0, 0.3, -1.0),
}
VIEWS = [
    (np.pi / 3, np.pi / 4, 1.0),
    (0.02, 1.8, 2.0),
    (np.pi / 2, -0.4, None),
    (2.7, 0.3, 0.5),
]
SETTINGS = dict(T=0.001, dt=15.0, Phi_phi0=0.3, Phi_theta0=0.4, Phi_r0=0.5)


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    output, artifact = (
        args.output_prefix.with_suffix(".json"),
        args.output_prefix.with_suffix(".npz"),
    )
    if output.exists() or artifact.exists():
        raise FileExistsError(output)
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "host": "linux",
        "base_commit": "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        "adapter": False,
        "scope": "Synthetic positive PSD selection; not physical detector/TDI acceptance",
        "script": identity(__file__),
        "sources": [
            identity(inspect.getsourcefile(cls))
            for cls in (SphericalHarmonicWaveformBase, ModeSelector)
        ],
        "parameters": CASES,
        "views": VIEWS,
        "settings": SETTINGS,
        "psd": "1 + 1000 * f_Hz**2",
        "runs": [],
        "passes": False,
    }

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["data"] = {
            name: identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        }
        arrays = {}
        for backend_name in ("cpu", "cuda12x"):
            backend = few.get_backend(backend_name)
            xp = backend.xp
            row = {
                "backend": backend_name,
                "native": identity(
                    importlib.import_module(
                        f"few_backend_{backend_name}.pyinterp"
                    ).__file__
                ),
                "cases": [],
                "psd_calls": 0,
            }
            report["runs"].append(row)

            def sensitivity(f):
                if not isinstance(f, xp.ndarray) or f.dtype != xp.float64:
                    raise TypeError(
                        "PSD did not receive the requested array backend/FP64"
                    )
                row["psd_calls"] += 1
                return 1 + 1000 * f**2

            model = FastKerrEccentricEquatorialFlux(
                force_backend=backend,
                sum_kwargs={"pad_output": True, "odd_len": True},
                mode_selector_kwargs={
                    "sensitivity_fn": sensitivity,
                    "mode_selection": "threshold",
                },
            )
            for name, parameters in CASES.items():
                for index, (theta, phi, dist) in enumerate(VIEWS):
                    wave = model(*parameters, theta, phi, dist=dist, **SETTINGS)
                    if not isinstance(wave, xp.ndarray):
                        raise TypeError("Output backend mismatch")
                    host = wave.get() if backend.uses_cupy else np.asarray(wave)
                    if (
                        host.ndim != 1
                        or host.dtype != np.complex128
                        or not host.size
                        or not np.all(np.isfinite(host))
                    ):
                        raise ValueError("Invalid waveform output")
                    modes = np.stack(
                        [
                            to_host(getattr(model, field))
                            for field in ("ls", "ms", "ks", "ns")
                        ]
                    )
                    key = f"{backend_name}_{name}_{index}"
                    arrays[key], arrays[key + "_modes"] = host.copy(), modes
                    item = {
                        "case": name,
                        "view": index,
                        "samples": host.size,
                        "mode_count": modes.shape[1],
                        "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
                    }
                    if backend_name == "cuda12x":
                        ref_key = f"cpu_{name}_{index}"
                        ref = arrays[ref_key]
                        np.testing.assert_array_equal(modes, arrays[ref_key + "_modes"])
                        if ref.shape != host.shape:
                            raise ValueError("CPU/CUDA shape mismatch")
                        scale = np.max(np.abs(ref))
                        if scale <= 0:
                            raise ValueError("Unexpected zero reference")
                        diff = (host - ref) / scale
                        l2 = float(np.linalg.norm(diff) / np.linalg.norm(ref / scale))
                        maximum = float(np.max(np.abs(diff)))
                        if not np.isfinite(l2 + maximum) or max(l2, maximum) > 2e-11:
                            raise AssertionError("CPU/CUDA PSD waveform mismatch")
                        item.update(relative_l2=l2, normalized_max=maximum)
                    row["cases"].append(item)
            if row["psd_calls"] != len(CASES) * len(VIEWS):
                raise AssertionError("PSD callback coverage mismatch")
            del model
            gc.collect()
            save()
            print(f"{backend_name}: 12 source-strain cases passed", flush=True)
        arrays["metadata_json"] = np.asarray(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "timestamp_cst",
                        "host",
                        "parameters",
                        "views",
                        "settings",
                        "psd",
                        "data",
                    )
                }
            )
        )
        np.savez_compressed(artifact, **arrays)
        report["artifact"] = identity(artifact)
        report["passes"] = True
        save()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


def to_host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


if __name__ == "__main__":
    main()
