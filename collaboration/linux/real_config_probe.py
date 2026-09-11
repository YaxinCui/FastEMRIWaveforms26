"""Real Kerr CPU waveform regression for call-time mode-selection overrides."""

# 2026-09-11 14:26 CST (linux): Reuse one immutable amplitude table, but use
# independent stock selector instances as references. No frontend adapter.
import argparse
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
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_commit": "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        "backend": "cpu",
        "adapter": False,
        "passes": False,
        "scope": "Actual short source waveforms; configuration regression, not scientific acceptance or timing",
        "sources": [
            identity(__file__),
            identity(inspect.getsourcefile(SphericalHarmonicWaveformBase)),
            identity(inspect.getsourcefile(ModeSelector)),
        ],
        "native": identity(
            importlib.import_module("few_backend_cpu.pyinterp").__file__
        ),
        "cases": [],
    }
    settings = dict(T=0.0001, dt=15.0, Phi_phi0=0.3, Phi_theta0=0.4, Phi_r0=0.5)
    report["settings"] = settings
    calls = []

    def psd(f):
        if (
            not isinstance(f, np.ndarray)
            or f.dtype != np.float64
            or not np.all(np.isfinite(f))
        ):
            raise ValueError("Invalid frequency array at PSD boundary")
        calls.append(f.shape)
        return 1 + 1000 * f**2

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["data"] = [
            identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        ]
        model = FastKerrEccentricEquatorialFlux(
            force_backend="cpu",
            mode_selector_kwargs={"mode_selection": "all", "sensitivity_fn": psd},
        )
        original = model.mode_selector
        reference = ModeSelector(
            model.amplitude_generator,
            model.ylm_gen,
            mode_selection="threshold",
            sensitivity_fn=psd,
            force_backend="cpu",
        )
        for name, intrinsic in {
            "nominal": (1e6, 10.0, 0.6, 8.0, 0.3, 1.0),
            "high_e": (1e6, 10.0, 0.6, 12.0, 0.8, 1.0),
            "retrograde": (1e6, 10.0, 0.6, 12.0, 0.3, -1.0),
        }.items():
            row = {"case": name, "intrinsic": intrinsic, "calls": []}
            report["cases"].append(row)

            def run(selection):
                before = len(calls)
                wave = model(
                    *intrinsic, 1.1, 0.7, dist=1.0, mode_selection=selection, **settings
                ).copy()
                modes = np.stack(
                    [getattr(model, field) for field in ("ls", "ms", "ks", "ns")]
                ).copy()
                if (
                    wave.dtype != np.complex128
                    or wave.ndim != 1
                    or not wave.size
                    or not np.all(np.isfinite(wave))
                    or np.max(np.abs(wave)) <= 0
                ):
                    raise AssertionError("Invalid real waveform")
                row["calls"].append(
                    {
                        "selection": selection,
                        "constructor": model.mode_selector.mode_selection,
                        "psd_calls": len(calls) - before,
                        "modes": modes.shape[1],
                        "samples": wave.size,
                        "sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                    }
                )
                return wave, modes

            full = run(None)
            filtered = run("threshold")
            full_after = run(None)
            model.mode_selector = reference
            filtered_reference = run(None)
            full_override = run("all")
            model.mode_selector = original
            for left, right in (
                (full, full_after),
                (filtered, filtered_reference),
                (full, full_override),
            ):
                for first, second in zip(left, right):
                    if (
                        first.shape != second.shape
                        or first.dtype != second.dtype
                        or first.tobytes() != second.tobytes()
                    ):
                        raise AssertionError(
                            "Configuration reference is not byte-identical"
                        )
            if filtered[1].shape[1] >= full[1].shape[1]:
                raise AssertionError("Threshold did not actually remove modes")
            if [entry["psd_calls"] for entry in row["calls"]] != [0, 1, 0, 1, 0]:
                raise AssertionError("Unexpected PSD callback path")
            if (
                original.mode_selection != "all"
                or reference.mode_selection != "threshold"
            ):
                raise AssertionError("Call override mutated constructor defaults")
            row["passes"] = True
            save()
            print(
                f"{name}: all/threshold/default restoration and independent selector references pass",
                flush=True,
            )
        report["passes"] = True
        save()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
