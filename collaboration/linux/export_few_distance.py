"""Generate actual master FEW SSB strains without altering its Python stack."""

# 2026-09-11 14:32 CST (linux): Small authenticated NPZ bridge to isolated
# response environment. Detector-frame here means SSB, not actual LISA TDI.
import argparse
import hashlib
import importlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.waveform import GenerateEMRIWaveform
from few.waveform.base import SphericalHarmonicWaveformBase


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
    parameters = [1e6, 10.0, 0.6, 8.0, 0.3, 1.0, 1.0, 1.1, 0.7, 0.8, 1.5, 0.3, 0.4, 0.5]
    distances = [1.0, 0.5, 2.0, 1.3, 1.0 + 1e-10]
    settings = dict(T=0.003, dt=15.0)
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        base_commit="47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        backend="cpu",
        parameters=parameters.copy(),
        distances_Gpc=distances,
        settings=settings,
        polarization="SSB h_plus - i*h_cross; conjugate for fastlisaresponse",
        passes=False,
        runs=[],
    )
    report["sources"] = [
        identity(__file__),
        identity(inspect.getsourcefile(GenerateEMRIWaveform)),
        identity(inspect.getsourcefile(SphericalHarmonicWaveformBase)),
    ]
    report["native"] = identity(
        importlib.import_module("few_backend_cpu.pyinterp").__file__
    )

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["data"] = [
            identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        ]
        start = perf_counter()
        model = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            force_backend="cpu",
            frame="detector",
            return_list=False,
        )
        report["construction_seconds"] = perf_counter() - start
        arrays = {}
        reference_modes = None
        for index, distance in enumerate(distances):
            parameters[6] = distance
            start = perf_counter()
            wave = np.asarray(model(*parameters, **settings)).copy()
            duration = perf_counter() - start
            modes = np.stack(
                [
                    getattr(model.waveform_generator, field)
                    for field in ("ls", "ms", "ks", "ns")
                ]
            ).copy()
            if (
                wave.ndim != 1
                or wave.dtype != np.complex128
                or not wave.size
                or not np.all(np.isfinite(wave))
            ):
                raise AssertionError("Invalid FEW SSB strain")
            if reference_modes is None:
                reference_modes = modes.copy()
            np.testing.assert_array_equal(modes, reference_modes)
            arrays[f"strain_{index}"] = wave
            reference = arrays["strain_0"] / distance
            scale = np.max(np.abs(reference))
            if not scale > 0 or reference.shape != wave.shape:
                raise AssertionError("Invalid scaling reference")
            error = float(
                np.linalg.norm((wave - reference) / scale)
                / np.linalg.norm(reference / scale)
            )
            if not np.isfinite(error) or error > 2e-12:
                raise AssertionError("FEW distance scaling mismatch")
            report["runs"].append(
                dict(
                    distance_Gpc=distance,
                    samples=wave.size,
                    modes=modes.shape[1],
                    full_waveform_seconds=duration,
                    relative_l2=error,
                    sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                )
            )
            save()
            print(
                f"D={distance}: {wave.size} samples, relative scaling error {error:.3e}",
                flush=True,
            )
        arrays["modes"] = reference_modes
        arrays["metadata_json"] = np.asarray(json.dumps(report))
        np.savez_compressed(artifact, **arrays)
        report["artifact"] = identity(artifact)
        report["passes"] = True
        save()
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
