"""Real Kerr prepared-observer regression and amortized timing evidence."""

# 2026-09-11 13:32 CST (linux): Include preparation costs and reselect modes at
# each angle. CUDA master references require an explicit reported host adapter.
import argparse
import gc
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
import weakref
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.waveform import FastKerrEccentricEquatorialFlux
from few.waveform.base import SphericalHarmonicWaveformBase
from few.waveform.prepared import PreparedKerrWaveform

CASES = {
    "nominal": (1e6, 10.0, 0.6, 8.0, 0.3, 1.0),
    "high_e": (1e6, 10.0, 0.6, 12.0, 0.8, 1.0),
    "retrograde": (1e6, 10.0, 0.6, 12.0, 0.3, -1.0),
}
OBSERVERS = [
    (np.pi / 3, np.pi / 4, 1.0),
    (0.02, 1.8, 2.0),
    (np.pi / 2, -0.4, None),
    (2.7, 0.3, 0.5),
]


# 2026-09-11 13:43 CST (linux): Preserve the first pilot; strengthen fail-closed
# comparisons, real owner invalidation tests and measured warm preparation.
def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def to_host(value):
    # CUDA snapshots deliberately contain both CPU time knots and GPU arrays.
    return value.get() if hasattr(value, "get") else np.asarray(value)


def compare(ref, got):
    if (
        ref.ndim != 1
        or ref.shape != got.shape
        or ref.dtype != got.dtype
        or ref.dtype != np.complex128
    ):
        raise ValueError("Waveform shape/dtype mismatch")
    if not ref.size or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        raise ValueError("Non-finite or empty waveform")
    scale = np.max(np.abs(ref))
    if scale == 0:
        if np.any(got != 0):
            raise AssertionError("Nonzero result for zero reference")
        l2 = maximum = 0.0
    else:
        difference = (got - ref) / scale
        l2 = float(np.linalg.norm(difference) / np.linalg.norm(ref / scale))
        maximum = float(np.max(np.abs(difference)))
    if not np.isfinite(l2) or not np.isfinite(maximum) or max(l2, maximum) > 2e-11:
        raise AssertionError(f"Prepared waveform gate failed: {l2}, {maximum}")
    return {
        "relative_l2": l2,
        "normalized_max": maximum,
        "raw_bytes_equal": ref.tobytes() == got.tobytes(),
    }


def telemetry():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda12x"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--duration", type=float, default=0.001)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--reference-host-adapter", action="store_true")
    parser.add_argument("--backwards", action="store_true")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="State/selection tests, intended for short T",
    )
    parser.add_argument(
        "--selection", choices=["threshold", "fixed"], default="threshold"
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Freeze and test a mutable synthetic PSD",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("Require repetitions >= 1 and positive finite duration")
    if args.output.exists():
        parser.error("Output exists; choose a new evidence filename")
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    backend = few.get_backend(args.backend)
    xp = backend.xp
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_commit": "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        "source": few.__file__,
        "backend": args.backend,
        "argv": sys.argv,
        "numpy": np.__version__,
        "reference_host_adapter": args.reference_host_adapter,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Source strain only; no TDI/likelihood. Model construction and first preparation separate; warm prepare measured. No cold-to-cold claim.",
        "cases": [],
        "passes": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def sync():
        if backend.uses_cuda:
            xp.cuda.runtime.deviceSynchronize()

    def host(value):
        return to_host(value)

    def snapshot_digest(prepared):
        digest = hashlib.sha256()
        arrays = [
            *prepared._trajectory,
            prepared._phase,
            prepared._phase_t,
            *prepared._frequencies.values(),
            prepared._selector.amplitude_generator._values,
        ]
        if prepared._selector.sensitivity_fn is not None:
            arrays.extend(
                [
                    prepared._selector.sensitivity_fn._values,
                    prepared._selector.sensitivity_fn._frequencies,
                ]
            )
        for value in arrays:
            array = host(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def must_raise(kind, callback):
        try:
            callback()
        except kind:
            return
        raise AssertionError(f"Expected {kind.__name__}")

    try:
        report["source_identities"] = [
            identity(inspect.getsourcefile(cls))
            for cls in (
                FastKerrEccentricEquatorialFlux,
                SphericalHarmonicWaveformBase,
                PreparedKerrWaveform,
            )
        ]
        report["native_interpolation"] = identity(
            importlib.import_module(f"few_backend_{args.backend}.pyinterp").__file__
        )
        report["data"] = {
            name: identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        }
        if backend.uses_cuda:
            report["telemetry_before"] = telemetry()
            report["cuda"] = {
                "cupy": xp.__version__,
                "runtime": xp.cuda.runtime.runtimeGetVersion(),
                "driver": xp.cuda.runtime.driverGetVersion(),
            }
        sensitivity_state = {"slope": 1000.0, "calls": 0}

        def sensitivity(frequencies):
            sensitivity_state["calls"] += 1
            return 1.0 + sensitivity_state["slope"] * frequencies**2

        selection = (
            [(2, 2, 0, 0), (3, 2, 0, 1), (2, 0, 0, 1)]
            if args.selection == "fixed"
            else "threshold"
        )
        start = time.perf_counter()
        model = FastKerrEccentricEquatorialFlux(
            force_backend=backend,
            sum_kwargs={"pad_output": True, "odd_len": True},
            mode_selector_kwargs={
                "mode_selection": selection,
                "sensitivity_fn": sensitivity if args.sensitivity else None,
            },
        )
        if args.reference_host_adapter:
            original = model.inspiral_generator.inspiral_generator.eval_integrator_derivative_spline

            def adapter(times, *pos, **kw):
                return original(
                    times.get() if hasattr(times, "get") else times, *pos, **kw
                )

            model.inspiral_generator.inspiral_generator.eval_integrator_derivative_spline = adapter
        sync()
        report["construction_seconds"] = time.perf_counter() - start
        settings = dict(
            T=args.duration,
            dt=15.0,
            Phi_phi0=0.3,
            Phi_theta0=0.4,
            Phi_r0=0.5,
            inspiral_kwargs={"integrate_backwards": args.backwards},
        )
        for name in args.cases:
            parameters = CASES[name]
            owner_config = deepcopy(model.inspiral_kwargs)
            sync()
            start = time.perf_counter()
            prepared = model.prepare_observer_scan(*parameters, **settings)
            sync()
            prepare_seconds = time.perf_counter() - start
            if owner_config != model.inspiral_kwargs:
                raise AssertionError("Preparation mutated owner inspiral kwargs")
            row = {
                "case": name,
                "parameters": parameters,
                "settings": settings,
                "preparation_seconds": prepare_seconds,
                "metadata": prepared.metadata,
                "observers": [],
                "samples_seconds": {"fresh": [], "prepared": []},
            }
            report["cases"].append(row)
            before_digest = snapshot_digest(prepared)
            saved = None
            for theta, phi, dist in [*OBSERVERS, OBSERVERS[0]]:
                reference = model(*parameters, theta, phi, dist=dist, **settings)
                reference_modes = np.stack(
                    [host(getattr(model, field)) for field in ("ls", "ms", "ks", "ns")]
                )
                candidate = prepared(theta, phi, dist=dist)
                sync()
                if not isinstance(reference, xp.ndarray) or not isinstance(
                    candidate, xp.ndarray
                ):
                    raise TypeError("Waveform backend does not match requested backend")
                ref, got = host(reference), host(candidate)
                candidate_modes = np.stack([host(v) for v in prepared.selected_modes])
                np.testing.assert_array_equal(reference_modes, candidate_modes)
                metrics = compare(ref, got)
                row["observers"].append(
                    {
                        "theta": theta,
                        "phi": phi,
                        "dist": dist,
                        "modes": candidate_modes.shape[1],
                        "mode_hash": hashlib.sha256(
                            candidate_modes.tobytes()
                        ).hexdigest(),
                        "samples": got.size,
                        **metrics,
                    }
                )
                if saved is None:
                    saved = got.copy()
            if saved.tobytes() != got.tobytes():
                raise AssertionError("A -> B -> A observer replay changed")
            if args.extended:
                theta, phi, dist = OBSERVERS[0]
                row["extra_selection"] = []
                for options in (
                    {"mode_selection": "all"},
                    {"mode_selection": "threshold", "mode_selection_threshold": 1e-3},
                    {"mode_selection": [(2, 0, 0, 1), (3, 2, 0, -1)]},
                    {
                        "mode_selection": [(3, -2, 0, 1), (2, 1, 0, -2)],
                        "include_minus_mkn": False,
                    },
                ):
                    reference = model(
                        *parameters, theta, phi, dist=dist, **settings, **options
                    )
                    reference_modes = np.stack(
                        [
                            host(getattr(model, field))
                            for field in ("ls", "ms", "ks", "ns")
                        ]
                    )
                    candidate = prepared(theta, phi, dist=dist, **options)
                    np.testing.assert_array_equal(
                        reference_modes,
                        np.stack([host(v) for v in prepared.selected_modes]),
                    )
                    row["extra_selection"].append(
                        {
                            "options": options,
                            **compare(host(reference), host(candidate)),
                        }
                    )
                row["phi_periodicity"] = compare(
                    saved, host(prepared(theta, phi + 2 * np.pi, dist=dist))
                )
                row["distance_scaling"] = compare(
                    saved, 2 * host(prepared(theta, phi, dist=2 * dist))
                )
                # Change intrinsic parameters, duration, sampling and phases in
                # owner state, not only the observer. Alternate integrator direction.
                other_settings = dict(
                    T=0.0007,
                    dt=7.0,
                    Phi_phi0=1.7,
                    Phi_theta0=0.1,
                    Phi_r0=2.1,
                    inspiral_kwargs={"integrate_backwards": not args.backwards},
                )
                other = (1.1e6, 12.0, 0.4, 13.0, 0.25, -1.0)
                model(*other, theta, phi, dist=dist, **other_settings)
                row["owner_A_B_A"] = compare(
                    saved, host(prepared(theta, phi, dist=dist))
                )
                other_prepared = model.prepare_observer_scan(*other, **other_settings)
                other_prepared(theta, phi, dist=dist)
                row["prepare_A_B_A"] = compare(
                    saved, host(prepared(theta, phi, dist=dist))
                )
                del other_prepared
                returned = prepared(theta, phi, dist=dist)
                returned[:] = 0
                modes_before = tuple(host(v).copy() for v in prepared.selected_modes)
                for value in prepared.selected_modes:
                    value[:] = -99
                for expected_modes, actual_modes in zip(
                    modes_before, prepared.selected_modes
                ):
                    np.testing.assert_array_equal(expected_modes, host(actual_modes))
                metadata_before = prepared.metadata
                metadata = prepared.metadata
                metadata["parameters"] = None
                if prepared.metadata != metadata_before:
                    raise AssertionError("Metadata getter aliases snapshot state")
                row["mutated_return_replay"] = compare(
                    saved, host(prepared(theta, phi, dist=dist))
                )
                for threshold in (np.nan, np.inf, -1, 1):
                    must_raise(
                        ValueError,
                        lambda: prepared(
                            theta, phi, mode_selection_threshold=threshold
                        ),
                    )
                for bad_dist in (np.nan, np.inf, 0, -1):
                    must_raise(ValueError, lambda: prepared(theta, phi, dist=bad_dist))
                must_raise(TypeError, lambda: prepared(theta, phi, T=0.1))
                prepared._render_lock.acquire()
                try:
                    must_raise(RuntimeError, lambda: prepared(theta, phi))
                finally:
                    prepared._render_lock.release()
                if backend.uses_cuda:
                    with xp.cuda.Stream(non_blocking=True):
                        must_raise(NotImplementedError, lambda: prepared(theta, phi))
                        must_raise(
                            NotImplementedError,
                            lambda: model.prepare_observer_scan(
                                *parameters, **settings
                            ),
                        )
                if args.sensitivity:
                    calls = sensitivity_state["calls"]
                    sensitivity_state["slope"] = 9e12
                    row["mutated_psd_replay"] = compare(
                        saved, host(prepared(theta, phi, dist=dist))
                    )
                    if calls != sensitivity_state["calls"]:
                        raise AssertionError(
                            "Prepared render called external PSD callback"
                        )
                    sensitivity_state["slope"] = 1000.0
                row["rejected_invalid_inputs"] = True
            if before_digest != snapshot_digest(prepared):
                raise AssertionError(
                    "Rendering or owner activity mutated intrinsic snapshot"
                )
            row["snapshot_sha256"] = before_digest
            # Restore full original preparation settings after the intrinsic-B
            # check; original master merges call-time kwargs persistently.
            model.inspiral_kwargs.clear()
            model.inspiral_kwargs.update(owner_config)

            # Warm all complete four-observer workloads, then alternate the
            # order. Measure preparation+scan directly, not just a median sum.
            def run(use_prepared, reprepare=False):
                active = (
                    model.prepare_observer_scan(*parameters, **settings)
                    if reprepare
                    else prepared
                )
                for theta, phi, dist in OBSERVERS:
                    if use_prepared:
                        active(theta, phi, dist=dist)
                    else:
                        model(*parameters, theta, phi, dist=dist, **settings)
                sync()

            run(False)
            run(True)
            run(True, reprepare=True)
            row["samples_seconds"]["prepare_and_scan"] = []
            row["samples_seconds"]["warm_preparation"] = []
            for _ in range(args.repetitions):
                sync()
                start = time.perf_counter()
                temporary = model.prepare_observer_scan(*parameters, **settings)
                sync()
                row["samples_seconds"]["warm_preparation"].append(
                    time.perf_counter() - start
                )
                del temporary
            for repeat in range(args.repetitions):
                for choice in (0, 1, 2) if repeat % 2 == 0 else (2, 1, 0):
                    sync()
                    start = time.perf_counter()
                    run(choice != 0, reprepare=choice == 2)
                    label = ("fresh", "prepared", "prepare_and_scan")[choice]
                    row["samples_seconds"][label].append(time.perf_counter() - start)
            fresh = float(np.median(row["samples_seconds"]["fresh"]))
            cached = float(np.median(row["samples_seconds"]["prepared"]))
            warm_prepare = float(np.median(row["samples_seconds"]["warm_preparation"]))
            row.update(
                {
                    "fresh_scan_seconds": fresh,
                    "prepared_scan_seconds": cached,
                    "warm_scan_speedup": fresh / cached,
                    "warm_preparation_seconds": warm_prepare,
                    "first_prepare_plus_warm_scan_vs_warm_fresh_ratio": fresh
                    / (cached + prepare_seconds),
                    "warm_prepare_plus_scan_speedup": fresh / (cached + warm_prepare),
                    "measured_prepare_and_scan_speedup": fresh
                    / float(np.median(row["samples_seconds"]["prepare_and_scan"])),
                    "estimated_first_prepare_break_even_scans": (
                        int(np.floor(prepare_seconds / (fresh - cached))) + 1
                        if fresh > cached
                        else None
                    ),
                    "amortized_warm_estimates": {
                        str(scans): scans * fresh / (warm_prepare + scans * cached)
                        for scans in (1, 4, 16, 64)
                    },
                }
            )
            save()
            print(json.dumps(row, allow_nan=False), flush=True)
        # An explicit lifetime check proves the snapshot does not keep either
        # the waveform owner or its high-memory amplitude provider alive.
        model_ref, amplitude_ref = (
            weakref.ref(model),
            weakref.ref(model.amplitude_generator),
        )
        model = None
        gc.collect()
        report["owner_released"] = model_ref() is None
        report["amplitude_provider_released"] = amplitude_ref() is None
        if not report["owner_released"] or not report["amplitude_provider_released"]:
            raise AssertionError(
                "Prepared snapshot retains the owner/amplitude provider"
            )
        theta, phi, dist = OBSERVERS[0]
        report["post_owner_release"] = compare(
            saved, host(prepared(theta, phi, dist=dist))
        )
        if backend.uses_cuda:
            report["telemetry_after"] = telemetry()
        report["passes"] = True
        save()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
