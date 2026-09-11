"""Real ESA-orbit second-generation TDI and conditional-distance reuse."""

# 2026-09-11 14:35 CST (linux): Actual A/E/T signals, explicitly synthetic
# fixed Gaussian metric. Timing is conditional likelihood, not whole FEW speed.
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits

from distance_likelihood import (
    DistanceConditionalLikelihood,
    one_sided_weights,
    weighted_inner,
)


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def longdouble_residual(distance, data, signal, weights):
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise RuntimeError("No genuinely wider longdouble oracle on this host")
    ld = np.longdouble
    alpha = ld(1) / ld(distance)
    w = weights.astype(ld)
    current, reference = ld(0), ld(0)
    for d, s in ((data.real, signal.real), (data.imag, signal.imag)):
        d, s = d.astype(ld), s.astype(ld)
        current += np.sum(w * (d - alpha * s) ** 2, dtype=ld)
        reference += np.sum(w * (d - s) ** 2, dtype=ld)
    return float(-0.5 * (current - reference))


def longdouble_actual_residual(candidate, data, signal, weights):
    """Independent norms using a freshly generated FEW -> TDI -> FFT signal."""
    ld = np.longdouble
    w = weights.astype(ld)
    current, reference = ld(0), ld(0)
    for c, d, s in (
        (candidate.real, data.real, signal.real),
        (candidate.imag, data.imag, signal.imag),
    ):
        c, d, s = c.astype(ld), d.astype(ld), s.astype(ld)
        current += np.sum(w * (d - c) ** 2, dtype=ld)
        reference += np.sum(w * (d - s) ** 2, dtype=ld)
    return float(-0.5 * (current - reference))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    output, artifact = (
        args.output_prefix.with_suffix(".json"),
        args.output_prefix.with_suffix(".npz"),
    )
    if output.exists() or artifact.exists():
        raise FileExistsError(output)
    meta_path, input_path = (
        args.input_prefix.with_suffix(".json"),
        args.input_prefix.with_suffix(".npz"),
    )
    meta = json.loads(meta_path.read_text())
    if (
        not meta["passes"]
        or identity(input_path)["sha256"] != meta["artifact"]["sha256"]
    ):
        raise AssertionError("FEW bridge did not pass or artifact identity changed")
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        scope="Fixed detector-frame masses, sky, orientation, ESA orbit response, sampling, data and Gaussian metric. Not distance marginalization or physical LISA noise validation.",
        input=identity(input_path),
        input_report=identity(meta_path),
        sources=[
            identity(__file__),
            identity(Path(__file__).with_name("distance_likelihood.py")),
            identity(inspect.getsourcefile(pyResponseTDI)),
            identity(inspect.getsourcefile(ESAOrbits)),
        ],
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "fastlisaresponse",
                "lisaanalysistools",
                "h5py",
                "gpubackendtools",
            )
        },
        host=platform.platform(),
        tdi="2nd generation",
        channels=["A", "E", "T"],
        backend="cpu",
        lagrange_order=25,
        buffer_requested_seconds=20000.0,
        response_scaling=[],
        likelihood_checks=[],
    )
    report["native"] = [
        identity(importlib.import_module(module).__file__)
        for module in (
            "lisatools_backend_cpu.pycppdetector",
            "fastlisaresponse_backend_cpu.responselisa",
        )
    ]

    report["execution_environment"] = {
        name: os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
    }
    report["numpy_configuration"] = np.show_config(mode="dicts")
    report["noise_generation"] = (
        "Independent N(0,1/w) real/imaginary components at interior bins; real-only DC/Nyquist. Consistent with -0.5*sum(w*abs(residual)^2)."
    )

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        with np.load(input_path, allow_pickle=False) as loaded:
            waves = [
                loaded[f"strain_{i}"].copy() for i in range(len(meta["distances_Gpc"]))
            ]
        for wave, row in zip(waves, meta["runs"]):
            if hashlib.sha256(wave.tobytes()).hexdigest() != row["sha256"]:
                raise AssertionError("Individual FEW bridge strain hash mismatch")
        dt = meta["settings"]["dt"]
        qS, phiS = meta["parameters"][7:9]
        orbit_file = (
            Path(inspect.getsourcefile(ESAOrbits)).parent
            / "orbit_files/esa-trailing-orbits.h5"
        )
        if not orbit_file.is_file():
            raise FileNotFoundError("Packaged ESA orbit absent; no automatic download")
        report["orbit"] = identity(orbit_file)
        start = perf_counter()
        orbits = ESAOrbits(force_backend="cpu")
        if Path(orbits.filename).resolve() != orbit_file.resolve():
            raise AssertionError("Unexpected orbit file resolution")
        orbits.configure(linear_interp_setup=True)
        response = pyResponseTDI(
            1 / dt,
            len(waves[0]),
            order=25,
            tdi="2nd generation",
            orbits=orbits,
            tdi_chan="AET",
            force_backend="cpu",
        )
        report["orbit_and_response_setup_seconds"] = perf_counter() - start

        def respond(wave):
            response.get_projections(
                np.ascontiguousarray(wave.conj()), phiS, np.pi / 2 - qS, t0=20000.0
            )
            cut = response.tdi_start_ind
            if cut <= 0 or 2 * cut >= wave.size:
                raise AssertionError("No valid nonempty TDI interior")
            channels = np.stack(
                [channel[cut:-cut].copy() for channel in response.get_tdi_delays()]
            )
            if (
                channels.shape[0] != 3
                or channels.dtype != np.float64
                or not np.all(np.isfinite(channels))
            ):
                raise AssertionError("Invalid TDI channels")
            return channels

        all_tdi = []
        for index, (distance, wave) in enumerate(zip(meta["distances_Gpc"], waves)):
            start = perf_counter()
            channels = respond(wave)
            seconds = perf_counter() - start
            all_tdi.append(channels)
            reference = all_tdi[0] / distance
            errors = []
            for actual, expected in zip(channels, reference):
                scale = np.max(np.abs(expected))
                if not scale > 0:
                    raise AssertionError("Unexpected identically zero TDI channel")
                error = float(
                    np.linalg.norm((actual - expected) / scale)
                    / np.linalg.norm(expected / scale)
                )
                if not np.isfinite(error) or error > 2e-10:
                    raise AssertionError("FEW to TDI distance scaling failed")
                errors.append(error)
            report["response_scaling"].append(
                dict(
                    distance_Gpc=distance,
                    relative_l2_per_channel=errors,
                    response_seconds=seconds,
                    samples=channels.shape[1],
                )
            )
            save()
        reference_tdi = all_tdi[0]
        n_time = reference_tdi.shape[1]
        report["crop"] = dict(
            samples_each_end=response.tdi_start_ind,
            first_time_seconds=response.tdi_start_ind * dt,
            samples=n_time,
            last_time_seconds=(response.tdi_start_ind + n_time - 1) * dt,
        )
        signal = dt * np.fft.rfft(reference_tdi, axis=-1)
        actual_signals = [dt * np.fft.rfft(channel, axis=-1) for channel in all_tdi]
        report["full_chain_likelihood_checks"] = []
        frequencies = np.fft.rfftfreq(n_time, dt)
        base_psd = (
            1e-42
            * np.asarray([1.0, 2.0, 3.0])[:, None]
            * (1 + (frequencies[None, :] / 0.01) ** 2)
        )
        base_weights = one_sided_weights(base_psd, dt, n_time)
        report["metric"] = (
            "Synthetic diagonal A/E/T one-sided PSD = 1e-42*[1,2,3]*(1+(f/0.01Hz)^2), globally rescaled per target SNR. This is a chosen Gaussian fixture, NOT the ESA constellation's physical noise covariance. No window or additional frequency mask."
        )
        rng = np.random.default_rng(93099)
        for rho in (20.0, 100.0, 1000.0):
            psd = base_psd * (weighted_inner(signal, signal, base_weights) / rho**2)
            weights = one_sided_weights(psd, dt, n_time)
            gaussian = (
                rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
            ) / np.sqrt(weights)
            # 2026-09-11 14:39 CST (linux): Match -0.5*w*(Re^2+Im^2):
            # each unconstrained real component has variance 1/w. Real-FFT
            # endpoints have only one degree of freedom, not a complex draw.
            gaussian[:, 0] = gaussian[:, 0].real
            if n_time % 2 == 0:
                gaussian[:, -1] = gaussian[:, -1].real
            for residual_name, residual in (
                ("zero", np.zeros_like(signal)),
                ("parallel", 0.01 * signal),
                ("gaussian", gaussian),
            ):
                data = signal + residual
                model = DistanceConditionalLikelihood.from_data(
                    1.0, data, signal, weights
                )
                distances = [1 / (1 + j / rho) for j in (-10, -3, -1, 0, 1, 3, 10)] + [
                    1.0 - 1e-6,
                    1.0 + 1e-6,
                    0.1,
                    10.0,
                ]
                for distance in distances:
                    expected = longdouble_residual(distance, data, signal, weights)
                    actual = model(distance)
                    delta = (1 - distance) / distance
                    scale = (
                        abs(delta * model.residual_projection)
                        + 0.5 * delta**2 * model.signal_norm2
                    )
                    gate = (
                        1e-8
                        if abs(delta) * rho <= 10.000000001
                        else 1e-12 * max(1.0, scale)
                    )
                    error = abs(actual - expected)
                    row = dict(
                        rho=rho,
                        residual=residual_name,
                        distance=distance,
                        delta_logl=actual,
                        oracle=expected,
                        absolute_error=error,
                        gate=gate,
                        passes=error <= gate,
                    )
                    report["likelihood_checks"].append(row)
                    if not row["passes"]:
                        raise AssertionError("Real TDI likelihood oracle mismatch")
                for distance, candidate in zip(meta["distances_Gpc"], actual_signals):
                    expected = longdouble_actual_residual(
                        candidate, data, signal, weights
                    )
                    actual = model(distance)
                    delta = (1 - distance) / distance
                    scale = (
                        abs(delta * model.residual_projection)
                        + 0.5 * delta**2 * model.signal_norm2
                    )
                    gate = (
                        1e-8
                        if abs(delta) * rho <= 10.000000001
                        else 1e-12 * max(1.0, scale)
                    )
                    error = abs(actual - expected)
                    row = dict(
                        rho=rho,
                        residual=residual_name,
                        distance=distance,
                        delta_logl=actual,
                        freshly_generated_oracle=expected,
                        absolute_error=error,
                        gate=gate,
                        passes=error <= gate,
                    )
                    report["full_chain_likelihood_checks"].append(row)
                    if not row["passes"]:
                        raise AssertionError(
                            "Fresh full-chain distance likelihood mismatch"
                        )
        # One fair repeated workload: both paths already own a cached frequency
        # waveform and prewhitened data. Baseline does no validation per bin.
        weights = base_weights * (
            100.0**2 / weighted_inner(signal, signal, base_weights)
        )
        timing_noise = (
            rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
        ) / np.sqrt(weights)
        timing_noise[:, 0] = timing_noise[:, 0].real
        if n_time % 2 == 0:
            timing_noise[:, -1] = timing_noise[:, -1].real
        data = signal + timing_noise
        model = DistanceConditionalLikelihood.from_data(1.0, data, signal, weights)

        def prepare_direct():
            root_weights = np.sqrt(weights)
            s, d = signal * root_weights, data * root_weights
            r = d - s
            return s, d, float(np.vdot(r, r).real)

        whitened_signal, whitened_data, reference_norm = prepare_direct()

        def direct_from(s, d, norm0):
            values = []
            for distance in distances:
                residual = d - s * (1.0 / distance)
                values.append(-0.5 * (float(np.vdot(residual, residual).real) - norm0))
            return np.asarray(values)

        def direct_scan():
            return direct_from(whitened_signal, whitened_data, reference_norm)

        def direct_batch_from(s, d, norm0):
            # Give the original O(N_distance*N_bin) algorithm a vectorized
            # implementation too; do not attribute all Python-loop savings to
            # sufficient statistics. Fixed chunks bound peak temporary memory.
            values = []
            for start in range(0, len(distances), 32):
                alpha = (1.0 / distances[start : start + 32])[:, None, None]
                residual = d[None, :, :] - alpha * s[None, :, :]
                norm = np.sum(residual.real**2 + residual.imag**2, axis=(1, 2))
                values.append(-0.5 * (norm - norm0))
            return np.concatenate(values)

        def direct_batch_scan():
            return direct_batch_from(whitened_signal, whitened_data, reference_norm)

        def scalar_scan():
            return np.asarray([model(distance) for distance in distances])

        def prepared_quadratic_scalar():
            prepared = DistanceConditionalLikelihood.from_data(
                1.0, data, signal, weights
            )
            return np.asarray([prepared(distance) for distance in distances])

        functions = {
            "cached_waveform_direct_residual": direct_scan,
            "cached_waveform_direct_batched": direct_batch_scan,
            "quadratic_scalar": scalar_scan,
            "quadratic_vector": lambda: model(distances),
            "prepare_and_direct": lambda: direct_from(*prepare_direct()),
            "prepare_and_direct_batched": lambda: direct_batch_from(*prepare_direct()),
            "prepare_and_quadratic_scalar": prepared_quadratic_scalar,
            "prepare_and_quadratic_vector": lambda: (
                DistanceConditionalLikelihood.from_data(1.0, data, signal, weights)(
                    distances
                )
            ),
        }
        report["timing"] = dict(
            scope="Fixed-distance conditional scans from the same cached frequency-domain TDI waveform; excludes FEW, orbit/response setup, FFT and data creation for all paths. prepare_and_* directly times its preparation plus scan, not a sum of separate medians.",
            rounds=9,
            vectorized_direct_chunk_size=32,
            workloads=[],
        )
        for count in (1, 4, 16, 256):
            distances = 1 / (1 + np.linspace(-0.1, 0.1, count))
            for _ in range(2):
                for fn in functions.values():
                    fn()
            timings = {key: [] for key in functions}
            reference_values = direct_scan()
            max_errors = {key: 0.0 for key in functions}
            for repeat in range(9):
                names = (
                    list(functions) if repeat % 2 == 0 else list(reversed(functions))
                )
                for name in names:
                    start = perf_counter()
                    values = functions[name]()
                    seconds = perf_counter() - start
                    timings[name].append(seconds)
                    error = float(np.max(np.abs(values - reference_values)))
                    max_errors[name] = max(max_errors[name], error)
                    if (
                        values.shape != reference_values.shape
                        or not np.all(np.isfinite(values))
                        or error > 1e-8
                    ):
                        raise AssertionError("Timed likelihood output mismatch")
            medians = {key: float(np.median(value)) for key, value in timings.items()}
            report["timing"]["workloads"].append(
                dict(
                    distances=count,
                    raw_seconds=timings,
                    median_seconds=medians,
                    max_delta_logl_difference=max_errors,
                    scalar_speedup=medians["cached_waveform_direct_residual"]
                    / medians["quadratic_scalar"],
                    vector_speedup=min(
                        medians["cached_waveform_direct_residual"],
                        medians["cached_waveform_direct_batched"],
                    )
                    / medians["quadratic_vector"],
                    scalar_preparation_inclusive_speedup=medians["prepare_and_direct"]
                    / medians["prepare_and_quadratic_scalar"],
                    vector_preparation_inclusive_speedup=min(
                        medians["prepare_and_direct"],
                        medians["prepare_and_direct_batched"],
                    )
                    / medians["prepare_and_quadratic_vector"],
                )
            )
        np.savez_compressed(
            artifact,
            tdi=np.stack(all_tdi),
            distance_Gpc=np.asarray(meta["distances_Gpc"]),
            frequencies_Hz=frequencies,
            likelihood_distances=distances,
            direct_delta_logl=reference_values,
            quadratic_delta_logl=model(distances),
        )
        report["artifact"] = identity(artifact)
        report["passes"] = True
        save()
        print(
            json.dumps(
                dict(
                    response_scaling=report["response_scaling"],
                    checks=len(report["likelihood_checks"]),
                    full_chain_checks=len(report["full_chain_likelihood_checks"]),
                    timings=[
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("raw_seconds", "max_delta_logl_difference")
                        }
                        for row in report["timing"]["workloads"]
                    ],
                ),
                indent=2,
            ),
            flush=True,
        )
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
