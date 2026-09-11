"""Actual FEW->ESA TDI phase scan: joint prepare+scan and stronger cached baseline."""

# 2026-09-11 17:07 CST (linux): New evidence only. FEW remains on its original
# Python stack; response on the preexisting isolated CPU stack. No dependency
# installation, downloads or production mutation. Serial single-thread timing.
import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits

from phase_likelihood import PhaseLikelihood, deltas


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("prograde", "retrograde"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Do not overwrite evidence")
    sibling = Path(__file__).resolve().parents[3]
    project = sibling.parent / "FastEMRIWaveforms"
    script = Path(__file__).with_name("few_phase_server.py")
    env = dict(
        os.environ,
        PYTHONPATH=str(sibling / "master-reference/src")
        + ":"
        + str(sibling / "artifacts/runtime-master"),
        LD_LIBRARY_PATH="/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib",
        FEW_ENABLED_BACKENDS="cpu",
    )
    command = [
        str(project / ".venv/bin/python"),
        str(script),
        "--case",
        args.case,
        "--data-dir",
        "/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download",
    ]
    worker = None
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=False,
        passes=False,
        case=args.case,
        benchmark=args.benchmark,
        command=command,
        host=platform.platform(),
        sources=[
            identity(__file__),
            identity(Path(__file__).with_name("phase_likelihood.py")),
            identity(script),
            identity(inspect.getsourcefile(pyResponseTDI)),
            identity(inspect.getsourcefile(ESAOrbits)),
        ],
        native=[
            identity(importlib.import_module(n).__file__)
            for n in (
                "lisatools_backend_cpu.pycppdetector",
                "fastlisaresponse_backend_cpu.responselisa",
            )
        ],
        versions={
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "h5py", "fastlisaresponse", "lisaanalysistools")
        },
        environment={
            n: os.environ.get(n)
            for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        metric="Synthetic fixed diagonal one-sided A/E/T PSD, not physical ESA noise covariance or posterior validation",
        accuracy=[],
        full_chain_timings=[],
        cached_basis_timings=[],
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        worker = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
        )
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("FEW service startup failed")
        meta = json.loads(line)
        report["few"] = meta
        if [e["sha256"] for e in meta["data"]] != [
            "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834",
            "db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014",
        ]:
            raise AssertionError("Unexpected Kerr data")
        dt, m = meta["settings"]["dt"], meta["m"]
        if not 1 <= m <= 10:
            raise AssertionError("Unexpected harmonic band")
        orbit_file = (
            Path(inspect.getsourcefile(ESAOrbits)).parent
            / "orbit_files/esa-trailing-orbits.h5"
        )
        if not orbit_file.is_file():
            raise FileNotFoundError("No orbit download permitted")
        report["orbit"] = identity(orbit_file)
        start = perf_counter()
        orbits = ESAOrbits(force_backend="cpu")
        if Path(orbits.filename).resolve() != orbit_file.resolve():
            raise AssertionError("Orbit resolution changed")
        orbits.configure(linear_interp_setup=True)
        response = pyResponseTDI(
            1 / dt,
            meta["samples"],
            order=25,
            tdi="2nd generation",
            orbits=orbits,
            tdi_chan="AET",
            force_backend="cpu",
        )
        report["response_setup_seconds"] = perf_counter() - start
        calls = 0

        def signal(phase):
            nonlocal calls
            worker.stdin.write(json.dumps(dict(phase=float(phase))) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError("FEW service failed")
            wave = np.frombuffer(
                base64.b64decode(json.loads(line)["wave"]), dtype=np.complex128
            )
            if wave.size != meta["samples"]:
                raise AssertionError("Bad transfer size")
            # FEW uses h+ - i hx, real response interface expects conjugate.
            response.get_projections(
                np.ascontiguousarray(wave.conj()),
                meta["parameters"][8],
                np.pi / 2 - meta["parameters"][7],
                t0=20000.0,
            )
            cut = response.tdi_start_ind
            channels = np.stack([v[cut:-cut].copy() for v in response.get_tdi_delays()])
            if (
                channels.shape[0] != 3
                or not channels.size
                or not np.all(np.isfinite(channels))
            ):
                raise AssertionError("Invalid TDI")
            calls += 1
            return dt * np.fft.rfft(channels, axis=-1), channels.shape[1]

        initial, nt = signal(0.0)
        report["crop"] = dict(
            samples_each_end=response.tdi_start_ind,
            first_time_seconds=response.tdi_start_ind * dt,
            samples=nt,
        )
        f = np.fft.rfftfreq(nt, dt)
        psd = (
            1e-42
            * np.asarray([1.0, 2.0, 3.0])[:, None]
            * (1 + (f[None, :] / 0.01) ** 2)
        )
        endpoint = np.ones(f.size)
        endpoint[0] = 0.5
        if nt % 2 == 0:
            endpoint[-1] = 0.5
        weights = 4 / (nt * dt) * endpoint / psd
        weights /= np.sum(weights * np.abs(initial) ** 2)

        def whiten(value):
            z = value * np.sqrt(weights)
            return np.concatenate((z.real.ravel(), z.imag.ravel()))

        reference = whiten(initial)
        rng = np.random.default_rng(14091)
        noise_complex = rng.normal(size=initial.shape) + 1j * rng.normal(
            size=initial.shape
        )
        noise_complex[:, 0] = noise_complex[:, 0].real
        if nt % 2 == 0:
            noise_complex[:, -1] = noise_complex[:, -1].real
        noise = np.concatenate((noise_complex.real.ravel(), noise_complex.imag.ravel()))
        grid = 2 * np.pi * np.arange(2 * m + 1) / (2 * m + 1)
        start = perf_counter()
        samples = np.stack([whiten(signal(v)[0]) for v in grid])
        report["first_basis_generation_seconds"] = perf_counter() - start
        # 2026-09-11 17:14 CST (linux): Explicitly authenticate the reference
        # anchor, including every timed re-preparation. Preserve pilot sources.
        if not np.array_equal(samples[0], reference):
            raise AssertionError("Repeated zero-phase reference differs")
        report["reference_anchor_byte_equal"] = (
            samples[0].tobytes() == reference.tobytes()
        )
        phases = np.concatenate(
            (
                rng.uniform(-2 * np.pi, 4 * np.pi, 16),
                [0.0, 1e-12, 1e-8, 1e-4, 2 * np.pi - 1e-8, 2 * np.pi + 1e-8],
            )
        )
        actual = np.stack([whiten(signal(v)[0]) for v in phases])
        basis_unit = PhaseLikelihood.prepare(samples, reference + noise)
        reconstructed = reference + deltas(phases, m) @ basis_unit.basis
        report["waveform_whitened_relative_l2_max"] = float(
            np.max(
                np.linalg.norm(actual - reconstructed, axis=1)
                / np.linalg.norm(actual, axis=1)
            )
        )
        if report["waveform_whitened_relative_l2_max"] > 1e-8:
            raise AssertionError(
                "Actual FEW/response violates phase reconstruction gate"
            )
        for rho in (20.0, 100.0, 1000.0):
            data = rho * reference + noise
            p = PhaseLikelihood.prepare(rho * samples, data).compact()
            residual0 = data.astype(np.longdouble) - rho * reference.astype(
                np.longdouble
            )
            for phase, true in zip(phases, actual):
                residual = data.astype(np.longdouble) - rho * true.astype(np.longdouble)
                expected = float(-0.5 * np.sum(residual**2 - residual0**2))
                estimated = float(p(phase))
                error = abs(estimated - expected)
                limit = 1e-5 + 2e-10 * abs(expected)
                row = dict(
                    rho=rho,
                    phase=float(phase),
                    direct_delta_logl=expected,
                    fourier_delta_logl=estimated,
                    absolute_error=error,
                    limit=limit,
                    passes=error <= limit,
                )
                report["accuracy"].append(row)
                if not row["passes"]:
                    raise AssertionError("Regenerated full-chain likelihood mismatch")
        data = 100 * reference + noise
        prepared = PhaseLikelihood.prepare(100 * samples, data)
        timing_residual = data - 100 * reference
        report["noise_vs_rounded_residual_max_difference"] = float(
            np.max(np.abs(noise - timing_residual))
        )
        compact = prepared.compact()
        report["compact_payload_bytes"] = compact.cosine.nbytes + compact.sine.nbytes
        report["retained_preparation_payload_bytes_for_diagnostics"] = sum(
            x.nbytes
            for x in (
                prepared.basis,
                prepared.residual,
                prepared.projection,
                prepared.gram,
                prepared.fourier_cos,
                prepared.fourier_sin,
            )
        )
        test_x = deltas(0.71, m)
        diagonal = test_x @ prepared.projection - 0.5 * np.sum(
            test_x * test_x * np.diag(prepared.gram)
        )
        report["diagonal_gram_negative_control_abs_error"] = float(
            abs(diagonal - prepared(0.71))
        )
        if report["diagonal_gram_negative_control_abs_error"] < 1e-3:
            raise AssertionError("Weak cross-term negative control")
        if args.benchmark:
            report["timing_scope"] = (
                "Warm loaded FEW model and real response, fixed data/weights shared. Full chain includes FEW generation, pipe, response, FFT and preparation+scan measured jointly. Excludes process/model/orbit setup and initial JIT on BOTH sides. No shared trajectory/FEW-mode-basis baseline yet. Cached basis comparison starts with identical real whitened basis samples."
            )
            scan_rng = np.random.default_rng(14914)
            for count in (1, 16, 64, 256):
                points = scan_rng.uniform(0.0, 2 * np.pi, count)

                def full_direct():
                    result = []
                    for phase in points:
                        dh = 100 * whiten(signal(phase)[0]) - 100 * reference
                        result.append(float(dh @ timing_residual - 0.5 * (dh @ dh)))
                    return np.asarray(result)

                def full_prepared():
                    basis_samples = np.stack([100 * whiten(signal(v)[0]) for v in grid])
                    if basis_samples[0].tobytes() != (100 * reference).tobytes():
                        raise AssertionError("Timed reference anchor changed")
                    return PhaseLikelihood.prepare(basis_samples, data).compact()(
                        points
                    )

                durations = {"full_direct": [], "full_prepared": []}
                outputs = {}
                repeat_errors = []
                for repeat in range(3):
                    order = [
                        ("full_direct", full_direct),
                        ("full_prepared", full_prepared),
                    ]
                    if repeat % 2:
                        order.reverse()
                    for name, fn in order:
                        start = perf_counter()
                        outputs[name] = fn()
                        durations[name].append(perf_counter() - start)
                        if not np.all(np.isfinite(outputs[name])):
                            raise AssertionError("Nonfinite timing result")
                    repeat_error = float(
                        np.max(
                            np.abs(outputs["full_direct"] - outputs["full_prepared"])
                        )
                    )
                    repeat_errors.append(repeat_error)
                    if repeat_error > 2e-5:
                        raise AssertionError("Timed full-chain repeat differs")
                err = float(
                    np.max(np.abs(outputs["full_direct"] - outputs["full_prepared"]))
                )
                if err > 2e-5:
                    raise AssertionError("Timed full-chain outputs differ")
                row = dict(
                    count=count,
                    seconds=durations,
                    speed_ratio=float(
                        np.median(durations["full_direct"])
                        / np.median(durations["full_prepared"])
                    ),
                    max_delta_logl_error=err,
                    repeat_max_delta_logl_errors=repeat_errors,
                )
                report["full_chain_timings"].append(row)
                print(json.dumps(dict(case=args.case, **row)), flush=True)
                save()

                # Stronger baseline: same cached detector basis, direct residual
                # calculation is vectorized in chunks rather than a Python loop.
                def cached_direct():
                    # Direct competitor does NOT pay for unused Gram/Fourier.
                    spectrum = np.fft.rfft(100 * samples, axis=0) / len(samples)
                    basis = np.concatenate(
                        (
                            spectrum[:1].real,
                            2 * spectrum[1:].real,
                            -2 * spectrum[1:].imag,
                        ),
                        axis=0,
                    )
                    residual = data - 100 * samples[0]
                    result = []
                    for i in range(0, count, 32):
                        dh = deltas(points[i : i + 32], m) @ basis
                        result.append(dh @ residual - 0.5 * np.sum(dh * dh, axis=1))
                    return np.concatenate(result)

                def cached_fourier():
                    return PhaseLikelihood.prepare(100 * samples, data).compact()(
                        points
                    )

                times = {"direct": [], "fourier": []}
                for repeat in range(7):
                    order = [("direct", cached_direct), ("fourier", cached_fourier)]
                    if repeat % 2:
                        order.reverse()
                    for name, fn in order:
                        start = perf_counter()
                        result = fn()
                        times[name].append(perf_counter() - start)
                        if np.max(np.abs(result - prepared(points))) > 1e-7:
                            raise AssertionError("Cached baseline output mismatch")
                report["cached_basis_timings"].append(
                    dict(
                        count=count,
                        seconds=times,
                        speed_ratio=float(
                            np.median(times["direct"]) / np.median(times["fourier"])
                        ),
                    )
                )
                save()
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            grid=grid,
            samples_unit_snr=samples,
            phases=phases,
            regenerated_unit_snr=actual,
            reference_unit_snr=reference,
            noise=noise,
            modes=np.asarray(meta["modes"]),
            fourier_cos=prepared.fourier_cos,
            fourier_sin=prepared.fourier_sin,
        )
        report["archive"] = identity(args.output.with_suffix(".npz"))
        report["full_signal_calls"] = calls
        report["basis_shape"] = list(prepared.basis.shape)
        report["fourier_real_coefficients_including_constant"] = 4 * m + 1
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.stdin.write('{"quit":true}\n')
                worker.stdin.flush()
                worker.wait(timeout=30)
            report["worker_returncode"] = worker.returncode
            if worker.returncode:
                report["passes"] = False
        report["terminal"] = True
        save()
        print(
            json.dumps(dict(case=args.case, passes=report["passes"], terminal=True)),
            flush=True,
        )


if __name__ == "__main__":
    main()
