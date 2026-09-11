"""New-vs-old phase preparation with one shared real FEW/response service."""

# 2026-09-11 17:33 CST (linux): Replace this branch's unexecuted exp14 producer
# copy with the bounded exp15 A/B. Original exp14 source/evidence stay untouched.
# Both paths transfer the same array count in one request and run real TDI.
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
from single_pass_basis import phase_coefficients


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def samples_to_basis(samples):
    n = len(samples)
    transformed = np.fft.fft(samples, axis=0) / n
    m = (n - 1) // 2
    # Input SSB is COMPLEX, so use both signed Fourier coefficients; rfft is
    # only valid after converting to a real data representation.
    return np.concatenate(
        (
            transformed[:1],
            transformed[1 : m + 1] + transformed[:m:-1],
            1j * (transformed[1 : m + 1] - transformed[:m:-1]),
        ),
        axis=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("prograde", "retrograde"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    archive = args.output.with_suffix(".npz")
    if args.output.exists() or archive.exists():
        raise FileExistsError("Reports must be new")
    here = Path(__file__).resolve().parent
    sibling = here.parents[2]
    worker_env = dict(
        os.environ,
        PYTHONPATH=str(sibling / "master-reference/src")
        + ":"
        + str(sibling / "artifacts/runtime-master"),
        LD_LIBRARY_PATH="/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib",
        FEW_ENABLED_BACKENDS="cpu",
    )
    command = [
        str(sibling.parent / "FastEMRIWaveforms/.venv/bin/python"),
        str(here / "few_phase_server.py"),
        "--case",
        args.case,
        "--data-dir",
        "/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download",
    ]
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=False,
        passes=False,
        case=args.case,
        benchmark=args.benchmark,
        command=command,
        platform=platform.platform(),
        sources=[
            identity(here / name)
            for name in (
                "phase_real_probe.py",
                "single_pass_basis.py",
                "phase_likelihood.py",
                "few_phase_server.py",
            )
        ]
        + [identity(inspect.getsourcefile(x)) for x in (pyResponseTDI, ESAOrbits)],
        native=[
            identity(importlib.import_module(x).__file__)
            for x in (
                "lisatools_backend_cpu.pycppdetector",
                "fastlisaresponse_backend_cpu.responselisa",
            )
        ],
        versions={
            x: importlib.metadata.version(x)
            for x in ("numpy", "scipy", "h5py", "fastlisaresponse", "lisaanalysistools")
        },
        environment={
            x: os.environ.get(x)
            for x in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        metric="Fixed SYNTHETIC diagonal A/E/T one-sided PSD, not physical LISA noise covariance",
        preparation_timings=[],
        scan_timings=[],
        cached_timings=[],
        accuracy=[],
        operations=[],
    )
    worker = None
    arrays = {}

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=worker_env,
        )
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("FEW service failed during startup")
        meta = json.loads(line)
        report["few"] = meta
        if [x["sha256"] for x in meta["data"]] != [
            "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834",
            "db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014",
        ]:
            raise AssertionError("Wrong data")
        m, dt = meta["m"], meta["settings"]["dt"]
        if not 1 <= m <= 10:
            raise AssertionError("Unexpected phase degree")

        def request(operation, phase=0.0):
            worker.stdin.write(
                json.dumps(dict(operation=operation, phase=float(phase))) + "\n"
            )
            worker.stdin.flush()
            result = worker.stdout.readline()
            if not result:
                raise RuntimeError("FEW operation failed")
            result = json.loads(result)
            value = (
                np.frombuffer(base64.b64decode(result.pop("wave")), dtype=np.complex128)
                .reshape(result["shape"])
                .copy()
            )
            if not np.all(np.isfinite(value)):
                raise AssertionError("Nonfinite FEW reply")
            if operation == "grouped":
                if not all(
                    result["metadata"][k] == 1
                    for k in ("trajectory_calls", "amplitude_calls", "outer_calls")
                ):
                    raise AssertionError("Grouped path repeated intrinsic evaluation")
            elif operation == "blackbox":
                if result["metadata"]["outer_calls"] != 2 * m + 1:
                    raise AssertionError("Blackbox count mismatch")
            report["operations"].append(result)
            return value, result

        orbit_file = (
            Path(inspect.getsourcefile(ESAOrbits)).parent
            / "orbit_files/esa-trailing-orbits.h5"
        )
        if not orbit_file.is_file():
            raise FileNotFoundError("No orbit downloads")
        report["orbit"] = identity(orbit_file)
        orbits = ESAOrbits(force_backend="cpu")
        if Path(orbits.filename).resolve() != orbit_file.resolve():
            raise AssertionError("Unexpected orbit")
        start = perf_counter()
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
        response_calls = 0

        def respond(wave):
            nonlocal response_calls
            if wave.shape != (meta["samples"],):
                raise AssertionError("Wrong SSB length")
            response.get_projections(
                np.ascontiguousarray(wave.conj()),
                meta["parameters"][8],
                np.pi / 2 - meta["parameters"][7],
                t0=20000.0,
            )
            cut = response.tdi_start_ind
            if cut <= 0 or 2 * cut >= wave.size:
                raise AssertionError("Invalid response crop")
            channels = np.stack([x[cut:-cut].copy() for x in response.get_tdi_delays()])
            if not np.all(np.isfinite(channels)):
                raise AssertionError("Invalid TDI")
            response_calls += 1
            return dt * np.fft.rfft(channels, axis=-1), channels.shape[1]

        ssb_reference, _ = request("wave")
        frequency_reference, nt = respond(ssb_reference)
        f = np.fft.rfftfreq(nt, dt)
        endpoint = np.ones(f.size)
        endpoint[0] = 0.5
        if nt % 2 == 0:
            endpoint[-1] = 0.5
        psd = (
            1e-42 * np.array([1.0, 2.0, 3.0])[:, None] * (1 + (f[None, :] / 0.01) ** 2)
        )
        weights = 4 / (nt * dt) * endpoint / psd
        weights /= np.sum(weights * np.abs(frequency_reference) ** 2)
        root_weights = np.sqrt(weights)

        def whiten(z):
            z = z * root_weights
            return np.concatenate((z.real.ravel(), z.imag.ravel()))

        reference = whiten(frequency_reference)
        report["crop"] = dict(
            samples=nt,
            samples_each_end=response.tdi_start_ind,
            first_time_seconds=response.tdi_start_ind * dt,
        )
        rng = np.random.default_rng(14091)
        random_noise = rng.normal(size=frequency_reference.shape) + 1j * rng.normal(
            size=frequency_reference.shape
        )
        random_noise[:, 0] = random_noise[:, 0].real
        if nt % 2 == 0:
            random_noise[:, -1] = random_noise[:, -1].real
        noise = np.concatenate((random_noise.real.ravel(), random_noise.imag.ravel()))
        data = 100 * reference + noise
        grid = 2 * np.pi * np.arange(2 * m + 1) / (2 * m + 1)

        def prepare(method):
            # Every invocation regenerates intrinsic state, including warmup.
            response_start = response_calls
            start = perf_counter()
            raw, worker_result = request(method)
            transfer_end = perf_counter()
            converted = np.stack([whiten(respond(row)[0]) for row in raw])
            response_end = perf_counter()
            if method == "blackbox":
                if (
                    raw[0].tobytes() != ssb_reference.tobytes()
                    or converted[0].tobytes() != reference.tobytes()
                ):
                    raise AssertionError("Blackbox reference anchor changed")
                basis = samples_to_basis(converted).real
            else:
                basis = converted
            diagnostic = PhaseLikelihood.from_basis(100 * basis, data, 100 * reference)
            compact = diagnostic.compact()
            end = perf_counter()
            return (
                diagnostic,
                compact,
                raw,
                basis,
                dict(
                    method=method,
                    joint_seconds=end - start,
                    worker_seconds=worker_result["operation_seconds"],
                    worker_transfer_seconds=transfer_end - start,
                    response_fft_seconds=response_end - transfer_end,
                    statistics_seconds=end - response_end,
                    response_calls=response_calls - response_start,
                    metadata=worker_result["metadata"],
                ),
            )

        # Same fresh preparation for both algorithms; no old benchmark timing reuse.
        old, old_compact, old_ssb, old_basis, old_cost = prepare("blackbox")
        new, new_compact, new_ssb, new_basis, new_cost = prepare("grouped")
        report["initial_preparation"] = [old_cost, new_cost]
        arrays.update(
            ssb_reference=ssb_reference,
            old_ssb_samples=old_ssb,
            new_ssb_basis=new_ssb,
            old_tdi_basis_unit=old_basis,
            new_tdi_basis_unit=new_basis,
            reference_unit=reference,
            noise=noise,
            grid=grid,
            modes=np.asarray(meta["modes"]),
        )
        expected_ssb_basis = samples_to_basis(old_ssb)
        report["ssb_basis_global_relative_l2"] = float(
            np.linalg.norm(new_ssb - expected_ssb_basis)
            / np.linalg.norm(expected_ssb_basis)
        )
        report["tdi_basis_global_relative_l2"] = float(
            np.linalg.norm(new_basis - old_basis) / np.linalg.norm(old_basis)
        )
        if (
            max(
                report["ssb_basis_global_relative_l2"],
                report["tdi_basis_global_relative_l2"],
            )
            > 1e-8
        ):
            raise AssertionError("New basis differs from blackbox basis")
        phases = np.concatenate(
            (
                rng.uniform(-2 * np.pi, 4 * np.pi, 16),
                [0.0, 1e-12, 1e-8, 1e-4, 2 * np.pi - 1e-8, 2 * np.pi + 1e-8],
            )
        )
        actual_ssb = []
        actual = []
        for phase in phases:
            wave, _ = request("wave", phase)
            actual_ssb.append(wave)
            actual.append(whiten(respond(wave)[0]))
        actual_ssb, actual = np.stack(actual_ssb), np.stack(actual)
        model_ssb = phase_coefficients(phases, m) @ new_ssb
        model_tdi = reference + deltas(phases, m) @ new_basis
        report["ssb_regeneration_relative_l2_max"] = float(
            np.max(
                np.linalg.norm(actual_ssb - model_ssb, axis=1)
                / np.linalg.norm(actual_ssb, axis=1)
            )
        )
        report["tdi_regeneration_relative_l2_max"] = float(
            np.max(
                np.linalg.norm(actual - model_tdi, axis=1)
                / np.linalg.norm(actual, axis=1)
            )
        )
        if (
            max(
                report["ssb_regeneration_relative_l2_max"],
                report["tdi_regeneration_relative_l2_max"],
            )
            > 1e-8
        ):
            raise AssertionError("Fresh full phase regeneration mismatch")
        # Nonzero REFERENCE phase tests catch omitted initial offsets/retrograde pi.
        offset = 0.37
        shifted, _ = request("grouped", offset)
        for delta in (0.0, 0.71, -0.4):
            wave, _ = request("wave", offset + delta)
            prediction = phase_coefficients(delta, m) @ shifted
            error = float(np.linalg.norm(wave - prediction) / np.linalg.norm(wave))
            report.setdefault("offset_reference_checks", []).append(
                dict(offset=offset, delta=delta, relative_l2=error)
            )
            if error > 1e-8:
                raise AssertionError("Nonzero reference phase mismatch")
        arrays.update(
            phases=phases, regenerated_ssb=actual_ssb, regenerated_unit=actual
        )
        for rho in (20.0, 100.0, 1000.0):
            observed = rho * reference + noise
            r0 = observed.astype(np.longdouble) - rho * reference.astype(np.longdouble)
            compact = PhaseLikelihood.from_basis(
                rho * new_basis, observed, rho * reference
            ).compact()
            for phase, truth in zip(phases, actual):
                residual = observed.astype(np.longdouble) - rho * truth.astype(
                    np.longdouble
                )
                expected = float(-0.5 * np.sum(residual**2 - r0**2))
                value = float(compact(phase))
                error = abs(value - expected)
                limit = 1e-5 + 2e-10 * abs(expected)
                report["accuracy"].append(
                    dict(
                        rho=rho,
                        phase=float(phase),
                        direct=expected,
                        grouped=value,
                        error=error,
                        limit=limit,
                        passes=error <= limit,
                    )
                )
                if not error <= limit:
                    raise AssertionError("Regenerated conditional likelihood failure")
        # A full wave call after basis creation must not change the saved arrays.
        before = hashlib.sha256(new_ssb.tobytes()).hexdigest()
        request("wave", 1.9)
        report["basis_ownership_passes"] = (
            hashlib.sha256(new_ssb.tobytes()).hexdigest() == before
        )
        points = np.random.default_rng(15915).uniform(0, 2 * np.pi, 256)
        if args.benchmark:
            report["timing_scope"] = (
                "Same warm CPU FEW/response service and fixed data, one batch request and identical array counts for each preparation. Includes fresh intrinsic generation, groups, transfer, real response, FFT, statistics. Common model/data/orbit/JIT excluded. Ratios compare exp15 with its freshly reproduced exp14-style baseline, not historical medians."
            )
            # Explicit full warmup for BOTH paths.
            for method in ("blackbox", "grouped"):
                prepare(method)
            for repeat in range(5):
                timings = {}
                values = {}
                order = (
                    ("blackbox", "grouped")
                    if repeat % 2 == 0
                    else ("grouped", "blackbox")
                )
                for method in order:
                    obj, compact, _, _, cost = prepare(method)
                    timings[method] = cost
                    values[method] = compact(points)
                err = float(np.max(np.abs(values["blackbox"] - values["grouped"])))
                if (
                    not all(np.all(np.isfinite(v)) for v in values.values())
                    or err > 2e-5
                ):
                    raise AssertionError("Preparation repeat outputs failed")
                row = dict(
                    repeat=repeat,
                    order=order,
                    timings=timings,
                    max_delta_logl_error=err,
                )
                report["preparation_timings"].append(row)
                print(
                    json.dumps(
                        dict(
                            case=args.case,
                            repeat=repeat,
                            blackbox=timings["blackbox"]["joint_seconds"],
                            grouped=timings["grouped"]["joint_seconds"],
                            error=err,
                        )
                    ),
                    flush=True,
                )
                save()
            for count in (1, 16, 256):
                chosen = points[:count]
                times = {"blackbox": [], "grouped": []}
                errors = []
                for repeat in range(3):
                    values = {}
                    order = (
                        ("blackbox", "grouped")
                        if repeat % 2 == 0
                        else ("grouped", "blackbox")
                    )
                    for method in order:
                        start = perf_counter()
                        _, compact, _, _, _ = prepare(method)
                        values[method] = compact(chosen)
                        times[method].append(perf_counter() - start)
                    error = float(
                        np.max(np.abs(values["blackbox"] - values["grouped"]))
                    )
                    errors.append(error)
                    if (
                        not all(np.all(np.isfinite(v)) for v in values.values())
                        or error > 2e-5
                    ):
                        raise AssertionError("Prepare+scan accuracy failure")
                report["scan_timings"].append(
                    dict(
                        count=count,
                        seconds=times,
                        errors=errors,
                        speed_ratio=float(
                            np.median(times["blackbox"]) / np.median(times["grouped"])
                        ),
                    )
                )

                # Strong same-already-prepared detector basis baseline.
                def direct():
                    residual = data - 100 * reference
                    output = []
                    for i in range(0, count, 32):
                        dh = deltas(chosen[i : i + 32], m) @ (100 * new_basis)
                        output.append(dh @ residual - 0.5 * np.sum(dh * dh, axis=1))
                    return np.concatenate(output)

                def compact_scan():
                    return PhaseLikelihood.from_basis(
                        100 * new_basis, data, 100 * reference
                    ).compact()(chosen)

                durations = {"direct": [], "compact": []}
                for repeat in range(7):
                    order = (("direct", direct), ("compact", compact_scan))
                    if repeat % 2:
                        order = order[::-1]
                    for name, fn in order:
                        start = perf_counter()
                        result = fn()
                        durations[name].append(perf_counter() - start)
                        if (
                            not np.all(np.isfinite(result))
                            or np.max(np.abs(result - new_compact(chosen))) > 1e-7
                        ):
                            raise AssertionError("Same-basis direct comparison failed")
                report["cached_timings"].append(
                    dict(
                        count=count,
                        seconds=durations,
                        speed_ratio=float(
                            np.median(durations["direct"])
                            / np.median(durations["compact"])
                        ),
                    )
                )
                save()
        arrays.update(compact_cosine=new_compact.cosine, compact_sine=new_compact.sine)
        np.savez_compressed(archive, **arrays)
        report["archive"] = identity(archive)
        report["response_calls"] = response_calls
        report["basis_shape"] = list(new_ssb.shape)
        report["compact_bytes"] = new_compact.cosine.nbytes + new_compact.sine.nbytes
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if worker is not None:
            if worker.poll() is None:
                try:
                    worker.stdin.write('{"quit":true}\n')
                    worker.stdin.flush()
                except BrokenPipeError:
                    pass
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
