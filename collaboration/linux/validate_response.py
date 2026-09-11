"""Source-bound real/synthetic response verification and fair CPU batch timing."""

# 2026-09-11 21:20 CST (linux): Serial frozen-input response experiment. Fresh
# FEW holdouts are exported separately. Preserve all sources, native files/data.
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits
from lisatools.utils.utility import AET

from prepared_tdi import PreparedTDI, make_response


def identity(path):
    path = Path(path).resolve()
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    here = Path(__file__).resolve().parent
    # 2026-09-11 21:24 CST (linux): Original slower registered/hash-every-plan
    # formal run and its exact producers remain; use a distinct final filename.
    output = here / "response_immutable_final.json"
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=False,
        passes=False,
        sources=[
            identity(p)
            for p in (
                Path(__file__),
                here / "prepared_tdi.py",
                here.parent.parent / "knowledge/LISAResponse.cu",
                Path(inspect.getsourcefile(pyResponseTDI)),
                Path(inspect.getsourcefile(ESAOrbits)),
                Path(inspect.getsourcefile(AET)),
            )
        ],
        native=[
            identity(importlib.import_module(x).__file__)
            for x in (
                "fastlisaresponse_backend_cpu.responselisa",
                "lisatools_backend_cpu.pycppdetector",
            )
        ],
        versions={
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "fastlisaresponse", "lisaanalysistools")
        },
        environment={
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        cases=[],
        timings=[],
        inputs=[],
        checks=[],
        metric="Fixed synthetic diagonal A/E/T PSD, not physical noise covariance",
    )
    saved = {}

    def check(label, ok):
        report["checks"].append(dict(label=label, passes=bool(ok)))
        if not ok:
            raise AssertionError(label)

    def authenticate(item):
        check(item["path"], identity(item["path"]) == item)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        holdout = json.loads((here / "holdout_final.json").read_text())
        check("fresh holdout terminal", holdout["terminal"] and holdout["passes"])
        authenticate(holdout["archive"])
        for entry in [holdout["producer"], *holdout["few_sources"], *holdout["native"]]:
            authenticate(entry)
        report["inputs"].append(identity(here / "holdout_final.json"))
        with np.load(holdout["archive"]["path"], allow_pickle=False) as a:
            newwaves = {k: a[k].copy() for k in a.files}
        rng = np.random.default_rng(16091)
        for case in ("prograde", "retrograde"):
            old = here.parents[2] / "exp15-single-pass-phase-basis/collaboration/linux"
            evidence = json.loads((old / f"real_{case}_final.json").read_text())
            check(
                "old input report terminal", evidence["passes"] and evidence["terminal"]
            )
            authenticate(evidence["archive"])
            authenticate(evidence["orbit"])
            report["inputs"].extend(
                (identity(old / f"real_{case}_final.json"), evidence["archive"])
            )
            with np.load(evidence["archive"]["path"], allow_pickle=False) as a:
                basis = a["new_ssb_basis"].copy()
                reference = a["ssb_reference"].copy()
            dt = 15.0
            n = len(reference)
            orbit = ESAOrbits(force_backend="cpu")
            check(
                "actual orbit filename",
                Path(orbit.filename).resolve()
                == Path(evidence["orbit"]["path"]).resolve(),
            )
            orbit.configure(linear_interp_setup=True)
            response = make_response(
                1 / dt,
                n,
                order=25,
                tdi="2nd generation",
                tdi_chan="AET",
                orbits=orbit,
                force_backend="cpu",
            )
            # This stock response/configuration setup is common to both paths.
            response.get_projections(
                reference.conj().copy(), 0.7, np.pi / 2 - 1.1, t0=20000.0
            )
            # 2026-09-11 21:26 CST (linux): Compare immutable initialization
            # with a separate unmodified public instance before sparse tests.
            plain = pyResponseTDI(
                1 / dt,
                n,
                order=25,
                tdi="2nd generation",
                tdi_chan="AET",
                orbits=orbit,
                force_backend="cpu",
            )
            for wave in (reference, basis[3], newwaves[f"{case}_0"]):
                for r in (plain, response):
                    r.get_projections(
                        wave.conj().copy(), 0.7, np.pi / 2 - 1.1, t0=20000.0
                    )
                check(
                    "plain/frozen native projection byte equality",
                    plain.y_gw.tobytes() == response.y_gw.tobytes(),
                )
                check(
                    "plain/frozen native TDI byte equality",
                    np.stack(plain.get_tdi_delays()).tobytes()
                    == np.stack(response.get_tdi_delays()).tobytes(),
                )
            del plain
            start = perf_counter()
            plan = PreparedTDI(response)
            preparation = perf_counter() - start
            calls = {"projection": 0, "stock_tdi": 0, "sparse_tdi": 0}

            def respond(wave, variant, prepared=None):
                response.get_projections(
                    np.ascontiguousarray(wave.conj()), 0.7, np.pi / 2 - 1.1, t0=20000.0
                )
                calls["projection"] += 1
                if variant == "stock":
                    full = response.get_tdi_delays()
                    calls["stock_tdi"] += 1
                    channels = np.stack([x[plan.cut : -plan.cut].copy() for x in full])
                else:
                    channels = prepared(response.y_gw)
                    calls["sparse_tdi"] += 1
                if not np.all(np.isfinite(channels)):
                    raise ValueError("Nonfinite TDI")
                return dt * np.fft.rfft(channels, axis=-1)

            ref = respond(reference, "stock")
            freq = np.fft.rfftfreq(plan.nt, dt)
            endpoint = np.ones(len(freq))
            endpoint[0] = 0.5
            if plan.nt % 2 == 0:
                endpoint[-1] = 0.5
            psd = (
                1e-42
                * np.array([1.0, 2.0, 3.0])[:, None]
                * (1 + (freq[None, :] / 0.01) ** 2)
            )
            weights = 4 / (plan.nt * dt) * endpoint / psd
            weights /= np.sum(weights * np.abs(ref) ** 2)
            root_weights = np.sqrt(weights)

            def whiten(z):
                z = z * root_weights
                return np.concatenate((z.real.ravel(), z.imag.ravel()))

            ref_unit = whiten(ref)
            noise = rng.normal(size=ref.shape) + 1j * rng.normal(size=ref.shape)
            noise[:, 0] = noise[:, 0].real
            if plan.nt % 2 == 0:
                noise[:, -1] = noise[:, -1].real
            noise = np.concatenate((noise.real.ravel(), noise.imag.ravel()))
            waves = np.concatenate(
                (basis, np.stack([newwaves[f"{case}_{i}"] for i in range(3)]))
            )
            native = []
            sparse = []
            rows = []
            for index, wave in enumerate(waves):
                old_tdi = respond(wave, "stock")
                new_tdi = respond(wave, "sparse", plan)
                old_unit = whiten(old_tdi)
                new_unit = whiten(new_tdi)
                relative = float(
                    np.linalg.norm(new_unit - old_unit) / np.linalg.norm(old_unit)
                )
                channels = [
                    float(np.linalg.norm(a - b) / np.linalg.norm(b))
                    for a, b in zip(new_tdi, old_tdi)
                ]
                check("actual all channels", relative <= 1e-9 and max(channels) <= 1e-9)
                likelihood = []
                for rho in (20.0, 100.0, 1000.0):
                    data = (rho * ref_unit + noise).astype(np.longdouble)
                    r0 = data - rho * ref_unit.astype(np.longdouble)
                    ro = data - rho * old_unit.astype(np.longdouble)
                    rn = data - rho * new_unit.astype(np.longdouble)
                    lo = float(-0.5 * np.sum(ro**2 - r0**2))
                    ln = float(-0.5 * np.sum(rn**2 - r0**2))
                    error = abs(ln - lo)
                    limit = 1e-5 + 2e-10 * abs(lo)
                    check("actual likelihood", error <= limit)
                    likelihood.append(
                        dict(rho=rho, stock=lo, sparse=ln, error=error, limit=limit)
                    )
                rows.append(
                    dict(
                        index=index,
                        kind="basis" if index < 13 else "fresh_FEW_holdout",
                        relative_l2=relative,
                        channel_relative_l2=channels,
                        likelihood=likelihood,
                    )
                )
                native.append(old_unit)
                sparse.append(new_unit)
            saved.update(
                {
                    case + "_stock": np.stack(native),
                    case + "_sparse": np.stack(sparse),
                    case + "_reference": ref_unit,
                    case + "_noise": noise,
                }
            )
            report["cases"].append(
                dict(
                    case=case,
                    rows=rows,
                    plan=plan.metadata,
                    preparation_seconds=preparation,
                    orbit=evidence["orbit"],
                    calls_before_timing=calls.copy(),
                )
            )
            # Independent synthetic probes at changed sky, sampling and order.
            if case == "prograde":
                for order, newdt, t0 in ((15, 7.5, 15000.0), (35, 30.0, 36000.0)):
                    rr = make_response(
                        1 / newdt,
                        n,
                        order=order,
                        tdi="2nd generation",
                        tdi_chan="AET",
                        orbits=orbit,
                        force_backend="cpu",
                    )
                    time = np.arange(n) * newdt
                    chirp = 1e-23 * np.sin(0.002 * time + 1e-9 * time * time)
                    probes = (
                        np.zeros(n, dtype=complex),
                        chirp.astype(complex),
                        1j * chirp,
                    )
                    rr.get_projections(probes[0], 1.9, -0.5, t0=t0)
                    pp = PreparedTDI(rr)
                    extra = []
                    for wave in probes:
                        rr.get_projections(wave.copy(), 1.9, -0.5, t0=t0)
                        truth = np.stack(
                            [x[pp.cut : -pp.cut].copy() for x in rr.get_tdi_delays()]
                        )
                        actual = pp(rr.y_gw)
                        err = float(
                            np.linalg.norm(actual - truth)
                            / max(np.linalg.norm(truth), 1e-300)
                        )
                        ch = [
                            float(
                                np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
                            )
                            for a, b in zip(actual, truth)
                        ]
                        check("changed sky/order/dt", err <= 1e-9 and max(ch) <= 1e-9)
                        extra.append(dict(relative_l2=err, channels=ch))
                    report.setdefault("synthetic", []).append(
                        dict(order=order, dt=newdt, t0=t0, rows=extra, plan=pp.metadata)
                    )

                def batch(variant, count, cached):
                    pp = plan if cached else None
                    if variant == "sparse" and not cached:
                        pp = PreparedTDI(response)
                    return np.stack(
                        [
                            whiten(
                                respond(
                                    w, "stock" if variant == "stock" else "sparse", pp
                                )
                            )
                            for w in basis[:count]
                        ]
                    )

                for count in (1, 13):
                    for cached in (False, True):
                        batch("stock", count, cached)
                        batch("sparse", count, cached)
                        timings = {"stock": [], "sparse": []}
                        errors = []
                        counters = []
                        for repeat in range(5):
                            order = (
                                ("stock", "sparse")
                                if repeat % 2 == 0
                                else ("sparse", "stock")
                            )
                            results = {}
                            for variant in order:
                                before = calls.copy()
                                start = perf_counter()
                                results[variant] = batch(variant, count, cached)
                                timings[variant].append(perf_counter() - start)
                                counters.append(
                                    dict(
                                        variant=variant,
                                        counts={k: calls[k] - before[k] for k in calls},
                                    )
                                )
                            error = float(
                                np.linalg.norm(results["sparse"] - results["stock"])
                                / np.linalg.norm(results["stock"])
                            )
                            check("timed result", error <= 1e-9)
                            errors.append(error)
                        item = dict(
                            count=count,
                            preparation_included=not cached,
                            seconds=timings,
                            errors=errors,
                            calls=counters,
                            speed_ratio=float(
                                np.median(timings["stock"])
                                / np.median(timings["sparse"])
                            ),
                        )
                        report["timings"].append(item)
                        print(
                            json.dumps(
                                dict(
                                    count=count,
                                    cached=cached,
                                    ratio=item["speed_ratio"],
                                    medians={
                                        k: float(np.median(v))
                                        for k, v in timings.items()
                                    },
                                )
                            ),
                            flush=True,
                        )
                        save()
            report["cases"][-1]["calls_after_timing"] = calls.copy()
            save()
        np.savez_compressed(archive, **saved)
        report["archive"] = identity(archive)
        report["timing_scope"] = (
            "CPU fixed preloaded stock response/orbit with immutable geometry; SSB arrays already available. Both include stock projection, TDI or sparse application, copy/FFT/whitening/stack. Fresh-plan rows additionally include full sparse construction with immutable argument identity/shape/dtype/stride checks. No FEW generation, orbit/model load, GPU or cold setup claim."
        )
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        save()
        print(json.dumps(dict(passes=report["passes"], terminal=True)), flush=True)


if __name__ == "__main__":
    main()
