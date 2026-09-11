"""Include response/frozen-geometry construction in a stronger stock comparison."""

# 2026-09-11 21:29 CST (linux): Serial fresh-response AB/BA; configured ESA
# orbit is common. This is not cold OS/data load or FEW waveform generation.
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits

from prepared_tdi import make_response, PreparedTDI
from validate_response import identity


def main():
    here = Path(__file__).resolve().parent
    output = here / "lifecycle_final.json"
    if output.exists():
        raise FileExistsError(output)
    source = (
        here.parents[2]
        / "exp15-single-pass-phase-basis/collaboration/linux/real_prograde_final.json"
    )
    old = json.loads(source.read_text())
    assert old["terminal"] and old["passes"]
    assert identity(old["archive"]["path"]) == old["archive"]
    assert identity(old["orbit"]["path"]) == old["orbit"]
    formal = json.loads((here / "response_immutable_final.json").read_text())
    assert formal["terminal"] and formal["passes"]
    for entry in (*formal["sources"], *formal["native"]):
        assert identity(entry["path"]) == entry
    with np.load(old["archive"]["path"], allow_pickle=False) as a:
        basis = a["new_ssb_basis"].copy()
        reference = a["ssb_reference"].copy()
    orbit = ESAOrbits(force_backend="cpu")
    assert Path(orbit.filename).resolve() == Path(old["orbit"]["path"]).resolve()
    orbit.configure(linear_interp_setup=True)
    n = basis.shape[1]
    dt = 15.0
    cut = 1333
    r = pyResponseTDI(
        1 / dt,
        n,
        order=25,
        tdi="2nd generation",
        tdi_chan="AET",
        orbits=orbit,
        force_backend="cpu",
    )
    r.get_projections(reference.conj().copy(), 0.7, np.pi / 2 - 1.1, t0=20000.0)
    rf = dt * np.fft.rfft(np.stack([x[cut:-cut] for x in r.get_tdi_delays()]), axis=-1)
    nt = n - 2 * cut
    f = np.fft.rfftfreq(nt, dt)
    ep = np.ones(f.size)
    ep[0] = 0.5
    if nt % 2 == 0:
        ep[-1] = 0.5
    w = (
        4
        / (nt * dt)
        * ep
        / (1e-42 * np.array([1.0, 2.0, 3.0])[:, None] * (1 + (f[None, :] / 0.01) ** 2))
    )
    w /= np.sum(w * np.abs(rf) ** 2)
    rootw = np.sqrt(w)
    del r

    def run(variant, count):
        start = perf_counter()
        factory = pyResponseTDI if variant == "stock" else make_response
        response = factory(
            1 / dt,
            n,
            order=25,
            tdi="2nd generation",
            tdi_chan="AET",
            orbits=orbit,
            force_backend="cpu",
        )
        constructed = perf_counter() - start
        results = []
        plan = None
        prep = 0.0
        for wave in basis[:count]:
            response.get_projections(
                wave.conj().copy(), 0.7, np.pi / 2 - 1.1, t0=20000.0
            )
            if variant == "stock":
                # Stronger stock packing: no redundant per-channel .copy()
                # before stack; stack already produces owned output.
                values = np.stack([x[cut:-cut] for x in response.get_tdi_delays()])
            else:
                if plan is None:
                    before = perf_counter()
                    plan = PreparedTDI(response)
                    prep = perf_counter() - before
                values = plan(response.y_gw)
            freq = dt * np.fft.rfft(values, axis=-1) * rootw
            results.append(np.concatenate((freq.real.ravel(), freq.imag.ravel())))
        return np.stack(results), dict(
            construction_seconds=constructed, plan_seconds=prep
        )

    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        source=identity(source),
        inputs=[
            old["archive"],
            old["orbit"],
            identity(here / "response_immutable_final.json"),
        ],
        producers=[
            identity(__file__),
            identity(here / "prepared_tdi.py"),
            identity(here / "validate_response.py"),
        ],
        scope="Fresh stock vs immutable response construction + plan + all projections/TDI/FFT/whitening/packing/return/cleanup. Same preconfigured orbit and SSB; excludes orbit configure/H5 load/FEW. Stock avoids redundant channel copies.",
        timings=[],
    )
    try:
        for count in (1, 13):
            run("stock", count)
            run("sparse", count)
            times = {"stock": [], "sparse": []}
            stages = {"stock": [], "sparse": []}
            errors = []
            for repeat in range(5):
                outputs = {}
                for variant in (
                    ("stock", "sparse") if repeat % 2 == 0 else ("sparse", "stock")
                ):
                    start = perf_counter()
                    outputs[variant], stage = run(variant, count)
                    times[variant].append(perf_counter() - start)
                    stages[variant].append(stage)
                err = float(
                    np.linalg.norm(outputs["sparse"] - outputs["stock"])
                    / np.linalg.norm(outputs["stock"])
                )
                if not np.isfinite(err) or err > 1e-9:
                    raise AssertionError(err)
                errors.append(err)
            item = dict(
                count=count,
                seconds=times,
                stages=stages,
                errors=errors,
                ratio=float(np.median(times["stock"]) / np.median(times["sparse"])),
            )
            report["timings"].append(item)
            print(
                json.dumps(
                    dict(
                        count=count,
                        ratio=item["ratio"],
                        medians={k: float(np.median(v)) for k, v in times.items()},
                    )
                ),
                flush=True,
            )
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
