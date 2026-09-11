"""First actual archived-input native-vs-sparse TDI experiment."""

# 2026-09-11 21:14 CST (linux): Preserve bounded pilot, no performance claims
# beyond fixed CPU response stage; write only this experiment's JSON/NPZ.
from datetime import datetime
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits

from prepared_tdi import PreparedTDI


def identity(path):
    p = Path(path).resolve()
    return dict(
        path=str(p),
        bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    )


def main():
    here = Path(__file__).resolve().parent
    out = here / "tdi_first.json"
    if out.exists():
        raise FileExistsError(out)
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        sources=[
            identity(__file__),
            identity(here / "prepared_tdi.py"),
            identity(inspect.getsourcefile(pyResponseTDI)),
            identity(inspect.getsourcefile(ESAOrbits)),
        ],
        native=[
            identity(importlib.import_module(x).__file__)
            for x in (
                "fastlisaresponse_backend_cpu.responselisa",
                "lisatools_backend_cpu.pycppdetector",
            )
        ],
        cases=[],
    )
    try:
        for case in ("prograde", "retrograde"):
            old = here.parents[2] / "exp15-single-pass-phase-basis/collaboration/linux"
            evidence = json.loads((old / f"real_{case}_final.json").read_text())
            archive = evidence["archive"]
            assert identity(archive["path"]) == archive
            assert identity(evidence["orbit"]["path"]) == evidence["orbit"]
            with np.load(archive["path"], allow_pickle=False) as a:
                waves = a["new_ssb_basis"].copy()
            orbit = ESAOrbits(force_backend="cpu")
            orbit.configure(linear_interp_setup=True)
            response = pyResponseTDI(
                1 / 15.0,
                waves.shape[1],
                order=25,
                tdi="2nd generation",
                tdi_chan="AET",
                orbits=orbit,
                force_backend="cpu",
            )
            response.get_projections(
                waves[0].conj().copy(), 0.7, np.pi / 2 - 1.1, t0=20000.0
            )
            start = perf_counter()
            plan = PreparedTDI(response)
            prep = perf_counter() - start
            rows = []
            for wave in waves:
                response.get_projections(
                    wave.conj().copy(), 0.7, np.pi / 2 - 1.1, t0=20000.0
                )
                start = perf_counter()
                full = response.get_tdi_delays()
                old_s = perf_counter() - start
                cut = response.tdi_start_ind
                truth = np.stack([a[cut:-cut] for a in full])
                start = perf_counter()
                prediction = plan(response.y_gw)
                new_s = perf_counter() - start
                rel = float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))
                by_channel = [
                    float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
                    for a, b in zip(prediction, truth)
                ]
                rows.append(
                    dict(
                        native_seconds=old_s,
                        sparse_seconds=new_s,
                        relative_l2=rel,
                        channels=by_channel,
                    )
                )
                if rel > 1e-9:
                    raise AssertionError(rows[-1])
            report["cases"].append(
                dict(
                    case=case,
                    input=archive,
                    orbit=evidence["orbit"],
                    preparation_seconds=prep,
                    plan=plan.metadata,
                    rows=rows,
                )
            )
            print(
                json.dumps(
                    dict(
                        case=case,
                        prep=prep,
                        metadata=plan.metadata,
                        max_error=max(r["relative_l2"] for r in rows),
                        native=float(np.median([r["native_seconds"] for r in rows])),
                        sparse=float(np.median([r["sparse_seconds"] for r in rows])),
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
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
