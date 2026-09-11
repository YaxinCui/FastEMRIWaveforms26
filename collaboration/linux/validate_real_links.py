"""Authenticated FEW holdout replay through unchanged CPU projection and TDI."""

# 2026-09-11 22:12 CST (linux): Reuse exp16's six fresh master-generated source
# waveforms with hashes; no FEW generation, H5 mutation, GPU or timing claim.
from datetime import datetime
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits
from lisatools.utils.utility import AET

from contract_support import EXPECTED_SOURCE, HERE, identity, load, orbit, response


def main():
    output = HERE / "real_final.json"
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    old = HERE.parents[2] / "exp16-prepared-detector-response/collaboration/linux"
    original, candidate = load("original"), load("candidate")
    report = dict(timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  terminal=False, passes=False, checks=[], cases=[], inputs=[],
                  scope="CPU FP64 interface repair; exp16 holdout reuse, not new FEW/H5 validation; no timing",
                  sources=[identity(p) for p in (__file__, HERE / "contract_support.py",
                           inspect.getsourcefile(original), inspect.getsourcefile(candidate),
                           inspect.getsourcefile(pyResponseTDI), inspect.getsourcefile(ESAOrbits),
                           inspect.getsourcefile(AET))],
                  native=[identity(importlib.import_module(m).__file__) for m in
                          ("fastlisaresponse_backend_cpu.responselisa", "lisatools_backend_cpu.pycppdetector")],
                  versions={v: importlib.metadata.version(v) for v in
                            ("numpy", "scipy", "fastlisaresponse", "lisaanalysistools")},
                  environment={k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")})
    saved = {}

    def check(label, passed):
        report["checks"].append(dict(label=label, passes=bool(passed)))
        if not passed:
            raise AssertionError(label)

    def authenticate(entry):
        check(entry["path"], identity(entry["path"]) == entry)
        report["inputs"].append(entry)

    def equal(label, left, right):
        check(label, left.shape == right.shape and np.all(np.isfinite(left))
              and left.tobytes() == right.tobytes())

    try:
        check("original/installed exact source", all(
            identity(inspect.getsourcefile(c))["sha256"] == EXPECTED_SOURCE
            for c in (original, pyResponseTDI)))
        holdout = json.loads((old / "holdout_final.json").read_text())
        check("authenticated holdout terminal", holdout["terminal"] and holdout["passes"])
        report["inputs"].append(identity(old / "holdout_final.json"))
        for entry in [holdout["archive"], holdout["producer"], *holdout["few_sources"], *holdout["native"]]:
            authenticate(entry)
        with np.load(holdout["archive"]["path"], allow_pickle=False) as a:
            waves = {k: a[k].copy() for k in a.files}
        geometry = orbit()
        old_real = json.loads((HERE.parents[2] / "exp15-single-pass-phase-basis/collaboration/linux/real_prograde_final.json").read_text())
        authenticate(old_real["orbit"])
        check("configured actual orbit", Path(geometry.filename).resolve() == Path(old_real["orbit"]["path"]).resolve())
        report["orbit"] = identity(geometry.filename)
        report["holdout_cases"] = holdout["cases"]
        for row in holdout["cases"]:
            key = row["key"]
            wave = waves[key]
            other_key = key.rsplit("_", 1)[0] + "_" + str((int(key[-1])+1) % 3)
            other_wave = waves[other_key]
            n, dt, t0 = len(wave), row["settings"]["dt"], 20000.
            check(key+" length", n == row["samples"])
            for tdi in ("1st generation", "2nd generation"):
                for channel in ("XYZ", "AET", "AE"):
                    label = f"{key}_{tdi[0]}_{channel}"
                    ref = response(original, geometry, n=n, dt=dt, tdi=tdi, channel=channel)
                    normal = response(candidate, geometry, n=n, dt=dt, tdi=tdi, channel=channel)
                    direct = response(candidate, geometry, n=n, dt=dt, tdi=tdi, channel=channel)
                    # FEW uses h+ - i hx; installed response takes its conjugate.
                    for r in (ref, normal):
                        r.get_projections(wave.conj().copy(), .7, np.pi/2-1.1, t0=t0)
                    equal(label+" projection", ref.y_gw, normal.y_gw)
                    y = ref.y_gw.copy()
                    truth = np.stack(ref.get_tdi_delays())
                    implicit = np.stack(normal.get_tdi_delays())
                    explicit = np.stack(direct.get_tdi_delays(y, t0=t0))
                    equal(label+" unchanged implicit", implicit, truth)
                    equal(label+" fresh explicit", explicit, truth)
                    equal(label+" input unchanged", direct.y_gw, y)
                    ref.get_projections(other_wave.conj().copy(), .7, np.pi/2-1.1, t0=t0)
                    second_truth = np.stack(ref.get_tdi_delays())
                    second = np.stack(direct.get_tdi_delays(ref.y_gw.copy()))
                    equal(label+" replacement B", second, second_truth)
                    check(label+" replacement differs from A", not np.array_equal(second, truth))
                    equal(label+" resident B", np.stack(direct.get_tdi_delays()), second_truth)
                    cut = int(t0/dt)
                    equal(label+" cropped interior", explicit[:, cut:-cut], truth[:, cut:-cut])
                    report["cases"].append(dict(key=label, tdi=tdi, channel=channel, samples=n,
                                               cut=cut, replacement_key=other_key,
                                               max_absolute_error=float(np.max(np.abs(explicit-truth))),
                                               relative_l2=float(np.linalg.norm(explicit-truth)/np.linalg.norm(truth)),
                                               passes=True))
                    saved.update({label+"_stock": truth, label+"_explicit": explicit,
                                  label+"_implicit": implicit, label+"_second_stock": second_truth,
                                  label+"_second": second})
                    del ref, normal, direct
        check("all physical interface configurations", len(report["cases"]) == 36)
        np.savez_compressed(archive, **saved)
        report["archive"] = identity(archive)
        report["passes"] = True
    finally:
        report["terminal"] = True
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k: report[k] for k in ("terminal", "passes", "scope")}))
        print("checks", len(report["checks"]), "cases", len(report["cases"]))


if __name__ == "__main__":
    main()
