"""Independent saved-array/provenance audit and actual-native stencil boundaries."""

# 2026-09-11 22:12 CST (linux): Run separately after producer jobs terminate.
# No producer flags alone certify numerical equality. Root-only serial runner.
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from contract_support import HERE, identity, load, orbit, response


def main():
    output = HERE / "audit_final.json"
    if output.exists():
        raise FileExistsError(output)
    report = dict(timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  terminal=False, passes=False, checks=[], files={}, stencil_cases=[],
                  source=identity(__file__))

    def check(label, passed):
        report["checks"].append(dict(label=label, passes=bool(passed)))
        if not passed:
            raise AssertionError(label)

    def authenticate(entry):
        actual = identity(entry["path"])
        check(entry["path"], actual == entry)
        report["files"][entry["path"]] = actual

    def read(name):
        path = HERE / name
        authenticate(identity(path))
        value = json.loads(path.read_text())
        check(name+" terminal and pass", value["terminal"] and value["passes"])
        return value

    try:
        baseline = read("original_failure.json")
        for entry in baseline["sources"]:
            authenticate(entry)
        check("three preserved original failures", len(baseline["cases"]) == 3
              and all(x["exception"] == "AttributeError" for x in baseline["cases"]))
        for name, optimized in (("unit_first.json", False), ("unit_optimized.json", True)):
            unit = read(name)
            for entry in unit["sources"]:
                authenticate(entry)
            check(name+" test count / optimization", unit["tests"] == 10
                  and unit["optimized_python"] is optimized and "\nOK\n" in unit["log"])
        real = read("real_final.json")
        for entry in [*real["sources"], *real["native"], *real["inputs"], real["archive"], real["orbit"]]:
            authenticate(entry)
        check("producer all checks", len(real["checks"]) == 351
              and all(x["passes"] for x in real["checks"]))
        check("36 distinct configurations", len(real["cases"]) == 36
              and len({x["key"] for x in real["cases"]}) == 36)
        with np.load(real["archive"]["path"], allow_pickle=False) as saved:
            check("archive exact array count", len(saved.files) == 180)
            for row in real["cases"]:
                key = row["key"]
                truth, second_truth = saved[key+"_stock"], saved[key+"_second_stock"]
                n, cut = row["samples"], row["cut"]
                expected_shape = (2 if row["channel"] == "AE" else 3, n)
                for suffix, reference in (("explicit", truth), ("implicit", truth), ("second", second_truth)):
                    value = saved[key+"_"+suffix]
                    check(key+suffix+" finite FP64 shape", value.dtype == np.dtype(np.float64)
                          and value.shape == expected_shape and np.all(np.isfinite(value)))
                    check(key+suffix+" exact bytes", value.tobytes() == reference.tobytes())
                    error = np.linalg.norm(value.astype(np.longdouble)-reference.astype(np.longdouble))
                    check(key+suffix+" independent high-precision residual", error == 0)
                    check(key+suffix+" zero crop margins", np.count_nonzero(value[:, :cut]) == 0
                          and np.count_nonzero(value[:, -cut:]) == 0)
                check(key+" B not stale A", np.linalg.norm(second_truth-truth) > 0)
                check(key+" reported metrics", row["max_absolute_error"] == 0 and row["relative_l2"] == 0)

        # Independently query the actual injected orbit C++ object at all output
        # times, then replay the native sequential subtraction/d/e arithmetic.
        cls, original = load("candidate"), load("original")
        geometry = orbit()
        configurations = [(.1, 1501, 15, "2nd generation"),
                          (1., 513, 25, "2nd generation"),
                          (15., 257, 35, "1st generation"),
                          (40., 257, 40, "2nd generation"),
                          (1., 2049, 25, [dict(link=12, links_for_delay=[12]*64, sign=1)])]
        for dt, n, order, topology in configurations:
            r = response(cls, geometry, n=n, dt=dt, order=order, tdi=topology)
            args = r.response_orbits.pycppdetector_args
            units, starts, lengths = np.unique(r.tdi_operation_index, return_index=True, return_counts=True)
            kmax = max(np.count_nonzero(r.tdi_link_combinations[a:a+b] != -11) for a, b in zip(starts, lengths))
            h = (order+1)//2
            cut = int(np.ceil(kmax*args[3].max()/dt)) + h + 2
            time = np.arange(n, dtype=np.float64)[cut:-cut] * dt
            check("nonempty bounded stencil domain", len(time) > 0)
            native_orbit = r.response_orbits.backend.OrbitsWrap(*args)
            travel = {}
            for link in (12, 23, 31, 13, 32, 21):
                out = np.empty_like(time)
                native_orbit.get_light_travel_time_wrap(out, time, np.full(time.size, link, dtype=np.int32), time.size)
                check("actual finite positive light time", np.all(np.isfinite(out)) and np.all(out > 0))
                travel[link] = out
            minimum, maximum = n, -1
            for a, b in zip(starts, lengths):
                delay = time.copy()
                for link in r.tdi_link_combinations[a:a+b]:
                    if link != -11:
                        delay -= travel[int(link)]
                x = delay * (1/dt)
                d = np.ceil(x).astype(np.int64)-1
                fraction = 1.0+d-x
                table = (fraction/r.deps).astype(np.int64)
                check("actual full native stencil inside input", np.all(d-h+1 >= 0) and np.all(d+h < n))
                check("actual native A table accesses valid", np.all(fraction >= 0) and np.all(fraction < 1)
                      and np.all(table >= 0) and np.all(table+1 < len(r.A_in)))
                minimum, maximum = min(minimum, int((d-h+1).min())), max(maximum, int((d+h).max()))
            y = np.random.default_rng(17131+order).normal(size=(6, n))
            # Noninteger buffer seconds select cut reliably even for dt=.1.
            predicted = np.stack(r.get_tdi_delays(y, t0=(cut+.25)*dt))
            ref = response(original, geometry, n=n, dt=dt, order=order, tdi=topology)
            ref.y_gw_flat, ref.y_gw_length, ref.nlinks, ref.tdi_start_ind = y.flatten(), n, 6, cut
            truth = np.stack(ref.get_tdi_delays())
            check("lowest accepted cut native equality", predicted.tobytes() == truth.tobytes())
            try:
                r.get_tdi_delays(y, t0=(cut-.75)*dt)
            except ValueError:
                rejected = True
            else:
                rejected = False
            check("one sample below conservative bound rejected", rejected)
            report["stencil_cases"].append(dict(dt=dt, n=n, order=order, max_delays=int(kmax),
                                                 cut=cut, units=len(units), min_read=minimum, max_read=maximum))
        # Negative audit self-test: edited bytes cannot retain the original hash.
        candidate_bytes = (HERE / "candidate/response.py").read_bytes()
        check("tamper detection", hashlib.sha256(candidate_bytes+b" ").hexdigest()
              != identity(HERE / "candidate/response.py")["sha256"])
        report["passes"] = True
    finally:
        report["terminal"] = True
        report["check_count"] = len(report["checks"])
        report["file_count"] = len(report["files"])
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k: report[k] for k in ("terminal", "passes", "check_count", "file_count")}))


if __name__ == "__main__":
    main()
