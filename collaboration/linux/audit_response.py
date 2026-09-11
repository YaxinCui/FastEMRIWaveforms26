"""Post-run source/lifecycle/numerical audit; preserve original negative runs."""

# 2026-09-11 21:30 CST (linux): Run only after FEW and timing jobs terminate.
# Read current/preserved sources and small archives; do not reread the Kerr H5.
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    output = here / "audit_final.json"
    math = here / "math_final.json"
    if output.exists() or math.exists():
        raise FileExistsError(output)
    checks = []
    files = {}

    def check(label, ok):
        checks.append(dict(label=label, passes=bool(ok)))
        if not ok:
            raise AssertionError(label)

    def auth(entry, path=None):
        p = Path(path or entry["path"])
        check(str(p), p.stat().st_size == entry["bytes"] and sha(p) == entry["sha256"])
        files[str(p)] = entry["sha256"]

    def read(name):
        p = here / name
        files[str(p)] = sha(p)
        return json.loads(p.read_text())

    final = read("response_immutable_final.json")
    check("formal terminal", final["passes"] and final["terminal"])
    for entry in [
        *final["sources"],
        *final["native"],
        *final["inputs"],
        final["archive"],
    ]:
        auth(entry)
    check(
        "producer internal gates",
        len(final["checks"]) == 227 and all(r["passes"] for r in final["checks"]),
    )
    check("two cases", len(final["cases"]) == 2)
    max_error = 0.0
    with np.load(final["archive"]["path"], allow_pickle=False) as a:
        for key in a.files:
            check(key + " finite", np.all(np.isfinite(a[key])))
        for case in final["cases"]:
            name = case["case"]
            auth(case["orbit"])
            check(name + " rows", len(case["rows"]) == 16)
            check(
                name + " modes/holdouts",
                sum(r["kind"] == "fresh_FEW_holdout" for r in case["rows"]) == 3,
            )
            check(
                name + " dimensions",
                a[name + "_stock"].shape == a[name + "_sparse"].shape == (16, 10944),
            )
            check(
                name + " plan",
                case["plan"]["orbit_queries"] == 6
                and case["plan"]["native_tdi_calls"] == 0
                and case["plan"]["bytes"] == 15006940
                and case["plan"]["nnz"] == 1246932,
            )
            for row, old, new in zip(
                case["rows"], a[name + "_stock"], a[name + "_sparse"]
            ):
                relative = float(np.linalg.norm(new - old) / np.linalg.norm(old))
                check(
                    "relative replay",
                    abs(relative - row["relative_l2"]) < 1e-20 and relative <= 1e-9,
                )
                check("weak channel gate", max(row["channel_relative_l2"]) <= 1e-9)
                for logged in row["likelihood"]:
                    rho = logged["rho"]
                    data = (rho * a[name + "_reference"] + a[name + "_noise"]).astype(
                        np.longdouble
                    )
                    ref = data - rho * a[name + "_reference"].astype(np.longdouble)
                    ro = data - rho * old.astype(np.longdouble)
                    rn = data - rho * new.astype(np.longdouble)
                    lo = float(-0.5 * np.sum(ro**2 - ref**2))
                    ln = float(-0.5 * np.sum(rn**2 - ref**2))
                    error = abs(ln - lo)
                    max_error = max(max_error, error)
                    check(
                        "longdouble direct norm replay",
                        lo == logged["stock"]
                        and ln == logged["sparse"]
                        and error == logged["error"],
                    )
                    check(
                        "likelihood tolerance",
                        logged["limit"] == 1e-5 + 2e-10 * abs(lo)
                        and error <= logged["limit"],
                    )
    for row in final["timings"]:
        check(
            "five alternating pairs",
            len(row["seconds"]["stock"]) == len(row["seconds"]["sparse"]) == 5,
        )
        check(
            "ratio",
            row["speed_ratio"]
            == np.median(row["seconds"]["stock"]) / np.median(row["seconds"]["sparse"]),
        )
        check("timing outputs", len(row["errors"]) == 5 and max(row["errors"]) <= 1e-9)
        for invocation in row["calls"]:
            expected = dict(
                projection=row["count"],
                stock_tdi=row["count"] if invocation["variant"] == "stock" else 0,
                sparse_tdi=row["count"] if invocation["variant"] == "sparse" else 0,
            )
            check("both paths retain projections", invocation["counts"] == expected)
    check(
        "changed configuration tests",
        len(final["synthetic"]) == 2
        and all(
            max(r["channels"]) <= 1e-9 for c in final["synthetic"] for r in c["rows"]
        ),
    )
    holdout = read("holdout_final.json")
    check(
        "fresh FEW terminal",
        holdout["terminal"] and holdout["passes"] and len(holdout["cases"]) == 6,
    )
    for entry in [
        holdout["producer"],
        holdout["archive"],
        *holdout["few_sources"],
        *holdout["native"],
    ]:
        auth(entry)
    check(
        "data identities recorded by worker",
        [x["sha256"] for x in holdout["data"]]
        == [
            "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834",
            "db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014",
        ],
    )
    life = read("lifecycle_final.json")
    check("lifecycle terminal", life["terminal"] and life["passes"])
    for entry in [life["source"], *life["inputs"], *life["producers"]]:
        auth(entry)
    for row in life["timings"]:
        check(
            "lifecycle pairs",
            len(row["seconds"]["stock"]) == len(row["seconds"]["sparse"]) == 5,
        )
        check(
            "lifecycle ratio",
            row["ratio"]
            == np.median(row["seconds"]["stock"]) / np.median(row["seconds"]["sparse"]),
        )
        check("lifecycle outputs", max(row["errors"]) <= 1e-9)
        for side in ("stock", "sparse"):
            for total, stage in zip(row["seconds"][side], row["stages"][side]):
                check(
                    "lifecycle stage scope",
                    total > stage["construction_seconds"] + stage["plan_seconds"] > 0,
                )
    first = read("tdi_first.json")
    for entry in first["sources"]:
        name = Path(entry["path"]).name
        auth(
            entry,
            here / "pilot_sources" / name
            if name in ("probe_tdi.py", "prepared_tdi.py")
            else None,
        )
    registered = read("response_final.json")
    for entry in registered["sources"]:
        name = Path(entry["path"]).name
        mapped = {
            "prepared_tdi.py": "prepared_tdi_registered.py",
            "validate_response.py": "validate_response_registered.py",
        }
        auth(entry, here / "pilot_sources" / mapped[name] if name in mapped else None)
    check(
        "slower original retained",
        next(
            x
            for x in registered["timings"]
            if x["count"] == 13 and x["preparation_included"]
        )["speed_ratio"]
        < 1,
    )
    run = subprocess.run(
        [sys.executable, str(here / "test_prepared_tdi.py")],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        ),
    )
    math.write_text(
        json.dumps(
            dict(
                timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                returncode=run.returncode,
                passes=run.returncode == 0,
                log=run.stdout + run.stderr,
                test_sha256=sha(here / "test_prepared_tdi.py"),
                implementation_sha256=sha(here / "prepared_tdi.py"),
            ),
            indent=2,
        )
        + "\n"
    )
    files[str(math)] = sha(math)
    files[str(here / "test_prepared_tdi.py")] = sha(here / "test_prepared_tdi.py")
    check(
        "seven final test methods", run.returncode == 0 and "Ran 7 tests" in run.stderr
    )
    branch = (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=here)
        .decode()
        .strip()
    )
    check("independent branch", branch == "codex/exp16-prepared-detector-response")
    check(
        "master base",
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here).decode().strip()
        == "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
    )
    check(
        "production unchanged",
        subprocess.check_output(["git", "diff", "--name-only"], cwd=here).strip()
        == b"",
    )
    result = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=True,
        checks=checks,
        files=files,
        audit_sha256=sha(__file__),
        max_logl_error=max_error,
        warm_timings=final["timings"],
        lifecycle_timings=life["timings"],
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                passes=True,
                checks=len(checks),
                files=len(files),
                max_logl_error=max_error,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
