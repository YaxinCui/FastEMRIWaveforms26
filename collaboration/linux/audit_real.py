"""Post-run source/archive/dtype audit without repeating high-memory calls."""

# 2026-09-11 17:11 CST (linux): Read only small sources/archives; emit a new
# audit. Explicitly check separate master all/raw failures and fixed FP64 types.
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    output = here / "audit_real_final.json"
    if output.exists():
        raise FileExistsError(output)
    checks, files = [], {}
    def check(name, condition):
        checks.append(dict(name=name, passes=bool(condition)))
        if not condition:
            raise AssertionError(name)
    def authenticate(entry):
        path = Path(entry["path"])
        check(str(path), path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"])
        files[str(path)] = entry["sha256"]
    for backend in ("cpu", "cuda12x"):
        for variant in ("master", "fixed"):
            path = here / f"real_{variant}_{backend}_v2.json"
            files[str(path)] = digest(path)
            report = json.loads(path.read_text())
            check("terminal/passes", report["terminal"] and report["passes"])
            check("identity", report["variant"] == variant and report["backend"] == backend)
            for entry in [report["producer"], report["archive"], *report["sources"].values(), *report["native"]]:
                authenticate(entry)
            source_root = Path(report["source_root"])
            init = source_root / "src/few/__init__.py"
            original = subprocess.check_output(["git", "show", report["base_commit"]+":src/few/__init__.py"], cwd=source_root)
            check("top-level init pinned", init.read_bytes() == original)
            files[str(init)] = digest(init)
            check("all-contract separate", all(v["passes"] == (variant == "fixed") for v in report["checks"] if v["label"].endswith("_all_rotation")))
            check("raw-contract separate", all(v["passes"] == (variant == "fixed") for v in report["checks"] if v["label"].endswith("_raw_rotation")))
            check("tuple preserved oracle", all(v["passes"] for v in report["checks"] if v["label"].endswith("_tuple_rotation")))
            with np.load(report["archive"]["path"], allow_pickle=False) as stored:
                check("34 arrays", set(stored.files) == set(report["array_manifest"]) and len(stored.files) == 34)
                for key in stored.files:
                    arr, entry = stored[key], report["array_manifest"][key]
                    check(key+" finite/hash", np.all(np.isfinite(arr)) and hashlib.sha256(arr.tobytes()).hexdigest() == entry["sha256"])
                    check(key+" shape/dtype", list(arr.shape) == entry["shape"] and str(arr.dtype) == entry["dtype"])
                    if not key.endswith("_modes"):
                        check(key+" FP64 complex", arr.dtype == np.complex128)
            if variant == "fixed":
                rows = report["comparisons"]
                check("24 preserved", sum(v["byte_equal"] and not v["allowed_contract_change"] for v in rows) == 24)
                check("10 corrected", sum(not v["byte_equal"] and v["allowed_contract_change"] for v in rows) == 10)
    result = dict(timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), passes=True,
                  scope="Sources/native/archive authentication and actual recorded contracts; no new H5 physics solve",
                  checks=checks, files=files, audit_sha256=digest(__file__))
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(passes=True,checks=len(checks),files=len(files))))


if __name__ == "__main__":
    main()
