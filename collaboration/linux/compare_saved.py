"""Compare independent-build waveform artifacts and record their identities."""

# 2026-09-11 12:49 CST (linux): Never overwrite reference arrays; emit metrics
# and content hashes only. The two directories are explicit CLI inputs.
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np


# 2026-09-11 13:00 CST (linux): Fail closed on NaN, zero denominators, missing
# cases and incompatible arrays; a metrics JSON alone is not a passing gate.
def compare_arrays(ref, got):
    if ref.shape != got.shape or ref.dtype != got.dtype:
        raise ValueError("Waveform shapes or dtypes differ")
    if ref.ndim != 1 or not ref.size or ref.dtype != np.complex128:
        raise ValueError("Require nonempty complex128 waveform vectors")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
        raise ValueError("Non-finite waveform")
    absolute = float(np.max(np.abs(ref - got)))
    scale = float(np.max(np.abs(ref)))
    if scale == 0.0:
        normalized, relative = None, None
        passed = absolute == 0.0
    else:
        normalized = absolute / scale
        relative = float(np.linalg.norm((ref - got) / scale) / np.linalg.norm(ref / scale))
        if not np.isfinite(normalized) or not np.isfinite(relative):
            raise ValueError("Non-finite error metric")
        passed = normalized <= 5e-13 and relative <= 5e-13
    # 2026-09-11 13:13 CST (linux): Signed zero differs bitwise, though equal().
    return {"bitwise_equal": ref.tobytes() == got.tobytes(),
            "absolute_max": absolute, "normalized_max": normalized,
            "relative_l2": relative, "passes": passed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", nargs="+", default=[
        "nominal_0.001_legacy.npy", "nominal_1.0_legacy.npy"])
    args = parser.parse_args()
    rows = []
    for name in args.expected:
        ref_path = args.reference_dir / name
        candidate = args.candidate_dir / ref_path.name
        ref, got = np.load(ref_path, allow_pickle=False), np.load(candidate, allow_pickle=False)
        rows.append({"name":ref_path.name,"reference":str(ref_path),"candidate":str(candidate),
                     "reference_sha256":hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                     "candidate_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest(),
                     **compare_arrays(ref, got)})
    report = {"timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "comparisons": rows, "passes": bool(rows) and all(x["passes"] for x in rows)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passes"]:
        raise AssertionError("Independent-build comparison failed")


if __name__ == "__main__":
    main()
