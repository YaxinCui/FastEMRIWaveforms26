"""Expose direct-retrograde selection disagreement from saved master outputs."""

# 2026-09-11 16:22 CST (linux): Diagnostic only, no FEW imports, no H5 access,
# no production mutation. Explicitly distinguish valid evidence from a failing
# API contract; an expected master bug is not labelled scientific acceptance.
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def identity(path):
    path = Path(path).resolve()
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=False,
        evidence_valid=False,
        contract_passes=False,
        bug_reproduced=False,
        scope="Existing same-backend master archive only, no new Teukolsky solve or full-waveform correctness claim",
        script=identity(__file__),
        reference_report=identity(args.reference),
        rows=[],
    )
    try:
        data = json.loads(args.reference.read_text())
        if (
            not data["passes"]
            or not data["terminal"]
            or data["policy"] != "master"
            or data["benchmark"]
            or data["base_commit"] != "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
        ):
            raise AssertionError("Need successful formal pinned-master reference")
        if (
            data["sources"][0]["sha256"]
            != "629d2751f694957f9f1e2b4beef20fc50e6005ee0362acadf7276e3054cfc887"
        ):
            raise AssertionError(
                "Unexpected producer; mode/parameter contract must be rechecked"
            )
        report["backend"] = data["backend"]
        # 2026-09-11 16:28 CST (linux): Authenticate the pinned model and
        # frontend too, not merely the producer script's claimed base label.
        expected_sources = (
            "8997e9d29abb026cdc0abc6fabbf07eaa0f244bdd3cd414c3f6b98c463655d78",
            "765ae14fb0c65ec0f6775739a80a6271df88287c50046999c7de605507193559",
        )
        for entry, expected_sha in zip(data["sources"][1:3], expected_sources):
            if entry["sha256"] != expected_sha or identity(entry["path"]) != entry:
                raise AssertionError("Pinned master model/frontend changed")
        archive = identity(data["archive"]["path"])
        if archive != data["archive"]:
            raise AssertionError("Reference archive changed")
        report["archive"] = archive
        report["data_identities_recorded_by_producer"] = data["data"]
        # Metadata identities are preserved; this diagnostic intentionally does
        # not rehash the large table or native code while a benchmark is live.
        # 2026-09-11 16:26 CST (linux): Bind the producer's MASTER source, not
        # this candidate branch, so a later repair cannot erase the diagnostic.
        source = Path(data["sources"][1]["path"]).with_name("base.py")
        original = subprocess.check_output(
            ["git", "show", data["base_commit"] + ":src/few/amplitude/base.py"],
            cwd=root,
        )
        if source.read_bytes() != original:
            raise AssertionError("Producer no longer has pinned master selection code")
        report["selection_source"] = identity(source)
        layout_source = source.parent.parent / "utils/baseclasses.py"
        original_layout = subprocess.check_output(
            ["git", "show", data["base_commit"] + ":src/few/utils/baseclasses.py"],
            cwd=root,
        )
        if layout_source.read_bytes() != original_layout:
            raise AssertionError("Producer's mode ordering changed")
        report["mode_layout_source"] = identity(layout_source)
        # Independent explicit mode-label order, as documented in the pinned
        # SphericalHarmonic constructor: m=0 first, then m>0, stable within each.
        labels = [
            (ell, m, 0, n)
            for ell in range(2, 11)
            for m in range(ell + 1)
            for n in range(-55, 56)
        ]
        labels = [mode for mode in labels if mode[1] == 0] + [
            mode for mode in labels if mode[1] > 0
        ]
        indexes = {mode: i for i, mode in enumerate(labels)}
        modes = [
            (2, 2, 0, 0),
            (3, 1, 0, 2),
            (3, -1, 0, -2),
            (2, 0, 0, -1),
            (10, 10, 0, 55),
        ]
        entries = {item["key"]: item for item in data["amplitudes"]}
        with np.load(archive["path"], allow_pickle=False) as stored:
            arrays = {
                key: stored[key].copy()
                for key in (
                    "canonical_retrograde_all",
                    "direct_retrograde_all",
                    "direct_retrograde_tuples",
                    "direct_retrograde_raw",
                )
            }
        for key, value in arrays.items():
            entry = entries[key]
            if (
                not np.all(np.isfinite(value))
                or list(value.shape) != entry["shape"]
                or str(value.dtype) != entry["dtype"]
                or hashlib.sha256(value.tobytes()).hexdigest() != entry["sha256"]
            ):
                raise AssertionError("Invalid archived array: " + key)
        canonical = arrays["canonical_retrograde_all"]
        if canonical.shape != (3, len(labels)):
            raise AssertionError("Unexpected canonical mode order/shape")
        for i, mode in enumerate(modes):
            ell, m, k, n = mode
            # D_lmn=(-1)^l C_l,-m,n, combining coordinate rotation with
            # C_l,-m,-n=(-1)^l conj(C_lmn) for m>0. k=0 only here.
            expected = (
                canonical[:, indexes[(ell, m, k, -n)]].conj()
                if m > 0
                else (-1) ** ell * canonical[:, indexes[(ell, -m, k, n)]]
            )
            named = arrays["direct_retrograde_tuples"][:, i]
            raw = arrays["direct_retrograde_raw"][:, i]
            all_value = (
                arrays["direct_retrograde_all"][:, indexes[mode]] if m >= 0 else None
            )
            same = lambda a, b: (
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            )
            row = dict(
                mode=mode,
                tuple_matches_rotation=same(named, expected),
                raw_matches_tuple=same(raw, named),
                raw_max_abs_difference=float(np.max(np.abs(raw - named))),
                tuple_first=[float(named[0].real), float(named[0].imag)],
                raw_first=[float(raw[0].real), float(raw[0].imag)],
            )
            if all_value is not None:
                row.update(
                    all_matches_tuple=same(all_value, named),
                    all_max_abs_difference=float(np.max(np.abs(all_value - named))),
                    all_first=[float(all_value[0].real), float(all_value[0].imag)],
                )
            report["rows"].append(row)
        if not all(row["tuple_matches_rotation"] for row in report["rows"]):
            raise AssertionError(
                "Physical-coordinate hypothesis does not match existing tuple convention"
            )
        report["contract_passes"] = all(
            row["raw_matches_tuple"] and row.get("all_matches_tuple", True)
            for row in report["rows"]
        )
        report["bug_reproduced"] = not report["contract_passes"]
        if not report["bug_reproduced"]:
            raise AssertionError("Expected master contract discrepancy not reproduced")
        report["evidence_valid"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "terminal",
                        "evidence_valid",
                        "contract_passes",
                        "bug_reproduced",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
