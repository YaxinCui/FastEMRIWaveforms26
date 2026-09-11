"""Fail-closed audit of exp12 code identity, arrays, cache accounting and timings."""

# 2026-09-11 16:15 CST (linux): Run only AFTER all performance workers finish;
# full data hashing here must not contend with their IO. Write a NEW audit only.
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    notes = root / "collaboration/linux"
    base = "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        checks=[],
        files={},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def check(condition, label):
        report["checks"].append(dict(label=label, passes=bool(condition)))
        if not condition:
            raise AssertionError(label)

    def git(*command, cwd=root):
        return subprocess.check_output(["git", *command], cwd=cwd, text=True).strip()

    def digest(path):
        path = Path(path).resolve()
        key = str(path)
        if key not in report["files"]:
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            report["files"][key] = dict(sha256=sha, bytes=path.stat().st_size)
        return report["files"][key]

    def authenticate(entry):
        found = digest(entry["path"])
        check(found["sha256"] == entry["sha256"], "hash:" + entry["path"])
        if "bytes" in entry:
            check(found["bytes"] == entry["bytes"], "size:" + entry["path"])

    def read(path):
        digest(path)
        return json.loads(Path(path).read_text())

    def archive(data, wanted=None):
        authenticate(data["archive"])
        with np.load(data["archive"]["path"], allow_pickle=False) as stored:
            values = {key: stored[key].copy() for key in stored.files}
        entries = list(data["amplitudes"])
        for call in data["calls"]:
            entries += [call["wave"], call["modes"]]
        check(len(entries) == len(values), "archive-entry-count")
        check({item["key"] for item in entries} == set(values), "archive-keys")
        for item in entries:
            value = values[item["key"]]
            check(value.size and np.all(np.isfinite(value)), "finite:" + item["key"])
            check(
                list(value.shape) == item["shape"]
                and str(value.dtype) == item["dtype"],
                "shape/dtype:" + item["key"],
            )
            check(
                hashlib.sha256(value.tobytes()).hexdigest() == item["sha256"],
                "array-hash:" + item["key"],
            )
            if wanted is not None:
                target = wanted[item["key"]]
                check(
                    value.shape == target.shape
                    and value.dtype == target.dtype
                    and value.tobytes() == target.tobytes(),
                    "master-bytes:" + item["key"],
                )
        return values

    def real_report(path, backend, policy, benchmark=False, wanted=None):
        data = read(path)
        check(data["passes"] and data["terminal"], "terminal-pass:" + str(path))
        check(
            data["backend"] == backend
            and data["policy"] == policy
            and data["benchmark"] == benchmark
            and not data["verified_first"],
            "worker-mode:" + str(path),
        )
        check(
            data["base_commit"] == base
            and data["host_derivative_adapter"] == (backend == "cuda12x"),
            "worker-base/adapter",
        )
        for field in ("sources", "native", "data"):
            for entry in data[field]:
                authenticate(entry)
        if policy != "master":
            authenticate(data["reference"])
        values = archive(data, wanted)
        check(len(values) == (14 if benchmark else 39), "workload-array-count")
        if policy == "lazy":
            check(
                all(
                    row["loads"] == 0 for row in data["cache_at_construction"].values()
                ),
                "lazy-constructor-no-coefficients",
            )
            for call in data["calls"]:
                for region, row in call["cache"].items():
                    check(
                        row["capacity"] == 2 and len(row["resident_indices"]) <= 2,
                        "steady-cache-bound:" + call["label"] + region,
                    )
                    check(
                        row["coefficient_bytes"]
                        == len(row["resident_indices"]) * slice_bytes[region],
                        "logical-cache-bytes:" + call["label"] + region,
                    )
                    check(
                        row["bytes_read"] == row["loads"] * slice_bytes[region],
                        "load-bytes:" + call["label"] + region,
                    )
        return data, values

    try:
        check(
            git("branch", "--show-current") == "codex/exp12-spin-local-amplitudes",
            "experiment-branch",
        )
        check(
            git("rev-parse", "HEAD") == base and git("rev-parse", "master") == base,
            "master-base",
        )
        check(
            git("diff", "HEAD", "--name-only").splitlines()
            == ["src/few/amplitude/ampinterp2d.py"],
            "tracked-change-scope",
        )
        check(
            not git("status", "--porcelain", "--", "collaboration/mac"), "mac-read-only"
        )
        root_checkout = root.parent.parent / "FastEMRIWaveforms"
        check(
            not git("diff", "HEAD", "--name-only", "--", "src", cwd=root_checkout),
            "root-production-unchanged",
        )
        pilot = read(notes / "real_master_cpu.json")
        check(
            pilot["terminal"] and not pilot["passes"] and bool(pilot.get("error")),
            "failed-p8-pilot-preserved",
        )
        for backend, suffix in (("cpu", "cpu"), ("cuda12x", "cuda")):
            small = read(notes / f"spin_{suffix}_stream_hardened.json")
            check(
                small["passes"]
                and small["terminal"]
                and small["returncode"] == 0
                and small["backend"] == backend
                and small["imported"]["backend"] == backend,
                "small-backend-terminal:" + backend,
            )
            check(
                "Ran 8 tests" in small["log"] and "\nOK\n" in small["log"],
                "eight-methods:" + backend,
            )
            for path, sha in small["imported"]["sources"].items():
                authenticate(dict(path=path, sha256=sha))
        cpu_ref = read(notes / "real_master_cpu_replay.json")
        table = next(item for item in cpu_ref["data"] if "ZNAmps" in item["path"])
        authenticate(table)
        slice_bytes = {}
        with h5py.File(table["path"], "r") as file:
            for region in ("A", "B"):
                ds = file[f"region{region}/CoeffsRegion{region}"]
                check(
                    ds.dtype == np.float64 and ds.shape[:3] == (33, 6993, 2),
                    "actual-table-shape:" + region,
                )
                check(
                    ds.chunks is None and ds.compression is None,
                    "contiguous-table:" + region,
                )
                slice_bytes[region] = math.prod(ds.shape[1:]) * ds.dtype.itemsize
        report["slice_bytes"] = slice_bytes
        for backend, suffix, baseline in (
            ("cpu", "cpu", "real_master_cpu_replay.json"),
            ("cuda12x", "cuda", "real_master_cuda.json"),
        ):
            master, reference = real_report(notes / baseline, backend, "master")
            for policy in ("eager", "lazy"):
                data, values = real_report(
                    notes / f"real_{policy}_{suffix}.json",
                    backend,
                    policy,
                    wanted=reference,
                )
                check(set(values) == set(reference), "real-complete-reference-keys")
                check(
                    data["data"] == master["data"]
                    and data["native"] == master["native"],
                    "same-real-data/native",
                )
                boundary = next(
                    row
                    for row in data["mapping_cases"]
                    if row["label"] == "region_boundary"
                )
                check(
                    boundary["region_A"] == [True, True, False],
                    "adjacent-representable-region-dispatch",
                )
            bench_path = notes / f"benchmark_{suffix}.json"
            bench = read(bench_path)
            check(
                bench["passes"]
                and bench["terminal"]
                and bench["pairs"] == 7
                and len(bench["runs"]) == 14
                and bench["backend"] == backend
                and not bench["verified_first"],
                "terminal-seven-pair-benchmark:" + backend,
            )
            check(
                bench["script_sha256"] == digest(notes / "spin_benchmark.py")["sha256"],
                "benchmark-driver-identity",
            )
            records = {"master": [], "lazy": []}
            expected_order = [
                (pair, policy)
                for pair in range(7)
                for policy in (
                    ("master", "lazy") if pair % 2 == 0 else ("lazy", "master")
                )
            ]
            check(
                [(row["pair"], row["policy"]) for row in bench["runs"]]
                == expected_order,
                "AB/BA-order",
            )
            for row in bench["runs"]:
                check(
                    row["state"] == "terminal" and row["returncode"] == 0,
                    "terminal-worker",
                )
                check(
                    row["warm_file_sha256"] == table["sha256"],
                    "explicit-page-cache-warm-pass",
                )
                authenticate(row["report"])
                data, _ = real_report(
                    row["report"]["path"], backend, row["policy"], True, reference
                )
                manifests = {
                    field: sorted(
                        [
                            [entry["path"], entry["sha256"], entry["bytes"]]
                            for entry in data[field]
                        ]
                    )
                    for field in ("sources", "native", "data")
                }
                check(
                    manifests == bench["expected_manifests"][row["policy"]],
                    "cross-process-recorded-manifests",
                )
                records[row["policy"]].append(data)
            for field in (
                "construction_seconds",
                "first_seconds",
                "construction_plus_first_seconds",
                "verified_setup_plus_first_seconds",
            ):
                medians = {
                    policy: statistics.median([data[field] for data in group])
                    for policy, group in records.items()
                }
                ratios = [
                    records["master"][i][field] / records["lazy"][i][field]
                    for i in range(7)
                ]
                summary = bench["summary"][field]
                check(
                    medians == summary["median_seconds"]
                    and ratios == summary["paired_ratios"],
                    "recomputed-stage:" + field,
                )
                check(
                    statistics.median(ratios) == summary["paired_median"]
                    and medians["master"] / medians["lazy"]
                    == summary["ratio_of_medians"]
                    and min(ratios) == summary["paired_min"]
                    and max(ratios) == summary["paired_max"],
                    "recomputed-ratios:" + field,
                )
            for label in (
                "first",
                "repeat1",
                "repeat2",
                "near_spin",
                "far_spin",
                "retrograde",
                "return",
            ):
                medians = {
                    policy: statistics.median(
                        [
                            next(
                                call["seconds"]
                                for call in data["calls"]
                                if call["label"] == label
                            )
                            for data in group
                        ]
                    )
                    for policy, group in records.items()
                }
                check(
                    medians == bench["summary"]["call_" + label]["median_seconds"]
                    and medians["master"] / medians["lazy"]
                    == bench["summary"]["call_" + label]["ratio_of_medians"],
                    "recomputed-call:" + label,
                )
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                dict(
                    passes=report["passes"],
                    checks=len(report["checks"]),
                    files=len(report["files"]),
                )
            )
        )


if __name__ == "__main__":
    main()
