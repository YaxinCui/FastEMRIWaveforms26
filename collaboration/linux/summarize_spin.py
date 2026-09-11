"""Derive human-facing MiB and timing summaries from terminal exp12 workers."""

# 2026-09-11 16:21 CST (linux): Read small reports only, write a new derived
# report; authenticate inputs. Snapshots are NOT continuously sampled peaks.
import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    notes = Path(__file__).resolve().parent
    result = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="Serial seven-pair statistical pilots, warmed OS file cache, model construction/first call after backend initialization. Memory = after-call snapshots, not peaks.",
        MiB_bytes=2**20,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={},
        backends={},
    )
    for suffix in ("cpu", "cuda"):
        path = notes / f"benchmark_{suffix}.json"
        data = json.loads(path.read_text())
        if not data["passes"] or not data["terminal"] or len(data["runs"]) != 14:
            raise AssertionError("Need terminal seven-pair benchmark")
        result["inputs"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = {"master": [], "lazy": []}
        for run in data["runs"]:
            child = Path(run["report"]["path"])
            if (
                hashlib.sha256(child.read_bytes()).hexdigest()
                != run["report"]["sha256"]
            ):
                raise AssertionError("Worker report changed")
            rows[run["policy"]].append(json.loads(child.read_text()))
        output = dict(
            timing=data["summary"],
            memory={},
            lazy_cache_sequence=rows["lazy"][0]["calls"],
        )
        output["lazy_cache_sequence"] = [
            dict(label=call["label"], cache=call["cache"])
            for call in output["lazy_cache_sequence"]
        ]
        fields = ["rss_MiB", "pss_MiB"]
        if suffix == "cuda":
            fields += ["pool_used_MiB", "pool_reserved_MiB"]
        for label in ("first", "repeat2", "retrograde", "return"):
            samples = {policy: [] for policy in rows}
            for policy, records in rows.items():
                for record in records:
                    snap = next(
                        call["memory"]
                        for call in record["calls"]
                        if call["label"] == label
                    )
                    row = dict(
                        rss_MiB=int(snap["smaps_rollup"]["Rss"].split()[0]) / 1024,
                        pss_MiB=int(snap["smaps_rollup"]["Pss"].split()[0]) / 1024,
                    )
                    if suffix == "cuda":
                        row.update(
                            pool_used_MiB=snap["cupy_pool"]["used_bytes"] / 2**20,
                            pool_reserved_MiB=snap["cupy_pool"]["total_bytes"] / 2**20,
                        )
                    samples[policy].append(row)
            output["memory"][label] = {}
            for field in fields:
                medians = {
                    policy: statistics.median([row[field] for row in values])
                    for policy, values in samples.items()
                }
                output["memory"][label][field] = dict(
                    median=medians,
                    reduction_percent=100 * (1 - medians["lazy"] / medians["master"]),
                    samples={
                        policy: [row[field] for row in values]
                        for policy, values in samples.items()
                    },
                )
        result["backends"][data["backend"]] = output
    result["passes"] = result["terminal"] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: dict(
                    joint=value["timing"]["construction_plus_first_seconds"],
                    first_memory=value["memory"]["first"],
                )
                for key, value in result["backends"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
