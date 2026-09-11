"""Serial fresh-process AB/BA benchmark with byte-checked master references."""

# 2026-09-11 15:58 CST (linux): Explicitly warmed file cache, not cold disk.
# 2026-09-11 16:10 CST (linux): Each fresh worker times model construction
# plus first waveform AFTER backend/native import and CUDA context setup.
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda12x"), required=True)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--verified-first", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError("Refuse to overwrite benchmark evidence")
    if args.pairs < 2:
        raise ValueError("Need at least two pairs for both AB and BA ordering")
    args.artifact_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    worker = root / "collaboration/linux/spin_real_probe.py"
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        backend=args.backend,
        pairs=args.pairs,
        verified_first=args.verified_first,
        protocol="Serial fresh processes, alternating master/lazy and lazy/master. Full H5 SHA256 pass before EVERY worker explicitly warms file cache. ONCE FEW integrity remains enabled. Process is statistical unit.",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        runs=[],
        summary={},
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    records = {"master": [], "lazy": []}
    expected_manifests = {}
    expected_workload = None
    reference = None
    try:
        for pair in range(args.pairs):
            for policy in ("master", "lazy") if pair % 2 == 0 else ("lazy", "master"):
                row = dict(pair=pair, policy=policy, state="running")
                report["runs"].append(row)
                save()
                table = args.data_dir / "ZNAmps_l10_m10_n55_DS2Outer.h5"
                with table.open("rb") as stream:
                    table_sha = hashlib.file_digest(stream, "sha256").hexdigest()
                if (
                    table_sha
                    != "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834"
                ):
                    raise AssertionError("Unexpected amplitude data identity")
                row["warm_file_sha256"] = table_sha
                output = args.artifact_dir / f"pair{pair}_{policy}.json"
                source = (
                    root.parent / "master-reference/src"
                    if policy == "master"
                    else root / "src"
                )
                env = dict(
                    os.environ,
                    FEW_ENABLED_BACKENDS=args.backend,
                    PYTHONPATH=f"{source}:{root.parent / 'artifacts/runtime-master'}",
                    LD_LIBRARY_PATH="/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                )
                command = [
                    sys.executable,
                    str(worker),
                    "--backend",
                    args.backend,
                    "--policy",
                    policy,
                    "--data-dir",
                    str(args.data_dir.resolve()),
                    "--output",
                    str(output.resolve()),
                    "--benchmark",
                ]
                if args.verified_first:
                    command.append("--verified-first")
                if reference is not None:
                    command += ["--reference", str(reference.resolve())]
                row["command"] = command
                print(
                    f"{args.backend} pair {pair + 1}/{args.pairs}: {policy}", flush=True
                )
                save()
                result = subprocess.run(
                    command, cwd=root, env=env, capture_output=True, text=True
                )
                row.update(
                    state="terminal",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                save()
                if result.returncode:
                    raise RuntimeError(
                        "Benchmark worker failed; preserved terminal log"
                    )
                data = json.loads(output.read_text())
                if (
                    not data["passes"]
                    or not data["terminal"]
                    or data["policy"] != policy
                    or data["backend"] != args.backend
                    or data["verified_first"] != args.verified_first
                    or not data["benchmark"]
                    or data["base_commit"] != "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
                    or data["host_derivative_adapter"] != (args.backend == "cuda12x")
                ):
                    raise AssertionError("Unexpected terminal worker identity")
                if data["sources"][0]["sha256"] != report["worker_sha256"]:
                    raise AssertionError("Worker source changed during benchmark")
                # 2026-09-11 16:12 CST (linux): Bind recorded production,
                # native and data identities across every independent process.
                manifests = {}
                for field in ("sources", "native", "data"):
                    entries = [
                        (item["path"], item["sha256"], item["bytes"])
                        for item in data[field]
                    ]
                    if not entries or len({item[0] for item in entries}) != len(
                        entries
                    ):
                        raise AssertionError("Invalid identity manifest: " + field)
                    manifests[field] = sorted(entries)
                if (
                    policy in expected_manifests
                    and manifests != expected_manifests[policy]
                ):
                    raise AssertionError(policy + " source/native/data changed")
                expected_manifests[policy] = manifests
                workload = {
                    field: data[field]
                    for field in ("settings", "platform", "python", "numpy", "xp")
                }
                workload["calls"] = [
                    (
                        call["label"],
                        call.get("parameters"),
                        [
                            (
                                call[field]["key"],
                                call[field]["shape"],
                                call[field]["dtype"],
                                call[field]["sha256"],
                            )
                            for field in ("wave", "modes")
                        ],
                    )
                    for call in data["calls"]
                ]
                if expected_workload is not None and workload != expected_workload:
                    raise AssertionError("Workload or environment changed")
                expected_workload = workload
                report["expected_manifests"] = expected_manifests
                report["expected_workload"] = expected_workload
                row["report"] = {
                    "path": str(output.resolve()),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
                records[policy].append(data)
                if reference is None:
                    reference = output.with_suffix(".npz")
                save()
        for field in (
            "construction_seconds",
            "first_seconds",
            "construction_plus_first_seconds",
            "verified_setup_plus_first_seconds",
        ):
            medians = {
                policy: statistics.median([record[field] for record in records[policy]])
                for policy in records
            }
            ratios = [
                records["master"][i][field] / records["lazy"][i][field]
                for i in range(args.pairs)
            ]
            report["summary"][field] = dict(
                median_seconds=medians,
                ratio_of_medians=medians["master"] / medians["lazy"],
                paired_ratios=ratios,
                paired_median=statistics.median(ratios),
                paired_min=min(ratios),
                paired_max=max(ratios),
            )
        labels = [row["label"] for row in records["master"][0]["calls"]]
        for label in labels:
            medians = {
                policy: statistics.median(
                    [
                        next(
                            row["seconds"]
                            for row in record["calls"]
                            if row["label"] == label
                        )
                        for record in records[policy]
                    ]
                )
                for policy in records
            }
            report["summary"]["call_" + label] = dict(
                median_seconds=medians,
                ratio_of_medians=medians["master"] / medians["lazy"],
            )
        # Preserve complete worker snapshots; summarize only like-for-like
        # first-result RSS and allocator values, not high-water differences.
        report["first_memory_samples"] = {
            policy: [record["calls"][0]["memory"] for record in records[policy]]
            for policy in records
        }
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        save()
        print(
            json.dumps(
                dict(passes=report["passes"], terminal=True, runs=len(report["runs"]))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
