"""Serial real-table runs: no concurrent GPU/CPU timing workloads."""

# 2026-09-11 13:46 CST (linux): Persist commands, live PIDs and terminal gates;
# every run owns new evidence paths. Failed/old pilots are never overwritten.
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JOBS = {
    "cpu_short": ["--backend", "cpu", "--extended"],
    "cuda_short": ["--backend", "cuda12x", "--reference-host-adapter", "--extended"],
    "cuda_year": [
        "--backend",
        "cuda12x",
        "--reference-host-adapter",
        "--duration",
        "1",
    ],
    "cpu_backwards": [
        "--backend",
        "cpu",
        "--backwards",
        "--extended",
        "--cases",
        "nominal",
        "retrograde",
    ],
    "cuda_backwards": [
        "--backend",
        "cuda12x",
        "--reference-host-adapter",
        "--backwards",
        "--extended",
        "--cases",
        "nominal",
        "retrograde",
    ],
    "cpu_fixed": ["--backend", "cpu", "--selection", "fixed", "--cases", "nominal"],
    "cuda_fixed": [
        "--backend",
        "cuda12x",
        "--reference-host-adapter",
        "--selection",
        "fixed",
        "--cases",
        "nominal",
    ],
    "cpu_psd": [
        "--backend",
        "cpu",
        "--sensitivity",
        "--extended",
        "--cases",
        "nominal",
    ],
    "cuda_psd": [
        "--backend",
        "cuda12x",
        "--reference-host-adapter",
        "--sensitivity",
        "--extended",
        "--cases",
        "nominal",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", choices=JOBS, default=list(JOBS))
    parser.add_argument("--tag", default="final")
    args = parser.parse_args()
    if not args.tag.replace("_", "").isalnum():
        parser.error("Tag must contain letters/digits/underscore")
    root = Path(__file__).resolve().parents[2]
    directory = root / "collaboration/linux"
    manifest = directory / f"validation_suite_{args.tag}.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    script = directory / "observer_scan_probe.py"
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "jobs": [],
        "passes": False,
    }

    def save():
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for name in args.jobs:
        output = directory / f"observer_scan_{name}_{args.tag}.json"
        logfile = directory / f"observer_scan_{name}_{args.tag}.log"
        if output.exists() or logfile.exists():
            raise FileExistsError(output)
        command = [
            sys.executable,
            str(script),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output",
            str(output),
            "--repetitions",
            "5",
            *JOBS[name],
        ]
        env = os.environ.copy()
        env.pop("FEW_ENABLED_BACKENDS", None)
        run = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        row = {
            "name": name,
            "command": command,
            "pid": run.pid,
            "state": "running",
            "output": str(output),
            "log": str(logfile),
        }
        report["jobs"].append(row)
        save()
        print(f"START {name}: PID {run.pid}", flush=True)
        log, _ = run.communicate()
        logfile.write_text(log)
        row.update(state="terminal", returncode=run.returncode)
        payload = json.loads(output.read_text()) if output.exists() else {}
        row["passes"] = (
            run.returncode == 0
            and payload.get("passes") is True
            and payload.get("script_sha256") == report["script_sha256"]
        )
        row["summary"] = [
            {
                "case": item["case"],
                "warm_reuse": item["warm_scan_speedup"],
                "prepare_and_scan": item["measured_prepare_and_scan_speedup"],
            }
            for item in payload.get("cases", [])
            if "warm_scan_speedup" in item
        ]
        save()
        print(json.dumps(row, allow_nan=False), flush=True)
        if not row["passes"]:
            raise RuntimeError(f"{name} failed; preserved log: {logfile}")
    report["passes"] = True
    save()


if __name__ == "__main__":
    main()
