"""Run source-bound contract tests and preserve expected master failures."""

# 2026-09-11 16:40 CST (linux): New reports only. Separate successful repair
# tests from a successful reproduction of the pre-existing contract defect.
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda12x"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    command = [sys.executable, "-m", "unittest", "tests.test_retrograde_contract", "-v"]
    run = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=dict(
            os.environ,
            FEW_ENABLED_BACKENDS=args.backend,
            FEW_RETROGRADE_TEST_BACKEND=args.backend,
        ),
        capture_output=True,
        text=True,
    )
    metadata = [
        line.split("=", 1)[1]
        for line in run.stdout.splitlines()
        if line.startswith("RETROGRADE_META=")
    ]
    actual = json.loads(metadata[0]) if len(metadata) == 1 else None
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=True,
        passes=run.returncode == 0
        and actual is not None
        and actual["backend"] == args.backend,
        backend=args.backend,
        returncode=run.returncode,
        imported=actual,
        command=command,
        log=run.stdout + run.stderr,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(passes=report["passes"], returncode=run.returncode, imported=actual)
        ),
        flush=True,
    )
    print("\n".join(report["log"].splitlines()[-5:]), flush=True)
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
