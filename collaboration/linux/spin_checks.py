"""Run tiny H5/native tests and preserve their terminal result and provenance."""

# 2026-09-11 15:49 CST (linux): No large data or downloads. New reports only.
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
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "unittest", "tests.test_spin_cache", "-v"]
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(
            os.environ,
            FEW_ENABLED_BACKENDS=args.backend,
            FEW_SPIN_CACHE_TEST_BACKEND=args.backend,
        ),
        capture_output=True,
        text=True,
    )
    metadata = [
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("SPIN_CACHE_META=")
    ]
    actual = json.loads(metadata[0]) if len(metadata) == 1 else None
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=True,
        passes=result.returncode == 0
        and actual is not None
        and actual["backend"] == args.backend,
        returncode=result.returncode,
        backend=args.backend,
        imported=actual,
        command=command,
        log=result.stdout + result.stderr,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["log"], flush=True)
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
