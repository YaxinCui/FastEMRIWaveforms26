"""Preserve pre-fix failures and post-fix backend-authenticated results."""

# 2026-09-11 13:56 CST (linux): Report actual source identities and terminal
# status. Expected baseline failure is retained, never relabeled as success.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda12x"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/few/waveform/base.py",
        root / "src/few/utils/modeselector.py",
        root / "tests/test_selector_frequency_boundary.py",
        Path(__file__),
    ]
    env = os.environ.copy()
    env["FEW_ENABLED_BACKENDS"] = args.backend
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_selector_frequency_boundary",
        "-v",
    ]
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    log = result.stdout + result.stderr
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "backend": args.backend,
        "command": command,
        "returncode": result.returncode,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "log": log,
        "passes": result.returncode == 0
        and f"running with backend '{args.backend}'" in log,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(log, flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
