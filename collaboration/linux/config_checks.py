"""Record actual test execution and source identities before/after repair."""

# 2026-09-11 14:13 CST (linux): Baseline failure is valid evidence, not a pass;
# separate output paths preserve every observed result.
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
    parser.add_argument("--include-stock-selector", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    # 2026-09-11 14:42 CST (linux): Include the existing selector regression
    # without allowing test-driven downloads or changing backend selection.
    env = dict(os.environ, FEW_ENABLED_BACKENDS="cpu", FEW_FILE_ALLOW_DOWNLOAD="false")
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_effective_mode_selection",
        "-v",
    ]
    if args.include_stock_selector:
        command.insert(-1, "tests.test_mode_selector")
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    sources = [
        root / "src/few/waveform/base.py",
        root / "tests/test_effective_mode_selection.py",
        Path(__file__),
    ]
    if args.include_stock_selector:
        sources.append(root / "tests/test_mode_selector.py")
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "command": command,
        "returncode": result.returncode,
        "passes": result.returncode == 0,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "log": result.stdout + result.stderr,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["log"], flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
