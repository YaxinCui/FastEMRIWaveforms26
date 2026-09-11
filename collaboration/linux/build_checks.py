"""Preserve the expected initial regression failure and BARE wheel build log."""

# 2026-09-11 15:06 CST (linux): A BARE wheel proves metadata only. Do not
# install it over any validated native FEW runtime.
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.wheel_dir and args.wheel_dir.exists()):
        raise FileExistsError("Output paths must be new")
    root = Path(__file__).resolve().parents[2]
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        commands=[],
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                root / "pyproject.toml",
                root / "tests/test_sampling_dependencies.py",
            )
        },
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        commands = [
            [sys.executable, str(root / "tests/test_sampling_dependencies.py"), "-v"]
        ]
        if args.wheel_dir:
            commands.append(
                [
                    "/home/ubuntu/.local/bin/uv",
                    "build",
                    "--wheel",
                    "--python",
                    sys.executable,
                    "--out-dir",
                    str(args.wheel_dir.resolve()),
                    "--config-setting",
                    "cmake.define.FEW_WITH_GPU=BARE",
                    str(root),
                ]
            )
        for command in commands:
            item = dict(command=command, state="running")
            report["commands"].append(item)
            save()
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            item.update(
                returncode=result.returncode,
                state="terminal",
                stdout=result.stdout,
                stderr=result.stderr,
            )
            print(result.stdout + result.stderr, flush=True)
            save()
        report["passes"] = all(row["returncode"] == 0 for row in report["commands"])
    finally:
        report["terminal"] = True
        save()
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
