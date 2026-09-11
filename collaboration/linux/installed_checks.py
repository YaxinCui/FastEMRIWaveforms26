"""Run selected packaged tests with backend settings and captured evidence."""

# 2026-09-11 13:18 CST (linux): Use installed package only, no src overlay;
# disable downloads and keep test selection consistent with enabled backends.
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
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "wheel": str(args.wheel.resolve()),
              "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
              "runtime": str(args.runtime.resolve()),
              "scope": "Packaged common-phase oracle plus legacy mode-summation tests, not full FEW suite",
              "passes": False, "runs": []}
    command = [sys.executable, "-m", "unittest", "few.tests.test_common_phase",
               "few.tests.test_mode_summation", "-v"]
    for backend in ("cpu", "cuda12x"):
        env = dict(os.environ, PYTHONPATH=str(args.runtime.resolve()),
                   FEW_TEST_BACKENDS=backend, FEW_ENABLED_BACKENDS=backend,
                   FEW_FILE_EXTRA_PATHS=str(args.data_dir.resolve()),
                   FEW_FILE_ALLOW_DOWNLOAD="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        result = subprocess.run(command, env=env, cwd=args.runtime.resolve().parent,
                                capture_output=True, text=True, timeout=300)
        report["runs"].append({"backend": backend, "command": command,
            "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{backend}: exit={result.returncode}\n{result.stderr}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    report["passes"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
