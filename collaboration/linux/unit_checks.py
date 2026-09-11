"""Run and retain backend-authenticated low-memory tests."""

# 2026-09-11 13:45 CST (linux): FEW master tests choose the first enabled
# backend, not FEW_TEST_BACKENDS. Assert the backend printed by each subprocess.
import hashlib
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    # 2026-09-11 13:51 CST (linux): Preserve earlier test evidence after moving
    # probe-only tests out of the package; use an explicit new output path.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    files = [
        root / "src/few/waveform/prepared.py",
        root / "tests/test_prepared_observer.py",
        root / "collaboration/linux/observer_scan_probe.py",
        root / "collaboration/linux/test_observer_probe.py",
        Path(__file__),
    ]
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "runs": [],
        "passes": False,
    }
    for backend in ("cpu", "cuda12x"):
        env = os.environ.copy()
        env["FEW_ENABLED_BACKENDS"] = backend
        command = [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_prepared_observer",
            "collaboration.linux.test_observer_probe",
            "-v",
        ]
        run = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
        log = run.stdout + run.stderr
        accepted = (
            run.returncode == 0
            and f"running with backend '{backend}'" in log
            and "Ran 8 tests" in log
        )
        report["runs"].append(
            {
                "backend": backend,
                "command": command,
                "returncode": run.returncode,
                "passes": accepted,
                "log": log,
            }
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{backend}: {'PASS' if accepted else 'FAIL'}", flush=True)
        if not accepted:
            raise RuntimeError(log)
    report["passes"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
