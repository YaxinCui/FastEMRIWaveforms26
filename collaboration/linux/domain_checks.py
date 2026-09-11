"""Record backend-authenticated public-domain test results, including failures."""

# 2026-09-11 14:57 CST (linux): Preserve before/after results and exact source
# identity; independent from the previous waveform-interface repairs.
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
    env = dict(
        os.environ,
        FEW_ENABLED_BACKENDS=args.backend,
        FEW_MODE_DOMAIN_TEST_BACKEND=args.backend,
        FEW_FILE_ALLOW_DOWNLOAD="false",
    )
    command = [sys.executable, "-m", "unittest", "tests.test_mode_domain", "-v"]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    files = [
        root / "src/few/utils/modeselector.py",
        root / "src/few/amplitude/base.py",
        root / "src/few/utils/baseclasses.py",
        root / "tests/test_mode_domain.py",
        Path(__file__),
    ]
    helper = root / "src/few/utils/modeindices.py"
    if helper.exists():
        files.append(helper)
    metadata_lines = [
        line[len("DOMAIN_META=") :]
        for line in result.stdout.splitlines()
        if line.startswith("DOMAIN_META=")
    ]
    actual = json.loads(metadata_lines[0]) if len(metadata_lines) == 1 else None
    provenance_ok = (
        actual is not None
        and actual["backend"] == args.backend
        and actual["xp"] == ("numpy" if args.backend == "cpu" else "cupy")
    )
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        backend=args.backend,
        passes=result.returncode == 0 and provenance_ok,
        provenance_ok=provenance_ok,
        imported=actual,
        returncode=result.returncode,
        command=command,
        log=result.stdout + result.stderr,
        # 2026-09-11 15:18 CST (linux): Separate harness worktree files from
        # imported source identity, particularly during clean-master replay.
        harness_worktree_sha256={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["log"], flush=True)
    raise SystemExit(0 if report["passes"] else (result.returncode or 1))


if __name__ == "__main__":
    main()
