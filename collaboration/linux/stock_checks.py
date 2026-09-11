"""Record limited unchanged upstream regressions with backend/source identity."""

# 2026-09-11 15:28 CST (linux): CPU selector test uses Schwarzschild data;
# stock amplitude test checks only its last fixture due to the known exp03 bug.
import argparse
import hashlib
import importlib
import inspect
import io
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import few
from few.amplitude.base import AmplitudeBase
from few.utils.modeselector import ModeSelector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda12x"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    # 2026-09-11 15:32 CST (linux): A directly executed script does not put
    # worktree root on sys.path. Import tests from this tree, not installed FEW.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    names = ["tests.test_amplitudes"]
    if args.backend == "cpu":
        names.append("tests.test_mode_selector")
    modules = [importlib.import_module(name) for name in names]
    source_paths = (
        [Path(__file__).resolve()]
        + [Path(module.__file__).resolve() for module in modules]
        + [
            Path(inspect.getsourcefile(cls)).resolve()
            for cls in (AmplitudeBase, ModeSelector)
        ]
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(names)
    )
    # Class backend is removed by tearDownClass; authenticate supported backend
    # selection independently via the same FEW resolver used by stock tests.
    from few.utils.globals import get_first_backend
    from few.amplitude.ampinterp2d import AmpInterpKerrEccEq

    selected = get_first_backend(AmpInterpKerrEccEq.supported_backends()).name
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        terminal=True,
        passes=result.wasSuccessful()
        and not result.skipped
        and selected == args.backend,
        backend=selected,
        tests=result.testsRun,
        skipped=len(result.skipped),
        log=stream.getvalue(),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        scope="Unmodified stock tests only; Kerr fixture indentation defect is not repaired or certified here",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), flush=True)
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
