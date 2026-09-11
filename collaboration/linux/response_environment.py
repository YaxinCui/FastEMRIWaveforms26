"""Authenticate the response-only CPU dependency failure and scoped rebuild."""

# 2026-09-11 14:30 CST (linux): No main FEW environment mutation. Disable core
# files and bytecode; preserve original imports, ISA crash and all build logs.
import argparse
import hashlib
import json
import os
import resource
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        CMAKE_BUILD_PARALLEL_LEVEL="2",
    )
    python = str(args.environment.resolve() / "bin/python")
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "environment": str(args.environment.resolve()),
        "terminal": False,
        "commands": [],
        "sources": {
            str(Path(__file__).resolve()): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(command):
        print("Running: " + " ".join(command), flush=True)
        item = {"command": command, "state": "running"}
        report["commands"].append(item)
        save()
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        item.update(
            state="terminal",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        save()
        print(f"returncode={result.returncode}", flush=True)
        return result

    modules = [
        "numpy",
        "scipy.interpolate",
        "h5py",
        "llvmlite.binding",
        "numba",
        "gpubackendtools",
        "multispline",
        "fastlisaresponse_backend_cpu.responselisa",
        "lisatools_backend_cpu.pycppdetector",
        "fastlisaresponse",
        "lisatools.detector",
    ]
    try:
        run(["lscpu"])
        run(
            [
                python,
                "-B",
                "-c",
                "import importlib.metadata as m,json; print(json.dumps(sorted([(d.metadata['Name'],d.version) for d in m.distributions()])))",
            ]
        )
        report["original_binary_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.environment.glob(
                "lib/python*/site-packages/lisatools_backend_cpu/*.so"
            )
        }
        for module in modules:
            run([python, "-B", "-c", "import " + module])
        run(
            [
                "gdb",
                "--batch",
                "-ex",
                "run",
                "-ex",
                "x/6i $pc",
                "-ex",
                "bt",
                "--args",
                python,
                "-B",
                "-c",
                "import lisatools_backend_cpu.pycppdetector",
            ]
        )
        if args.source_directory:
            source = args.source_directory.resolve()
            if source.exists():
                raise FileExistsError(source)
            result = run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "v1.2.8",
                    "--no-recurse-submodules",
                    "https://github.com/mikekatz04/LISAanalysistools.git",
                    str(source),
                ]
            )
            if result.returncode:
                raise RuntimeError("Source checkout failed")
            run(["git", "-C", str(source), "rev-parse", "HEAD"])
            for name in ("CMakeLists.txt", "pyproject.toml"):
                report["sources"][str(source / name)] = hashlib.sha256(
                    (source / name).read_bytes()
                ).hexdigest()
            result = run(
                [
                    "/home/ubuntu/.local/bin/uv",
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--no-deps",
                    "--reinstall-package",
                    "lisaanalysistools",
                    "--config-setting",
                    "cmake.define.LISATOOLS_MARCH=x86-64",
                    "--config-setting",
                    "cmake.define.LISATOOLS_WITH_GPU=OFF",
                    str(source),
                ]
            )
            if result.returncode:
                raise RuntimeError("Scoped response dependency source build failed")
            for module in (
                "lisatools_backend_cpu.pycppdetector",
                "lisatools.detector",
                "fastlisaresponse",
            ):
                if run([python, "-B", "-c", "import " + module]).returncode:
                    raise RuntimeError("Rebuilt response dependency still fails")
            report["rebuilt_binary_sha256"] = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.environment.glob(
                    "lib/python*/site-packages/lisatools_backend_cpu/*.so"
                )
            }
            report["rebuild_passes"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["terminal"] = True
        save()


if __name__ == "__main__":
    main()
