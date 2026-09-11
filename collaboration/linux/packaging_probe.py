"""Audit real wheel metadata, namespace collisions and dry-run resolution."""

# 2026-09-11 15:05 CST (linux): Inspect downloaded archives without importing
# them. Resolve both extras in a separate pip env; never install either extra.
import argparse
import email
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def metadata(path):
    with zipfile.ZipFile(path) as wheel:
        candidates = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1:
            raise AssertionError("Ambiguous wheel metadata")
        message = email.message_from_bytes(wheel.read(candidates[0]))
        all_requirements = message.get_all("Requires-Dist", [])
        sampling = []
        for text in all_requirements:
            req = Requirement(text)
            if (
                req.marker is not None
                and req.marker.evaluate({"extra": "sampling"})
                and not req.marker.evaluate({"extra": ""})
            ):
                sampling.append(canonicalize_name(req.name))
        return dict(
            identity=identity(path),
            name=message["Name"],
            version=message["Version"],
            requirements=all_requirements,
            # 2026-09-11 15:21 CST (linux): Permit the expected dirty-version
            # suffix only; preserve every other metadata header and the body.
            stable_headers={
                key: message.get_all(key)
                for key in message.keys()
                if key.lower() not in ("version", "requires-dist")
            },
            description=message.get_payload(),
            sampling=sorted(sampling),
            native_files=[name for name in wheel.namelist() if name.endswith(".so")],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-wheel", type=Path, required=True)
    parser.add_argument("--after-wheel", type=Path, required=True)
    parser.add_argument("--resolver-python", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--archive-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError("Output or artifact directory already exists")
    args.artifact_dir.mkdir(parents=True)
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        scope="BARE metadata-only wheels and dry-run dependency resolution, not runnable FEW/sampling acceptance",
        script=identity(__file__),
        driver_executable=sys.executable,
        commands=[],
        pypi=[],
    )
    # 2026-09-11 15:23 CST (linux): No inherited alternate pip indexes,
    # constraints or Python module overrides. Keep network proxy configuration.
    cleaned = [
        key
        for key in os.environ
        if key.startswith("PIP_") or key in ("PYTHONPATH", "PYTHONHOME")
    ]
    child_env = {key: value for key, value in os.environ.items() if key not in cleaned}
    child_env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1")
    report["resolver_config"] = dict(
        removed_variable_names=cleaned,
        pip_config_file=os.devnull,
        index="https://pypi.org/simple",
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(command):
        row = dict(command=command, state="running")
        report["commands"].append(row)
        save()
        print("Running: " + " ".join(command), flush=True)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=child_env,
        )
        row.update(
            state="terminal",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        save()
        if result.returncode:
            raise RuntimeError("Resolver command failed; see preserved log")
        return result.stdout

    try:
        # 2026-09-11 15:18 CST (linux): Do NOT resolve the venv Python symlink
        # into the base interpreter. Authenticate isolation in the child itself.
        resolver = str(args.resolver_python.absolute())
        environment = json.loads(
            run(
                [
                    resolver,
                    "-c",
                    "import json,sys; print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix)))",
                ]
            )
        )
        report["resolver_environment"] = environment
        if (
            Path(environment["prefix"]).resolve()
            != args.resolver_python.absolute().parent.parent.resolve()
            or environment["prefix"] == environment["base_prefix"]
        ):
            raise AssertionError(
                "Resolver is not running in the requested isolated virtual environment"
            )
        inventory_before = json.loads(
            run([resolver, "-m", "pip", "list", "--format=json"])
        )
        report["inventory_before"] = inventory_before
        for label, path in (("before", args.before_wheel), ("after", args.after_wheel)):
            report[label] = metadata(path)
            if report[label]["native_files"]:
                raise AssertionError("Expected intentionally nonfunctional BARE wheels")
        if set(report["before"]["sampling"]) - set(report["after"]["sampling"]) != {
            "lisatools"
        } or set(report["after"]["sampling"]) - set(report["before"]["sampling"]):
            raise AssertionError(
                "Sampling dependencies changed beyond removing lisatools"
            )
        for field in ("stable_headers", "description"):
            if report["before"][field] != report["after"][field]:
                raise AssertionError(f"Unexpected metadata change in {field}")
        old = Counter(
            str(Requirement(item)) for item in report["before"]["requirements"]
        )
        new = Counter(
            str(Requirement(item)) for item in report["after"]["requirements"]
        )
        removed = old - new
        if (
            sum(removed.values()) != 1
            or canonicalize_name(Requirement(next(iter(removed))).name) != "lisatools"
            or new - old
        ):
            raise AssertionError("Unexpected non-sampling dependency change")
        archives = {}
        for name, version in (("lisatools", "0.5.1"), ("lisaanalysistools", "1.2.8")):
            url = f"https://pypi.org/pypi/{name}/{version}/json"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
            data = json.loads(raw)
            (args.artifact_dir / f"{name}-{version}-pypi.json").write_bytes(raw)
            wheels = [
                entry for entry in data["urls"] if entry["filename"].endswith(".whl")
            ]
            if name == "lisaanalysistools":
                wheels = [
                    entry
                    for entry in wheels
                    if "cp312-cp312" in entry["filename"]
                    and "manylinux" in entry["filename"]
                    and "x86_64" in entry["filename"]
                ]
            if len(wheels) != 1:
                raise AssertionError("Ambiguous audited PyPI wheel")
            entry = wheels[0]
            cached = (
                args.archive_cache / entry["filename"] if args.archive_cache else None
            )
            if cached is not None and cached.is_file():
                payload = cached.read_bytes()
            else:
                with urllib.request.urlopen(entry["url"], timeout=60) as response:
                    payload = response.read()
            if hashlib.sha256(payload).hexdigest() != entry["digests"]["sha256"]:
                raise AssertionError("Downloaded wheel digest mismatch")
            target = args.artifact_dir / entry["filename"]
            target.write_bytes(payload)
            with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
                names = {
                    path
                    for path in wheel.namelist()
                    if path.startswith("lisatools/") and not path.endswith("/")
                }
                archives[name] = {
                    path: hashlib.sha256(wheel.read(path)).hexdigest() for path in names
                }
            report["pypi"].append(
                dict(
                    distribution=name,
                    version=version,
                    metadata_url=url,
                    metadata_sha256=hashlib.sha256(raw).hexdigest(),
                    summary=data["info"]["summary"],
                    project_urls=data["info"].get("project_urls"),
                    wheel_url=entry["url"],
                    cache_source=identity(cached)
                    if cached is not None and cached.is_file()
                    else None,
                    wheel=identity(target),
                )
            )
        overlap = sorted(
            set(archives["lisatools"]) & set(archives["lisaanalysistools"])
        )
        report["namespace_overlap"] = {
            path: {name: mapping[path] for name, mapping in archives.items()}
            for path in overlap
        }
        if (
            "lisatools/__init__.py" not in overlap
            or len(set(report["namespace_overlap"]["lisatools/__init__.py"].values()))
            != 2
        ):
            raise AssertionError("Expected differing import initialization files")
        for label, wheel in (
            ("before", args.before_wheel),
            ("after", args.after_wheel),
        ):
            resolution = args.artifact_dir / f"{label}_resolution.json"
            run(
                [
                    resolver,
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--ignore-installed",
                    "--only-binary=:all:",
                    "--index-url",
                    "https://pypi.org/simple",
                    "--report",
                    str(resolution.resolve()),
                    str(wheel.resolve()) + "[sampling]",
                ]
            )
            resolved = json.loads(resolution.read_text())
            names = {
                canonicalize_name(row["metadata"]["name"]): row["metadata"]["version"]
                for row in resolved["install"]
            }
            report[label]["resolution"] = dict(
                identity=identity(resolution), distributions=names
            )
            # 2026-09-11 15:21 CST (linux): Bind namespace inspection to the
            # actual resolver archives, not merely to distribution names.
            for audited in report["pypi"]:
                name = audited["distribution"]
                if name == "lisatools" and label == "after":
                    continue
                rows = [
                    row
                    for row in resolved["install"]
                    if canonicalize_name(row["metadata"]["name"]) == name
                ]
                if (
                    len(rows) != 1
                    or rows[0]["metadata"]["version"] != audited["version"]
                    or rows[0]["download_info"]["archive_info"]["hashes"]["sha256"]
                    != audited["wheel"]["sha256"]
                ):
                    raise AssertionError(
                        "Resolved version/archive differs from the namespace-audited wheel"
                    )
            if not {
                "eryn",
                "fastlisaresponse",
                "lisaanalysistools",
            } <= names.keys() or ("lisatools" in names) != (label == "before"):
                raise AssertionError(
                    "Resolved extra does not match the expected before/after boundary"
                )
        inventory_after = json.loads(
            run([resolver, "-m", "pip", "list", "--format=json"])
        )
        report["inventory_after"] = inventory_after
        if inventory_before != inventory_after:
            raise AssertionError(
                "Installed distribution inventory changed across dry runs"
            )
        installed = {canonicalize_name(item["name"]) for item in inventory_after}
        if installed & {
            "fastemriwaveforms",
            "lisatools",
            "lisaanalysistools",
            "fastlisaresponse",
            "eryn",
        }:
            raise AssertionError("Dry-run unexpectedly installed scoped packages")
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        save()
        print(json.dumps(dict(passes=report["passes"], terminal=True)), flush=True)


if __name__ == "__main__":
    main()
