"""Audit current exp10/11 evidence; never treat earlier failed pilots as passes."""

# 2026-09-11 15:32 CST (linux): Root coordinator verifies sibling sources,
# actual output bytes and packaging provenance without changing either runtime.
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    parent = root.parent / "FastEMRIWaveforms-experiments"
    ten = parent / "exp10-mode-index-domain"
    eleven = parent / "exp11-sampling-dependency"
    checks, identities, cache = [], {}, {}

    def check(name, condition):
        checks.append(dict(name=name, passes=bool(condition)))
        if not condition:
            raise AssertionError(name)

    def digest(path):
        path = Path(path).resolve()
        if path not in cache:
            with path.open("rb") as stream:
                cache[path] = hashlib.file_digest(stream, "sha256").hexdigest()
            identities[str(path)] = cache[path]
        return cache[path]

    def authenticated(entry):
        check(
            "identity:" + entry["path"],
            digest(entry["path"]) == entry["sha256"]
            and Path(entry["path"]).stat().st_size == entry["bytes"],
        )

    def read(path):
        digest(path)
        return json.loads(path.read_text())

    def git(path, *arguments):
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments], text=True
        ).strip()

    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        scope="Current exp10/11 source-boundary, recorded-test and actual-artifact audit; not whole-goal or scientific acceptance",
        checks=checks,
        file_sha256=identities,
    )
    try:
        digest(__file__)
        master = "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
        check(
            "root branch untouched",
            git(root, "branch", "--show-current") == "deep-optimization",
        )
        check(
            "root production untouched",
            git(root, "diff", "--name-only", "HEAD", "--", "src") == "",
        )
        for tree, branch in (
            (ten, "codex/exp10-mode-index-domain"),
            (eleven, "codex/exp11-sampling-dependency"),
        ):
            check(
                branch + " independent base",
                git(tree, "rev-parse", "HEAD") == master
                and git(tree, "branch", "--show-current") == branch,
            )
            check(
                branch + " whitespace",
                subprocess.run(
                    ["git", "-C", str(tree), "diff", "--check"], capture_output=True
                ).returncode
                == 0,
            )
            digest(tree / "collaboration/linux/HANDOFF.md")
        check(
            "exp10 tracked source scope",
            set(git(ten, "diff", "--name-only", "HEAD").splitlines())
            == {
                "src/few/amplitude/base.py",
                "src/few/utils/baseclasses.py",
                "src/few/utils/modeselector.py",
            },
        )
        check(
            "exp11 tracked source scope",
            git(eleven, "diff", "--name-only", "HEAD") == "pyproject.toml",
        )
        for backend, tag, count in (("cpu", "cpu", 43), ("cuda12x", "cuda", 27)):
            for prefix, passed in (("domain_", True), ("master_domain_", False)):
                evidence = read(
                    ten / f"collaboration/linux/{prefix}{tag}_extended.json"
                )
                check(
                    prefix + tag + " terminal test outcome",
                    evidence["passes"] is passed
                    and (evidence["returncode"] == 0) is passed
                    and "Ran 9 tests" in evidence["log"],
                )
                check(
                    prefix + tag + " real backend",
                    evidence["provenance_ok"]
                    and evidence["imported"]["backend"] == backend,
                )
                for path, sha in evidence["imported"]["source_sha256"].items():
                    check("imported:" + path, digest(path) == sha)
                for path, sha in evidence["harness_worktree_sha256"].items():
                    check("harness:" + path, digest(path) == sha)
            before = read(ten / f"collaboration/linux/real_master_{tag}_replay.json")
            after = read(ten / f"collaboration/linux/real_fixed_{tag}.json")
            for label, evidence in (("before", before), ("after", after)):
                check(
                    tag + label + " actual successful terminal",
                    evidence["passes"]
                    and evidence["terminal"]
                    and evidence["backend"] == backend,
                )
                for key in ("sources", "native", "data"):
                    for entry in evidence[key]:
                        authenticated(entry)
                authenticated(evidence["archive"])
                check(
                    tag + label + " output inventory", len(evidence["arrays"]) == count
                )
                check(
                    tag + label + " invalid request behavior",
                    len(evidence["invalid_requests"]) == 2
                    and all(
                        row["rejected"] is (label == "after")
                        for row in evidence["invalid_requests"]
                    ),
                )
                if label == "before":
                    check(
                        tag + " master alias recorded",
                        all(
                            row["byte_equal"] and row["aliases"] == [2, 2, 0, -55]
                            for row in evidence["invalid_requests"]
                        ),
                    )
            check(
                tag + " exact shared baseline inputs",
                before["data"] == after["data"]
                and before["native"] == after["native"]
                and before["sources"][0] == after["sources"][0],
            )
            authenticated(after["reference"])
            check(tag + " reference bound", after["reference"] == before["archive"])
            with (
                np.load(before["archive"]["path"], allow_pickle=False) as old,
                np.load(after["archive"]["path"], allow_pickle=False) as new,
            ):
                check(
                    tag + " archive inventory",
                    set(old.files)
                    == set(new.files)
                    == {row["key"] for row in after["arrays"]},
                )
                for row in after["arrays"]:
                    key = row["key"]
                    a, b = old[key], new[key]
                    check(
                        tag + ":" + key,
                        a.shape == b.shape
                        and a.dtype == b.dtype
                        and a.tobytes() == b.tobytes()
                        and np.all(np.isfinite(b))
                        and row["sha256"] == hashlib.sha256(b.tobytes()).hexdigest()
                        and row["master_byte_equal"],
                    )
            check(
                tag + " new API compatibility",
                len(after["compatibility"]) == (29 if tag == "cpu" else 21)
                and all(row["byte_equal"] for row in after["compatibility"]),
            )
            stock = read(ten / f"collaboration/linux/stock_{tag}.json")
            check(
                tag + " stock regression scope",
                stock["passes"]
                and stock["terminal"]
                and stock["backend"] == backend
                and stock["skipped"] == 0
                and stock["tests"] == (2 if tag == "cpu" else 1),
            )
            for path, sha in stock["source_sha256"].items():
                check("stock-source:" + path, digest(path) == sha)
        build = read(eleven / "collaboration/linux/fixed_sampling_build.json")
        check(
            "candidate metadata test and build terminal",
            build["passes"]
            and build["terminal"]
            and all(
                row["state"] == "terminal" and row["returncode"] == 0
                for row in build["commands"]
            ),
        )
        for path, sha in build["source_sha256"].items():
            check("build-source:" + path, digest(path) == sha)
        failed = read(eleven / "collaboration/linux/master_sampling_failure.json")
        check(
            "expected metadata regression preserved",
            not failed["passes"]
            and failed["terminal"]
            and failed["commands"][0]["returncode"] != 0,
        )
        package = read(eleven / "collaboration/linux/packaging_final.json")
        check(
            "packaging final terminal",
            package["passes"]
            and package["terminal"]
            and all(
                row["state"] == "terminal" and row["returncode"] == 0
                for row in package["commands"]
            ),
        )
        authenticated(package["script"])
        for label in ("before", "after"):
            authenticated(package[label]["identity"])
            authenticated(package[label]["resolution"]["identity"])
            check(label + " BARE only", package[label]["native_files"] == [])
        check(
            "expected sampling dependency boundary",
            package["before"]["sampling"]
            == ["eryn", "fastlisaresponse", "lisaanalysistools", "lisatools"]
            and package["after"]["sampling"]
            == ["eryn", "fastlisaresponse", "lisaanalysistools"],
        )
        for entry in package["pypi"]:
            authenticated(entry["wheel"])
            if entry["cache_source"]:
                authenticated(entry["cache_source"])
            raw_path = (
                Path(entry["wheel"]["path"]).parent
                / f"{entry['distribution']}-{entry['version']}-pypi.json"
            )
            check(
                "pypi-json:" + entry["distribution"],
                digest(raw_path) == entry["metadata_sha256"],
            )
        check(
            "namespace collision recorded",
            len(set(package["namespace_overlap"]["lisatools/__init__.py"].values()))
            == 2,
        )
        check(
            "environment inventory unchanged",
            package["inventory_before"]
            == package["inventory_after"]
            == [{"name": "pip", "version": "26.2.1"}],
        )
        environment = package["resolver_environment"]
        check(
            "actual isolated interpreter",
            Path(environment["prefix"]).resolve()
            == parent / "artifacts/exp11-resolver-env"
            and environment["prefix"] != environment["base_prefix"]
            and Path(environment["executable"]).parent
            == Path(environment["prefix"]) / "bin",
        )
        index = root / "collaboration/linux/EXPERIMENTS_2026-09-11.md"
        digest(index)
        links = re.findall(r"\[exp\d+ handoff\]\(([^)]+)\)", index.read_text())
        check(
            "eleven handoff links",
            len(links) == 11 and all((index.parent / link).is_file() for link in links),
        )
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["terminal"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                dict(passes=report["passes"], checks=len(checks), files=len(identities))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
