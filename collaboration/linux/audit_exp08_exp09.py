"""Authenticate this checkpoint's actual evidence; not a scientific validator."""

# 2026-09-11 14:48 CST (linux): Read both independent worktrees and artifacts,
# preserve original failures, and write a fresh coordinator-only audit record.
import argparse
import hashlib
import json
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
    experiments = root.parent / "FastEMRIWaveforms-experiments"
    exp08 = experiments / "exp08-effective-mode-selection"
    exp09 = experiments / "exp09-distance-conditional-likelihood"
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        checks=[],
        file_sha256={},
    )

    def check(condition, label):
        report["checks"].append(dict(label=label, passes=bool(condition)))
        if not condition:
            raise AssertionError(label)

    def digest(path):
        path = Path(path).resolve()
        key = str(path)
        if key not in report["file_sha256"]:
            with path.open("rb") as stream:
                report["file_sha256"][key] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        return report["file_sha256"][key]

    def walk_identities(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                check(
                    digest(value["path"]) == value["sha256"],
                    "identity: " + value["path"],
                )
            for nested in value.values():
                walk_identities(nested)
        elif isinstance(value, list):
            for nested in value:
                walk_identities(nested)

    def load(path):
        digest(path)
        value = json.loads(path.read_text())
        check(value.get("passes") is True, "terminal passing report: " + str(path))
        walk_identities(value)
        for name, expected in value.get("source_sha256", {}).items():
            candidate = Path(name) if Path(name).is_absolute() else path.parent / name
            check(digest(candidate) == expected, "source: " + str(candidate))
        return value

    def git(path, *command):
        return subprocess.check_output(
            ["git", "-C", str(path), *command], text=True
        ).strip()

    try:
        for path, branch, changed in (
            (exp08, "codex/exp08-effective-mode-selection", "src/few/waveform/base.py"),
            (exp09, "codex/exp09-distance-conditional-likelihood", ""),
        ):
            check(git(path, "branch", "--show-current") == branch, "branch: " + branch)
            check(
                git(path, "rev-parse", "HEAD")
                == "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
                "independent pinned master base: " + branch,
            )
            check(
                git(path, "diff", "--name-only", "--", "src") == changed,
                "production scope: " + branch,
            )
        check(
            git(root, "branch", "--show-current") == "deep-optimization",
            "root branch unchanged",
        )
        check(
            not git(root, "diff", "--name-only", "--", "src", "collaboration/mac"),
            "root source/Mac tracked files unchanged",
        )
        baseline = json.loads(
            (exp08 / "collaboration/linux/master_config_failure.json").read_text()
        )
        check(
            baseline["passes"] is False and baseline["returncode"] != 0,
            "original exp08 failing baseline preserved",
        )
        load(exp08 / "collaboration/linux/config_stock_audited.json")
        real = load(exp08 / "collaboration/linux/real_config_audited.json")
        check(
            len(real["cases"]) == 3
            and all(row["passes"] and len(row["calls"]) == 5 for row in real["cases"]),
            "15 actual Kerr configuration calls",
        )
        for name in ("math_main_audited.json", "math_response_audited.json"):
            math = load(exp09 / "collaboration/linux" / name)
            check(
                math["tests_run"] == 7
                and len(math["rows"]) == 298
                and all(row["passes"] for row in math["rows"]),
                "seven mathematical methods and 298 numeric rows: " + name,
            )
        bridge = load(exp09 / "collaboration/linux/few_ssb_distance.json")
        with np.load(bridge["artifact"]["path"], allow_pickle=False) as arrays:
            embedded = json.loads(str(arrays["metadata_json"]))
            check(
                all(
                    value == bridge[key]
                    for key, value in embedded.items()
                    if key != "passes"
                ),
                "bridge embedded metadata agrees with final report",
            )
            for index, row in enumerate(bridge["runs"]):
                array = arrays[f"strain_{index}"]
                check(
                    array.dtype == np.complex128
                    and np.all(np.isfinite(array))
                    and hashlib.sha256(array.tobytes()).hexdigest() == row["sha256"],
                    f"authenticated bridge array {index}",
                )
        tdi = load(exp09 / "collaboration/linux/tdi_distance_audited.json")
        check(
            len(tdi["likelihood_checks"]) == 99
            and all(row["passes"] for row in tdi["likelihood_checks"]),
            "99 independent residual norm comparisons",
        )
        check(
            len(tdi["full_chain_likelihood_checks"]) == 45
            and all(row["passes"] for row in tdi["full_chain_likelihood_checks"]),
            "45 regenerated-chain likelihood comparisons",
        )
        check(
            len(tdi["timing"]["workloads"]) == 4
            and all(
                len(values) == 9
                for row in tdi["timing"]["workloads"]
                for values in row["raw_seconds"].values()
            ),
            "four workload sizes, eight paths, nine raw timings each",
        )
        with np.load(tdi["artifact"]["path"], allow_pickle=False) as arrays:
            check(
                all(np.all(np.isfinite(arrays[key])) for key in arrays.files),
                "finite saved TDI/likelihood arrays",
            )
            check(
                np.max(
                    np.abs(arrays["direct_delta_logl"] - arrays["quadratic_delta_logl"])
                )
                <= 1e-8,
                "saved timing outputs agree",
            )
        env = json.loads(
            (exp09 / "collaboration/linux/response_environment_build.json").read_text()
        )
        check(
            env["terminal"] and env["rebuild_passes"],
            "response-only rebuild terminated successfully",
        )
        check(
            sum(row["returncode"] == -4 for row in env["commands"]) == 3,
            "three original SIGILL imports preserved",
        )
        for path, expected in env["rebuilt_binary_sha256"].items():
            check(digest(path) == expected, "rebuilt binary identity: " + path)
        for path in (exp08, exp09, root):
            result = subprocess.run(
                ["git", "-C", str(path), "diff", "--check"],
                capture_output=True,
                text=True,
            )
            check(result.returncode == 0, "tracked whitespace: " + str(path))
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["script_sha256"] = digest(__file__)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    passes=report["passes"],
                    checks=len(report["checks"]),
                    files=len(report["file_sha256"]),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
