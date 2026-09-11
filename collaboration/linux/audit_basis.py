"""Post-run exp15 provenance, numeric replay, operation counts and timing audit."""

# 2026-09-11 17:41 CST (linux): Run after serial benchmark jobs terminate.
# Read sources/native/small NPZ, not the H5 payload. Write new Linux reports.
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from phase_likelihood import PhaseLikelihood


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    output = here / "audit_final.json"
    if output.exists():
        raise FileExistsError(output)
    checks, files, summaries = [], {}, []

    def check(label, condition):
        checks.append(dict(label=label, passes=bool(condition)))
        if not condition:
            raise AssertionError(label)

    def authenticate(entry):
        p = Path(entry["path"])
        check(str(p), p.stat().st_size == entry["bytes"] and sha(p) == entry["sha256"])
        files[str(p)] = entry["sha256"]

    for case in ("prograde", "retrograde"):
        path = here / f"real_{case}_final.json"
        report = json.loads(path.read_text())
        files[str(path)] = sha(path)
        check(
            case + " terminal",
            report["terminal"]
            and report["passes"]
            and report["worker_returncode"] == 0,
        )
        for entry in [
            *report["sources"],
            *report["native"],
            report["archive"],
            report["orbit"],
            *report["few"]["sources"].values(),
            *report["few"]["native"],
            report["few"]["producer"],
            report["few"]["basis_implementation"],
        ]:
            authenticate(entry)
        diagnostic = report["worker_diagnostics"]
        check(
            "worker ownership/restore",
            all(
                diagnostic[k]
                for k in (
                    "worker_basis_ownership_passes",
                    "exception_caught",
                    "callable_restoration_passes",
                    "post_failure_wave_byte_equal",
                )
            ),
        )
        check(
            "frozen native",
            len(diagnostic["frozen_native_checks"]) == 3
            and max(x["relative_l2"] for x in diagnostic["frozen_native_checks"])
            <= 1e-10,
        )
        check(
            "nonzero reference SSB",
            len(report["offset_reference_checks"]) == 3
            and max(x["relative_l2"] for x in report["offset_reference_checks"])
            <= 1e-8,
        )
        m = report["few"]["m"]
        for op in report["operations"]:
            if op["operation"] not in ("blackbox", "grouped"):
                continue
            expected = 1 if op["operation"] == "grouped" else 2 * m + 1
            check(
                "symmetric intrinsic count",
                all(
                    op["metadata"][key] == expected
                    for key in ("trajectory_calls", "amplitude_calls", "outer_calls")
                ),
            )
            check(
                "batch payload shape",
                op["shape"] == [2 * m + 1, report["few"]["samples"]],
            )
        with np.load(report["archive"]["path"], allow_pickle=False) as arrays:
            for key in arrays.files:
                check(key + " finite", np.all(np.isfinite(arrays[key])))
            check("compact size", report["compact_bytes"] == 4 * m * 8)
            check(
                "shape",
                arrays["new_ssb_basis"].shape == (2 * m + 1, report["few"]["samples"]),
            )
            check(
                "reference anchor",
                arrays["old_ssb_samples"][0].tobytes()
                == arrays["ssb_reference"].tobytes(),
            )
            check(
                "phase coverage",
                len(arrays["phases"]) == 22 and len(report["accuracy"]) == 66,
            )
            for rho in (20.0, 100.0, 1000.0):
                data = rho * arrays["reference_unit"] + arrays["noise"]
                r0 = data.astype(np.longdouble) - rho * arrays["reference_unit"].astype(
                    np.longdouble
                )
                compact = PhaseLikelihood.from_basis(
                    rho * arrays["new_tdi_basis_unit"],
                    data,
                    rho * arrays["reference_unit"],
                ).compact()
                rows = [row for row in report["accuracy"] if row["rho"] == rho]
                for phase, truth, row in zip(
                    arrays["phases"], arrays["regenerated_unit"], rows
                ):
                    residual = data.astype(np.longdouble) - rho * truth.astype(
                        np.longdouble
                    )
                    expected = float(-0.5 * np.sum(residual**2 - r0**2))
                    value = float(compact(phase))
                    check(
                        "direct residual replay", abs(expected - row["direct"]) < 1e-10
                    )
                    check("compact replay", abs(value - row["grouped"]) < 1e-8)
                    check(
                        "full regenerated accuracy",
                        row["passes"] and abs(expected - value) <= row["limit"],
                    )
            compact = PhaseLikelihood.from_basis(
                100 * arrays["new_tdi_basis_unit"],
                100 * arrays["reference_unit"] + arrays["noise"],
                100 * arrays["reference_unit"],
            ).compact()
            check(
                "coefficient identity",
                np.array_equal(compact.cosine, arrays["compact_cosine"])
                and np.array_equal(compact.sine, arrays["compact_sine"]),
            )
        timing_summary = {}
        if report["benchmark"]:
            check("five preparation pairs", len(report["preparation_timings"]) == 5)
            for row in report["preparation_timings"]:
                check(
                    "prep output accuracy",
                    np.isfinite(row["max_delta_logl_error"])
                    and row["max_delta_logl_error"] <= 2e-5,
                )
                for variant, values in row["timings"].items():
                    check("response not skipped", values["response_calls"] == 2 * m + 1)
                    check(
                        "stage timing",
                        0
                        < values["worker_seconds"]
                        <= values["worker_transfer_seconds"]
                        <= values["joint_seconds"],
                    )
            for variant in ("blackbox", "grouped"):
                timing_summary[variant] = {
                    key: float(
                        np.median(
                            [
                                r["timings"][variant][key]
                                for r in report["preparation_timings"]
                            ]
                        )
                    )
                    for key in (
                        "joint_seconds",
                        "worker_seconds",
                        "worker_transfer_seconds",
                        "response_fft_seconds",
                        "statistics_seconds",
                    )
                }
            timing_summary["speed_ratio"] = (
                timing_summary["blackbox"]["joint_seconds"]
                / timing_summary["grouped"]["joint_seconds"]
            )
            for row in report["scan_timings"]:
                check(
                    "scan output accuracy",
                    len(row["errors"]) == 3 and max(row["errors"]) <= 2e-5,
                )
                check(
                    "scan arithmetic",
                    row["speed_ratio"]
                    == np.median(row["seconds"]["blackbox"])
                    / np.median(row["seconds"]["grouped"]),
                )
            for row in report["cached_timings"]:
                check(
                    "cached arithmetic",
                    row["speed_ratio"]
                    == np.median(row["seconds"]["direct"])
                    / np.median(row["seconds"]["compact"]),
                )
        summaries.append(
            dict(
                case=case,
                ssb_error=report["ssb_regeneration_relative_l2_max"],
                tdi_error=report["tdi_regeneration_relative_l2_max"],
                max_logl_error=max(x["error"] for x in report["accuracy"]),
                preparation=timing_summary,
                scan=report["scan_timings"],
                cached=report["cached_timings"],
            )
        )
    pilot = json.loads((here / "real_prograde_first.json").read_text())
    for name in ("single_pass_basis.py", "few_phase_server.py", "phase_real_probe.py"):
        entry = next(x for x in pilot["sources"] if Path(x["path"]).name == name)
        check(
            "preserved pilot " + name,
            sha(here / "pilot_sources" / name) == entry["sha256"],
        )
        files[str(here / "pilot_sources" / name)] = entry["sha256"]
    for stack, python in (
        ("main", "/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python"),
        (
            "response",
            "/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python",
        ),
    ):
        target = here / f"math_{stack}_final.json"
        if target.exists():
            raise FileExistsError(target)
        runs = []
        for name in ("test_single_pass_basis.py", "test_phase_likelihood.py"):
            command = [python, str(here / name)]
            run = subprocess.run(
                command,
                cwd=here,
                capture_output=True,
                text=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
            )
            runs.append(
                dict(
                    command=command,
                    returncode=run.returncode,
                    log=run.stdout + run.stderr,
                    test_sha256=sha(here / name),
                )
            )
            check(stack + " " + name, run.returncode == 0)
        target.write_text(
            json.dumps(
                dict(
                    timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    passes=True,
                    runs=runs,
                ),
                indent=2,
            )
            + "\n"
        )
        files[str(target)] = sha(target)
    result = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=True,
        checks=checks,
        files=files,
        summaries=summaries,
        audit_sha256=sha(__file__),
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(passes=True, checks=len(checks), files=len(files), summaries=summaries)
        )
    )


if __name__ == "__main__":
    main()
