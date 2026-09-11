"""Authenticate formal phase evidence and independently replay residual norms."""

# 2026-09-11 17:16 CST (linux): Execute after timed jobs terminate. Read small
# archives/source/native files only; no H5 load, GPU or production mutation.
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
        path = Path(entry["path"])
        check(
            str(path),
            path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
        )
        files[str(path)] = entry["sha256"]

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
        check(case + " anchor", report["reference_anchor_byte_equal"])
        check(
            case + " CPU single thread",
            report["environment"]["OMP_NUM_THREADS"]
            == report["environment"]["OPENBLAS_NUM_THREADS"]
            == "1",
        )
        for entry in [
            report["archive"],
            report["orbit"],
            *report["sources"],
            *report["native"],
            report["few"]["producer"],
            *report["few"]["sources"].values(),
            *report["few"]["native"],
        ]:
            authenticate(entry)
        few_root = Path(report["few"]["source_root"])
        initial = few_root / "src/few/__init__.py"
        pinned = subprocess.check_output(
            [
                "git",
                "show",
                "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191:src/few/__init__.py",
            ],
            cwd=few_root,
        )
        check("top-level source", initial.read_bytes() == pinned)
        files[str(initial)] = sha(initial)
        with np.load(report["archive"]["path"], allow_pickle=False) as arrays:
            for key in arrays.files:
                check(key + " finite", np.all(np.isfinite(arrays[key])))
            m = report["few"]["m"]
            check("phase nodes", arrays["samples_unit_snr"].shape[0] == 2 * m + 1)
            check(
                "coefficients",
                report["fourier_real_coefficients_including_constant"] == 4 * m + 1,
            )
            check("compact bytes", report["compact_payload_bytes"] == 4 * m * 8)
            check(
                "reference content",
                arrays["reference_unit_snr"].tobytes()
                == arrays["samples_unit_snr"][0].tobytes(),
            )
            check(
                "full phase count",
                len(arrays["phases"]) == 22 and len(report["accuracy"]) == 66,
            )
            for rho in (20.0, 100.0, 1000.0):
                reference = rho * arrays["reference_unit_snr"]
                data = reference + arrays["noise"]
                prepared = PhaseLikelihood.prepare(
                    rho * arrays["samples_unit_snr"], data
                ).compact()
                r0 = data.astype(np.longdouble) - rho * arrays[
                    "reference_unit_snr"
                ].astype(np.longdouble)
                rows = [v for v in report["accuracy"] if v["rho"] == rho]
                for phase, actual, row in zip(
                    arrays["phases"], arrays["regenerated_unit_snr"], rows
                ):
                    residual = data.astype(np.longdouble) - rho * actual.astype(
                        np.longdouble
                    )
                    expected = float(-0.5 * np.sum(residual**2 - r0**2))
                    value = float(prepared(phase))
                    check(
                        "independent residual replay",
                        abs(expected - row["direct_delta_logl"]) < 1e-10,
                    )
                    check(
                        "Fourier replay", abs(value - row["fourier_delta_logl"]) < 1e-8
                    )
                    check(
                        "recorded accuracy",
                        row["passes"] and abs(value - expected) <= row["limit"],
                    )
            # Saved coefficients correspond to rho=100, including fixed noise;
            # they do not simply scale as rho^2 when noise remains unchanged.
            p100 = PhaseLikelihood.prepare(
                100 * arrays["samples_unit_snr"],
                100 * arrays["reference_unit_snr"] + arrays["noise"],
            ).compact()
            check(
                "saved compact coefficients",
                np.array_equal(p100.cosine, arrays["fourier_cos"])
                and np.array_equal(p100.sine, arrays["fourier_sin"]),
            )
        for row in report["full_chain_timings"]:
            check(
                "three authenticated repetitions",
                len(row["repeat_max_delta_logl_errors"]) == 3
                and max(row["repeat_max_delta_logl_errors"]) <= 2e-5,
            )
            check(
                "positive durations",
                all(len(v) == 3 and min(v) > 0 for v in row["seconds"].values()),
            )
            ratio = np.median(row["seconds"]["full_direct"]) / np.median(
                row["seconds"]["full_prepared"]
            )
            check("timing arithmetic", ratio == row["speed_ratio"])
        summaries.append(
            dict(
                case=case,
                m=report["few"]["m"],
                compact_bytes=report["compact_payload_bytes"],
                relative_l2_max=report["waveform_whitened_relative_l2_max"],
                max_delta_logl_error=max(
                    v["absolute_error"] for v in report["accuracy"]
                ),
                timings=report["full_chain_timings"],
                cached_timings=report["cached_basis_timings"],
            )
        )
    # Preserve exact legacy producer bytes rather than retroactively associating
    # the initial timing pilot with the hardened formal implementation.
    pilot = json.loads((here / "real_prograde_first.json").read_text())
    for entry in pilot["sources"][:2]:
        archive = here / "pilot_sources" / Path(entry["path"]).name
        check("preserved pilot source", sha(archive) == entry["sha256"])
        files[str(archive)] = sha(archive)
    for name, python in (
        ("main", "/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python"),
        (
            "response",
            "/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python",
        ),
    ):
        target = here / f"math_{name}_final.json"
        if target.exists():
            raise FileExistsError(target)
        command = [python, str(here / "test_phase_likelihood.py")]
        run = subprocess.run(
            command,
            cwd=here,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
            capture_output=True,
            text=True,
        )
        math = dict(
            timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            command=command,
            passes=run.returncode == 0,
            returncode=run.returncode,
            log=run.stdout + run.stderr,
            test_sha256=sha(here / "test_phase_likelihood.py"),
            implementation_sha256=sha(here / "phase_likelihood.py"),
        )
        target.write_text(json.dumps(math, indent=2) + "\n")
        check("math " + name, run.returncode == 0)
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
