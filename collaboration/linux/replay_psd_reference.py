"""Replay an independently repaired master reference through prepared scans."""

# 2026-09-11 14:03 CST (linux): Keep exp06 independent: consume authenticated
# small arrays from exp07, without importing its production source or adapters.
import argparse
import hashlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.waveform import FastKerrEccentricEquatorialFlux
from few.waveform.prepared import PreparedKerrWaveform

from observer_scan_probe import compare, identity, to_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, required=True, help="exp07 JSON manifest"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference.read_text())
    if reference.get("passes") is not True or reference.get("adapter") is not False:
        raise ValueError("Require a completed, adapter-free independent reference")
    artifact = reference["artifact"]
    if identity(artifact["path"]) != artifact:
        raise ValueError("Reference artifact identity mismatch")
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "host": "linux",
        "reference": identity(args.reference),
        "artifact": artifact,
        "script": identity(__file__),
        "prepared_source": identity(inspect.getsourcefile(PreparedKerrWaveform)),
        "adapter": False,
        "scope": "Source strain + synthetic PSD selection; no timing/TDI inference claim",
        "comparisons": [],
        "passes": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
        few.get_config_setter().disable_file_download().set_file_integrity_check("once")
        for name, expected in reference["data"].items():
            actual = identity(few.get_file_manager().get_file(name))
            if (
                actual["sha256"] != expected["sha256"]
                or actual["bytes"] != expected["bytes"]
            ):
                raise ValueError("Reference/runtime data identity mismatch")
        backend = few.get_backend("cuda12x")
        state = {"slope": 1000.0, "calls": 0}

        def sensitivity(f):
            state["calls"] += 1
            return 1 + state["slope"] * f**2

        model = FastKerrEccentricEquatorialFlux(
            force_backend=backend,
            sum_kwargs={"pad_output": True, "odd_len": True},
            mode_selector_kwargs={
                "sensitivity_fn": sensitivity,
                "mode_selection": "threshold",
            },
        )
        with np.load(artifact["path"], allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata_json"]))
            for key in ("parameters", "views", "settings", "psd", "data"):
                if metadata[key] != reference[key]:
                    raise ValueError("Embedded reference metadata mismatch")
            for name, parameters in reference["parameters"].items():
                before = state["calls"]
                prepared = model.prepare_observer_scan(
                    *parameters, **reference["settings"]
                )
                if state["calls"] != before + 1:
                    raise AssertionError(
                        "PSD must be frozen exactly once per preparation"
                    )
                for index, (theta, phi, dist) in enumerate(reference["views"]):
                    candidate = to_host(prepared(theta, phi, dist=dist))
                    candidate_modes = np.stack(
                        [to_host(v) for v in prepared.selected_modes]
                    )
                    key = f"cuda12x_{name}_{index}"
                    np.testing.assert_array_equal(
                        candidate_modes, arrays[key + "_modes"]
                    )
                    report["comparisons"].append(
                        {
                            "case": name,
                            "view": index,
                            "cuda_reference": compare(arrays[key], candidate),
                            "cpu_reference": compare(
                                arrays[f"cpu_{name}_{index}"], candidate
                            ),
                            "mode_hash": hashlib.sha256(
                                candidate_modes.tobytes()
                            ).hexdigest(),
                        }
                    )
                theta, phi, dist = reference["views"][0]
                state["slope"] = 9e12
                replay = to_host(prepared(theta, phi, dist=dist))
                if not compare(arrays[f"cuda12x_{name}_0"], replay)["raw_bytes_equal"]:
                    raise AssertionError("Mutating the external PSD changed a snapshot")
                if state["calls"] != before + 1:
                    raise AssertionError("Prepared render called the external PSD")
                state["slope"] = 1000.0
        if len(report["comparisons"]) != 12:
            raise AssertionError("Expected three intrinsic cases times four views")
        report["passes"] = True
        save()
        print(
            "12 CUDA prepared PSD cases passed independent CPU/CUDA references",
            flush=True,
        )
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
