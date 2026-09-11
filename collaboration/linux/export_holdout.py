"""Export fresh pinned-master FEW holdouts without importing any experiment."""

# 2026-09-11 21:18 CST (linux): Serial CPU computation, actual original Kerr
# table, no downloads. Write only new Linux JSON/NPZ provenance and SSB inputs.
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import few
from few.waveform import GenerateEMRIWaveform


def identity(path):
    p = Path(path).resolve()
    with p.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha)


def main():
    here = Path(__file__).resolve().parent
    output = here / "holdout_final.json"
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    os.environ["FEW_FILE_EXTRA_PATHS"] = (
        "/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download"
    )
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    r = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=False,
        terminal=False,
        producer=identity(__file__),
        cases=[],
    )
    try:
        model = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            force_backend="cpu",
            frame="detector",
            return_list=False,
        )
        waves = {}
        for case in ("prograde", "retrograde"):
            params = [
                1e6,
                10.0,
                0.6,
                8.0,
                0.3,
                1.0,
                1.0,
                1.1,
                0.7,
                0.8,
                1.5,
                0.0,
                0.4,
                0.5,
            ]
            if case == "retrograde":
                params[3:6] = [12.0, 0.4, -1.0]
            for index, phase in enumerate((0.1234567, -2.3456789, 5.9182736)):
                params[11] = phase
                key = f"{case}_{index}"
                waves[key] = np.asarray(model(*params, T=0.003, dt=15.0)).copy()
                if not np.all(np.isfinite(waves[key])):
                    raise ValueError("Invalid SSB")
                r["cases"].append(
                    dict(
                        key=key,
                        parameters=params.copy(),
                        settings=dict(T=0.003, dt=15.0),
                        samples=waves[key].size,
                    )
                )
        source_root = Path(few.__file__).resolve().parents[2]
        expected = here.parents[2] / "master-reference"
        if source_root != expected:
            raise AssertionError("Not clean master overlay")
        r["few_sources"] = []
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if (
                (name == "few" or name.startswith("few."))
                and path
                and path.endswith(".py")
            ):
                relative = Path(path).resolve().relative_to(source_root)
                item = identity(path)
                if str(relative) != "src/few/_version.py":
                    raw = subprocess.check_output(
                        [
                            "git",
                            "show",
                            "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191:" + str(relative),
                        ],
                        cwd=source_root,
                    )
                    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                        raise AssertionError(str(relative))
                r["few_sources"].append(item)
        native = Path(
            importlib.import_module("few_backend_cpu.pyinterp").__file__
        ).parent
        r["native"] = [identity(p) for p in sorted(native.glob("*.so"))]
        r["data"] = [
            identity(few.get_file_manager().get_file(n))
            for n in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        ]
        if [x["sha256"] for x in r["data"]] != [
            "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834",
            "db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014",
        ]:
            raise AssertionError("Data identity")
        np.savez_compressed(archive, **waves)
        r["archive"] = identity(archive)
        r["passes"] = True
    except Exception as exc:
        r["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        r["terminal"] = True
        output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(dict(passes=r["passes"], terminal=True, cases=len(r["cases"]))),
            flush=True,
        )


if __name__ == "__main__":
    main()
