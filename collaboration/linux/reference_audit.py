"""Expose all fixture outcomes without replacing disputed physical values."""

# 2026-09-11 12:51 CST (linux): Read-only physical data; small JSON evidence is
# generated separately. SciPy uses the same table, not an independent BHPT solver.
import argparse
import ast
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.request import urlopen
import h5py
import numpy as np
from scipy.interpolate import bisplev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-raw", action="store_true")
    args = parser.parse_args()
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir)
    import few
    from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "tests/test_amplitudes.py").read_text())
    fixture = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "KERRECCEQ_AMP_TEST_POINTS")
    module = AmpInterpKerrEccEq(force_backend="cpu")
    encode = lambda x:[float(complex(x).real),float(complex(x).imag)]
    report = {"timestamp_cst":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "base_commit":"47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
              "source":few.__file__,"backend":"cpu","fixtures":[],
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "expected_values_changed":False,"rtol":0.,"atol":1e-9}
    for i, point in enumerate(fixture):
        a, p, e = point["ape"]
        actual = module(a,p,e,1.,specific_modes=[point["lmkn"]])[point["lmkn"]].item()
        error = abs(actual-point["Almkn"])
        row = {"index":i,"mode":point["lmkn"],"ape":point["ape"],
               "expected":encode(point["Almkn"]),"actual":encode(actual),
               "absolute_error":float(error),"passes":bool(error<=1e-9)}
        report["fixtures"].append(row)
        print(json.dumps(row),flush=True)
    with h5py.File(args.data_dir / "ZNAmps_l10_m10_n55_DS2Outer.h5", "r") as h5:
        group = h5["regionA"]
        # m=0-first ordering is checked against FEW's actual mode metadata.
        match = np.flatnonzero((module.l_arr==2)&(module.m_arr==2)&(module.n_arr==0))
        if len(match) != 1:
            raise RuntimeError("Mode (2,2,0) is not uniquely identified")
        index = int(match[0])
        # 2026-09-11 13:02 CST (linux): Bind the selected raw grid node to
        # its physical coordinates and fixture identity, not just array offsets.
        exact_ape = group["ParamsRegionA"][3,10,17]
        if tuple(fixture[2]["lmkn"]) != (2,2,0,0):
            raise ValueError("Disputed fixture identity changed")
        np.testing.assert_allclose(exact_ape, fixture[2]["ape"], rtol=0., atol=5e-9)
        coeff = group["CoeffsRegionA"][3,index]
        w, u = group["w_knots"][()], group["u_knots"][()]
        exact = complex(bisplev(17/32,10/32,(w,u,coeff[0],3,3)),
                        bisplev(17/32,10/32,(w,u,coeff[1],3,3)))
        report["independent_table_evaluation"] = {
            "method":"SciPy bisplev of one spin/mode; no FEW interpolation kernel or forward map",
            "mode_index":index,"spin_index":3,"u":10/32,"w":17/32,
            "exact_ape":exact_ape.tolist(),
            "actual":encode(exact),"fixture_difference":float(abs(exact-fixture[2]["Almkn"])),
            "warning":"Same-table consistency evidence, not an independent physical truth."}
    if args.published_raw:
        # Author's paper-data release: immutable revision, 2.88 MB, held only in
        # memory. Do not copy this into the runtime table or replace old fixtures.
        revision = "7edb856bae835d4540e7aa18b9229d6ae41aceef"
        remote = ("https://raw.githubusercontent.com/cchapmanbird/"
                  f"KerrEccentricEquatorialFigures/{revision}/scripts/Validation/"
                  "Amplitude/BicubicTricubicComparison/A220_slice.h5")
        with urlopen(remote, timeout=30) as response:
            raw = response.read(4_000_000)
        digest = hashlib.sha256(raw).hexdigest()
        expected_digest = "37b190b52013e432099e4fc14318755a1c2fa87a231ab393cb1e6641175f563f"
        if digest != expected_digest:
            raise ValueError("Published raw slice identity mismatch")
        with h5py.File(io.BytesIO(raw), "r") as published:
            parameters = published["paramsA"][10,17,3]
            real, imag = published["modeA"][10,17,3]
            raw_amplitude = complex(real, imag)
        np.testing.assert_array_equal(parameters, exact_ape)
        exact_few = module(*parameters, 1., specific_modes=[(2,2,0,0)])[(2,2,0,0)].item()
        np.testing.assert_allclose(exact_few, raw_amplitude, rtol=0., atol=1e-9)
        report["published_raw_node"] = {
            "url":remote,"revision":revision,"sha256":digest,"bytes":len(raw),
            "release_doi":"10.5281/zenodo.15631641",
            "axes":"u,w,z,[Re,Im]; params last axis a,p,e",
            "node_index":[10,17,3],"exact_ape":parameters.tolist(),
            "published_amplitude":encode(raw_amplitude),
            "few_at_exact_ape":encode(exact_few),
            "few_vs_published_absolute_error":float(abs(exact_few-raw_amplitude)),
            "scipy_vs_published_absolute_error":float(abs(exact-raw_amplitude)),
            "old_fixture_vs_published_absolute_error":float(abs(fixture[2]["Almkn"]-raw_amplitude)),
            "warning":"Published author dataset anchor, not an independently rerun Teukolsky solver; old fixture preserved.",
        }
    report["passes"] = sum(x["passes"] for x in report["fixtures"])
    report["failures"] = len(fixture)-report["passes"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
