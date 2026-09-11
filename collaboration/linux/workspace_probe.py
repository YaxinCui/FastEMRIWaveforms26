"""Isolated synthetic and real FP64 trajectory timing with saved evidence."""

# 2026-09-11 12:48 CST (linux): Keep failed/neutral speed results; compare values,
# dense coefficients and knot arrays, not merely a single final coordinate.
import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import few
from few.trajectory.dopr853 import DOPR853
from few.trajectory.inspiral import EMRIInspiral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["FEW_FILE_EXTRA_PATHS"] = args.data_dir
    few.get_config_setter().disable_file_download()
    report = {"timestamp_cst":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "base_commit":"47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
              "source":few.__file__, "cases":[],
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "real_trajectory_T_years":1.,"real_trajectory_err":1e-12,
              "parameters":{}}
    def synthetic(reuse):
        def rhs(t, y, out, extra):
            out[:] = -.2 * y
        stepper = DOPR853(rhs, reuse_stage_workspace=reuse)
        stepper.fix_step = True
        y, t, h = np.ones((6, 1)), np.zeros(1), np.full(1, .001)
        for _ in range(1000):
            stepper.take_step(t, h, y, np.zeros((0, 1)))
        return (y.copy(), t.copy())
    trajectory = EMRIInspiral(func="KerrEccEqFlux", force_backend="cpu")
    def real(reuse, parameters, duration):
        trajectory.inspiral_generator.dopr.reuse_stage_workspace = reuse
        output = trajectory(*parameters, T=duration, err=1e-12)
        return (*[x.copy() for x in output],
                trajectory.inspiral_generator.integrator_spline_coeff.copy())
    workloads = [("synthetic_1000_steps", synthetic)]
    for name, pars in [("nominal", (1e6,10.,.6,8.,.3,1.)),
                       ("high_e", (1e6,10.,.6,12.,.8,1.)),
                       ("retrograde", (1e6,10.,.6,12.,.3,-1.))]:
        report["parameters"][name] = pars
        workloads.append((name, lambda reuse, pars=pars:real(reuse, pars, 1.)))
    for name, function in workloads:
        for reuse in (False, True):
            function(reuse)
            function(reuse)
        samples, outputs = {False:[], True:[]}, {}
        for i in range(15):
            for reuse in ((False, True) if i%2 == 0 else (True, False)):
                start = time.perf_counter()
                outputs[reuse] = function(reuse)
                samples[reuse].append(1e3*(time.perf_counter()-start))
        row = {"name":name,"legacy_ms":samples[False],"workspace_ms":samples[True],
               "legacy_median_ms":float(np.median(samples[False])),
               "workspace_median_ms":float(np.median(samples[True])),
               "speedup":float(np.median(samples[False])/np.median(samples[True])),
               "all_outputs_bitwise_equal":all(a.shape == b.shape and a.dtype == b.dtype
                    and a.tobytes() == b.tobytes() for a,b in zip(outputs[False],outputs[True], strict=True))}
        report["cases"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        # 2026-09-11 13:10 CST (linux): Fail closed after preserving evidence;
        # an inconsistent trajectory must never be reported as a successful run.
        if not row["all_outputs_bitwise_equal"]:
            raise AssertionError(f"Trajectory output changed: {name}")
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
