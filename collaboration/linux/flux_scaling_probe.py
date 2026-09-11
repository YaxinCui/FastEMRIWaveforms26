"""Controlled trajectory error study, not a new astrophysical flux model."""

# 2026-09-11 12:59 CST (linux): Keep this experiment outside production FEW.
# A known time-reparameterization identity separates numerical error from an
# intentionally imposed uniform dissipative-rate change. Reports are generated.
import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.utils.constants import YRSID_SI

BASE = "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
CASES = {
    "nominal": (1e6, 10.0, 0.7, 12.0, 0.3, 1.0),
    "retrograde": (1e6, 10.0, 0.9, 15.0, 0.3, -1.0),
    "high_e": (1e6, 10.0, 0.6, 12.0, 0.8, 1.0),
}
PHASES = {"Phi_phi0": 0.3, "Phi_theta0": 0.4, "Phi_r0": 0.5}


class UniformRateScaledKerr(KerrEccEqFlux):
    """Multiply only (pdot, edot) by 1+delta; retain orbital frequencies."""

    def add_fixed_parameters(self, m1, m2, a, additional_args=None):
        values = np.asarray(additional_args, dtype=np.float64)
        if values.shape != (1,):
            raise ValueError("Supply exactly one positional rate perturbation.")
        delta = float(values[0])
        if not np.isfinite(delta) or delta <= -1.0:
            raise ValueError("delta must be finite and greater than -1.")
        super().add_fixed_parameters(m1, m2, a, additional_args=values)
        self.rate_scale = 1.0 + delta

    def modify_rhs(self, ydot, y, **kwargs):
        if self.use_ELQ or self.flux_output_convention != "pex":
            raise ValueError("This experiment requires pex states and rates.")
        # ODEBase.__call__ ignores the hook return value: mutate in place.
        ydot[:2] *= self.rate_scale
        return ydot


def make_run(parameters, delta, err, duration, *, stock=False):
    trajectory = EMRIInspiral(
        func="KerrEccEqFlux" if stock else UniformRateScaledKerr,
        integrate_constants_of_motion=False,
        flux_output_convention="pex",
        force_backend="cpu",
    )
    extra = () if stock else (delta,)
    started = time.perf_counter()
    output = trajectory(
        *parameters, *extra, T=duration, dt=10.0, err=err,
        DENSE_STEPPING=False, **PHASES,
    )
    return trajectory, output, time.perf_counter() - started


def evaluate(trajectory, times):
    end = float(trajectory.integrator_spline_t[-1])
    if times.ndim != 1 or not np.all((times >= 0.0) & (times <= end)):
        raise ValueError("Never accept the interpolator's out-of-range zeros.")
    values = trajectory.inspiral_generator.eval_integrator_spline(times)
    # 2026-09-11 13:10 CST (linux): Empty phase columns must not pass finite().
    if values.shape != (len(times), 6):
        raise ValueError("Dense trajectory must have six state columns")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite trajectory output")
    # Public dense interpolation already restores phases to physical radians.
    return values


def compare(reference, candidate, parameters, delta, points, requested_seconds):
    scale = 1.0 + delta
    end_ref = float(reference.integrator_spline_t[-1])
    end_cand = float(candidate.integrator_spline_t[-1])
    # 2026-09-11 13:04 CST (linux): Adaptive integration can retain a dense
    # interval beyond requested T. Cap the analysis horizon to avoid comparing
    # different durations when tightening tolerances; retain the pilot report.
    end = 0.9 * min(requested_seconds, end_ref, scale * end_cand)
    if not np.isfinite(end) or end <= 0.0:
        raise RuntimeError("No positive common time interval")
    times = np.linspace(0.0, end, points)
    ref = evaluate(reference, times)
    got = evaluate(candidate, times / scale)
    phase_ref = ref[:, 3:] - ref[0, 3:]
    phase_got = got[:, 3:] - got[0, 3:]
    state = np.array([*parameters[3:], 0.0, 0.0, 0.0])
    # The ODE instances have been bound by the completed runs. get_rhs_ode()
    # would rebind extra arguments in master, so do not use it for this check.
    rhs_ref = reference.func(state.copy())
    rhs_got = candidate.func(state.copy())
    same_times = np.linspace(0.0, 0.9 * min(requested_seconds, end_ref, end_cand), points)
    same_ref = evaluate(reference, same_times)
    same_got = evaluate(candidate, same_times)
    same_phase_difference = same_got[:, 3:] - same_ref[:, 3:]
    return {
        "time_reparameterized_end_seconds": end,
        "same_time_end_seconds": float(same_times[-1]),
        "reference_end_seconds": end_ref,
        "candidate_end_seconds": end_cand,
        "max_abs_orbit_invariant_residual": np.max(np.abs(got[:, :3] - ref[:, :3]), axis=0).tolist(),
        "max_abs_phase_invariant_residual_rad": np.max(np.abs(phase_got - phase_ref / scale), axis=0).tolist(),
        "max_abs_same_time_phase_difference_rad": np.max(np.abs(same_phase_difference), axis=0).tolist(),
        "end_same_time_phase_difference_rad": same_phase_difference[-1].tolist(),
        "reference_phase_sweep_rad": phase_ref[-1].tolist(),
        "rhs_reference": rhs_ref.tolist(),
        "rhs_rate_residual": (rhs_got[:2] - scale * rhs_ref[:2]).tolist(),
        "rhs_frequency_residual": (rhs_got[3:] - rhs_ref[3:]).tolist(),
        "rhs_rate_scaling_exact": bool(np.array_equal(rhs_got[:2], scale * rhs_ref[:2])),
        "rhs_frequencies_unchanged": bool(np.array_equal(rhs_got[3:], rhs_ref[3:])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, -1e-4, 1e-4, -1e-6, 1e-6])
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-10, 1e-11, 1e-12])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=1001)
    args = parser.parse_args()
    if args.points < 2 or args.duration <= 0.0:
        parser.error("Require >=2 samples and a positive duration")
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    flux_file = Path(few.get_file_manager().get_file("KerrEccEqFluxData.h5"))
    report = {
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_commit": BASE, "few_source": few.__file__,
        "few_version": few.__version__, "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "flux_file_sha256": hashlib.sha256(flux_file.read_bytes()).hexdigest(),
        "resolved_flux_file": str(flux_file.resolve()),
        "scope": "Controlled 0PA p/e rate perturbation; no new physical model, waveform, PSD, or posterior gate",
        "identity": "q_delta(t/c)=q_0(t); DeltaPhi_delta(t/c)=DeltaPhi_0(t)/c; c=1+delta",
        "phase_columns": ["phi", "theta", "r"],
        "orbit_columns": ["p", "e", "x"],
        "horizon_policy": "90% of requested time, capped by both valid dense intervals",
        "rhs_units": "internal tau=nu*t/M_time; first 3 physical rates multiply nu/M_time, last 3 divide M_time",
        "points": args.points, "duration_years": args.duration, "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        for name in args.cases:
            for err in args.tolerances:
                ref, ref_out, ref_seconds = make_run(CASES[name], 0.0, err, args.duration)
                stock, stock_out, _ = make_run(CASES[name], 0.0, err, args.duration, stock=True)
                # 2026-09-11 13:13 CST (linux): Check bytes as well as values
                # so the bitwise label includes signed-zero representations.
                stock_equal = all(a.dtype == b.dtype and a.shape == b.shape
                                  and a.tobytes() == b.tobytes()
                                  for a, b in zip(ref_out, stock_out, strict=True))
                ref_coeff = ref.inspiral_generator.integrator_spline_coeff
                stock_coeff = stock.inspiral_generator.integrator_spline_coeff
                stock_dense_equal = (ref_coeff.shape == stock_coeff.shape
                                     and ref_coeff.dtype == stock_coeff.dtype
                                     and ref_coeff.tobytes() == stock_coeff.tobytes())
                for delta in args.deltas:
                    got, got_out, seconds = make_run(CASES[name], delta, err, args.duration)
                    row = {
                        "case": name, "parameters": CASES[name], "delta": delta,
                        "err": err, "stock_zero_delta_bitwise_equal": stock_equal,
                        "stock_zero_delta_dense_coefficients_equal": stock_dense_equal,
                        "reference_knots": len(ref_out[0]), "candidate_knots": len(got_out[0]),
                        "reference_run_seconds_diagnostic_only": ref_seconds,
                        "candidate_run_seconds_diagnostic_only": seconds,
                        **compare(ref, got, CASES[name], delta, args.points, args.duration * YRSID_SI),
                    }
                    report["cases"].append(row)
                    save()
                    if not (stock_equal and stock_dense_equal
                            and row["rhs_rate_scaling_exact"] and row["rhs_frequencies_unchanged"]):
                        raise AssertionError("Zero-perturbation or RHS identity failed")
                    print(json.dumps(row, allow_nan=False), flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    report["status"] = "measurements_complete_not_scientific_acceptance"
    save()


if __name__ == "__main__":
    main()
