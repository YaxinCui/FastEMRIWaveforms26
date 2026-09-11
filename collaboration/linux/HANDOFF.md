# exp05 — controlled flux-rate perturbation and phase error

<!-- 2026-09-11 13:11 CST (linux): Record a physics/numerics validation
experiment with an analytic identity, preserving the preliminary horizon issue. -->

- Branch: `codex/exp05-flux-scaling-invariants`; independent master base
  `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Linux root owns editing. Local/uncommitted, no host switch.
- Production FEW is untouched. A script-local `KerrEccEqFlux` subclass scales
  only pdot/edot in-place by a constant c=1+delta > 0; frequencies stay unchanged.

## Why this test has an answer independent of a copied waveform

For the autonomous stock 0PA equations dq/dt=F(q), dPhi/dt=Omega(q), changing
F to cF traverses the same orbit curve at a different speed. Integration gives

```text
q_delta(t/c) = q_0(t)
Phi_delta(t/c) - Phi_delta(0) = [Phi_0(t) - Phi_0(0)] / c
```

This follows by substituting u=c*t in the phase integral. It tests parameter
binding, the RHS modification hook, time/phase units and dense integration
against a mathematical invariant. It does NOT certify the flux table's true
physical accuracy, 1PA, an environment model, resonances or plunge dynamics.
If pdot and edot are scaled differently, or c depends on time/state, this
particular constant-scaling identity is no longer applicable.

Source contracts: `src/few/trajectory/ode/base.py` calls the hook in-place;
`ode/flux.py` returns p/e rates and geodesic frequencies in the pex convention;
`integrate.py` applies dimensionless time and restores dense-output phases.
Internally tau=nu*t/M_time and phases are nu*Phi. Direct RHS first three
components multiply nu/M_time for physical per-second rates, while its last
three divide by M_time. Public dense phases are already physical radians.

## Results and interpretation

`flux_scaling_fixed_horizon.json` completed 45 combinations: 3 real Kerr
trajectories x 5 deltas (0, +/-1e-4, +/-1e-6) x tolerances 1e-10/1e-11/1e-12.
Each uses 1001 common dense points up to 0.9 requested year. Zero perturbation
matches the stock trajectory and all dense coefficients bitwise. Every direct
RHS rate-scaling check is exact and every frequency-preservation check passes.

Worst phase-invariant residual over the delta values and three phase columns:

| Case | err=1e-10 | err=1e-11 | err=1e-12 |
| --- | ---: | ---: | ---: |
| nominal | 5.60e-6 rad | 2.30e-6 rad | 4.63e-7 rad |
| retrograde | 9.30e-6 rad | 2.50e-6 rad | 1.50e-6 rad |
| high-e | 1.84e-6 rad | 7.29e-7 rad | 3.34e-7 rad |

At err=1e-12, maximum orbit residual is <=2.00e-12. Convergence improves
the envelope, not necessarily every individual point monotonically.

At the SAME physical time, not the same orbital position, delta=+1e-6 gives
azimuthal phase differences about +0.00410, -0.00210 and +0.00222 rad across
the three cases. A tiny local rate perturbation can accumulate into a much
larger phase difference. This controlled perturbation is NOT a measured FP32
error, and these three phase numbers do not establish a waveform mismatch or
scientific acceptance threshold. No speedup is claimed by this experiment.

`flux_scaling_probe.json` is the preliminary report: it used the end of the
available dense spline and therefore compared different durations across
tolerances. The spline can legitimately extend past requested T by an accepted
step. Do not use that pilot for fixed-horizon convergence claims.

Two fast analytic/harness tests passed. They include an independent autonomous
decay/frequency example, parameter rejection, in-place hook behavior, time-range
checks and six-column shape enforcement. The last shape/actual-file-resolution
guards were added after the 45-row run; original report identities are preserved
rather than rewritten. All reported phase-residual vectors have three entries.

<!-- 2026-09-11 13:20 CST (linux): Repeat the complete matrix with final guards
and actual byte comparisons; preserve both earlier reports as historical data. -->

`flux_scaling_final.json` is the full 45-row replay with shape enforcement,
actual resolved data-file identity and strict byte equality for zero-delta
stock/dense output. All identity checks passed; the phase/orbit error values
match the preceding fixed-horizon matrix. The resolved flux file is in the
existing dev93 data cache; SHA256
`db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014`.
Timing fields remain diagnostic only, not a performance comparison.

## Reproduce and continue

Only `KerrEccEqFluxData.h5` is required, not the 5 GB amplitude table.
Use the pinned master CPU package:

```bash
python -m unittest discover -s collaboration/linux -p test_flux_scaling_probe.py -v
python collaboration/linux/flux_scaling_probe.py --data-dir "$FEW_DATA" \
  --output collaboration/linux/replay.json
```

Useful next experiments: fixed intrinsic-parameter trajectory/amplitude reuse
across observation angles; exact distance scaling; finer tolerance sweeps and
mode-weighted phase-error consequences. Each implementation must start on its
own branch from master, not accumulate into this validation branch.
