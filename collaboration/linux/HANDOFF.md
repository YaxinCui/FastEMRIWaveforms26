# exp14 — phase-structured conditional likelihood

<!-- 2026-09-11 17:04 CST (linux): Read master phase/RHS/selector, exp01/06/09
and primary papers. New isolated scripts, no production changes. -->

Master-based branch `codex/exp14-phase-likelihood-fourier`. Linux root sole
writer; all changes LOCAL/UNCOMMITTED, no push or host transfer.

## Hypothesis and bounded test

Fix intrinsic parameters, radial initial phase, sky/orientation, absolute time,
linear response, sampling/window, data and Gaussian noise metric. Vary only
initial azimuthal phase alpha. FEW modes carry exp[-i*m*alpha]; with |m|<=M,
each real detector signal is a degree-M trigonometric polynomial. The Gaussian
log likelihood is degree <=2M. Hence 2M+1 full signal samples determine a real
basis; complete Gram cross terms give at most 4M+1 real Fourier coefficients
for the conditional log likelihood. This is an algebraic consequence of the
existing model, not FP32, extra mode pruning, or an invented physical law.

Use the existing separate FEW/response Python environments via a local pipe,
without changing dependencies. Measure the actual wall time of prepare+scan,
including full FEW generation, pipe transfer, real ESA second-generation TDI,
FFT and coefficient preparation. Common loaded model, data, orbit setup and
warm JIT are excluded on both sides and reported. Also compare with the stronger
baseline using the same already prepared detector basis and direct residuals.

First test short prograde and retrograde sources; off-grid phases and SNR
20/100/1000. Full regenerated FEW->TDI is the implementation reference;
independent longdouble synthetic finite-series sums are the math oracle.
Metric initially synthetic, explicitly not physical LISA covariance acceptance.
Preserve a diagonal-Gram negative control and preparation-cost counterexamples.
No universal speedup claim, full posterior/marginalization or yearly validation.

## Sources and non-novelty boundary

- [FEW Kerr paper](https://arxiv.org/html/2506.09470v2), waveform/phase formulae.
- [Pankow et al. 2015](https://arxiv.org/abs/1502.04370), precomputed mode
  overlaps for extrinsic inference: related strategy already exists.
- [Roulet et al. 2024](https://arxiv.org/abs/2404.02435), multiharmonic
  likelihood covariance and extrinsic marginalization.
- [Katz et al. response](https://arxiv.org/html/2204.06633), full time-varying
  LISA response. It is real-linear in polarizations, not naively complex-linear.
- [Babak et al. 2009](https://arxiv.org/abs/0902.4133), EMRI harmonic search
  statistics. We retain the common physical phase, not free phases per mode.

Research agents only queried/read these sources, exp01/06/09 handoffs and the
master waveform, mode selector and integrator sources at 16:57–17:03 CST.
No PDF downloads, changes to the 5 GB table, or Mac file edits.

<!-- 2026-09-11 17:20 CST (linux): Close the bounded round after formal
prograde/retrograde runs, six math methods on both stacks and the final audit.
All child processes are terminal; original pilots remain preserved. -->

## Completed result

`real_prograde_final.json` and `real_retrograde_final.json` are terminal/pass;
both FEW child services exited zero. Production FEW remains pinned master.
Both actual selections have max |m|=6, giving 13 detector basis samples and
25 real coefficients including a constant (24 FP64 coefficients for centered
Delta logL). Compact numeric payload is 192 bytes, **not process RSS**. The
diagnostic producer intentionally retains another 1,227,376 bytes of prepared
basis/residual/statistics; a weakref test proves the standalone compact object
does not retain those arrays. No peak-memory or GPU speed measurement is claimed.

Each source has 6312 FEW SSB samples; real ESA trailing-orbit second-generation
A/E/T, order-25 interpolation, dt=15 s, 20000 s requested buffer. Crop removes
1333 samples from each end, leaving 3646 samples/channel starting at 19995 s.
No additional mode truncation beyond master selection. No response changes,
physical noise substitution, extra installations or main-environment changes.

### Accuracy

Per source, 22 independent phase calls (random off-grid, negative, wrapping and
small increments) at SNR 20/100/1000 give 66 regenerated-chain comparisons:

| Case | Maximum weighted waveform relative L2 | Maximum absolute Delta-logL error |
| --- | --- | --- |
| Prograde, a=.6,p0=8,e0=.3,xI=+1 | 1.26506e-12 | 4.20551e-8 |
| Retrograde, a=.6,p0=12,e0=.4,xI=-1 | 2.73969e-12 | 7.22940e-8 |

The synthetic one-sided diagonal PSD is proportional to
`1e-42*[1,2,3]*(1+(f/0.01Hz)^2)`; reference weights are normalized to unit SNR,
then signal amplitude is scaled to the three SNRs. Correct real-FFT endpoint
half weights and real-only endpoint noise are used. This is not physical LISA
covariance or full posterior acceptance. The prespecified engineering gates
were relative L2 <=1e-8 and absolute Delta-logL <=1e-5+2e-10*abs(reference).
Small increments under this absolute gate do NOT establish relative sensitivity
near stationary points; a separate nonzero analytic sine test checks a narrower
tiny-phase property. Frozen full-chain and prepared zero-phase anchors are
byte-identical, including every timed preparation.

Dropping Gram cross terms is demonstrably wrong: at alpha=.71 it changes
Delta logL by 14.76 (prograde) / 207.55 (retrograde). Undersampling phase and
wrong harmonic reconstruction are separately covered by synthetic controls.

### Performance and limits

CPU only, prograde short source, one BLAS/OpenMP thread, 3 alternating-order
repetitions. BOTH sides start with the same loaded FEW/response/data state.
Times directly include full FEW generation, local pipe serialization, response,
FFT, coefficient preparation and scan; common setup/JIT is excluded and recorded.
Each full-chain repeat's outputs are checked, not just the final repeat.

| Phases | Direct median seconds | Prepare+scan seconds | Speed ratio |
| --- | --- | --- | --- |
| 1 | 0.125927 | 1.576457 | 0.0799x, slower |
| 16 | 1.936803 | 1.570060 | 1.2336x |
| 64 | 7.937274 | 1.589906 | 4.9923x |
| 256 | 31.369353 | 1.582280 | 19.8254x |

These gains are primarily amortization of phase-dependent full-chain work.
There is no claim of single-waveform 20x, cold-start 20x, GPU performance,
annual observations, all intrinsic parameters, or an optimal shared-trajectory
implementation. Avoid calling 4.9923x a demonstrated strict >=5x threshold.
The local two-environment IPC cost is part of both paths; its reduced count
contributes to the ratio. Retrograde has accuracy evidence only, not timing.

A stronger competitor starts from the SAME cached detector samples, constructs
the basis and calculates vectorized direct residuals in chunks; it does not
pay for unused Gram/Fourier. Against it, preparation-inclusive ratios for
1/16/64/256 points are 0.758/0.990/1.677/4.202x (7 alternating repetitions).
At 256, about 5.636 ms versus 1.341 ms. Thus not all of the full-chain ~20x
gain should be attributed to the final few-coefficient polynomial alone.

### Evidence and next decision

`audit_final.json`: 568 passing checks over 66 authenticated files, including
source/native/NPZ identities, independent longdouble direct-residual replays,
array/coefficient consistency and every full-chain repeat's error/arithmetic.
`math_main_final.json` and `math_response_final.json`: 6 methods each pass;
these include independent finite series, real mixing, alias/diagonal controls,
nonzero tiny-phase sine and compact ownership. Focused Ruff E9/F checks pass.

The initial `real_prograde_first` pilot remains. Its last-repeat-only validation,
implicit reference-anchor assumption and retained-basis object were hardened
before the formal runs; exact original scripts are under `pilot_sources/` and
their hashes match the pilot. Do not retroactively describe it as using the new
compact object or all-repeat checks.

Reproduce from this worktree using a NEW output path:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python \
  collaboration/linux/phase_real_probe.py --case prograde --benchmark \
  --output collaboration/linux/real_prograde_replay.json
```

The server uses the authenticated existing master source/runtime and main FEW
Python. Data downloads are disabled. The ordinary replay must not recreate or
modify the response environment. For retrograde accuracy omit `--benchmark`
and use `--case retrograde` plus another new output path. Audit scripts reject
existing reports; preserve current evidence rather than overwriting it.

Adoption recommendation: promising optional *fixed-parameter phase scan*
primitive. Next prioritize constructing the m-group basis from ONE trajectory
and amplitude call, compare its preparation with the current black-box sampler,
then decide whether sparse two-phase support is worthwhile. Physical covariance,
long-duration trials and integration/cache invalidation remain future work.
See [Chinese derivation and research note](../../knowledge/PHASE_STRUCTURED_LIKELIHOOD.md).
