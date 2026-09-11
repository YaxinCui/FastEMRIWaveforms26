# exp15 — one intrinsic evaluation for a phase basis

<!-- 2026-09-11 17:27 CST (linux): Read master waveform/selector/sum, exp06
prepared observer and exp14 producer/results; query primary FEW/Fourier papers.
No old experiment files, root source, Mac files or data are modified. -->

Independent master branch, scripts only under Linux ownership. The audited
exp14 `phase_likelihood.py` and mathematical tests are copied explicitly as
small experimental infrastructure, not production changes from another branch.
Any later modifications are recorded here. Original evidence remains on exp14.

Hypothesis: construct the same real phase basis with one stock trajectory,
amplitude and selection evaluation instead of 2M+1 full calls. At the original
native summation boundary, group columns by |m|. For each nonzero q, obtain C_q
using unchanged weights, and S_q with the first half of Ylm weights multiplied
by -i*sign(m), second half by +i*sign(m). Preserve native (-1)^ell handling;
m=0 contributes only the constant basis. This avoids modifying phase splines
and preserves the stock amplitudes and native interpolation.

The temporary sum adapter returns a matrix of basis strains through the STOCK
outer distance/source-sign/SSB-polarization conversion. Restore the original
sum module in `finally`. Scope is stock CPU sparse forward Kerr TD, default
unbatched invocation; unsupported configurations fail closed. This experiment
is not a generic/public cache API or integrated wheel.

Validation: independent synthetic pair sums (odd/even ell, m0 and signed m),
real actual-table basis reconstruction versus fresh stock phase calls, then
real ESA second-generation TDI and fixed Gaussian conditional likelihood.
Measure original black-box preparation versus freshly constructed single-pass
basis, including all calls/transfer/response/FFT/statistics. Report the new
incremental speed ratio, do not multiply it by exp14's old speed ratio.

Full response still costs up to 2M+1 basis calls. The innovation here is removing
repeated intrinsic work and reducing dense mode processing, not treating the
time-varying response as complex-linear or dropping cross terms. Coefficient
differences between fresh ODE initial phases may reflect numerical step/roundoff
differences; frozen algebra and full regenerated comparisons are distinct gates.

Sources:
- [FEW harmonics and phase structure](https://arxiv.org/html/2506.09470v2).
- [Related multiharmonic likelihood work](https://arxiv.org/html/2404.02435v2).
- Pinned `waveform/base.py`, `waveform/waveform.py`, `summation/base.py`,
  `summation/interpolatedmodesum.py`, `cutils/interpolate.cu` and `modeselector.py`.
  Actual CPU/CUDA native kernels test m!=0 for the negative branch; the Python
  DirectModeSum instead checks m>0, so do not use that as a signed-m oracle.

<!-- 2026-09-11 17:46 CST (linux): Close the bounded exp15 implementation,
serial actual-table A/B and post-run audit. Record read-only peer findings,
source queries and generated Linux JSON/NPZ; no production/Mac/data edits. -->

## Result: same phase basis, one intrinsic evaluation

Branch `codex/exp15-single-pass-phase-basis` is independently based on master
`47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`. Root stays `deep-optimization`.
Only experimental Linux scripts/evidence and this branch's explanatory note
are new. The main/response environments and all native binaries are unchanged.
Everything remains LOCAL/UNCOMMITTED: no merge, commit, push or host switch.

The formal prograde CPU preparation median falls **1.574754 -> 0.325235 s**,
**4.84189x** against a freshly reproduced exp14-style black-box preparation.
Five serial alternating A/B pairs share the same loaded FEW model, data,
response service, one-request IPC and identical 13-row transfer sizes. Both
paths compute all 13 real-response/FFT rows and retain full interference terms.
The counters on BOTH paths actually observe trajectory/amplitude calls 13->1.
There is no FP32 conversion, additional mode truncation or new physical model.

In the prograde source, 82 selected modes have max |m|=6. Both paths still
invoke the native sum 13 times, but the old path processes all 82 columns each
time. The grouped path processes two m=0 columns once and 80 others twice:
1066 -> 162 mode-column visits (not an independent measured speed ratio).
The stock outer model handles distance, sign, polarization and retrograde
canonicalization, including its Phi_phi0+pi shift; no separate rotation is added.

Preparation includes worker compute, IPC, real TDI, FFT, whitening/stacking,
and statistical preparation. It excludes common model/table loading, orbit
construction, data creation and warmup/JIT. Internal timing ends before return
construction and caller cleanup. The independently timed preparation+scan
includes those costs: 1/16/256 phase points give **4.8774/4.8807/4.8029x** versus
the same newly built black-box basis. This is NOT a single-waveform baseline;
an ordinary one-phase FEW call does not need either 13-row preparation.

Stage medians (not additive independent measurements):

| Stage, seconds | Black-box | Grouped |
| --- | ---: | ---: |
| Worker computation | 1.367846 | 0.126278 |
| Worker plus IPC (includes preceding row) | 1.375969 | 0.134324 |
| Response + FFT + whitening/stacking | 0.189839 | 0.190014 |
| Statistics preparation and checks | 0.001490 | 0.000567 |

The old statistics stage also includes its DFT and phase-zero anchor check;
neither timing is a pure Gram kernel. Response processing is now about 58% of
new preparation. This motivates profiling that stage, not assuming its geometry
is uncached or that all 13 separately computed basis responses can be discarded.

The stronger same-already-cached-detector-basis comparison starts BOTH methods
at the same basis. The compact method includes fresh statistic construction:
for 1/16/256 points ratios are **0.2513/0.8034/8.4789x**. Few-point queries lose
to direct vectorized evaluation. Do not multiply any of these ratios by exp14's
historical 19.825x or infer whole-FEW/GPU/annual-data performance.

## Accuracy and failure-path coverage

Formal evidence: [prograde JSON](real_prograde_final.json),
[retrograde JSON](real_retrograde_final.json), and matching saved NPZ files.
Retrograde has `benchmark=false`: accuracy only, no measured speed claim.
Both use actual unchanged 5 GB Kerr amplitudes, T=0.003 yr, dt=15 s, 6312 SSB
samples. Actual ESA trailing orbit and second-generation A/E/T response leave
3646 samples/channel after edge cropping. The one-sided PSD is a fixed SYNTHETIC
diagonal fixture, not a validated physical LISA covariance.

| Full regenerated reference comparison | Prograde | Retrograde |
| --- | ---: | ---: |
| Maximum SSB global relative L2 | 1.399e-13 | 9.322e-14 |
| Maximum weighted TDI global relative L2 | 1.187e-12 | 2.702e-12 |
| Maximum absolute Delta-logL error | 3.685e-8 | 4.122e-8 |

Each source checks 22 freshly regenerated phases: 21 off-grid phases (including
wrapped/near-zero values) and the phase-zero anchor. SNR 20/100/1000 gives
132 likelihood checks total against independently formed
longdouble direct residual-norm differences. Gates are global waveform L2
<=1e-8 and |Delta-logL error| <=1e-5+2e-10*|reference|. These are local numerical
regression gates, not a scientific certificate for all inference workloads.
The likelihood is anchored at the common stock phase-zero reference. Global
L2 does not guarantee relative accuracy of every weak mode or individual
weak T channel. Relative error at an almost-zero likelihood is not certified.

Separate frozen-native checks shift only the additive azimuthal phase-spline
coefficient, holding trajectory/amplitudes fixed. Three offsets per source
pass <=1e-10, maximum 1.017e-13. Fresh full FEW phase calls are also checked;
phase in the ODE state can affect floating-point step selection, so the two
checks answer different questions and bitwise equality is not promised.
Nonzero reference phase 0.37 with three offsets passes at the SSB layer only.

Five signed-pair/guard/exception test methods and six likelihood methods pass
on BOTH numerical stacks; logs are [main](math_main_final.json) and
[response](math_response_final.json). Independent finite sums cover signed m,
odd/even ell, m=0, missing groups and one-sided weights. Negative controls expose
the wrong whole-pair phasor and missing likelihood cross terms. A synthetic
renderer is an algebra oracle, not an independent Teukolsky computation.

Actual in-worker diagnostics verify owned basis arrays survive later ordinary
wave calls unchanged, injected Python sum failure restores callable identity
AND descriptor/instance-override lookup, and the recovered ordinary waveform
matches the pre-failure wave byte-for-byte. This is not native-crash recovery,
full internal model-state rollback or a thread-safe public cache API.

The final [source/numeric audit](audit_final.json) passes **680 checks over 68
authenticated files**. It replays all 132 likelihood values, authenticates
producer/helper/FEW Python/native/orbit/archive identities, checks measured
operation counts and all timing-output gates, and reruns both mathematical
suites. FEW Python is verified against pinned master, not this host's older
default installed package; generated `_version.py` is identified separately.
The full data digests were checked by each formal FEW worker; the post-run audit
authenticates the small reports, not a second full 5 GB read.

## Implementation history and reproduction

The initial `real_prograde_first.json/.npz` pilot remains. Its exact three
producer/helper sources are archived under `pilot_sources/` and hash-checked.
Before formal reruns we fixed restoration of the class-bound amplitude method,
counted the old path symmetrically, included top-level `few` identity, moved
ownership/failure probes into the worker, and strengthened the direct cached
competitor by scaling its basis once outside the chunk loop. Only the two
`*_final` reports support final claims; pilot timings are not mixed into them.
Copied exp14 mathematical infrastructure is explicitly retained; `from_basis`
now accepts an owned direct real basis with an explicit shared reference.

Run from this worktree, with NEW output names (existing reports are protected):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python \
  collaboration/linux/phase_real_probe.py --case prograde --benchmark \
  --output collaboration/linux/real_prograde_replay.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python \
  collaboration/linux/phase_real_probe.py --case retrograde \
  --output collaboration/linux/real_retrograde_replay.json
```

The parent selects clean `master-reference/src` plus `artifacts/runtime-master`
for its main-FEW worker and pins the toolchain library directory. The response
environment uses NumPy 2.5.3, SciPy 1.18.1, fastlisaresponse 1.1.17 and
lisaanalysistools 1.2.8 with the existing x86-64 CPU rebuild. Downloads are
disabled, ONCE integrity is retained. Exact versions/paths/hashes are in JSON.

`audit_basis.py` reads the protected final reports and creates `audit_final.json`
and both math logs; it intentionally refuses to overwrite these existing files.
All performance jobs ended before this post-run audit. No concurrent GPU/CPU
performance jobs were used. Diagnostic arrays remain resident during timings,
so 24 FP64 compact coefficients (192 payload bytes) do not measure process RSS
or single-call peak memory. Unsupported model/batching/output settings fail
closed; the implementation has not been packaged into FEW production.

Next bounded experiment: profile the 0.190 s response/FFT stage and its existing
cache behavior, then test shared/batched real-linear processing against all
13 separately computed basis responses. Broader parameter/long-duration tests,
physical covariance, GPU, two-phase searches and production integration remain
open. Linux retains the edit lock; no collaborator handover is requested.

<!-- 2026-09-11 17:50 CST (linux): Final read-only peer review corrects the
off-grid count (21 plus the zero anchor, still 22 total). Ruff E9/F and format
checks pass for seven scripts; root diff whitespace, new-note whitespace, 25
local links, six JSON reports and all 68 archived identities pass. No remaining
exp15 worker process observed. Only wording changes follow the frozen audit. -->
