# exp09 — fixed-response distance conditional likelihood

<!-- 2026-09-11 14:44 CST (linux): Record real TDI, independent mathematical
oracles, negative timing cases and the isolated dependency repair. No host swap. -->

- Branch: `codex/exp09-distance-conditional-likelihood`.
- Base: `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191` (`master`).
- Experiment scripts/evidence only under `collaboration/linux`; FEW production
  source is unchanged. No other experiment's production changes are imported.
- Local/uncommitted; Linux retains editing, Mac may read. No commit or push.

## What is being optimized

Fix detector-frame masses, intrinsic parameters, sky/orientation, phases,
sampling, linear detector response, window/frequency mask, data and Gaussian
noise metric. Then s(D)=(D0/D)s0. With r0=d-s0 and delta=(D0-D)/D,

    A = (r0|s0), B = (s0|s0)
    Delta logL(D) = delta * (A - 0.5*delta*B).

Prepare two FP64 coefficients once; later distance scans need no waveform-sized
arrays. This is a standard distance-amplitude/Gaussian identity, not a newly
invented physical approximation. Computing r0 first and the distance difference
directly avoids avoidable subtraction of large global terms near D0.

Not mixed precision, mode pruning, distance marginalization, a prior, or full
parameter inference. Fixed source-frame masses with distance-dependent redshift
can change waveform shape, invalidating this reuse. Changing any fixed response,
PSD, data or waveform-shape dependency requires new coefficients.

## Actual chain and numerical evidence

`export_few_distance.py` runs actual clean-master high-level FEW on the original
CPU stack and 5 GB Kerr table. One nominal intrinsic point, five distances
(1, 0.5, 2, 1.3, 1+1e-10 Gpc), T=0.003 yr, dt=15 s. All have 6312 complex128
SSB samples and the same 82 selected modes. Maximum relative scaling difference
is 1.45e-16. The 487 KB NPZ bridge is authenticated by source/native/data and
per-array hashes in `few_ssb_distance.json`. Timing there is descriptive, not
an independently controlled full-pipeline benchmark.

`tdi_distance_probe.py` uses the real CPU fastlisaresponse 1.1.17 implementation,
ESA trailing orbits, second-generation TDI, order-25 interpolation, and A/E/T.
FEW h_plus-i*h_cross is conjugated to the response convention. Latitude is
pi/2-qS. The requested buffer is 20000 s; stable API uses t0 for that buffer.
Each end is cropped by 1333 samples, leaving 3646 real samples per channel.
Physical first time is retained as 19995 s, not reset to the initial phase time.

Orbit file: 1,212,152 bytes; SHA256
`ab7afd7a85da955781896de70bb48cb1156b8a739d732d0e96a2cdc4345cacbe`.
`tdi_distance_audited.json/npz` are the current source-matching evidence:

- Five full FEW -> TDI distance realizations, maximum relative channel L2
  discrepancy 2.83038e-13 (the weaker T channel).
- 99 independent longdouble *direct residual norm difference* comparisons:
  SNR 20/100/1000; zero, parallel and seeded Gaussian residuals; posterior-local
  and wide-distance points. Largest local absolute Delta-logL error 5.69e-14;
  full matrix maximum 7.46e-9, including tails.
- Another 45 comparisons use the independently regenerated FEW -> response ->
  FFT signal at each distance, rather than scaling only a cached frequency
  signal. Maximum absolute Delta-logL difference 5.83e-11; local max 4.88e-12.

The metric is explicitly SYNTHETIC diagonal A/E/T one-sided PSD, proportional to
`1e-42*[1,2,3]*(1+(f/0.01Hz)^2)`, normalized to each reference SNR. It is not the
physical noise covariance of these unequal-arm ESA orbits. Real frequency bins
use Gaussian components with variance 1/w; DC and even-N Nyquist are real-only.
No window/frequency mask is applied. Actual response execution/linearity is
tested; an independent response solver or response-accuracy certification is not.

`math_main_audited.json` and `math_response_audited.json`: seven methods pass on
NumPy 2.5.2 and 2.5.3 respectively. 258 100-digit Decimal direct-residual checks
(including nonunit D0=2.3), 40 odd/even time-domain Parseval checks, plus invalid
inputs, ownership, zero-signal and numerical-range checks. Close zero-noise
distances also require nonzero answers and relative error <=1e-12.

Engineering gates: local absolute Delta-logL <=1e-8; tails use 1e-12 times the
absolute quadratic term scale. Some extreme Decimal tails have absolute error
0.03125 against enormous likelihood magnitudes, so do NOT summarize all errors
as <1e-8. These gates do not constitute universal scientific posterior acceptance.

## Fair timing and counterexamples

Both paths already have the SAME cached frequency-domain TDI, data and weights.
Direct residual loops and chunked vectorized residuals are both supplied.
Preparation-inclusive paths directly time their own prepare+scan; they do not
add unrelated medians. Two warmups and nine alternating-order repetitions,
single CPU/BLAS thread, raw samples and NumPy build configuration are recorded.

Current pilot, preparation included, fastest measured direct baseline versus
batch quadratic:

| Distance evaluations | Speed ratio |
| --- | --- |
| 1 | 0.265x (slower) |
| 4 | 0.503x (slower) |
| 16 | 1.16x |
| 256 | 14.74x |

At 256, direct prepare+scan is 1.968 ms, quadratic prepare+batch 0.1335 ms.
Cached-only batch ratio is about 152.7x, but scalar-call ratio is only 1.019x;
scalar preparation-inclusive ratio is 0.933x. Thus Python scalar dispatch can
erase the algebraic gain on this short input. This is NOT FEW waveform generation
14.7x/153x faster, nor a complete likelihood pipeline benchmark. No CPU clock
telemetry, separate-process confidence intervals, yearly data or CUDA timing yet.

Preserved pilot caveats: `tdi_distance_first` used a costly complex division in
the direct loop, lacked vectorized direct comparison and summed separately
measured preparation times. Its 458x cached-only ratio is not the current fair
claim. `first`/`extended` also generated interior noise with half the declared
variance; arbitrary fixed-data algebra checks remain valid, but that noise
description was corrected. Later `final`/`audited` use the consistent variance.
The initial math harness also incorrectly classified D/D0=0.1 at SNR=1000 as
posterior-local; preserved failure is 1.49e-8 against an inapplicable fixed
1e-8 gate. The corrected criterion uses amplitude displacement times SNR.

## Isolated environment and portability finding

Main `.venv` was not modified. A separate
`artifacts/exp09-response-env` uses fastlisaresponse 1.1.17,
lisaanalysistools 1.2.8, h5py 3.14.0, NumPy 2.5.3 and SciPy 1.18.1.
Main h5py is incompatible with the response wheel's <3.15 requirement, hence
the small bridge instead of changing the validated FEW environment.

The published lisatools CPU wheel imported with SIGILL (-4): GDB showed
`vmovdqu8` in pycppdetector, unsupported by this i5-13400F's ISA. NumPy, Numba
and the response's own CPU extension imported successfully. This identifies
the faulty binary, not its unknown original build flags.

`response_environment_build.json` preserves direct import returncodes, GDB,
package inventory, CPU flags, before/after binary hashes and all commands.
GDB itself returning zero was NOT interpreted as import success. Root cloned
official v1.2.8 without recursive SSH submodules and rebuilt ONLY this dependency
in the response env with LISATOOLS_MARCH=x86-64, LISATOOLS_WITH_GPU=OFF, --no-deps.
All three previously failing imports then passed. No core files intentionally
created, no main-environment downgrade and no GPU dependency installed.

The correct distribution name is `lisaanalysistools`, imported as `lisatools`;
PyPI's different distribution `lisatools` is unrelated. The master sampling
extra appears to list that wrong distribution: a separate future packaging task.

## Reproduce

From this worktree, with the clean master runtime overlay/data settings described
in the coordinator index, use main Python for `export_few_distance.py` and
`test_distance_likelihood.py`. Use the response environment for the real TDI:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python \
  collaboration/linux/tdi_distance_probe.py \
  --input-prefix collaboration/linux/few_ssb_distance \
  --output-prefix collaboration/linux/tdi_distance_replay
```

All scripts reject existing output targets. Environment diagnostic/rebuild script
also rejects an existing source checkout; do not rerun it merely for an ordinary
response replay. Production source is untouched, so no package integration is
claimed. Extreme weighted products that underflow/overflow explicitly fail;
ordinary FP64 cancellation still needs condition-aware interpretation.

## Primary references and code

- [Thrane & Talbot, Appendix C.3/C.4](https://arxiv.org/html/1809.02293v7):
  distance amplitude scaling, Gaussian likelihood, cosmological dependencies.
- [Katz et al., response model](https://arxiv.org/abs/2204.06633): numerical LISA
  response and importance of orbit modeling; not a speed transfer claim.
- [Official polarization conventions](https://mikekatz04.github.io/lisa-on-gpu/fast_LISA_response_tutorial.html).
- [NumPy real FFT conventions](https://numpy.org/doc/stable/reference/routines.fft.html):
  DC/Nyquist and normalization; time-domain Parseval is the independent check.
- [Official v1.2.8 build options](https://github.com/mikekatz04/LISAanalysistools/blob/v1.2.8/CMakeLists.txt).
- Local FEW distance/frame implementation: `src/few/waveform/base.py` and
  `src/few/waveform/waveform.py`, with exact file/native hashes in the bridge.

Next work remains: multiple intrinsic/sky cases and durations, physical noise
covariance, full pipeline workload measurements, cache invalidation APIs and
conditional-prior integration. Do not mark the open-ended exploration complete.
