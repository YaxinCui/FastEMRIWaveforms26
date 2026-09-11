# exp06 — fixed-intrinsic observer scans

<!-- 2026-09-11 13:48 CST (linux): Document the independent master-based
experiment, ownership, preliminary evidence and explicit validation limits.
Linux root is the sole editor; this is not a host switch or synchronization. -->

- Branch: `codex/exp06-prepared-observer-scan`.
- Base: `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191` (`master`).
- All changes LOCAL/UNCOMMITTED. Root checkout remains `deep-optimization`.
- No production files from another experiment were imported. No native code
  was changed. Mac-owned files and original user probes are untouched.

## What this experiment actually reuses

```python
prepared = model.prepare_observer_scan(
    m1, m2, a, p0, e0, xI0, T=1.0, dt=15.0,
    Phi_phi0=0.3, Phi_theta0=0.4, Phi_r0=0.5,
)
h1 = prepared(theta1, phi1, dist=1.0)
h2 = prepared(theta2, phi2, dist=2.0)
```

The normal `model(...)` API remains unchanged. The new method owns copies of
the full sparse trajectory, continuous phase coefficients/time grid, frequency
samples, full unweighted amplitudes and mode topology. Each view still runs
the stock angular functions, mode selection, interpolation and mode sum.
It does NOT reuse a previous observer's selected mode subset.

This follows the dependency structure in master `waveform/base.py`: intrinsic
trajectory before observer selection; `utils/modeselector.py`: angular weights
affect mode power and therefore selection; final distance scaling after sum.
The spherical-basis strain decomposition is described in the
[FEW paper, Section III.2, Eq. 12](https://arxiv.org/html/2506.09470v2#S3.SS2).
This is a workload/dependency optimization, NOT a new physical approximation
or a mixed-precision kernel. All cached numerical quantities remain FP64 /
complex128. The time-domain output is low-level source-frame strain, not TDI.

Preparation fixes intrinsic parameters, initial phases, duration, sampling,
effective ODE options and mode-selection sensitivity. Changing those requires
a new explicit snapshot. The same snapshot may change source viewing angles,
distance, threshold and requested mode list. No approximate cache keys exist.

For an optional sensitivity callback, the exact full frequency vector and its
PSD values are frozen. Subsequent selections verify identical frequency order.
Positive infinity PSD is allowed (zero weight); nonpositive/NaN values fail.
This permits mutable user callbacks without retaining a callback/owner alias.
It is only a mode-selection sensitivity, not likelihood validation.

## Ownership and rejected cases

- Copy complete allocations, not CUDA offset views. CPU cache arrays are
  read-only; metadata and selected modes are returned as independent copies.
- Preserve retrograde observer transformation, initial phases, backwards
  phase convention, padding and odd output length.
- Sparse stock Kerr ODE, stock selector/angular module and TD interpolated
  sum only; custom ODE, dense stepping and FD output are explicitly rejected.
- CUDA default stream only, on the original preparation device. Same-renderer
  concurrent calls are rejected. Multi-GPU guard has mock coverage, not actual
  hardware coverage on multiple devices.
- Preparation runs the owner's integrator once, but does not mutate its
  `inspiral_kwargs` dict. Later owner trajectories cannot overwrite a snapshot.
- A snapshot does not retain the amplitude-provider object / 5 GB table.
  Weak-reference lifetime checks do NOT prove that GPU allocator pools return
  their reserved memory to the driver. The H5 file itself is not compressed,
  modified or made unnecessary for initial preparation.

## Evidence and reproducibility

The matching master native build is installed outside Git in sibling
`artifacts/runtime-master`; current experiment `src` is overlaid on it.
Do not use the Python environment's default installed older FEW core as master.

```bash
export PYTHONPATH="$PWD/src:/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/runtime-master"
export LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

Python: `/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python`.
Existing data directory:
`/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download`.
The probe resolves and fingerprints both actual H5 files and native module.
It disables downloads and refuses to overwrite existing evidence paths.

- `unit_checks.json`: 8 tests each on authenticated CPU and CUDA, passed.
  Includes signed mode mapping, independent ownership, PSD mutation/frequency
  ordering, device/stream and threshold guards, and comparison-harness guards.
- `observer_scan_cpu_pilot.json`: original short nominal prototype, zero
  numerical differences; warm reuse 15.87x but first preparation 3.20 s.
  Its old one-scan ratio mixes first preparation with warm reference timings;
  it must NOT be interpreted as a cold-to-cold benchmark.
- `observer_scan_cpu_extended.json`: intermediate three-case expanded pass;
  retained as historical evidence before final harness tightening/formatting.
- `validation_suite_final.json`: serial nine-workload matrix. At this entry,
  CPU short completed; remaining jobs are in progress. Inspect terminal
  return codes, per-file `passes` and source hashes before claiming success.
- CPU short FINAL: nominal/high-e/retrograde warm four-view reuse
  14.963x / 4.805x / 17.175x; directly measured preparation-plus-four-view
  workloads 3.320x / 2.460x / 3.401x. These are warmed, same-process pilot
  medians, not whole-program cold-start or single-waveform speedups.

The initial manual component-test invocation set `FEW_TEST_BACKENDS=cpu` but
master's test harness ignores that variable and selected CUDA. The corrected
runner sets `FEW_ENABLED_BACKENDS` in separate processes and checks the printed
backend name. A prior direct test script invocation also lacked the project
root in the import path; `python -m unittest` resolves the test package.
Neither invocation mistake is counted as CPU coverage.

CUDA ordinary master waveform calls have the separately diagnosed CPU/CuPy
derivative-time boundary failure. CUDA reference probes explicitly install
the same host-time adapter recorded in their metadata. The NEW preparation
path already evaluates derivatives on host times and needs no adapter itself.
The exp04 production fix is NOT included in this branch.

## Interpretation and remaining work

Four-view timing separates fresh generation, reuse of an existing snapshot,
warm preparation, and directly measured preparation plus a new four-view scan.
First-observed preparation may include first-use/JIT initialization; that cost
has not been attributed to individual stages. Estimated break-even counts and
amortization curves are algebraic estimates from timings, not measured batches.

Fixed-mode lists must be evaluated separately: the ordinary path only computes
the few requested amplitudes, whereas preparation computes ALL modes. Longer
outputs can be dominated by summation; short-case gains cannot be extrapolated.
Distance-only reuse has an even simpler `D0/D` scaling baseline and should not
be advertised as a new benefit of this observer-scan API.

Pending: completion of the matrix, broader/installed-package regressions,
multi-process confidence intervals and controlled thermal/clock experiments.
Before/after GPU telemetry is not continuous clock control. No detector-noise
weighted or inference-level scientific acceptance is established. Do not
multiply this workload speedup with exp01 or call it general FEW acceleration.

<!-- 2026-09-11 14:04 CST (linux): Close the original matrix honestly at eight
passes and one reference failure; retain negatives and keep source independent. -->

## Terminal matrix and additional reference work

`validation_suite_final.json` is terminal: eight jobs passed, CUDA+PSD failed
inside ordinary master's reference selection, before any waveform comparison.
The whole original manifest intentionally remains `passes=false`. Its failure
is documented and repaired on independent `codex/exp07-cuda-selector-frequency-boundary`.
Do not relabel this original matrix as nine passing jobs.

| Workload | Warm cached four-view scan | Directly measured warm preparation + four views |
| --- | ---: | ---: |
| CPU short, three intrinsic cases | 4.81–17.18x | 2.46–3.40x |
| CUDA short, three intrinsic cases | 1.49–2.44x | 1.32–1.75x |
| CUDA one year, three intrinsic cases | 1.02–1.20x | 1.02–1.14x |
| CPU fixed three modes, nominal | 7.48x | **0.0996x (about 10x slower)** |
| CUDA fixed three modes, nominal | 3.54x | 1.25x |

All 16 intrinsic-case entries in the eight passing jobs have byte-identical
same-backend direct waveform comparisons. Extended short cases additionally
pass owner-intrinsic A→B→A, preparing B then reusing A, negative/list/all mode
selection, backwards evolution, distance/angle identities, invalid inputs,
external-return mutation and owner/amplitude-provider release checks.

The two probe-checker tests now live in `collaboration/linux/test_observer_probe.py`
instead of shipped FEW tests. `unit_checks_final.json` authenticates 6 package
tests + 2 local-harness tests on each backend, all passed, with current hashes.
This avoids making installed FEW tests depend on an unshipped research folder.

`replay_psd_reference.py` consumes the separately repaired exp07's small,
hashed CPU/CUDA waveforms. It imports NO exp07 production code, installs no
adapter and records its own source/data identities. Its result is pending at
this entry; this is numerical replay only, NOT a replacement CUDA+PSD timing.

<!-- 2026-09-11 14:07 CST (linux): Record the terminal independent replay and
source-matching audit; preserve the original failed matrix unchanged. -->

`psd_independent_replay.json` has now passed all 12 CUDA prepared PSD cases
against exp07's authenticated references. All are byte-identical to the CUDA
reference and agree with the CPU reference within 1.41e-15 relative L2.
Mutating the original PSD callback after preparation does not alter the
snapshot and does not invoke that callback again. This required no changes
to exp06 production source and installed no harness adapter.

Final audit confirms 80/80 primary comparisons from the eight original passing
jobs are byte-identical, unit report hashes match current source/tests, the
new replay's source hash matches current `prepared.py`, and Ruff E9/F and
whitespace checks pass. All observed processes for this checkpoint are terminal.
The original ninth job remains a documented reference failure; CUDA+PSD timing
was not measured after repair. Broader scientific/package acceptance and the
open-ended multilayer exploration remain unfinished. No commit/push occurred.
