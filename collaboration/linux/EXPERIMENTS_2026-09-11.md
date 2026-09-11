# Master-based multilayer experiments — Linux coordinator

<!-- 2026-09-11 13:13 CST (linux): Record the user-authorized persistent
experiment series, actual evidence and remaining work. Mac-owned files are
read-only; Linux root is the sole editor and performance runner. -->

Root checkout remains `deep-optimization` at `23eef90c`. Each attempt below
was independently created from master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`;
none contains another experiment's uncommitted changes. Worktrees are siblings
under `/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/`.
<!-- 2026-09-11 22:27 CST (linux): User-authorized publication supersedes
historical uncommitted status; preserve earlier experimental facts below. -->

The 17 experiments are now individually committed and pushed. All 17 remote
commit IDs were independently verified; see [GitHub sync record](GIT_SYNC_2026-09-11.md).
Historical LOCAL/UNCOMMITTED notes below describe their original timestamps.
No merge or host switch occurred; external runtime/data paths still need setup.

## Branches and evidence

| Branch | Layer and attempt | Verified result / boundary |
| --- | --- | --- |
| `codex/exp01-common-phase-fp64` | Algebra / dense summation | FP64 grouping, independent CPU/CUDA oracle tests; first one-year pilot 1.20–1.38x, ~1e-15 relative error. Short cases not reliably faster. Explicit shared host adapter required by master frontend. |
| `codex/exp02-trajectory-workspace` | Memory allocation / ODE implementation | Opt-in stage reuse; byte-equality regressions passed. Three runs show only about -0.3% to +1.5% real effect; not a major acceleration path. |
| `codex/exp03-kerr-reference-coverage` | Test design / scientific data provenance | All five old fixtures now checked; four pass, one fails. Author raw release agrees with local table, not the old constant. Failure preserved. |
| `codex/exp04-cuda-trajectory-boundary` | CPU/GPU interface correctness | Actual default CUDA frontend repaired. Three independent short CPU/CUDA comparisons pass with ~1e-16–1e-15 relative L2. |
| `codex/exp05-flux-scaling-invariants` | Physics dependencies / numerical error | New script-only controlled perturbation; 45 fixed-horizon cases and analytic harness tests. Tightening tolerance improves invariant residual envelope; no performance/1PA claim. |
| `codex/exp06-prepared-observer-scan` | Parameter dependencies / repeated workload | CPU short four-view preparation+scan 2.46–3.40x; CUDA one year 1.02–1.14x; CPU fixed-three-mode workload 0.0996x. Eight jobs pass; original CUDA+PSD reference failure preserved; separate 12-case PSD replay passes. Not single-waveform acceleration. |
| `codex/exp07-cuda-selector-frequency-boundary` | CPU/GPU PSD interface correctness | Original two boundary tests fail; repaired CPU/CUDA tests pass. Adapter-free 3 cases × 4 views on each backend agree to relative L2 <=1.41e-15; remaining construction-all/call-threshold configuration issue documented. |
| `codex/exp08-effective-mode-selection` | Configuration semantics | Effective call-time selection controls frequency sampling; 6 CPU tests and 15 real Kerr waveforms pass. Defaults restore correctly, byte-identical configuration references. No exp07 CUDA fix imported. |
| `codex/exp09-distance-conditional-likelihood` | Inference dependencies / numerical algebra | Real ESA second-generation TDI, 5 distance realizations, 99 algebra + 45 regenerated-chain likelihood checks pass. Preparation-inclusive 256-distance batch pilot 14.74x; one distance 0.265x, scalar scan ~no gain. Synthetic noise metric, not full FEW acceleration. |
| `codex/exp10-mode-index-domain` | Physical domain / public API correctness | Reject signed-coordinate aliasing and invalid IDs before amplitude lookup; 9 CPU + 9 CUDA methods pass. Actual 5 GB Kerr valid-input outputs: CPU 43 / CUDA 27 arrays byte-identical to master, including 8 CPU source waves. No speed or generic physics claim. |
| `codex/exp11-sampling-dependency` | Packaging / scientific workflow reliability | Remove unrelated portfolio distribution colliding with LISA import package. Before/after wheel metadata and isolated binary-only dry-run pass; no sampling package installed, no import/ABI/full sampling claim. |
| `codex/exp12-spin-local-amplitudes` | Storage / working-set residency | Opt-in FP64 spin-plane LRU; 8 methods per backend and 39-array real master regressions pass. Seven-pair CPU/CUDA first-result ratios 1.0395/1.0510x; CUDA active pool after both regions 4853.94→295.65 MiB. Reload can be ~5x slower. |
| `codex/exp13-retrograde-amplitude-contract` | Physical convention / direct API consistency | Minimal Kerr raw/all repair; 8 methods per backend and fresh CPU/CUDA actual-table regressions pass. Per backend 15 contract checks, 10 intentionally repaired arrays, 24 byte-identical arrays including 7 short source waves. No general physics/speed claim. |
| `codex/exp14-phase-likelihood-fourier` | Harmonic structure / conditional inference | Script-only real FEW->ESA TDI phase scan: 132 accuracy checks across short prograde/retrograde, max weighted L2 2.74e-12. Prograde warm prepare+scan 64/256 points 4.992/19.825x; same cached-sample competitor 1.677/4.202x. Single point slower; no universal/GPU/posterior claim. |
| `codex/exp15-single-pass-phase-basis` | Harmonic structure / eliminating repeated intrinsic work | FP64 direct phase bases, observed trajectory/amplitude calls 13->1. Same-session CPU preparation 1.575->0.325 s (4.842x); 132 prograde/retrograde likelihood checks, max weighted TDI L2 2.702e-12. Response still 13 calls; no universal/single-wave/GPU speed claim. |
| `codex/exp16-prepared-detector-response` | Detector operator / immutable data lifecycle | FP64 sparse TDI, original projections retained. 13-input warm plan+batch 1.402x; cached plan 2.725x; including response creation only 1.034x, one input slower. 32 real-basis/holdout response comparisons, 96 likelihood checks and 7 methods pass; no whole-FEW/GPU/cold-load claim. |
| `codex/exp17-explicit-link-input` | Dependency interface / safe reusable link data | Isolated CPU single-method repair. Original explicit-y AttributeError reproduced; 10 methods pass normally and with -O. Two physical systems × three phases × six TDI/channel combinations byte-identical to original; 983-check / 70-file audit passes. No speedup/GPU/physical-model claim. |

Per-worktree notes are self-contained:

- [exp01 handoff](../../../FastEMRIWaveforms-experiments/exp01-common-phase-fp64/collaboration/linux/HANDOFF.md)
- [exp02 handoff](../../../FastEMRIWaveforms-experiments/exp02-trajectory-workspace/collaboration/linux/HANDOFF.md)
- [exp03 handoff](../../../FastEMRIWaveforms-experiments/exp03-kerr-reference-coverage/collaboration/linux/HANDOFF.md)
- [exp04 handoff](../../../FastEMRIWaveforms-experiments/exp04-cuda-trajectory-boundary/collaboration/linux/HANDOFF.md)
- [exp05 handoff](../../../FastEMRIWaveforms-experiments/exp05-flux-scaling-invariants/collaboration/linux/HANDOFF.md)
- [exp06 handoff](../../../FastEMRIWaveforms-experiments/exp06-prepared-observer-scan/collaboration/linux/HANDOFF.md)
- [exp07 handoff](../../../FastEMRIWaveforms-experiments/exp07-cuda-selector-frequency-boundary/collaboration/linux/HANDOFF.md)
- [exp08 handoff](../../../FastEMRIWaveforms-experiments/exp08-effective-mode-selection/collaboration/linux/HANDOFF.md)
- [exp09 handoff](../../../FastEMRIWaveforms-experiments/exp09-distance-conditional-likelihood/collaboration/linux/HANDOFF.md)
- [exp10 handoff](../../../FastEMRIWaveforms-experiments/exp10-mode-index-domain/collaboration/linux/HANDOFF.md)
- [exp11 handoff](../../../FastEMRIWaveforms-experiments/exp11-sampling-dependency/collaboration/linux/HANDOFF.md)
- [exp12 handoff](../../../FastEMRIWaveforms-experiments/exp12-spin-local-amplitudes/collaboration/linux/HANDOFF.md)
- [exp13 handoff](../../../FastEMRIWaveforms-experiments/exp13-retrograde-amplitude-contract/collaboration/linux/HANDOFF.md)
- [exp14 handoff](../../../FastEMRIWaveforms-experiments/exp14-phase-likelihood-fourier/collaboration/linux/HANDOFF.md)
- [exp15 handoff](../../../FastEMRIWaveforms-experiments/exp15-single-pass-phase-basis/collaboration/linux/HANDOFF.md)
- [exp16 handoff](../../../FastEMRIWaveforms-experiments/exp16-prepared-detector-response/collaboration/linux/HANDOFF.md)
- [exp17 handoff](../../../FastEMRIWaveforms-experiments/exp17-explicit-link-input/collaboration/linux/HANDOFF.md)

<!-- 2026-09-11 22:17 CST (linux): Add completed exp17 interface experiment;
source-byte and independent native-stencil audits terminal. Root sole writer,
Mac read-only; preserve unrelated changes and prior experiment evidence. -->

Exp17 repairs old fastlisaresponse 1.1.17's explicit six-link entry in a local
single-method copy, with clear buffer-time semantics and pre-native contracts.
Normal constructed CPU inputs preserve native bytes; non-CPU is deliberately
rejected by this experimental method. No installed source/native/data change.
Both normal and optimized Python test suites pass; six existing FEW waveforms
produce 36 passing real interface configurations. Independent audit also checks
five actual native-orbit stencil boundaries at the lowest accepted cut. This
is reliability work, not a performance result, and does not close the open goal.
All seventeen worktrees remain local/uncommitted, root stays deep-optimization,
no commit/push/merge or host transfer. See per-worktree handoff for API bounds,
preserved failures, source/version mismatch and the real-check label caveat.

These filesystem links are local conveniences; on another host select the
named branch after its changes have actually been committed and synchronized.

<!-- 2026-09-11 17:20 CST (linux): Complete exp13 bounded real-table repair
and exp14 first phase-structure round. Fresh serial jobs and audits are terminal;
all worktrees remain independent, local/uncommitted, with Linux ownership. -->

Latest audit boundaries: exp13 `audit_real_final.json` passes 612 checks / 107
authenticated files; exp14 `audit_final.json` passes 568 / 66. Exp14 has six
math methods on each isolated stack. Its 192-byte compact coefficient payload
does not mean a 192-byte FEW process; the benchmark retains diagnostic arrays.
The latest per-branch handoffs supersede historical pending statuses below.

<!-- 2026-09-11 17:46 CST (linux): Add the user's next bounded exp15 round.
Read-only peers audit algebra/lifecycle/fairness; Linux root alone writes/runs. -->

Exp15 now closes the shared-trajectory baseline left open by exp14. Fresh A/B
preparation uses identical batch IPC/response counts; the new method invokes
trajectory/amplitudes once and emits signed-m real harmonic bases using the
unchanged native sum. Five CPU pairs give 4.84189x preparation; direct timed
prepare+256 scan gives 4.80291x. Do not multiply by historical exp14 ratios.
Same cached detector basis + fresh statistics gives 8.4789x for 256 points but
loses for 1/16. Retrograde is accuracy-only; the metric is synthetic fixed PSD.
The 680-check / 68-file audit authenticates all final sources and numerical
replays; 5+6 mathematical methods pass on each of two numerical stacks. Both
actual-table workers terminate normally and recover from injected Python sum
exceptions. No production source, native binary, environment or data changed.
The independent note explains the equations, full response, nonzero-reference
SSB-only test and remaining short-waveform/scientific boundaries. All fifteen
attempts remain local/uncommitted; Linux retains the lock and root branch.

<!-- 2026-09-11 21:34 CST (linux): Add independently master-based exp16 after
actual profile, immutable-geometry hardening, lifecycle timing and audit. -->

Exp16 caches the installed FP64 discrete TDI operator in a bounded CSR plan,
without changing native projections, orbit data, dependencies or FEW production.
Immutable factory-time geometry prevents stale Python/native state without
repeated full-content scans; its cost is explicitly included in a separate
fresh-response lifecycle benchmark. Warm 13-input batch times 0.1863->0.1329 s
with new plan; cached plan 0.1860->0.0683 s. Including response construction from
a configured orbit gives only 0.2616->0.2530 s; single-input use is slower.
Original hash-every-plan 0.923x negative result and exact sources are preserved.
Six fresh master FEW holdouts supplement archived real bases. Default max
weighted L2 is 5.668e-14; 96 likelihood checks differ by <=3.493e-10 in synthetic
PSD. The final audit passes 447 checks / 78 files and seven actual-native/unit
methods. Actual source-version/time-origin differences and a broken direct-y
dependency API are recorded, not silently repaired. All processes are terminal;
sixteen independent worktrees remain local/uncommitted with Linux ownership.

## Reproducible local environment

- Python executable: `/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python`.
  Its installed default FEW is an older collaboration build: do NOT use that
  FEW package as the clean-master reference.
- Clean master detached worktree: sibling `master-reference`.
- Toolchain prefix:
  `/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901`.
  GCC/G++/gfortran 14.4, nvcc 12.9.86, CMake 4.4.3, CUDA arch 75.
- GPU observed 2026-09-11: RTX 2080 Ti, 22528 MiB reported memory,
  driver 595.91.07. Do not substitute the older handoff's driver number.
- Native reference build: `artifacts/master-build`; reference installed wheel
  `artifacts/runtime-master` (master dev91, not PyPI 2.1.0).
- Native exp01 build: `artifacts/exp01-build`; installed matching wheel
  `artifacts/runtime-exp01`. Tests overlay latest experiment `src` on this
  native wheel. The final wheel was subsequently rebuilt and installed under
  `artifacts/runtime-exp01-final`, with 7 CPU and 7 CUDA packaged tests passing.
- Shared data directory:
  `/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download`.
  Runtime table SHA256:
  `3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`.
  No 5 GB file duplicated, edited or committed.

Master build initially failed with `FEW_LAPACKE_FETCH=OFF` because pkg-config
could not find LAPACKE. The recorded successful baseline used FETCH=ON and
Reference LAPACK revision `6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`.
Exp01 used FETCH=OFF, DETECT_WITH=DISABLE and EXACTLY that baseline's static
`liblapacke.a;liblapack.a;libblas.a`, with its include directory in CPATH.
Both used `FEW_WITH_GPU=ON`, `FEW_CUDA_ARCH=75`, four build jobs and the same
compiler prefix. This avoids a BLAS-vendor change masquerading as a kernel gain.

Example runtime overlay, after selecting the relevant experiment worktree:

```bash
export FEW_EXPERIMENTS=/home/ubuntu/Desktop/FastEMRIWaveforms-experiments
export FEW_DATA=/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download
export LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$FEW_EXPERIMENTS/artifacts/runtime-master"
```

For exp01 replace runtime-master with runtime-exp01. Use the explicit Python
executable above. No GPU performance probes were run concurrently; physics
checks are not timing evidence. Original timing pilots lack clocks/thermal
telemetry and statistical confidence intervals. Never backfill those as if
they had been measured during the original runs.

## New provenance finding

The [author's data release](https://zenodo.org/records/15631641) has a 2.88 MB
A220 raw slice matching the disputed third fixture's full-precision grid
coordinates. Current FEW vs published value is 1.21e-14 absolute; the old
constant differs by 0.408. See exp03's pinned URL, hash, script and JSON.
Two related author artifacts agreeing is not an independent physical solver
check; the origin of the old fixture still needs clarification.

## Next work, not claims of completion

1. Complete source/native/build provenance and package-level regressions for
   promising exp01; retain opt-in behavior. Evaluate acceleration at the true
   workload boundary and do not multiply unrelated speedups.
2. Continue from independent master branches with parameter-dependency reuse
   (trajectory/amplitude reuse, fixed-distance scaling), spin-local data
   policies, interpolation kernels and trajectory error control.
3. Treat numerical, waveform, detector-weighted and inference-level error as
   different gates. exp05's uniform rate perturbation is not a measured FP32
   error and does not certify mixed precision by itself.
4. Keep weak/negative results and known failures visible. No production merge,
   scientific blanket acceptance or goal-completion claim is justified yet.

Preserved unrelated root scripts: `deep_verify_lazy_loading.py`,
`roman_fusion_probe.py`, `roman_graph_probe.py`, `roman_mismatch_test.py`,
`roman_mixed_precision_probe.py`. Root production code and Mac-owned files
were not changed. Coordination lock remains Linux-owned.

<!-- 2026-09-11 13:20 CST (linux): Update completed replay/package gates without
turning this checkpoint into completion of the open-ended exploration goal. -->

Latest follow-ups: exp01 strict six-case replay passed (one-year 1.303x,
1.381x, 1.191x); its rebuilt installed wheel passed both seven-test backend
runs and the legacy-SASS comparison. exp05's final 45-row replay passed exact
RHS/zero-delta byte checks with source-matching script hash and resolved data
identity. The exp01 initial manual packaged-test command had contradictory
backend settings; the corrected recorded runner resolved that test-invocation
error. It had also downloaded only a small flux table into the new version
cache, noted in the exp01 handoff. No runtime amplitude table was duplicated.

<!-- 2026-09-11 13:23 CST (linux): Close this concrete validation batch, retain
Linux ownership and the open-ended goal, and preserve negative repeat results. -->

Final exp02 raw-byte comparison also passed all four workloads; retrograde
timing in this third run was 0.9970x, not faster. All observed test/build
processes for this batch have terminated. JSON parsing, changed tracked-file
whitespace checks, new collaboration-file whitespace checks, focused Ruff E9/F
checks and local handoff-link existence checks passed. No commit/push occurred.

<!-- 2026-09-11 13:48 CST (linux): Add the independently master-based exp06
attempt and distinguish warm cached-render gains from preparation-inclusive
workloads. The preceding terminal-process statement refers only to exp01–05. -->

Exp06 now owns immutable intrinsic arrays, phase/frequency data, complete
unweighted amplitudes and optional frozen PSD values. Recompute mode selection
for every observer; bind the CUDA device/default stream and reject invalid
thresholds. New component tests are authenticated separately on CPU and CUDA.
The nine-workload real-table suite is currently live; completed CPU short
results are preliminary timing evidence, not blanket scientific acceptance.

<!-- 2026-09-11 14:04 CST (linux): Preserve the exp06 negative/failure evidence
and open a seventh independent attempt for the newly isolated PSD boundary. -->

Exp06's original serial matrix has terminated, eight passes / one reference
failure. Exp07 independently fixes the CPU trajectory input and CUDA PSD
frequency-consumer boundaries, validated against 24 real CPU/CUDA source
strains. Its 734 KB reference artifact can be read by exp06 without importing
the fix into that branch. Exp06's independent PSD replay is currently running.
No commit/push, host switch or modification of root production source occurred.

<!-- 2026-09-11 14:07 CST (linux): Close this validation checkpoint, not the
persistent goal. Record the independent replay and actual remaining scope. -->

Exp06 independent PSD replay passed all 12 cases, byte-identical to exp07 CUDA
references. The eight original passing jobs also retain 80/80 byte-identical
primary waveform comparisons. Source/test/artifact hash checks and focused
Ruff/whitespace checks pass for exp06 and exp07. All currently observed test
processes are terminal. Next available work includes the effective call-time
mode-selection configuration bug, fixed-response distance likelihood reuse,
and broader package/scientific validation. No universal 5x claim is justified.

<!-- 2026-09-11 14:46 CST (linux): Extend the actual experiment inventory to
nine; retain bad pilots and explicit workload/precision acceptance boundaries. -->

Exp08 closes the effective constructor/call-time selection inconsistency with
one production file changed. Five new actual-frontend/stock-selector fixture
tests and the pre-existing selector test pass on CPU; three short actual Kerr
cases each exercise five configuration calls with byte-identical references.

Exp09 implements a script-only exact fixed-shape distance Gaussian likelihood.
The original FEW environment exports authenticated 487 KB SSB data; an isolated
response environment runs actual ESA orbit / second-generation A/E/T. Maximum
channel scaling discrepancy is 2.83e-13. Independent Decimal/Parseval tests pass
on both numerical stacks; longdouble direct norms and regenerated FEW/TDI
references also pass. Local Delta-logL errors are below 1e-8, but enormous
mathematical tails use condition-scaled gates, not that absolute tolerance.

The published lisaanalysistools CPU wheel failed import with unsupported
AVX-512 `vmovdqu8` on this CPU. Explicit x86-64 CPU-only rebuilding in the
response environment fixed it; import -4, GDB, source command, before/after
binary identities are preserved. Main FEW Python and its dependencies remain
unchanged. The original wheel build flags are not known.

Timing lessons: a vectorized quadratic is useful when many distance values are
evaluated together; scalar Python validation/dispatch erases most of the gain on
this short input. Four counts (1/4/16/256), vectorized direct alternatives and
directly timed preparation+scan replace the first optimistic 458x cached-only
pilot. Current preparation-inclusive batch ratios are 0.265/0.503/1.16/14.74x.
These exclude waveform generation, TDI, FFT and data creation for BOTH sides.
No whole-pipeline, full posterior, physical noise covariance or CUDA claim.

Remaining useful directions include mode-index domain validation (aliasing
through negative array indices), the misnamed master sampling dependency,
broader intrinsic/sky/length distance cases, physical noise covariance and full
workload amortization. Experiments 08/09 are not merged into other branches.

<!-- 2026-09-11 14:50 CST (linux): Complete the evidence audit for this checkpoint,
not the persistent exploration goal. All observed tool processes are terminal. -->

`EXPERIMENTS_AUDIT_2026-09-11_1448.json` passes 66 checks across 29 authenticated
files: pinned branch bases, source/native/data/artifact identities, preserved
original failure, actual numerical rows and bridge metadata, finite saved arrays,
and rebuilt dependency identity. Focused Ruff E9/F checks pass for the new
scripts/tests. Exp08's existing selector regression emitted its existing
sensitivity-spline edge-fill warning and still passed; the full log is retained.
No root production or Mac-owned tracked file changed, and no commit/push occurred.

<!-- 2026-09-11 15:35 CST (linux): Supersede the exp10/11 proposal status with
completed bounded evidence, not whole-goal completion or a host transfer. -->

Exp10 and exp11 are now implemented independently on their own master-based
branches, bringing this index to eleven attempts. The first prevents physical
mode-coordinate aliases and invalid raw IDs; the second removes the unrelated
portfolio package from FEW's sampling dependencies. Neither is a measured
speedup. Final per-branch handoffs describe unchanged legal signed modes, actual
5 GB A/B, preserved failures, data/native identities, and isolated dependency
resolution without installing sampling packages.

Exp10's CPU 43 / CUDA 27 arrays compare byte-for-byte to the same-backend master
entry, plus 50 new-interface equivalence checks. Actual CPU waveforms cover both
prograde and retrograde and four mode-selection cases. Nine label-oracle test
methods per backend and limited stock CPU 2 / CUDA 1 regressions pass. The old
upstream Kerr test only asserts its final fixture; no five-fixture certificate
is inferred. CUDA full-frontend and direct-retrograde convention work remain.

Exp11's final dependency probe binds actual resolver versions/hashes to the two
official namespace-inspected wheels, compares metadata fields and dependency
multiplicity, and authenticates the child virtual environment. The environment
contains only pip before and after its dry runs; full installation/ABI/sampling
acceptance remains a different task. Early interpreter-path and SSL failures
are retained, including the fact that the first intended venv was not used.

The source-bound final audit is `EXPERIMENTS_AUDIT_EXP10_EXP11_FINAL_2026-09-11.json`.
Next independent candidates: direct-retrograde public-amplitude conventions,
spin-local table residency, repeated explicit-selector validation cost, and a
normal native sampling environment with actual import/ABI smoke tests. Retain
separate numerical and performance gates; root remains `deep-optimization`,
all experiment edits remain uncommitted, and Linux retains the lock.

<!-- 2026-09-11 16:29 CST (linux): Close exp12's bounded measurement batch and
record a concrete exp13 failure, without completing the persistent goal. -->

Exp12's real-table CPU/CUDA eager and lazy candidates preserve all 39 arrays of
their own master references; hardened tiny native tests pass eight methods per
backend. The serial seven-pair benchmark on each backend is terminal: first
result joint latency improves only 1.0395x CPU / 1.0510x CUDA. CPU first-result
RSS falls 6045.85→1426.14 MiB. CUDA active allocation after visiting both regions
falls 4853.94→295.65 MiB, with reserved pool 5202.91→582.96 MiB; none is a
continuous peak or total GPU-memory measurement. Repeated cache-hit waveform
times are nearly unchanged, and CUDA retrograde/return reloads cost ~5x longer.
This is an opt-in storage-policy tradeoff, not universally faster computation.
Default eager policy, FP64 arithmetic, native binaries and the 5 GB data file
are unchanged. ONCE integrity checks are retained.

The exp12 audit has 4278 passing checks over 91 authenticated files. It retains
the invalid-p8 and initial cache/test pitfalls, the explicit identical CUDA
host-time adapter, limited physical coverage and untested optional diagnostics.

Exp13 is a fresh independent master branch with unchanged production source.
CPU and CUDA archived master probes authenticate five modes at three p/e
points and show that direct-retrograde tuple/raw/all selection can disagree.
They explicitly report a failing API contract, not a passing scientific model.
Next: independent finite-harmonic coordinate identities, a focused public-mode
repair, and unchanged high-level waveform regressions. No commit/push or host
switch occurred; root remains deep-optimization and the overall goal is active.
