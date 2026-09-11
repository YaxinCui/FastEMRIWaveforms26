# exp16 — prepared detector response

<!-- 2026-09-11 21:06 CST (linux): Read coordinator/exp15 handoffs, current
branches/processes, actual installed response.py/detector.py and newer reference
repository native source. Query primary response paper online; no downloads.
Create independent master worktree and Linux-only diagnostics. -->

Previous goal turn was progress: exp15 implements and verifies single-pass
phase bases. This attempt starts from master independently; exp15 archived
SSB inputs may be read after authentication, not silently imported as source.
Root checkout stays deep-optimization; all changes remain local/uncommitted.

Hypothesis: amortize waveform-independent response preparation while retaining
the existing FP64 discrete response and real second-generation A/E/T outputs.
First profile stock 13-row phase-basis response, then select a measured hotspot.
Fair timings must include preparation when claiming per-batch improvement.
Already-prepared repeated workloads need their own distinctly named baseline.

Version warning: installed fastlisaresponse 1.1.17 response.py SHA256
`7a40c9ffd992bf09b2e84d201ce5e9f5da0c2c19ea33efd821f7afa4d027af4a`
differs from the newer local reference repository. In the installed API `t0`
is a crop/buffer and time starts at zero; the newer API has separate physical
start and buffer plus a different native ABI. Do NOT overlay the newer source.
The newer native multi-delay formula also requires separate physical-convention
review; do not change its ordering while claiming a pure speed optimization.

Keep original orbit data and native response intact. Audit hp+ihx convention,
shape/length, sky, dt/order/buffer, input ownership, exceptional/invalid calls,
fixed vs changed configuration, actual prograde/retrograde bases and fresh
holdout SSB inputs, response/FFT/numerical likelihood, and memory cost. Avoid
claims about physical covariance, annual signals, GPU, global waveform speed,
or new physics from short conditional-response tests.

Primary source: [Katz et al., generic LISA response](https://arxiv.org/html/2204.06633).
The paper motivates retaining evolving orbit geometry and distinguishes
projection from TDI; our evidence must identify the exact installed algorithm.

<!-- 2026-09-11 21:34 CST (linux): Close bounded exp16 after serial FEW export,
two response A/B rounds, lifecycle comparison and source/numeric audit. Record
all read/query/download/generation scope; Linux remains sole editor/runner. -->

## Outcome and scope

`codex/exp16-prepared-detector-response` is independently based on master
`47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`. No FEW production source, installed
dependency/native binary, Mac-owned file or H5 data changed. Root remains
deep-optimization; sixteen experiment worktrees remain LOCAL/UNCOMMITTED.
No commit, push, merge or host switch occurred. The open exploration goal is
not complete; this is one finished, bounded response-layer attempt.

The new FP64 TDI plan is numerically validated and helps repeated responses.
It is **not a broadly faster one-shot FEW workflow**. Five serial alternating
pairs per row give these prograde CPU medians for 13 SSB basis inputs:

| Starting point and measured work | Stock (s) | Candidate (s) | Ratio |
| --- | ---: | ---: | ---: |
| Already frozen response; build plan + process batch | 0.186309 | 0.132883 | 1.4021x |
| Already prepared plan; process batch | 0.186007 | 0.068269 | 2.7246x |
| Configured orbit; create response + plan + batch | 0.261578 | 0.253039 | 1.0337x |

The last row includes immutable-geometry conversion and both response objects'
normal construction costs. It uses a stronger stock packer without redundant
per-channel copies. The ~3% difference is small, not a robust large cold-start
acceleration claim. Orbit configuration, OS/H5 cold loading and FEW generation
are excluded even here. For ONE waveform this lifecycle comparison is worse:
0.090432 -> 0.189511 s (0.4772x). On already frozen response, one fresh plan is
also worse, 0.014422 -> 0.069645 s (0.2071x). Reuse matters.

The main batch paths include all stock projections, native TDI or sparse TDI,
cropping/packing, FFT, whitening and returned batch ownership. No projection
is dropped. Each 13-input stock batch calls the stock TDI path 13 times;
the prepared batch instead applies its fixed matrix 13 times. Initial plan
construction makes six vector light-travel-time queries. All timed outputs
are checked and raw samples preserved. Do not multiply these ratios by exp15
or exp14 medians, or call them GPU/full-FEW/annual-data speedups.

## Algorithm and exact version boundary

The stock stage profile averages 14.419 ms per basis: projection 4.058 ms,
TDI 9.954 ms, packing/FFT 0.408 ms. Native TDI alone is 9.807 ms. This profile
includes diagnostic wrapping; it selects the hotspot, not the final speedup.
The final implementation keeps native projection, orbit interpolation and
stock A/E/T conversion, and expands the existing finite TDI interpolation into
a sparse matrix from six link streams to cropped XYZ. It retains FP64 and the
actual A/E lookup interpolation; it does not replace it with different analytic
weights, remove modes, assume constant arms, or use a complex-linear response.

The first COO implementation created 4,550,208 entries and then sorted and
combined duplicates. The final assembler first groups by channel/link and
accumulates narrow time bands in original TDI-unit order, then creates CSR
directly. Final 6312-sample/order-25 plan: 1,246,932 nonzeros, **15,006,940
bytes** across CSR values/indices/row pointers. This is not process RSS, total
response memory or measured construction peak. The plan constructor bounds
uncompressed work to 10,000,000 candidate entries. Long observations will
require a separately tested blocked/streamed policy; this is not an annual
full-size sparse response implementation.

Exp16 targets fastlisaresponse 1.1.17, whose Python and CPU `git_version.py`
record c16cd9cbf8d69719067040cf6c170a41c696c05d. The matching official native
source is downloaded under knowledge/, SHA256
`583a38e10ddee30707ba2e853c93474671d3fdfd3915ef78643d5e28e4c458c6`.
Current native binaries are separately authenticated in the reports. Its
projection and TDI share the originally injected response_orbits, not the
separate Python tdi_orbits property. Its time axis starts at zero: t0 is buffer.
The newer local reference repository cannot substitute for this ABI/API.

The code subtracts each L_link(t) in the original link order, then computes
ceil/sample fractions identically. Same-t chained delays are intentional in
the primary paper section IV.3 equation (20), for its GW response approximation.
This is not full recursive delay ordering or laser-noise-cancellation validation.
Sparse coefficient expansion/aggregation reorders floating operations, so the
claim is bounded numerical agreement, not bitwise equality of sparse output.

## Immutability, ownership and negative evidence

An arbitrary existing response can have Python orbit attributes diverge from
the orbit already passed into C++. The final helper therefore accepts only
its registered `_FixedResponse` factory. It converts orbit arrays to immutable
bytes-backed NumPy views **before their first native injection**, and freezes
the TDI topology and A/E tables similarly. Ordinary NumPy cannot re-enable
writeability. Plan creation checks original native/orbit identity, array
identity/shape/dtype/strides and configuration. This avoids rescanning the
122,442,744 bytes of orbit-array payload at every plan construction. It moves
copying to factory lifetime; the lifecycle benchmark explicitly charges it.

Plan outputs own their storage; the CSR does not retain the response or input
waveform. A plan is a fixed dt/N/order/orbit/topology/crop operator. Changing
sky alone is allowed through fresh stock projections because TDI does not
depend on sky. Rebuild for changed operator configuration; raw same-shaped
link inputs cannot convey changed time units or provenance. Explicit native
rebinding, ctypes mutation, concurrent mutation and private-attribute tampering
are outside this script prototype's contract. It is not a production cache API.

Preserved evidence:

- `profile_first.json` selects the native TDI hotspot; source unchanged.
- `tdi_first.json` is the original COO pilot, with exact producer/helper bytes
  in `pilot_sources/`. Its ~0.8 ms cached kernel is not a whole-response ratio.
- `response_final.json/.npz` is a successful numerical but slower fresh-plan
  run: registered mutable arrays required full geometry hashing; 13-input ratio
  **0.9232x** with preparation. Its exact two source files are archived with
  `_registered` names. This negative result was not overwritten.
- `FIRST_TEST_FAILURE.md` and archived original test source retain the missing
  upstream explicit-y attribute error and one over-exact integer-node test
  expectation. The final test preserves the actual stock A-table factor.

The dependency's direct `get_tdi_delays(y_gw=...)` references undefined
`link_space_craft_0_in`. Normal FEW response calls the no-argument path and is
unaffected. Random-link native tests populate the documented link buffer and
use that unchanged no-argument path; no installed-package fix is smuggled in.
This is a candidate for a separate independent API repair experiment.

## Numerical evidence

`export_holdout.py` freshly generates six clean-master source waves (three
off-grid phases each for prograde and retrograde), T=.003 yr, dt=15 s, actual
original 5 GB table. It checks imported FEW source against master and binds
all native files/data digests. No exp15 implementation is imported. Exp15's
13 phase bases per source are read only as authenticated numerical inputs.

Final `response_immutable_final.json/.npz` checks 32 basis/holdout responses and
96 direct Gaussian likelihood values (including basis-function test inputs,
not 96 distinct physical sources) at SNR 20/100/1000. Noise is a fixed SYNTHETIC
diagonal A/E/T PSD. Maximum weighted global relative L2 is **5.668e-14**,
maximum individual-channel relative L2 **5.577e-12**, and maximum absolute
Delta-logL discrepancy **3.493e-10**. Gates are 1e-9 for waveform/channel error
and 1e-5+2e-10*|reference| for likelihood. These local gates do not certify
physical covariance, self-force/model accuracy or full inference.

Additional zero/pure-hplus/pure-hcross synthetic chirps exercise changed sky,
dt=7.5/30 s, interpolation order=15/35, different buffers. Their maximum
individual-channel relative discrepancy is 8.379e-12. For two real sources,
three waves each also compare an unmodified public response with the immutable
factory: both native projection and native TDI are byte-identical, 12 checks.

Seven final unit methods cover an independent longdouble scalar-stencil oracle,
integer/near-integer points, independent six-link random/impulse/zero inputs,
real linearity, factory vs plain native bytes, owner release, unchanged results
after input changes, malformed inputs, budgets, forbidden array/config/orbit
replacement and immutable buffers. An intentional metadata-mutation test emits
the existing NumPy 2.5 shape-assignment deprecation warning; the log is retained.

`audit_final.json` passes **447 checks across 78 authenticated files**, replays
all 96 longdouble likelihood comparisons from saved arrays, verifies timing
arithmetic/call counts/lifecycle bounds, source/native/archive identities and
the original slower result, and runs all seven methods (`math_final.json`).
It does not reread the large Kerr file; its digest was checked by the fresh
FEW producer. Audits are checks, not 447 independent physical cases.

## Reproduction and next work

Commands actually used, from this worktree:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python \
  collaboration/linux/test_prepared_tdi.py
```

The same interpreter/environment ran `profile_response.py`, `validate_response.py`,
`lifecycle_response.py` and finally `audit_response.py` serially. These snapshot
producers deliberately reject existing output filenames. To reproduce the
generation runs, preserve this evidence and choose new output targets in an
independently recorded replay; do not delete/overwrite the originals. Exact
producer bytes and environment/native versions are in the report identities.

The FEW export used the main `.venv/bin/python`, `FEW_ENABLED_BACKENDS=cpu`,
`PYTHONPATH=.../master-reference/src:.../artifacts/runtime-master`, and
`LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib`.
File downloads were disabled, ONCE integrity preserved. Both installed numerical
environments remain unchanged. Every started FEW/performance/audit process is
terminal; Linux keeps the edit lock, no host transfer is requested.

Next useful work: independent explicit-y dependency repair; a lifecycle-aware
response service for repeated FEW evaluations; bounded-block long-observation
plans and actual GPU comparison. Do not equate a cached sparse kernel gain
with an end-to-end waveform gain or claim general scientific acceptance.

<!-- 2026-09-11 21:37 CST (linux): Final eight-script Ruff E9/F and format
checks, root/new-note whitespace checks, 22 local-link checks and fresh 78-file
SHA replay pass. Root production and Mac-owned tracked files are unchanged.
No source/test/report was edited after the numerical audit; only notes/indexes.
Leave the broad goal active, with next useful independent work identified. -->
