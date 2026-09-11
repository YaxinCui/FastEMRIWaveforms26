# exp17 — explicit six-link TDI input contract

<!-- 2026-09-11 21:43 CST (linux): Inspect current state and exp16 evidence;
read actual 1.1.17 response/detector interfaces and query official current source
through read-only peers. Copy only one dependency Python source into original
and candidate experimental areas. Root production and installed files untouched. -->

Previous goal turn is verified progress: exp16 source/numeric/lifecycle audit
passed. This attempt independently starts from master and fixes a dependency
API contract, not a measured acceleration. Root remains deep-optimization.

Installed fastlisaresponse 1.1.17 `response.py` SHA256 is
`7a40c9ffd992bf09b2e84d201ce5e9f5da0c2c19ea33efd821f7afa4d027af4a`.
Its direct `get_tdi_delays(y_gw=...)` references undefined
`link_space_craft_0_in`. A fresh object also lacks initialized link/crop state.
Normal get_projections->get_tdi_delays calls are unaffected by that attribute.

Plan: isolated single-file candidate, strict backend-native FP64 (6,N) input,
optional keyword-only t0 BUFFER initialization for standalone arrays (not
absolute start time in this old API), pre-native shape/finiteness/domain gates,
and commit output/buffer state only after a successful native call. Preserve
default native arithmetic, link order and XYZ/AET/AE conventions. Inputs may be
strided/read-only but must not be silently cast from complex/FP32/other backends.

Validation must not compare the explicit argument only with identical resident
input. Test different replacement arrays, independent single-link topology,
random/pulse/zero and real FEW-derived links, all channels, fresh/noarg behavior,
invalid calls before native, state after failures and native-injected exceptions.
No GPU or full LISA physics claim follows from CPU interface tests.

Source: [fixed installed-version implementation](https://github.com/mikekatz04/lisa-on-gpu/blob/c16cd9cbf8d69719067040cf6c170a41c696c05d/src/fastlisaresponse/response.py).
Current official master also retains the missing-attribute check, but has a
different time-origin API/native ABI and must not be overlaid into this version.

<!-- 2026-09-11 22:14 CST (linux): Resume after user status interruption.
Root read candidates/fixtures/reports/native source and paper Eq19/20; peers
reviewed read-only. Generate baseline, unit, optimized-Python, real replay and
audit reports; update experiment and root indexes. No old worker restarted. -->

## Completed bounded result

The one-method CPU repair passes. `candidate/response.py` changes ONLY
`pyResponseTDI.get_tdi_delays`; method-excluded AST and exact surrounding text
both match `original/response.py`. The original matches installed source bytes.
The package is loaded under separate relative-import-compatible aliases, never
overwriting canonical modules or installed files. Root remains deep-optimization;
this master-based worktree is codex/exp17-explicit-link-input. All edits LOCAL /
UNCOMMITTED. No commit/push/merge or host transfer is requested.

The accepted experimental domain is deliberately narrower than old arbitrary
calls: CPU only, ordinary native-endian NumPy float64 arrays, positive mutually
consistent FP64 dt/fs, 0<N<2**28, integer order 1..40, maximum 64 actual delayed
links per unit, normal unchanged constructed native orbit/configuration.
CUDA is explicitly rejected; its wider shared prefetch/stream safety was NOT
tested. This is not an all-backend production/upstream-ready replacement.

The optional keyword-only t0 requires explicit y and means a two-ended BUFFER
in seconds for old API 1.1.17, not physical start time. Without t0 a valid stored
cut is reused. Fresh standalone success initializes all link/crop/output fields.
Input validation and local buffer ownership occur before native; state commits
only after native+channel conversion return finite results. XYZ keeps its
original current-output view semantics. Invalid calls and injected recoverable
Python exceptions leave prior link/crop/output state unchanged; this does not
repair pre-existing failed get_projections state or native faults.

### Evidence

- `original_failure.json`: three expected AttributeErrors reproduced BEFORE
  repair, plus ordinary implicit native checksum. This PASS flag means the
  expected failure was captured, not that the original API was healthy.
- `FIRST_REVIEW.md`: preserved issues found before running the candidate:
  FP32/longdouble scalar promotion, MaskedArray/matrix subtypes, odd-N cut,
  overflow before state commit. Also corrects an estimated source-comment
  timestamp; generated JSON timestamps are authoritative.
- `unit_first.json` and `unit_optimized.json`: 10 methods PASS on ordinary and
  python -O interpreter. 36 synthetic order/topology/channel configurations
  (six orders × two topologies × three channels), each with zero/pulse/random
  inputs and explicit/resident comparisons. Includes standalone initialization,
  replacement B, independent undelayed-link row oracle, input/output ownership,
  malformed arrays/scalars/topology, missing buffer, orbit end and injected
  Python native/AET conversion exceptions. No invalid raw native call needed.
- `real_final.json/.npz`: terminal PASS, 351 producer checks, six authenticated
  existing exp16 FEW source holdouts × two generations × three channel outputs
  = 36 configurations, 180 saved arrays. Fresh explicit, implicit and distinct
  replacement B match original native CPU bytes; max residual = 0 for these
  cases. FEW holdouts use real Kerr table provenance but were NOT regenerated
  here, and the 5 GB table was not revalidated by this experiment.
- `audit_final.json`: terminal PASS, 983 checks / 70 authenticated files.
  Independently replays saved-array FP64/shape/bytes/longdouble residual/crop/B
  checks. Five additional native-orbit stencil configurations use dt .1/1/15/40,
  order 15/25/35/40 and up to 64 sequential delays at the lowest accepted cut;
  every actual d-h+1..d+h and A-table ind/ind+1 is valid, native bytes agree,
  one sample below conservative cut is rejected. This is domain testing, not
  an arbitrary-FP64 formal proof or full physical model certificate.
- Critical Ruff checks E9/F63/F7/F82 pass; original invalid-escape docstring
  SyntaxWarnings are preserved outside the changed method. No all-style lint
  claim. Imports may generate normal interpreter bytecode caches; package
  source, native binary, dependency installation and data remain unchanged.

### Reproduce (CPU only)

Use `/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp09-response-env/bin/python`
with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''.
Run `test_explicit_links.py NEW_NAME.json`, or `python -O` with another NEW_NAME.
Real/audit producers refuse to overwrite evidence; their existing reports are
the authoritative completed run. To repeat them, use a separately identified
output path and keep the prior producer/report bytes, never silently overwrite.
The local sibling exp16 holdouts and old orbit provenance must remain available.

Final candidate SHA256:
`5a440660abe331fc1df72c5b7feb9b3b6ea5a6a0b89759ea8b9f0cc8ed142647`.
Real report SHA256:
`2b89441d451b575abdeb7fd6d4e897ef85b2901ad281adecd08abe99107bab4c`.
Audit report SHA256:
`7a9806e53095a5427836c63f4a1109e9935091e75b9ed9100d6b144fbdfe9e5f`.

### Next direction

This repairs a dependency interface; no speedup was measured. Separate the
service lifetime from individual waveforms: fixed geometry/interpolation can
be prepared once, while sky/source inputs change without reusing stale data.
Next bounded performance experiment should compare batched operator application
and construction/cleanup-inclusive latency against an equally reused native
baseline. Do not multiply historical exp15/16 ratios or promise GPU benefit.
See [Chinese explanation](../../knowledge/EXPLICIT_LINK_CONTRACT.md).

<!-- 2026-09-11 22:17 CST (linux): Final read-only peer review found no
substantive false-positive in B replacement or independent link routing. -->

Coverage clarification: the six holdouts are TWO physical intrinsic systems
(prograde/retrograde) with three phases each, not six independent intrinsic
systems. The real producer's label "input unchanged" compares stored explicit
links with the passed array, not a before/after snapshot; read that individual
check as "stored explicit links match input". Actual before/after bytes and
bidirectional ownership are independently asserted in the passing unit tests.
The historical source/report label is retained and disclosed, not silently
edited after hashing. Native byte-equality is not an independent physical oracle.
