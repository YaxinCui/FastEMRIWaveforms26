# exp10 — signed mode domain and index safety

<!-- 2026-09-11 15:28 CST (linux): Record actual master A/B evidence and
limitations. Linux sole writer; no Mac transfer, commit or push. -->

- Branch: `codex/exp10-mode-index-domain`.
- Independent base: master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- All changes local/uncommitted. Linux keeps the experiment lock.
- Production changes: `amplitude/base.py`, `utils/modeselector.py`, new
  `utils/modeindices.py`; `utils/baseclasses.py` has documentation changes only.
  No waveform/frontend or native changes; no exp04/07/08 patches imported.

## Defect and repair

The signed index-map layout uses Python negative indexing internally. This is
NOT the physical/truncation domain. A Kerr request `(2,2,0,56)` currently indexes
the same slot as `(2,2,0,-55)` despite claiming a different harmonic label. The
actual 5 GB table reproduced this in both amplitude and selector entries on CPU
and CUDA. The corrected APIs reject it before amplitude evaluation.

Shared validation enforces integer, nonempty shapes; ell>=2, |m|<=ell; and
ell/m/k/n map coverage before gather. It rejects floats, booleans (including
mixed Python sequences before coercion), unsigned overflow, negative raw IDs,
out-of-range IDs, sentinels, and malformed arrays. A value already coerced into
an integer ndarray cannot be identified retrospectively as an original bool.
Legitimate negative m/k/n remain legal. k and n are NOT constrained by ell.
The full negative-m raw-ID tail remains supported. Parity uses integer modulo,
avoiding negative integer exponents for generic k<-ell.

Native/custom-map bounds are checked separately; a custom map cannot fill a
native catalogue hole. Legacy custom-only adapters remain supported, with their
provided map as the authoritative domain: unknown native coverage cannot be
verified. Constructor defaults are copied to a Python tuple list so later
caller mutations do not change the selection and N-by-4 ndarray defaults work
through the unchanged high-level waveform frontend. Raw-ID inputs are owned
copies, including read-only and noncontiguous arrays.

## Evidence, not just green status

- `master_domain_cpu_extended.json`: 9 methods, 53 failed subtests/33 errors.
- `master_domain_cuda_extended.json`: same test hash, 73 failures/18 errors.
  These totals include input-contract expectations, not that many independent
  physics bugs. Unique labels permit invalid-API tests without unsafe native IDs.
- `domain_cpu_extended.json`, `domain_cuda_extended.json`: 9/9 methods each pass,
  no skips. Imported files/native hashes match current sources. An independent
  closed-form mode-ID and conjugation oracle includes odd/even k, m=0, signed
  boundaries, custom-map holes/shape changes, and constructor input ownership.
- `real_master_cpu_replay.json/.npz`, `real_master_cuda_replay.json/.npz`:
  successful actual clean-master references, not the older installed FEW.
- `real_fixed_cpu.json/.npz`: all 43 valid-output arrays byte-identical to master.
  Includes 8 short CPU source waveforms (prograde/retrograde × threshold/all/
  positive-list/signed-list) and their mode lists, plus actual amplitudes/selector.
  Four list cases also pass array-call and array-constructor/default comparisons.
- `real_fixed_cuda.json/.npz`: all 27 amplitude/selector arrays byte-identical
  to CUDA master; no full CUDA waveform claim. Host raw IDs and device N-by-4
  selector arrays are compared with the corresponding legacy device/list entry.
- Both actual backends reject the two invalid n=56 entry tests after the repair.
  Baseline JSON preserves the alias and its measured amplitude, not an unsafe
  native out-of-bounds experiment.

The real probes compare each interface with ITS OWN baseline for `(a,xI)` equal
to `(.6,+1)`, `(-.6,+1)`, and `(.6,-1)`. Existing direct-retrograde tuple/raw/
default transformations differ; their physics convention has NOT been resolved
by this indexing change. Higher-level retrograde waveforms first transform spin
and observer coordinates, so they are a separate contract.

Data are shared in place, not copied or edited. Every real manifest records
resolved data and native hashes; candidate references require identical data,
native files and archive hash. Kerr table SHA256:
`3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`.
Snapshots are small output NPZs, not another copy of the 5 GB H5.

## Reproduce

Use Python `/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python` and:

```bash
export FEW_EXPERIMENTS=/home/ubuntu/Desktop/FastEMRIWaveforms-experiments
export FEW_DATA=/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download
export LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FEW_ENABLED_BACKENDS=cpu
export PYTHONPATH="$PWD/src:$FEW_EXPERIMENTS/artifacts/runtime-master"
/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python collaboration/linux/domain_checks.py --backend cpu --output collaboration/linux/new_domain_cpu.json
/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python collaboration/linux/real_domain_probe.py --backend cpu --data-dir "$FEW_DATA" --guarded --reference collaboration/linux/real_master_cpu_replay.npz --output collaboration/linux/new_real_cpu.json
```

For a new baseline replace `$PWD/src` with `$FEW_EXPERIMENTS/master-reference/src`,
omit `--guarded` and `--reference`, and use new output paths. For CUDA set both
`FEW_ENABLED_BACKENDS` and `--backend` to `cuda12x`, choosing its reference NPZ.
The baseline real script prints a generic list/default/array progress line;
only the guarded candidate runs extra array compatibility checks, as recorded
in the JSON `compatibility` list. Do not infer coverage from that console text.

Early failures remain: `real_master_cpu.json` caught the harness assuming that
a namespace package has `__file__`; the corrected harness anchors identities in
the actual `pyinterp` extension. Earlier small reports precede strengthened
oracles/ownership tests and are not final evidence. The initial default fixture
also used include-minus=False with threshold, exposing an unrelated old warning
that claims an override without applying it; final default controls use True.

## Scope and next work

No speedup is claimed. Repeated validation/copies can add overhead, especially
for constructor-selected repeated requests. No precision was reduced. This is
API correctness and valid-input regression, not generic-Kerr physical coverage,
full CPU/CUDA equivalence, TDI/likelihood acceptance, or a whole-suite certificate.
The independent CUDA derivative boundary, old amplitude-fixture indentation,
warning-only include-minus override, and direct-retrograde convention audit
remain separate candidates. Keep failures visible instead of importing fixes
from other experiments and calling the result master-independent.

Primary engineering reference: [CuPy indexing differences](https://docs.cupy.dev/en/stable/user_guide/difference.html)
documents advanced integer out-of-bounds wrapping. Physical coordinate limits
and signed mode symmetries here are grounded in pinned FEW `SphericalHarmonic`
and `AmplitudeBase` code, with independent label-oracle tests above.

<!-- 2026-09-11 15:35 CST (linux): Add completed limited stock regressions and
their test-invocation correction; do not imply the old fixture bug was fixed. -->

`stock_cpu.json` passes the unchanged upstream Kerr amplitude and Schwarzschild
mode-selector tests (2 methods); `stock_cuda.json` passes the unchanged Kerr
amplitude test (1 method). Source hashes and selected backends are recorded.
The CPU selector emits its pre-existing sensitivity-spline edge-fill warning.
The old Kerr amplitude test still asserts only its LAST tabulated fixture, so
these passes do NOT mean all five constants were validated. See exp03.

The first two direct `stock_checks.py` invocations terminated with
`ModuleNotFoundError: No module named 'tests'` before generating JSON: executing
a script in the collaboration subfolder did not put the worktree root on
`sys.path`. The corrected runner inserts that specific root and authenticates
the actual test sources; no installed older FEW tests are substituted.
