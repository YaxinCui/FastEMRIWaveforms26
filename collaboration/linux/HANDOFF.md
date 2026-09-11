# exp01 — FP64 common-phase aggregation

<!-- 2026-09-11 13:11 CST (linux): Preserve implementation, pilot evidence and
validation limits on an independent master-based branch. No host switch. -->

- Branch: `codex/exp01-common-phase-fp64`.
- Base: `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191` (`master`).
- Linux root owns editing; Mac may read. Changes are local and uncommitted.
- Production changes: `commonphase.py`, `interpolatedmodesum.py`, native
  `interpolate.cu/.hh`, and `pyinterp.pyx`. Default `common_phase=False`.

## What changes

After existing mode selection, group by the exact signed `(m,k,n)` tuple,
not by approximate instantaneous frequency. Different ell values share the
orbital phasor. At sparse nodes, form separate complex B+ and B- sums with
the original angular weights, conjugation, ell parity and m=0 convention.
Interpolate four real channels per group and evaluate each group phasor once
per dense output time. FP64 throughout; no additional mode threshold.

This is for final strain at a fixed observer. It is not an interface for
returning individual ell modes. Equivalent rearrangement requires a shared
time grid and linear spline construction. Roundoff may differ from the old sum.

## Evidence

- CPU and CUDA `tests/test_common_phase.py`: 4 methods passed, including
  independent SciPy/analytic oracle, complex +/- weights, k!=0, signed modes,
  nondefault stream, phase offset 1e8, changed topology, and native tile edges
  225/226/450/451. Legacy float arrays containing exact integer indices work.
- `common_phase_adapted.json`: 7 alternating synchronized AB/BA repetitions,
  two warmups, shared model/table, complete waveform generation excluding
  construction, TDI and likelihood. Three one-year cases: 1.298x, 1.381x,
  1.201x; short cases 0.969–1.029x. Relative L2 4.7e-16–1.4e-15.
  This is pilot evidence; GPU clocks/temperature were not recorded.
- `common_phase_pilot.json` preserves the unadapted master frontend failure.
  A pre-existing CPU/CuPy derivative boundary fails before summation. All
  adapted comparisons explicitly apply the same host-time bridge to BOTH
  variants. The independent production fix is exp04, not included here.
- `common_phase_strict.json` is the follow-up run with fail-closed finite,
  dtype, shape, mode-set and 2e-12 numerical gates and actual source/native/data
  hashes. All six cases passed. One-year speedups are 1.303x/1.381x/1.191x;
  short cases 0.980x/1.044x/0.968x. Do not infer success from file existence.
- `independent_build_comparison_strict.json`: short and one-year nominal
  legacy arrays match the separate master-native + exp04 production-fix run
  bitwise. The saved-waveform validator has independent NaN/Inf/zero tests.
- `sass_audit.json`: master legacy and exp01 `<false>` sm75 kernel have
  identical 1160 SASS instructions / 2320 machine-code words. This is NOT
  proof of identical whole modules, host wrappers, clocks or performance.

## Reproduce

Build CPU+CUDA from this worktree, architecture 75 on this host. A core from
this revision must not use the incompatible PyPI 2.1.0 CUDA plugin.
Local tests used current `src` on PYTHONPATH plus the matching native wheel
under sibling `artifacts/runtime-exp01`. Reference native binaries came from
clean master under `artifacts/runtime-master`. Both native builds link the
same pinned Reference LAPACK build; build details are in the coordinator index.

```bash
FEW_TEST_BACKENDS=cpu,cuda12x python tests/test_common_phase.py -v
python -m unittest discover -s collaboration/linux -p test_compare_saved.py -v
python collaboration/linux/common_phase_probe.py --data-dir "$FEW_DATA" \
  --output collaboration/linux/replay.json --host-derivative-adapter
```

`FEW_DATA` must identify the existing data directory. Downloads are disabled.
Do not overwrite preserved evidence when running a new trial.

Remaining: broader regression suite, independent process/clock-controlled benchmarks,
PSD/TDI error criteria, and CPU performance characterization. This branch
does not establish 5x speedup or global scientific acceptance.

<!-- 2026-09-11 13:20 CST (linux): Complete final installed-wheel checks and
retain the initial test-invocation mistake as a reproducibility note. -->

Final CPU+CUDA wheel rebuilt and installed under `artifacts/runtime-exp01-final`,
SHA256 `d342ce4aa49c1debcec0f2b4a7d39b7e104dc43454731f49641a3327f7be0c9a`.
`installed_checks.json`: without any source overlay, CPU 7/7 and CUDA 7/7
tests pass (new common-phase oracle plus existing spline/direct/interpolated
sum tests). `installed_checks.py` captures commands, logs, wheel identity and
backend selection. `sass_audit_final.json` also passes for the installed native
module; the final legacy SASS still matches the master reference.

The first manual installed-test invocation incorrectly enabled ONLY CUDA but
asked the new tests to exercise BOTH CPU and CUDA, so CPU subtests failed with
`BackendAccessException`, not a numerical failure. It also omitted the external
data path and automatically downloaded the small flux table into FEW's new
version cache (not the 5 GB amplitude table). The corrected runner explicitly
aligns backend/test selection, uses existing data, and disables downloads.
