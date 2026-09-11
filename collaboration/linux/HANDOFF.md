# exp07 — CUDA PSD frequency boundary

<!-- 2026-09-11 14:04 CST (linux): Preserve original failures, independent
master-based repair, actual reference artifacts and explicit remaining limits. -->

- Branch: `codex/exp07-cuda-selector-frequency-boundary`.
- Base: master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Linux root owns editing. All changes local/uncommitted; no push/host switch.
- This is a correctness/CPU–GPU interface attempt, not a speedup claim.

## Finding and change

Exp06's final CUDA+PSD reference failed before waveform comparison. Passing
sparse CUDA times to the CPU derivative spline was the first boundary bug;
even after a harness host-time adapter, the returned NumPy frequency vectors
were multiplied by CuPy mode indices in `get_mode_frequencies`, raising
`TypeError: Unsupported type <class 'numpy.ndarray'>`.

Two scoped production changes, independently made from master:

1. `waveform/base.py`: explicitly obtain host times for the CPU spline.
2. `utils/modeselector.py`: in the PSD branch only, apply `self.xp.asarray` to
   all three frequency vectors at the consumer. Preserve dtype, Hz units,
   formula, mode selection and original input dictionary. No full time×mode
   matrix transfer and no amplitude-table changes.

The first item overlaps exp04's input-boundary repair; this branch does not
contain exp04's worktree or commits. A future merge must account for that
overlap. No exp06 snapshot implementation is imported.

## Evidence

- `master_boundary_failure.json`: original master failed both tests as expected
  (mixed PSD arrays and CPU sampler input). Failure remains `passes=false`.
- `cpu_boundary_fixed.json`, `cuda_boundary_fixed.json`: both small tests pass
  on their authenticated backends. Extended consumer checks cover all NumPy,
  all backend-native and each single-host-vector mixed case; PSD spy checks an
  independent Hz/frequency-order oracle, dtype and backend. Non-PSD/all/list
  selection paths do not require frequency data. The second test evaluates
  the actual sampler-call AST and verifies one host download only for CUDA.
- `psd_waveform_reference.json`: 3 intrinsic cases × 4 views, separately on
  CPU and CUDA, all 24 generated waveforms passed. Matching selected modes;
  12 CPU/CUDA relative L2 differences range 2.29e-16–1.41e-15, with maximum
  normalized sample difference 2.75e-15. No harness adapter was used.
- `psd_waveform_reference.npz`: 733,966 bytes, SHA256
  `3ed5fa6ac1873290d9029655df801ea89d4b55167515fe4ccfb435e45bfed952`.
  Contains CPU/CUDA complex128 source strains, mode indices and embedded
  JSON metadata. It is an experiment reference, not independent physical truth.

The sensitivity is deliberately synthetic and positive, `1+1000*f_Hz**2`,
to exercise PSD-weighted selection without claiming a detector model. Original
Kerr H5 files are resolved and fingerprinted, not edited or duplicated.
The matching master CPU/CUDA native wheel is reused unchanged, with this
branch's Python source overlay. No installed-package acceptance is claimed.

## Reproduction

From this worktree, use the explicit Python executable
`/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python` and:

```bash
export PYTHONPATH="$PWD/src:/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/runtime-master"
export LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

`boundary_checks.py --backend cpu|cuda12x --output NEW.json` runs the small
tests; `psd_waveform_probe.py --data-dir PATH --output-prefix NEW_PREFIX`
generates the real-reference package. Existing outputs are never overwritten.

## Known remaining boundary

Constructing the ordinary model with `mode_selection='all'` and then overriding
it to `'threshold'` with a PSD still causes the waveform frontend to skip
frequency sampling: it checks the construction-time option, not the effective
call-time one. This pre-existing configuration bug is NOT fixed by this branch.
No arbitrary bidirectional conversion promise is made: a CPU selector is not
made to consume CuPy frequencies implicitly. Long-waveform, helper-specific,
installed-package and broad-regression coverage remain future work.

<!-- 2026-09-11 14:07 CST (linux): Bind the checkpoint to inspected terminal
evidence and preserve the independent nature of the cross-experiment replay. -->

Source/test hashes in both fixed small-test reports and source/script/artifact
hashes in the real-reference report match current files. Ruff E9/F and tracked
whitespace checks pass. Exp06 has now independently replayed all 12 CUDA PSD
reference waveforms with byte identity, without importing this branch's code.
All observed test processes are terminal. Linux retains editing for the active
research goal; no Git commit, push or Mac host transfer has occurred.
