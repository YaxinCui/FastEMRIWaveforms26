# exp08 — effective call-time mode-selection configuration

<!-- 2026-09-11 14:44 CST (linux): Record the independent master-based fix,
real-table evidence and limits. Root sole writer; no host switch or Git push. -->

- Branch: `codex/exp08-effective-mode-selection`.
- Base: `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191` (`master`).
- All changes are local/uncommitted. Linux retains editing. Mac may read.
- Production change: only `src/few/waveform/base.py`, seven added lines and one
  replacement. Other experiments' changes are not imported.

## Defect and repair

The frontend used the constructor's `mode_selector.mode_selection` to decide
whether to sample orbital frequencies, while the actual selector honored the
call-time override. Thus constructor `all` + call `threshold` + PSD entered a
PSD selector with no frequencies; the inverse override sampled unnecessarily.

Resolve the effective call-time value once, before batching, using the existing
None/default contract. Use it only in the existing frequency condition. Do not
mutate constructor defaults or change thresholds, PSD formula, dtype or modes.
List and non-PSD frequency redundancy are intentionally left unchanged.

## Evidence

- `master_config_failure.json`: original four tests, one pass / two errors /
  one failure, preserved. The original all-to-threshold case failed on missing
  frequency data; threshold-to-all made unnecessary derivative calls.
- `config_fixed.json`: those four tests passed after the production change.
- `config_extended.json`: five tests passed, adding a None constructor and
  full per-batch selected-array comparison with a separate default-threshold
  pipeline. Real frontend + stock selector are used, but trajectory/amplitudes
  are small fixtures and the sum is a recording fixture, not a physics oracle.
- `config_stock_audited.json`: final runner also includes the existing
  `tests.test_mode_selector` CPU regression, with downloads disabled: six tests
  passed, returncode zero. Existing sensitivity-spline out-of-bounds edge-fill
  warning is preserved in the log, not hidden.
- `real_config_audited.json`: actual 5 GB Kerr table, CPU, three intrinsic cases
  (nominal / high eccentricity / retrograde), five calls each. T=0.0001 yr,
  dt=15 s. All-to-threshold-to-default and independent constructor-threshold
  references agree byte for byte, including actual selected mode indices.
  Threshold genuinely removes modes; PSD call counts are [0,1,0,1,0].

The two selector instances share one amplitude generator/table, not a second
5 GB allocation. These are independent configuration references, not independent
physical models. The final probe was repeated after formatting to keep source
hashes authentic; `real_config_first.json` remains an earlier source snapshot.

## Reproduce

Use the main FEW Python but overlay this branch's sources and the clean master
native runtime. The default installed main FEW is older and not the reference.

```bash
export PYTHONPATH="$PWD/src:/home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/runtime-master"
export LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib
export FEW_ENABLED_BACKENDS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FEW_FILE_EXTRA_PATHS=/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download
/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python collaboration/linux/config_checks.py \
  --include-stock-selector --output collaboration/linux/config_replay.json
/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python collaboration/linux/real_config_probe.py \
  --data-dir "$FEW_FILE_EXTRA_PATHS" --output collaboration/linux/real_config_replay.json
```

Output paths must be new. Real evidence records source/native/data identities.
This branch does not include exp07's separate CPU/CuPy interface repairs, so no
CUDA frontend success is claimed. No installed-wheel test, long-duration error
study, physical TDI acceptance or acceleration factor is claimed for exp08.

Future separate attempt: invalid explicit mode indices can alias valid negative
indices in array maps; do not silently fold a new domain-validation policy into
this narrow configuration repair.
