# exp04 — explicit CUDA-to-CPU trajectory derivative boundary

<!-- 2026-09-11 13:11 CST (linux): Document the standalone default-path fix,
without importing other experiment implementations or assuming a GPU fallback. -->

- Branch: `codex/exp04-cuda-trajectory-boundary`; master base
  `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Linux root owns editing. Local/uncommitted, no host switch.
- Only production change: `src/few/waveform/base.py` explicitly calls `.get()`
  on CuPy sample times before the CPU-owned trajectory derivative evaluator.
  CPU input remains unchanged. No FP64 arithmetic is reduced.

On clean master, default Kerr CUDA waveform generation fails before summation:
NumPy trajectory knots reach CuPy `searchsorted` through device sample times.
The fix respects the existing architecture: orbit integration is CPU-side,
while amplitude selection/summation can use CUDA.

`tests/test_cuda_trajectory_boundary.py` now locates the actually imported
source, including after wheel installation. Its AST-isolated branch test
checks values/columns, exactly one explicit device copy, CPU behavior, and no
derivative evaluation for the `all` selector. Passed.

`cuda_boundary_probe.py` is self-contained: no exp01 import, no adapter, no
CPU fallback. It generates nominal, high-e and retrograde short Kerr waveforms
on both CPU and CUDA, then repeats nominal on each model. It records actual
source/native/table SHA256, versions and complete selected-mode lists, and
fails on nonfinite/incompatible outputs or error above 2e-11.
`cuda_boundary_regression.json`: all pass; relative L2 respectively
2.9865e-16, 7.3607e-16, 2.8782e-16, with identical selected modes.

```bash
python tests/test_cuda_trajectory_boundary.py -v
python collaboration/linux/cuda_boundary_probe.py --data-dir "$FEW_DATA" \
  --output collaboration/linux/replay.json
```

The older `boundary_fixed.json` long-waveform pilot used the exp01 benchmark
script in baseline-only mode, with no adapter; it is historical, not the
standalone reproduction entry point. Its saved short/year nominal outputs
are bitwise equal to the separate exp01 legacy build. Do not compare those
separate-process wall times as a controlled speedup result.

This is compatibility/regression repair, not GPU acceleration by itself.
No claim of full LISA/TDI, all-parameter-space or Mac acceptance is made.
