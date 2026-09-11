# exp02 — reusable FP64 DOP853 stage workspace

<!-- 2026-09-11 13:11 CST (linux): Record a small/low-yield optimization without
turning microbenchmark gains into end-to-end acceleration claims. -->

- Branch: `codex/exp02-trajectory-workspace`; independent base
  `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191` (`master`).
- Linux root owns editing. Local, uncommitted; no Mac edits or host switch.
- Production change: `src/few/trajectory/dopr853.py` only.

`reuse_stage_workspace=False` preserves allocation behavior by default.
Opt-in reuses a float64 `(10,nODE,numSys)` allocation and its stage views;
shape changes reallocate. Every new step clears the workspace so partial-write
RHS callbacks retain their original zero-initialized contract. Dense-output
scratch is not cleared until the next step; controller history is untouched.

`tests/test_trajectory_workspace.py` passed 3 tests covering shape changes,
analytic fixed/adaptive ODEs, rejected steps, poisoned workspace, partial-write
RHS, bitwise legacy comparison and retained dense-coefficient lifetimes.

`workspace_probe.json` contains the first 15-pair alternating run:
synthetic 1000 steps 1.030x; real nominal/high-e/retrograde 1.0088x/1.0056x/1.0001x.
The stricter repeat `workspace_probe_strict.json` records parameters, script
hash and a nonzero-exit consistency gate; 1.0144x synthetic and
1.0141x/1.0149x/1.0016x real. All returned arrays and dense coefficients match
bitwise in both runs. Differences this small do not justify a major speed claim.
No accepted/rejected-step instrumentation or confidence interval is claimed.

Run after building/installing this source, or overlay its `src` on the clean
master native wheel (there are no native modifications in this experiment):

```bash
python tests/test_trajectory_workspace.py -v
python collaboration/linux/workspace_probe.py --data-dir "$FEW_DATA" \
  --output collaboration/linux/replay.json
```

Keep as a low-priority allocation optimization, not the principal route to 5x.
Separately identified batch-controller index/state issues must receive their
own master-based experiment instead of being silently added here.

<!-- 2026-09-11 13:23 CST (linux): Strengthen the equality label to byte equality
and retain the third timing run including its slightly negative result. -->

`workspace_probe_bytes.json` is the final fail-closed replay comparing shape,
dtype and raw bytes, including signed zeros. All outputs match bitwise. Its
speedups are 1.0295x synthetic, 1.0114x nominal, 1.0098x high-e and **0.9970x
retrograde**. Across all three runs the real effect ranges from about -0.3% to
+1.5%, reinforcing the low-yield/noise-sensitive conclusion. No earlier report
was deleted or overwritten to select the best timing.
