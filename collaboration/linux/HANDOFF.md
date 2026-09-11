# exp03 — Kerr amplitude reference coverage and provenance

<!-- 2026-09-11 13:11 CST (linux): Preserve the failing old reference and the
independently retrieved author-release evidence. No physical oracle invented. -->

- Branch: `codex/exp03-kerr-reference-coverage`; independent master base
  `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Linux root owns editing; local/uncommitted, no host switch.
- Production FEW is unchanged. Only tests and Linux evidence are added/edited.

## The actual defect and finding

The old five-point test asserted outside its loop, so only the final point
was checked. Each reference now has its own `subTest`, preserving constants
and `rtol=0, atol=1e-9`. Four pass; index 2 fails with absolute error
0.4079364550559394. The test intentionally remains red; no xfail, relaxed
tolerance or current-output replacement was used.

The disputed `(ell,m,k,n)=(2,2,0,0)` fixture is near the grid point:

```text
a = 0.9909069337322672
p = 1.817083116219875
e = 0.2531312656783681
```

The author's [paper data release](https://zenodo.org/records/15631641) links
to `KerrEccentricEquatorialFigures` v1.0, immutable revision
`7edb856bae835d4540e7aa18b9229d6ae41aceef`.
Its [A220 slice](https://raw.githubusercontent.com/cchapmanbird/KerrEccentricEquatorialFigures/7edb856bae835d4540e7aa18b9229d6ae41aceef/scripts/Validation/Amplitude/BicubicTricubicComparison/A220_slice.h5)
is 2,879,056 bytes, SHA256
`37b190b52013e432099e4fc14318755a1c2fa87a231ab393cb1e6641175f563f`.
It was retrieved into memory, not copied into the runtime table.

`paramsA[10,17,3]` contains the full-precision coordinates above, and
`modeA[10,17,3]` contains `-0.05104331491159769 - 0.4048482662967619j`.
The [associated interpolation script](https://github.com/cchapmanbird/KerrEccentricEquatorialFigures/blob/7edb856bae835d4540e7aa18b9229d6ae41aceef/scripts/Validation/Amplitude/BicubicTricubicComparison/generate_data.py)
establishes the `(u,w,z)` axis order and A220 mode identity.

`reference_audit_published.json` records:

- Current FEW at full coordinates vs published raw node: 1.2135e-14 absolute.
- Independent SciPy evaluation of the local coefficient table vs raw node:
  5.55e-17 absolute.
- Old test constant vs published node: 0.4079363255271417 absolute.

Inference: the old fixture conflicts with the author's released data; this is
not evidence that a newly added optimization corrupted the table. Two author
artifacts can share ancestry, however, so agreement is not independent
Teukolsky-solver validation. The original solver version/tolerance and reason
for the old constant remain unknown. Eight-decimal coordinates alone cause
about 1.30e-7 amplitude change here, already larger than the 1e-9 test tolerance.

The old constant appeared in upstream commit
`e09ca8f53b797e17d2f9fef87515831a9a46e20e` (2025-03-09), with the assertion
already outside the loop. Do not silently rewrite this history.

## Reproduce

Use this branch's test source with the pinned master CPU package and the
existing hash-verified 5 GB runtime amplitude table:

```bash
FEW_ENABLED_BACKENDS=cpu python -m unittest discover -s tests -p test_amplitudes.py -v
python collaboration/linux/reference_audit.py --data-dir "$FEW_DATA" \
  --output collaboration/linux/replay.json --published-raw
```

Set `FEW_FILE_EXTRA_PATHS=$FEW_DATA` and `FEW_FILE_ALLOW_DOWNLOAD=false`
for the unittest command. Both commands are EXPECTED to exit 1 on the retained
old fixture; the audit still writes all five outcomes and the passing published
node comparison. The network option downloads only the pinned 2.88 MB slice.
Next decision: a separately documented reference-data correction, or upstream
clarification of the original constant—not merely making the test green.
