# exp11 — sampling distribution/import-name collision

<!-- 2026-09-11 15:28 CST (linux): Preserve packaging evidence without claiming
working sampling or automatically repairing any installed environment. -->

- Branch: `codex/exp11-sampling-dependency`.
- Independent base: master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Production change: remove only `lisatools` from `sampling` extra in
  `pyproject.toml`, plus a timestamped explanation. Keep `eryn`,
  `fastlisaresponse`, `lisaanalysistools` and all other dependency constraints.
- Local/uncommitted, Linux-owned. No commit, push or host transfer.

## Why this is a real defect

An installable distribution name is not necessarily its Python import name.
The correct LISA analysis distribution is `lisaanalysistools`, which provides
the `lisatools` Python package. PyPI's distribution `lisatools==0.5.1` is an
unrelated Lifetime ISA portfolio tool. No custom index/URL suggesting another
meaning was found in the pinned FEW tracked files.

This is more than an unused dependency: official wheel inspection found both
archives provide `lisatools/__init__.py`, with DIFFERENT bytes. Installing both
risks file overwrite or removal conflicts. This audit establishes shared paths,
not that a particular install order was observed to corrupt an existing env.

Primary references, read 2026-09-11:

- [PyPA: distribution vs import package](https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/).
- [The unrelated lisatools 0.5.1 release](https://pypi.org/project/lisatools/0.5.1/).
- [LISA analysis tools 1.2.8](https://pypi.org/project/lisaanalysistools/1.2.8/).

## What was actually verified

1. Pure TOML regression fails before (`master_sampling_failure.json`) and passes
   after (`fixed_sampling_build.json`). It does not import FEW or start a backend.
2. Both master and candidate BARE metadata-only wheels were built. The candidate
   build command, full logs and current source hashes are in the build report.
   These wheels intentionally contain no `.so` and MUST NOT replace working FEW.
3. `packaging_final.json` is terminal and passes. All `Requires-Dist` entries are
   compared as a multiset: only the unrelated sampling requirement is removed.
   Remaining METADATA headers and description match; Version is separately
   reported (candidate has the build-date dirty suffix). The whole wheels are
   NOT byte-identical: version code and the added test also differ.
4. The two official dependency wheels were checked against fresh PyPI JSON
   hashes and inspected as ZIPs, never imported. The resolver's actual versions
   AND archive hashes must equal those namespace-audited wheels. Raw PyPI JSON,
   wheels, and pip reports are under sibling `artifacts/exp11-packaging-final`.
5. Both extras resolve with `pip --dry-run --ignore-installed --only-binary=:all:`
   on this Linux x86_64 / CPython 3.12 environment. Before: 55 distributions;
   after: 51. This reduction is this resolver snapshot, not a permanent count.
6. Child Python `sys.prefix` proves the requested `exp11-resolver-env` was used,
   rather than its base interpreter. Inherited pip/Python overrides are removed,
   pip config is `/dev/null`, and the index is explicitly public PyPI. Installed
   inventory before and after is exactly `pip==26.2.1`; no sampling packages
   were installed. Main FEW Python and the exp09 response env were not modified.

## Reproduce

Use `/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python` as `FEW_PYTHON`.
For the pure test, from this worktree:

```bash
export FEW_PYTHON=/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python
"$FEW_PYTHON" tests/test_sampling_dependencies.py -v
"$FEW_PYTHON" collaboration/linux/build_checks.py --output collaboration/linux/new_build.json --wheel-dir /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/artifacts/exp11-new-bare
```

Exact before/after wheel paths, interpreter, commands and artifact identities
are in `packaging_final.json`; rerun `packaging_probe.py` with these same
`--before-wheel`, `--after-wheel`, `--resolver-python` arguments and NEW
`--output` / `--artifact-dir` paths. Optional `--archive-cache` reuses downloaded
wheels only when they match the freshly fetched official metadata digest.

## Preserved failures and limitations

- `packaging_first.json` incorrectly resolves the venv Python symlink to the
  base interpreter. Its dry-run/archive evidence is real, but its intended
  venv-isolation claim is invalid; use final child-prefix/inventory proof.
- `packaging_audited.json` is a terminal failure due to an SSL EOF fetching a
  wheel. A new run used existing archives validated against fresh official
  hashes. No TLS verification was disabled. `packaging_verified.json` passes
  before a timestamp-comment correction; final report matches current source.
- BARE metadata and successful resolution do NOT establish import, ABI,
  trajectory, TDI, sampler or posterior correctness. The separate exp09 LAT
  CPU-wheel SIGILL finding is NOT fixed by removing this dependency.
- Installing the corrected extra will not necessarily uninstall an already
  installed erroneous distribution. No existing environment was repaired here;
  any recovery should use an isolated rebuild and import/ABI tests, not an
  unverified uninstall that could remove shared package files.
- Future complete sampling acceptance needs a normal FEW native wheel, a fresh
  installation, `pip check`, imports, and a minimal real sampling workflow.
  No timing or full workflow acceptance is claimed by this packaging experiment.
