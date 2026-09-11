# exp12 — spin-local FP64 Kerr amplitude residency

<!-- 2026-09-11 16:13 CST (linux): Root is sole editor/GPU runner; Mac and
research agents are read-only. Record bounded evidence, not a host transfer. -->

- Branch: `codex/exp12-spin-local-amplitudes`.
- Independent base: master `47e4fea4bb3e5fbe1ba34e5e399ed37c97814191`.
- Status: LOCAL/UNCOMMITTED. No merge, commit, push or host switch.
- Production: opt-in additions to `src/few/amplitude/ampinterp2d.py`, plus
  `src/few/amplitude/_spincache.py`. No native or data-file changes.
- This follows the storage-layer proposal; it is NOT a mixed-precision or
  higher-fidelity interpolation implementation.

## Contract and implementation

Pass `amplitude_kwargs={"spin_cache_size": 2}` to
`FastKerrEccentricEquatorialFlux`, or `spin_cache_size=2` directly to
`AmpInterpKerrEccEq`. Default `None` retains the master's eager loading path.
The cache capacity is the number of resident spin planes **per region**, not
megabytes or a global CUDA memory-pool limit.

The existing model uses a single constant signed spin in each call. Its spin
interpolation evaluates either one exact grid node or two adjacent nodes;
each visited region needs only those nodes. Region A/B knots and metadata are
read at construction, but coefficient payloads are read on misses only with
HDF5 `[physical_spin_index, ...]` slicing. FP64 coefficients, mode selection,
coordinate mapping, native bicubic interpolation and final spin interpolation
arithmetic remain unchanged. Existing `downsample_Z` chooses the same original
physical node indices; an excluded last node is not silently restored.

An instance-owned LRU preserves needed resident nodes before upper-first pair
evaluation, avoiding eviction and immediate reload of the shared node. Arrays
own their contiguous payloads; no NumPy slice keeps an entire region alive.
No persistent HDF5 handle or parent-reference cycle is retained. File metadata
and both regions bind to one stat identity; ordinary replacement/modification
is rejected on a miss. The file must remain immutable for the object's life;
stat identity is not a cryptographic adversarial-tampering defense.

CUDA objects must be called on their construction device. Lazy nondefault
streams explicitly bridge the unchanged native null-stream launcher and copy
the result into the caller's allocation arena before return. This is a safety
bridge, not an asynchronous concurrency optimization. Objects are not designed
for simultaneous calls from multiple threads/streams.

`clear_spin_cache()` releases instance-held slices, not the shared CuPy pool.
`spin_cache_info()` reports logical coefficient bytes and cumulative loads,
hits, evictions and successful-load bytes. A new slice is constructed before
eviction, so transient residency can exceed capacity by one plane per loading
region, plus host staging, gather/flatten temporaries and other FEW arrays.

## Numerical evidence

All reports below require `passes=true`, `terminal=true` and authenticated
applicable identities: source/native for tiny generated fixtures, additionally
real data for actual-table probes. Mere file existence is not evidence.

- `spin_cpu_stream_hardened.json` and `spin_cuda_stream_hardened.json`: eight
  test methods each pass. Tiny generated H5 coefficients represent an analytic
  complex bilinear polynomial through Greville-node coefficients, evaluated by
  actual compiled CPU/CUDA interpolants. Coverage includes exact/interior spin,
  A/B/mixed regions, downsampling, duplicate/subset modes, LRU switching,
  parent/array release, failed loads, metadata safety, device guards, and
  CUDA delayed device consumers under same-sized null-stream pool pressure.
- `real_master_cpu_replay.json`, `real_eager_cpu.json`, `real_lazy_cpu.json`:
  39 arrays each, eager/lazy byte-identical to the same-backend master archive.
- `real_master_cuda.json`, `real_eager_cuda.json`, `real_lazy_cuda.json`: the
  same 39-array gates pass on CUDA. All variants use the SAME explicit host-time
  derivative adapter for a pre-existing master frontend CPU/CuPy boundary.
  This is not an unmodified-master CUDA frontend certificate.
- Each real probe has nine source-waveform calls plus their mode-index arrays,
  and seven amplitude cases through all/tuple/raw selection (21 arrays).
  The nine calls include repeats: five distinct intrinsic parameter sets, each
  `T=0.003 year`, `dt=15 s`, 6312 samples. No long-inspiral/TDI/likelihood gate.
- The real A/B boundary fixture uses adjacent representable values around
  `p_sep+9.001` and enters `[A,A,B]`. This proves original dispatch/output
  preservation, not interpolation continuity or a full orbit crossing.
- Positive extreme spin `a=+0.999` is tested only in region B; negative extreme
  spin reaches both regions. Tiny analytic tests cover both regions' endpoints,
  but real-table A-region positive-extreme coverage remains open.
- Direct retrograde all/tuple/raw outputs are each compared to the SAME master
  entry; they are not treated as mutually equivalent physical oracles. The
  public convention mismatch is a separate proposed independent experiment.

Data: existing shared `ZNAmps_l10_m10_n55_DS2Outer.h5`, 5,089,095,248 bytes,
SHA256 `3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`.
No table payload was edited, duplicated or downloaded.

## Performance protocol and current status

`spin_benchmark.py` runs fresh processes in alternating master/lazy, lazy/master
order, seven pairs per backend. CPU and CUDA performance suites run serially,
not concurrently. Each worker gets a full SHA256 file pass BEFORE launch to
explicitly warm the OS page cache; this is not cold-disk evidence. FEW's ONCE
integrity checks remain enabled inside every fresh process.

The measured interval is backend-initialized **model construction through first
source waveform**, including first-use work inside that interval. FEW/native
imports and CUDA context initialization precede it. It is not process-start to
first-result latency. An optional `--verified-first` separately times prior
file-manager validation; default measurements do not hide that cost. Joint
intervals are measured directly, not by adding component medians. Fixed-repeat,
near-spin, far-spin, retrograde and return requests expose hits and reloads.

Recorded memory is current RSS/PSS, cumulative VmHWM, CuPy active/reserved bytes
and driver memory snapshots AFTER calls, not continuously sampled peaks.
Post-GC and pool-trim snapshots are separate diagnostics outside timing; local
trajectory/output references can remain, so they do not prove all FEW objects
are gone. GPU clocks/temperature are observed, not locked. No universal speed,
5x, Mac or energy claim follows from these pilots.

Both seven-pair suites are now terminal and pass: `benchmark_cpu.json` and
`benchmark_cuda.json`, 14 fresh processes each, 28 total. Source and benchmark
scripts remained unchanged throughout. `performance_summary.json` is derived
from their authenticated worker reports. The preceding real-regression runs
were not isolated performance tests and their timings are not used here.

<!-- 2026-09-11 16:29 CST (linux): Terminal measured results, including
negative reload timings. Logical coefficient size is not total or peak memory. -->

| Metric, median | Master eager | Lazy capacity=2 | Interpretation |
| --- | ---: | ---: | --- |
| CPU construction through first waveform | 20.1047 s | 19.3403 s | 1.0395x ratio of medians |
| CUDA construction through first waveform | 22.4561 s | 21.3659 s | 1.0510x ratio of medians |
| CPU first-result RSS | 6045.85 MiB | 1426.14 MiB | 76.41% lower current process RSS |
| CUDA first-result CuPy used | 4853.94 MiB | 233.98 MiB | First request visits only A |
| CUDA after A/B and return, CuPy used | 4853.94 MiB | 295.65 MiB | 93.91% lower active allocation |
| CUDA after A/B and return, CuPy reserved | 5202.91 MiB | 582.96 MiB | 88.80% lower pool reservation, not total GPU memory |
| CPU repeat2 | 108.06 ms | 108.74 ms | Approximately unchanged |
| CUDA repeat2 | 16.95 ms | 17.17 ms | Approximately unchanged |
| CPU retrograde reload | 108.34 ms | 147.67 ms | Slower: new A bracket |
| CUDA retrograde reload | 17.53 ms | 88.45 ms | About 5.04x LONGER latency |
| CUDA return-to-original reload | 18.18 ms | 89.65 ms | About 4.93x LONGER latency |

Joint paired-ratio ranges are CPU 1.0333–1.0469x and CUDA 1.0385–1.0609x;
paired medians are 1.0427x and 1.0525x. These are seven-pair pilot ranges,
not confidence bounds or domain-wide guarantees. Use the directly measured
joint interval; do not sum independently reported component medians.

Actual first request loads A physical indices `[18,17]`, no B. Repeats and
near-spin `.61` reuse that bracket. Far-spin `.9` visits B `[11,10]` (changing
both spin and region, not a pure A-only bracket experiment). Retrograde loads
A `[30,29]`; returning to `.6` reloads `[18,17]`. Final logical payload is
308,363,328 bytes = 294.078 MiB, plus other active allocations. The full eager
coefficient payload is 4852.290 MiB. Data on disk still occupies 5.09 GB.

The measured conclusion is **substantial memory savings, modest fresh-model
latency savings, and potentially severe cache-miss latency penalties**. ONCE
integrity checking still scans the full file, and the first waveform includes
first-use work. We did not profile those first-use subcomponents or run the
optional `--verified-first` diagnostic; no new split-cost claim is made.
Do not replace full-file validation with unsafe skipping to improve a benchmark.

`audit_final.json` passes 4278 checks over 91 unique authenticated files:
source/native/data identity, preserved failed pilot, actual finite arrays and
their master bytes, cache accounting, serial AB/BA records, and recomputed
stage/call medians and ratios. The current production change remains opt-in.

## Reproduction

Use the existing main Python with this worktree's source overlay and the
independently built master native runtime, NOT the older installed FEW:

```bash
# 2026-09-11 16:13 CST (linux): Scope environment to the selected experiment.
cd /home/ubuntu/Desktop/FastEMRIWaveforms-experiments/exp12-spin-local-amplitudes
export FEW_EXPERIMENTS=/home/ubuntu/Desktop/FastEMRIWaveforms-experiments
export FEW_DATA=/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download
export PYTHONPATH="$PWD/src:$FEW_EXPERIMENTS/artifacts/runtime-master"
export LD_LIBRARY_PATH=/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python collaboration/linux/spin_checks.py \
  --backend cuda12x --output collaboration/linux/spin_cuda_new.json
/home/ubuntu/Desktop/FastEMRIWaveforms/.venv/bin/python collaboration/linux/spin_benchmark.py \
  --backend cuda12x --pairs 7 --data-dir "$FEW_DATA" \
  --output collaboration/linux/benchmark_cuda_new.json \
  --artifact-dir "$FEW_EXPERIMENTS/artifacts/exp12-benchmark-cuda-new"
```

Real regression: run `spin_real_probe.py --backend cpu --policy master` with
`master-reference/src` overlay and a NEW `--output`, then eager/lazy with this
worktree overlay and `--reference` pointing to that master NPZ. Supply the same
`--data-dir` to each. CUDA is analogous. Output overwrite is rejected.

## Preserved failures and boundaries

The first LRU test exposed four loads where an adjacent bracket only needs
three; `prepare()` fixed eviction of a needed shared node. The first real CPU
pilot (`real_master_cpu.json`, failed/terminal) used p=8 inside a retrograde
separatrix; formal replays use valid p=12/14/40. No failed pilot is counted as
passing. The first cross-stream fixture fetched before memory pressure; the
new hardened fixture fixes that weakness without changing production code.

Focused Ruff E9/F passes for new files and whitespace checks pass. The existing
unrelated Schwarzschild class has a master-preexisting F841 unused `file_path`;
this experiment does not repair it or claim a globally clean lint suite.

Full eager loading remains a reasonable policy when memory is ample and future
spins vary widely: it pays initialization once and avoids later disk/transfer
misses. This experiment offers a workload-dependent alternative, not proof the
authors' original loading policy was accidental or universally inferior.

## References

- [FEW Kerr paper, III.2.2 and Appendix B](https://arxiv.org/html/2506.09470v1):
  existing bicubic-plus-linear representation and spin/region grids. Local
  code/data, not this paper alone, establish the actual loading opportunity.
- [h5py dataset slicing](https://docs.h5py.org/en/stable/high/dataset.html):
  hyperslab reads versus `[()]` full reads; contiguous/chunked layout distinction.
- [CuPy memory management](https://docs.cupy.dev/en/stable/user_guide/memory.html):
  active versus reserved memory and why deleting arrays does not flush the pool.
- Local source: `ampinterp2d.py`, `_spincache.py`, `spin_real_probe.py`,
  `spin_benchmark.py`, `tests/test_spin_cache.py` and their report manifests.
