"""Read authenticated exp15 basis, profile stock installed CPU response only."""

# 2026-09-11 21:06 CST (linux): New independent diagnostic; preserve old sources,
# installed environment/orbits/H5 and write only a new Linux JSON report.
import cProfile
import hashlib
import inspect
import json
import os
from pathlib import Path
import pstats
from time import perf_counter
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits


def identity(path):
    path = Path(path).resolve()
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class MeteredResponse(pyResponseTDI):
    @property
    def response_gen(self):
        native = super().response_gen

        def invoke(*args):
            start = perf_counter()
            value = native(*args)
            self.native_projection_seconds += perf_counter() - start
            return value

        return invoke

    @property
    def tdi_gen(self):
        native = super().tdi_gen

        def invoke(*args):
            start = perf_counter()
            value = native(*args)
            self.native_tdi_seconds += perf_counter() - start
            return value

        return invoke


def main():
    here = Path(__file__).resolve().parent
    output = here / "profile_first.json"
    if output.exists():
        raise FileExistsError(output)
    old = here.parents[2] / "exp15-single-pass-phase-basis/collaboration/linux"
    report = json.loads((old / "real_prograde_final.json").read_text())
    archive = report["archive"]
    assert identity(archive["path"]) == archive
    assert identity(report["orbit"]["path"]) == report["orbit"]
    with np.load(archive["path"], allow_pickle=False) as arrays:
        basis = arrays["new_ssb_basis"].copy()
    dt = report["few"]["settings"]["dt"]
    orbit = ESAOrbits(force_backend="cpu")
    orbit.configure(linear_interp_setup=True)
    response = MeteredResponse(
        1 / dt,
        basis.shape[1],
        order=25,
        tdi="2nd generation",
        orbits=orbit,
        tdi_chan="AET",
        force_backend="cpu",
    )
    response.native_projection_seconds = 0.0
    response.native_tdi_seconds = 0.0
    rows = []

    def run():
        for wave in basis:
            before_p = response.native_projection_seconds
            before_t = response.native_tdi_seconds
            start = perf_counter()
            response.get_projections(
                np.ascontiguousarray(wave.conj()), 0.7, np.pi / 2 - 1.1, t0=20000.0
            )
            projection_end = perf_counter()
            channels = response.get_tdi_delays()
            tdi_end = perf_counter()
            cut = response.tdi_start_ind
            values = np.stack([x[cut:-cut].copy() for x in channels])
            freq = dt * np.fft.rfft(values, axis=-1)
            end = perf_counter()
            assert np.all(np.isfinite(freq))
            rows.append(
                dict(
                    projection=projection_end - start,
                    tdi=tdi_end - projection_end,
                    copy_fft=end - tdi_end,
                    total=end - start,
                    native_projection=response.native_projection_seconds - before_p,
                    native_tdi=response.native_tdi_seconds - before_t,
                )
            )

    run()  # separately discard warmup
    rows.clear()
    for _ in range(3):
        run()
    profiler = cProfile.Profile()
    profiler.runcall(run)
    stats = pstats.Stats(profiler)
    hot = []
    for key, values in sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)[
        :35
    ]:
        hot.append(
            dict(
                function=str(key),
                calls=values[1],
                self_seconds=values[2],
                cumulative_seconds=values[3],
            )
        )
    result = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        passes=True,
        producer=identity(__file__),
        input=archive,
        orbit=report["orbit"],
        sources=[
            identity(inspect.getsourcefile(x)) for x in (pyResponseTDI, ESAOrbits)
        ],
        environment={
            x: os.environ.get(x)
            for x in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        scope="CPU stage diagnostic, native property timing adds overhead; first 39 rows unprofiled, last 13 cProfile",
        rows=rows,
        hot=hot,
        mean_seconds={k: float(np.mean([r[k] for r in rows[:39]])) for k in rows[0]},
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(means=result["mean_seconds"], hot=hot[:12])), flush=True)


if __name__ == "__main__":
    main()
