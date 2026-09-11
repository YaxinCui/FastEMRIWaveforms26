"""Bounded CPU sparse representation of the installed 1.1.17 GW-TDI operator."""

# 2026-09-11 21:14 CST (linux): Cache only waveform-independent TDI stencils.
# Keep stock projection, orbit interpolation and approximate delay convention.
# FP64 coefficient expansion/reduction reorders arithmetic: not bitwise exact.
import hashlib
import inspect

import numpy as np
from scipy.sparse import coo_matrix
from fastlisaresponse.response import pyResponseTDI
from lisatools.utils.utility import AET

EXPECTED_SOURCE = "7a40c9ffd992bf09b2e84d201ce5e9f5da0c2c19ea33efd821f7afa4d027af4a"


def interpolation_weights(x, order, a_table, deps, e_table):
    """Expand the native finite stencil; reject undefined table/end indices."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("Finite one-dimensional sample coordinates required")
    if order not in (15, 25, 35):
        raise NotImplementedError("Only audited odd interpolation orders")
    if np.any(np.abs(x) > 2**30):
        raise ValueError("Sample coordinates outside bounded integer range")
    d = np.ceil(x).astype(np.int64) - 1
    e = 1.0 + d - x  # preserve native arithmetic order
    ind = (e / deps).astype(np.int64)
    if np.any(e < 0) or np.any(e >= 1) or np.any(ind + 1 >= len(a_table)):
        raise ValueError("Undefined native fraction/table lookup")
    frac = (e - ind * deps) / deps
    a = a_table[ind] * (1 - frac) + a_table[ind + 1] * frac
    h = (order + 1) // 2
    indices = [d + 1, d]
    weights = [a * (1 - e), a * e]
    for j in range(1, h):
        scale = a * (e * (1 - e)) * e_table[j - 1]
        indices.extend((d + 1 + j, d - j))
        weights.extend((scale / (j + e), scale / (j + (1 - e))))
    return np.stack(indices, axis=1), np.stack(weights, axis=1)


class PreparedTDI:
    """Owned fixed-configuration matrix mapping six real link streams to A/E/T.

    No model or input waveform is retained. Prepare again for changed dt,
    orbit, topology, length/order or crop. Sky may change: it enters the STOCK
    projection before this operator. This is a script prototype, not public API.
    """

    def __init__(self, response, max_entries=10_000_000):
        if type(response) is not pyResponseTDI or response.xp is not np:
            raise NotImplementedError("Exact stock CPU response required")
        with open(inspect.getsourcefile(pyResponseTDI), "rb") as stream:
            if hashlib.sha256(stream.read()).hexdigest() != EXPECTED_SOURCE:
                raise RuntimeError("Unvalidated response source/ABI")
        if response.tdi != "2nd generation" or response.tdi_chan != "AET":
            raise NotImplementedError("Second-generation AET only")
        if response.order not in (15, 25, 35):
            raise NotImplementedError("Unaudited interpolation order")
        n = response.num_pts
        cut = getattr(response, "tdi_start_ind", None)
        if (
            not isinstance(n, int)
            or not isinstance(cut, int)
            or cut <= 0
            or 2 * cut >= n
        ):
            raise ValueError("Initialize stock projection/crop before preparation")
        if not np.isfinite(response.dt) or response.dt <= 0:
            raise ValueError("Invalid dt")
        self.num_pts, self.cut, self.nt = n, cut, n - 2 * cut
        units, starts, lengths = np.unique(
            response.tdi_operation_index, return_index=True, return_counts=True
        )
        if not np.array_equal(units, np.arange(len(units))):
            raise ValueError("Non-contiguous TDI units")
        estimated = len(units) * self.nt * (response.order + 1)
        if estimated > max_entries:
            raise MemoryError("Explicit bounded COO preparation budget exceeded")
        time = np.arange(n, dtype=np.float64)[cut:-cut] * response.dt
        # Native 1.1.17 was given RESPONSE orbits, not the separate tdi_orbits.
        orbit = response.response_orbits
        links = list(map(int, orbit.LINKS))
        if len(links) != 6 or len(set(links)) != 6:
            raise ValueError("Six unique links required")
        travel = {link: orbit.get_light_travel_times(time, link) for link in links}
        if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in travel.values()):
            raise ValueError("Invalid light travel time")
        rows, cols, values = [], [], []
        min_fraction, max_fraction = 1.0, 0.0
        for unit, start, length in zip(units, starts, lengths):
            delay = time.copy()
            for link in response.tdi_link_combinations[start : start + length]:
                if link != -11:
                    delay -= travel[int(link)]  # same ordered subtraction, all L(t)
            x = delay * response.sampling_frequency
            frac = np.ceil(x) - x
            min_fraction = min(min_fraction, float(frac.min()))
            max_fraction = max(max_fraction, float(frac.max()))
            ind, weight = interpolation_weights(
                x, response.order, response.A_in, response.deps, response.E_in
            )
            if np.any(ind < 0) or np.any(ind >= n):
                raise ValueError("TDI stencil outside input time domain")
            channel = int(response.channels[unit])
            if channel not in (0, 1, 2):
                raise ValueError("Invalid TDI channel")
            base = links.index(int(response.tdi_base_links[unit]))
            rows.append(
                np.broadcast_to(
                    (channel * self.nt + np.arange(self.nt))[:, None], ind.shape
                ).ravel()
            )
            cols.append((base * n + ind).ravel())
            values.append((float(response.tdi_signs[unit]) * weight).ravel())
        matrix = coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(3 * self.nt, 6 * n),
        ).tocsr()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Invalid sparse coefficients")
        for a in (matrix.data, matrix.indices, matrix.indptr):
            a.flags.writeable = False
        self._matrix = matrix
        self.metadata = dict(
            n=n,
            cut=cut,
            order=response.order,
            dt=response.dt,
            input_entries=estimated,
            nnz=matrix.nnz,
            bytes=sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr)),
            fraction_min=min_fraction,
            fraction_max=max_fraction,
            orbit_queries=len(travel),
            native_tdi_calls=0,
        )

    def xyz(self, y):
        y = np.asarray(y)
        if (
            y.dtype != np.float64
            or y.shape != (6, self.num_pts)
            or not np.all(np.isfinite(y))
        ):
            raise ValueError("Finite FP64 (6,N) link array required")
        result = self._matrix @ y.ravel()
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite sparse response")
        return result.reshape(3, self.nt)

    def __call__(self, y):
        return np.stack(AET(*self.xyz(y)))
