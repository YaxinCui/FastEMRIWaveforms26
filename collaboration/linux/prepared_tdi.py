"""Bounded CPU sparse representation of the installed 1.1.17 GW-TDI operator."""

# 2026-09-11 21:12 CST (linux): Cache only waveform-independent TDI stencils.
# Keep stock projection, orbit interpolation and approximate delay convention.
# FP64 coefficient expansion/reduction reorders arithmetic: not bitwise exact.
import hashlib
import inspect
import weakref

import numpy as np
from scipy.sparse import csr_matrix
from fastlisaresponse.response import pyResponseTDI
from lisatools.utils.utility import AET

EXPECTED_SOURCE = "7a40c9ffd992bf09b2e84d201ce5e9f5da0c2c19ea33efd821f7afa4d027af4a"
_BINDINGS = weakref.WeakKeyDictionary()
_CONFIG = ("num_pts", "dt", "sampling_frequency", "order", "tdi", "tdi_chan", "deps")
_ARRAYS = (
    "tdi_operation_index",
    "tdi_base_links",
    "tdi_link_combinations",
    "tdi_signs",
    "channels",
    "A_in",
    "E_in",
)


def _signature(values):
    return tuple(
        (id(v), v.shape, v.dtype.str, v.strides) if isinstance(v, np.ndarray) else v
        for v in values
    )


def _immutable(array):
    """A bytes owner makes NumPy write-enable impossible, unlike flags alone."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _FixedResponse(pyResponseTDI):
    # 2026-09-11 21:24 CST (linux): Preserve registered/hash-every-plan evidence.
    # Freeze geometry BEFORE its first native injection; native/query share the
    # same immutable bytes. No installed source or interpolation code changes.
    def check_add_orbit_args(self, *args):
        checked = super().check_add_orbit_args(*args)
        immutable = tuple(
            _immutable(a) if isinstance(a, np.ndarray) else a for a in checked
        )
        self.response_orbits.pycppdetector_args = immutable
        return immutable


def make_response(*args, **kwargs):
    """Construct/register a stock response, freezing its original native inputs.

    Only this controlled factory is supported; arbitrary existing objects can
    have Python/native orbit state diverge. Direct C++ rebinding is unsupported.
    """
    response = _FixedResponse(*args, **kwargs)
    if response.xp is not np:
        raise NotImplementedError("CPU only")
    orbit_args = tuple(response.response_orbits.pycppdetector_args)
    for key in _ARRAYS:
        setattr(response, key, _immutable(getattr(response, key)))
    arrays = tuple(getattr(response, k) for k in _ARRAYS)
    _BINDINGS[response] = (
        response.cpp_response,
        response.response_orbits,
        orbit_args,
        tuple(getattr(response, k) for k in _CONFIG),
        arrays,
        _signature((*orbit_args, *arrays)),
    )
    return response


def interpolation_weights(x, order, a_table, deps, e_table):
    """Expand the native finite stencil; reject undefined table/end indices."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Finite one-dimensional sample coordinates required")
    if order not in (15, 25, 35):
        raise NotImplementedError("Only audited odd interpolation orders")
    if np.any(np.abs(x) > 2**30):
        raise ValueError("Sample coordinates outside bounded integer range")
    if (
        not np.isfinite(deps)
        or deps <= 0
        or np.asarray(a_table).ndim != 1
        or len(a_table) < 2
        or np.asarray(e_table).ndim != 1
        or len(e_table) < (order + 1) // 2 - 1
        or not np.all(np.isfinite(a_table))
        or not np.all(np.isfinite(e_table))
    ):
        raise ValueError("Invalid interpolation tables")
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
        if type(response) is not _FixedResponse or response.xp is not np:
            raise NotImplementedError("Registered immutable CPU response required")
        # 2026-09-11 21:15 CST (linux): Bind preparation to factory-time native
        # geometry and reject later Python orbit/config/table replacement.
        binding = _BINDINGS.get(response)
        if binding is None:
            raise ValueError(
                "Use make_response; arbitrary mutable native state is unsupported"
            )
        native, orbit, orbit_args, config, arrays, signature = binding
        if (
            response.cpp_response is not native
            or response.response_orbits is not orbit
            or tuple(getattr(response, k) for k in _CONFIG) != config
            or any(
                getattr(response, k) is not a or a.flags.writeable
                for k, a in zip(_ARRAYS, arrays)
            )
            or len(orbit.pycppdetector_args) != len(orbit_args)
            or any(
                (current is not old or old.flags.writeable)
                if isinstance(old, np.ndarray)
                else current != old
                for current, old in zip(orbit.pycppdetector_args, orbit_args)
            )
            or _signature((*orbit_args, *arrays)) != signature
        ):
            raise ValueError(
                "Response configuration/native geometry changed; construct a new response"
            )
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
        if (
            not np.array_equal(units, np.arange(len(units)))
            or not np.all(np.isin(np.diff(response.tdi_operation_index), (0, 1)))
            or len(response.tdi_link_combinations) != len(response.tdi_operation_index)
            or any(
                len(x) != len(units)
                for x in (
                    response.channels,
                    response.tdi_base_links,
                    response.tdi_signs,
                )
            )
        ):
            raise ValueError("Non-contiguous TDI units")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Positive integer preparation budget required")
        estimated = len(units) * self.nt * (response.order + 1)
        if estimated > max_entries:
            raise MemoryError("Explicit bounded COO preparation budget exceeded")
        time = np.arange(n, dtype=np.float64)[cut:-cut] * response.dt
        # Native 1.1.17 was given RESPONSE orbits, not the separate tdi_orbits.
        links = list(map(int, orbit_args[5]))
        if len(links) != 6 or len(set(links)) != 6:
            raise ValueError("Six unique links required")
        query = orbit.backend.OrbitsWrap(*orbit_args)
        travel = {}
        for link in links:
            value = np.empty_like(time)
            query.get_light_travel_time_wrap(
                value, time, np.full(time.size, link, dtype=np.int32), time.size
            )
            travel[link] = value
        if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in travel.values()):
            raise ValueError("Invalid light travel time")
        # 2026-09-11 21:12 CST (linux): Preserve the original COO pilot in
        # pilot_sources. Assemble overlapping narrow time bands directly in
        # native unit order, avoiding a large duplicate-coordinate sort.
        groups = {}
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
            channel = int(response.channels[unit])
            if channel not in (0, 1, 2):
                raise ValueError("Invalid TDI channel")
            base = links.index(int(response.tdi_base_links[unit]))
            groups.setdefault((channel, base), []).append(
                (x, float(response.tdi_signs[unit]))
            )
        index = np.arange(cut, n - cut)
        tap_rows = np.arange(self.nt)[:, None]
        data, columns, pointers = [], [], []
        total = 0
        h = (response.order + 1) // 2
        for channel in range(3):
            bands, band_columns = [], []
            for (ch, base), entries in sorted(groups.items()):
                if ch != channel:
                    continue
                lower = min(
                    int((np.ceil(x).astype(np.int64) - 1 - index).min()) - h + 1
                    for x, _ in entries
                )
                upper = max(
                    int((np.ceil(x).astype(np.int64) - 1 - index).max()) + h
                    for x, _ in entries
                )
                band = np.zeros((self.nt, upper - lower + 1))
                band_index = index[:, None] + np.arange(lower, upper + 1)
                if np.any(band_index < 0) or np.any(band_index >= n):
                    raise ValueError("TDI stencil outside input time domain")
                for x, sign in entries:
                    ind, weight = interpolation_weights(
                        x, response.order, response.A_in, response.deps, response.E_in
                    )
                    # Every index in a unit's row is unique, so advanced += is
                    # safe here; overlap between units is accumulated serially.
                    band[tap_rows, ind - index[:, None] - lower] += sign * weight
                bands.append(band)
                band_columns.append((base * n + band_index).astype(np.int32))
            if not bands:
                raise ValueError("Empty TDI output channel")
            channel_data = np.concatenate(bands, axis=1)
            data.append(channel_data.ravel())
            columns.append(np.concatenate(band_columns, axis=1).ravel())
            pointers.append(
                total + np.arange(self.nt, dtype=np.int64) * channel_data.shape[1]
            )
            total += channel_data.size
        matrix = csr_matrix(
            (
                np.concatenate(data),
                np.concatenate(columns),
                np.concatenate(pointers + [np.array([total])]),
            ),
            shape=(3 * self.nt, 6 * n),
        )
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
