"""Deterministic FP64 aggregation after selection of individual FEW modes."""

# 2026-09-11 12:04 CST (linux): Aggregate independent +/- branches on sparse
# nodes. Only mode topology is copied to the host; amplitudes stay on device.
import numpy as np


_CUDA_AGGREGATE = r'''
extern "C" __global__
void aggregate(double* out, const double2* amp, const double2* weights,
               const int* ell, const int* m, const int* order,
               const int* offsets, int nt, int nm, int ng) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nt * ng) return;
    int t = idx / ng, g = idx % ng;
    double pr = 0., pi = 0., mr = 0., mi = 0.;
    for (int q = offsets[g]; q < offsets[g + 1]; ++q) {
        int j = order[q];
        double2 a = amp[t * nm + j], p = weights[j];
        pr += a.x * p.x - a.y * p.y;
        pi += a.x * p.y + a.y * p.x;
        if (m[j] != 0) {
            double2 b = weights[nm + j];
            double parity = (ell[j] % 2 == 0) ? 1. : -1.;
            mr += parity * (a.x * b.x + a.y * b.y);
            mi += parity * (a.x * b.y - a.y * b.x);
        }
    }
    out[(0 * ng + g) * nt + t] = pr;
    out[(1 * ng + g) * nt + t] = pi;
    out[(2 * ng + g) * nt + t] = mr;
    out[(3 * ng + g) * nt + t] = mi;
}
'''


def aggregate_common_phase(amplitudes, ylms, ell, m, k, n, xp=np):
    """Return (4*groups, sparse_times) FP64 values and actual (m,k,n) keys.

    Channels are Re B+, Im B+, Re B-, Im B-. Mode order within each group
    is stable. No amplitude threshold, phase approximation or dtype reduction
    is introduced. Negative weights include (-1)**ell; m=0 has no extra branch.
    """
    a = xp.ascontiguousarray(amplitudes, dtype=xp.complex128)
    if a.ndim != 2:
        raise ValueError("amplitudes must have shape (sparse_times, modes)")
    nt, nm = a.shape
    weights = xp.ascontiguousarray(ylms, dtype=xp.complex128)
    if weights.shape != (2 * nm,):
        raise ValueError("ylms must contain two weights per selected mode")
    topology = []
    for values in (ell, m, k, n):
        values = xp.asarray(values)
        if values.shape != (nm,) or values.dtype.kind not in "iuf":
            raise ValueError("mode indices must be integer-valued arrays matching modes")
        host = values.get() if xp is not np else np.asarray(values)
        if not np.all(np.isfinite(host)) or np.any(host != np.floor(host)):
            raise ValueError("mode indices must be finite and exactly integer-valued")
        if np.any(host < np.iinfo(np.int32).min) or np.any(host > np.iinfo(np.int32).max):
            raise ValueError("mode index is outside int32 range")
        topology.append(host.astype(np.int32))
    ell_h, m_h, k_h, n_h = topology
    members = {}
    for j, key in enumerate(zip(m_h, k_h, n_h)):
        members.setdefault(key, []).append(j)
    ng = len(members)
    groups = np.asarray(list(members), dtype=np.int32).reshape(ng, 3)
    order = np.asarray([j for group in members.values() for j in group], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum([len(x) for x in members.values()]))).astype(np.int32)
    out = xp.empty((4 * ng, nt), dtype=xp.float64)
    if nt and ng:
        if xp is np:
            # The group member loop matches the CUDA reduction order and avoids
            # cumsum/subtraction cancellation or unordered floating atomics.
            for g, group in enumerate(members.values()):
                plus = np.zeros(nt, dtype=np.complex128)
                minus = np.zeros(nt, dtype=np.complex128)
                for j in group:
                    plus += a[:, j] * weights[j]
                    if m_h[j] != 0:
                        minus += a[:, j].conj() * weights[nm + j] * (-1.0 if ell_h[j] % 2 else 1.0)
                out[g], out[ng + g] = plus.real, plus.imag
                out[2 * ng + g], out[3 * ng + g] = minus.real, minus.imag
        else:
            kernel = xp.RawKernel(_CUDA_AGGREGATE, "aggregate", options=("--fmad=false",))
            args = (out, a, weights, xp.asarray(ell_h), xp.asarray(m_h),
                    xp.asarray(order), xp.asarray(offsets), np.int32(nt),
                    np.int32(nm), np.int32(ng))
            kernel(((nt * ng + 127) // 128,), (128,), args)
    return out, tuple(xp.asarray(groups[:, i]) for i in range(3))
