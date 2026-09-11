"""Validate signed mode coordinates before backend-specific array indexing."""

# 2026-09-11 15:00 CST (linux): Physical/truncation domains are not NumPy's
# negative-index domain or CuPy's wrapping advanced-index domain.
import numpy as np


def integer_array(values, xp, ndim, label):
    # 2026-09-11 15:18 CST (linux): Reject bool before mixed Python sequences
    # silently coerce it to an integer. Already-integer arrays carry no such tag.
    def contains_bool(value):
        if isinstance(value, (list, tuple)):
            return any(contains_bool(item) for item in value)
        return (
            isinstance(value, (bool, np.bool_))
            or getattr(getattr(value, "dtype", None), "kind", None) == "b"
        )

    if contains_bool(values):
        raise ValueError(f"{label} must not contain boolean indices")
    try:
        raw = xp.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer array") from exc
    if raw.ndim != ndim or raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a nonempty {ndim}D integer array")
    if raw.dtype.kind == "u" and bool(xp.any(raw > np.iinfo(np.int64).max)):
        raise ValueError(f"{label} exceeds the integer index range")
    return raw.astype(xp.int64, copy=True)


def mode_tuples(values, shape, xp, *, unique=False):
    """Return owned (N,4) signed indices for the existing signed-axis layout.

    Infer supported bounds from the map shape, so adapters need no new lmax/
    kmax/nmax metadata. Negative m/k/n are legal coordinates, never raw IDs.
    """
    if (
        len(shape) != 4
        or any(size < 1 for size in shape)
        or any(size % 2 != 1 for size in shape[1:])
    ):
        raise ValueError("Mode map must have four axes and odd signed-axis lengths")
    modes = integer_array(values, xp, 2, "Mode coordinates")
    if modes.shape[1] != 4:
        raise ValueError("Each mode must contain exactly (ell,m,k,n)")
    ell, m, k, n = modes.T
    lmax, mmax, kmax, nmax = (
        shape[0] - 1,
        (shape[1] - 1) // 2,
        (shape[2] - 1) // 2,
        (shape[3] - 1) // 2,
    )
    invalid = (
        (ell < 2)
        | (ell > lmax)
        | (m < -ell)
        | (m > ell)
        | (m < -mmax)
        | (m > mmax)
        | (k < -kmax)
        | (k > kmax)
        | (n < -nmax)
        | (n > nmax)
    )
    if bool(xp.any(invalid)):
        raise ValueError(
            "Requested mode is unphysical or outside the model's covered mode domain"
        )
    if unique and xp.unique(modes, axis=0).shape[0] != modes.shape[0]:
        raise ValueError("Mode selection has duplicate modes")
    return modes


def flat_indices(values, limit, xp):
    """Raw full-mode IDs: reject sentinels and wrapping indices before gather."""
    indices = integer_array(values, xp, 1, "Mode IDs")
    if bool(xp.any((indices < 0) | (indices >= limit))):
        raise ValueError("Mode ID is absent or outside the stored mode catalogue")
    return indices
