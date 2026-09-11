"""Experimental FP64 sufficient statistics for a fixed-shape distance scan."""

# 2026-09-11 14:16 CST (linux): Preserve the full fixed Gaussian likelihood;
# no mode pruning or precision reduction. Physical dependencies must be fixed.
from dataclasses import dataclass

import numpy as np


def _real_scalar(value, name):
    array = np.asarray(value)
    if array.ndim or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real scalar")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        value = float(array)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite in FP64")
    if array != 0 and value == 0:
        raise FloatingPointError(f"{name} underflowed during FP64 conversion")
    return value


def _positive_scalar(value, name):
    result = _real_scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _signal_array(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiuc":
        raise ValueError("Signal arrays must be numeric, not bool/string/object")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        return np.asarray(raw, dtype=np.complex128)


def weighted_inner(first, second, weights):
    """Real Gaussian inner product of pre-transformed channel/bin arrays.

    Caller supplies the complete fixed quadrature/noise weights. This does not
    infer FFT, PSD, channel independence or physical detector conventions.
    """
    first = _signal_array(first)
    second = _signal_array(second)
    raw_weights = np.asarray(weights)
    if (
        not first.size
        or first.shape != second.shape
        or raw_weights.dtype.kind not in "fiu"
    ):
        raise ValueError("Require equal nonempty signal shapes and real weights")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        weights = np.broadcast_to(
            np.asarray(raw_weights, dtype=np.float64), first.shape
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Signal arrays must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    # 2026-09-11 14:29 CST (linux): Fail closed on unsupported extreme dynamic
    # ranges rather than silently erasing nonzero weighted contributions.
    with np.errstate(over="raise", under="raise", invalid="raise"):
        root_weight = np.sqrt(weights)
        a = first * root_weight
        b = second * root_weight
        value = np.sum(a.real * b.real + a.imag * b.imag, dtype=np.float64)
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite inner product")
    return float(value)


@dataclass(frozen=True)
class DistanceConditionalLikelihood:
    """Delta logL(D) relative to D0, for s(D)=(D0/D)*s0.

    Fix detector-frame masses, sky/orientation, response, sampling, data,
    windows, PSD/covariance and all waveform-shape parameters. This is neither
    a distance prior nor marginalization, and stores no waveform-sized arrays.
    """

    distance0: float
    residual_projection: float
    signal_norm2: float

    def __post_init__(self):
        object.__setattr__(
            self, "distance0", _positive_scalar(self.distance0, "Reference distance")
        )
        object.__setattr__(
            self,
            "residual_projection",
            _real_scalar(self.residual_projection, "Residual projection"),
        )
        object.__setattr__(
            self, "signal_norm2", _real_scalar(self.signal_norm2, "Signal norm")
        )
        if self.signal_norm2 < 0:
            raise ValueError(
                "Coefficients must be finite; signal norm must be nonnegative"
            )
        if self.signal_norm2 == 0 and self.residual_projection != 0:
            raise ValueError("Zero signal norm requires zero projection")

    @classmethod
    def from_data(cls, distance0, data, reference_signal, weights):
        distance0 = _positive_scalar(distance0, "Reference distance")
        data = _signal_array(data)
        reference_signal = _signal_array(reference_signal)
        if data.shape != reference_signal.shape:
            raise ValueError("Data and reference must have identical shapes")
        # Form the residual first; subtracting two large global inner products
        # would lose accuracy near the best fit.
        with np.errstate(over="raise", invalid="raise"):
            residual = data - reference_signal
        a = weighted_inner(residual, reference_signal, weights)
        b = weighted_inner(reference_signal, reference_signal, weights)
        if b == 0 and np.any(
            (np.broadcast_to(weights, reference_signal.shape) > 0)
            & (reference_signal != 0)
        ):
            raise FloatingPointError("Nonzero weighted signal norm underflowed")
        return cls(distance0, a, b)

    def __call__(self, distance):
        raw = np.asarray(distance)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Distances must be real")
        distance = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(distance)) or np.any(distance <= 0):
            raise ValueError("Distances must be finite and positive")
        if self.signal_norm2 == 0:
            result = np.zeros_like(distance)
            return float(result) if result.ndim == 0 else result
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            delta = (self.distance0 - distance) / distance
            result = delta * (
                self.residual_projection - 0.5 * delta * self.signal_norm2
            )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Distance likelihood overflow")
        return float(result) if result.ndim == 0 else result


def one_sided_weights(psd, dt, n_time):
    """Quadrature weights for dt*rfft(real time series), including endpoints.

    Each supplied PSD is one-sided; channel covariance must be handled before
    use or supplied as an explicitly independent-channel model. +inf PSD gives
    zero weight. For odd lengths the last bin is NOT Nyquist.
    """
    dt = _positive_scalar(dt, "Time step")
    if (
        isinstance(n_time, (bool, np.bool_))
        or not isinstance(n_time, (int, np.integer))
        or n_time < 2
    ):
        raise ValueError("Time series length must be an integer >=2")
    raw = np.asarray(psd)
    if raw.dtype.kind not in "fiu" or raw.ndim < 1 or raw.shape[-1] != n_time // 2 + 1:
        raise ValueError("PSD must be real with one-sided FFT bin count")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        psd = np.asarray(raw, dtype=np.float64)
    if np.any(np.isnan(psd)) or np.any(psd <= 0):
        raise ValueError("PSD must be positive; +inf is allowed")
    endpoint = np.ones(n_time // 2 + 1)
    endpoint[0] = 0.5
    if n_time % 2 == 0:
        endpoint[-1] = 0.5
    # Exponent bookkeeping avoids overflow in n_time*dt or 1/dt even when the
    # final weight is representable. Infinite PSD bins remain exactly masked.
    finite = np.isfinite(psd)
    mantissa_dt, exponent_dt = np.frexp(dt)
    mantissa_psd, exponent_psd = np.frexp(np.where(finite, psd, 1.0))
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        prefactor = np.where(
            finite, (4.0 / n_time) * endpoint / (mantissa_dt * mantissa_psd), 0.0
        )
        weights = np.ldexp(prefactor, -exponent_dt - exponent_psd)
    weights = np.where(finite, weights, 0.0)
    if np.any(finite & (weights == 0)):
        raise FloatingPointError("Positive PSD quadrature weight underflowed")
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("Non-finite quadrature weights")
    return weights
