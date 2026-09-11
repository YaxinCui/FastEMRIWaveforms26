"""FP64 real-linear phase basis and centered conditional Gaussian likelihood."""

# 2026-09-11 17:04 CST (linux): Script-only finite Fourier/Gram experiment.
# Samples/data are already real-whitened coordinates; arbitrary fixed linear
# response and covariance whitening belong upstream. Keep every cross term.
from dataclasses import dataclass

import numpy as np


def finite_real(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
        raise ValueError("Require finite real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Outside FP64 range")
    return result


def deltas(phase, m):
    """[0, cos(k*alpha)-1, sin(k*alpha)], centered at alpha=0."""
    phase = finite_real(phase)
    angle = phase[..., None] * np.arange(1, m + 1)
    if not np.all(np.isfinite(angle)):
        raise ValueError("Phase product overflow")
    return np.concatenate(
        (np.zeros(phase.shape + (1,)), -2 * np.sin(angle / 2) ** 2, np.sin(angle)),
        axis=-1,
    )


@dataclass(frozen=True)
class FourierConditional:
    """Only phase-dependent sufficient statistics; no waveform/basis residency."""

    cosine: np.ndarray
    sine: np.ndarray

    def __post_init__(self):
        # 2026-09-11 17:14 CST (linux): Own the compact payload independently
        # so discarding the preparation object actually releases its large basis.
        cosine, sine = finite_real(self.cosine).copy(), finite_real(self.sine).copy()
        if cosine.ndim != 1 or cosine.shape != sine.shape:
            raise ValueError("Matching one-dimensional coefficient arrays required")
        cosine.setflags(write=False)
        sine.setflags(write=False)
        object.__setattr__(self, "cosine", cosine)
        object.__setattr__(self, "sine", sine)

    def __call__(self, phase):
        degree = self.cosine.size
        x = deltas(phase, degree)
        return x[..., 1 : 1 + degree] @ self.cosine + x[..., 1 + degree :] @ self.sine


@dataclass(frozen=True)
class PhaseLikelihood:
    basis: np.ndarray
    residual: np.ndarray
    projection: np.ndarray
    gram: np.ndarray
    fourier_cos: np.ndarray
    fourier_sin: np.ndarray

    @property
    def m(self):
        return (self.basis.shape[0] - 1) // 2

    @classmethod
    def prepare(cls, samples, data):
        samples, data = finite_real(samples), finite_real(data)
        if (
            samples.ndim != 2
            or samples.shape[0] % 2 != 1
            or not samples.size
            or data.shape != samples.shape[1:]
        ):
            raise ValueError(
                "Odd uniform phase sample count and matching flat data required"
            )
        n = samples.shape[0]
        spectrum = np.fft.rfft(samples, axis=0) / n
        basis = np.concatenate(
            (spectrum[:1].real, 2 * spectrum[1:].real, -2 * spectrum[1:].imag), axis=0
        )
        residual = data - samples[0]
        b, g = basis @ residual, basis @ basis.T
        m = (n - 1) // 2
        grid = 2 * np.pi * np.arange(4 * m + 1) / (4 * m + 1)
        x = deltas(grid, m)
        values = x @ b - 0.5 * np.einsum("ni,ij,nj->n", x, g, x)
        coeff = np.fft.rfft(values) / values.size
        result = cls(basis, residual, b, g, 2 * coeff[1:].real, -2 * coeff[1:].imag)
        for arr in (basis, residual, b, g, result.fourier_cos, result.fourier_sin):
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError("Nonfinite prepared statistic")
            arr.setflags(write=False)
        return result

    def gram_evaluate(self, phase):
        x = deltas(phase, self.m)
        return x @ self.projection - 0.5 * np.einsum(
            "...i,ij,...j->...", x, self.gram, x
        )

    def __call__(self, phase):
        # Subtract the reference analytically instead of two large logL values.
        # This forces exactly zero at alpha=0; tiny phase derivatives still
        # inherit the FP64 conditioning of Fourier coefficient preparation.
        x = deltas(phase, 2 * self.m)
        return (
            x[..., 1 : 1 + 2 * self.m] @ self.fourier_cos
            + x[..., 1 + 2 * self.m :] @ self.fourier_sin
        )

    def direct(self, phase):
        x = deltas(phase, self.m)
        dh = x @ self.basis
        return dh @ self.residual - 0.5 * np.sum(dh * dh, axis=-1)

    def compact(self):
        return FourierConditional(self.fourier_cos, self.fourier_sin)
