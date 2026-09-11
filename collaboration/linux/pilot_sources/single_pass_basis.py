"""Stock native m-pair basis adapter, independent of the FEW production tree."""

# 2026-09-11 17:27 CST (linux): Keep trajectory, amplitude selection and SSB
# conversion on the stock path. Only the TD summation's returned layout changes
# temporarily; every array is owned and the owner is restored even on failures.
import numpy as np


def phase_coefficients(phase, m):
    phase = np.asarray(phase)
    if phase.dtype.kind not in "fiu" or not np.all(np.isfinite(phase)):
        raise ValueError("Finite real phase required")
    angle = phase[..., None] * np.arange(1, m + 1)
    return np.concatenate(
        (np.ones(phase.shape + (1,)), np.cos(angle), np.sin(angle)), axis=-1
    )


class PairBasisSum:
    """Return [constant, cos(q), sin(q)] via the unchanged native renderer."""

    def __init__(self, renderer):
        if renderer.output_type != "td":
            raise NotImplementedError("TD only")
        self.renderer = renderer
        self.calls = 0
        self.group_calls = 0
        self.group_sizes = []
        self.output_type = "td"

    def __getattr__(self, name):
        return getattr(self.renderer, name)

    def __call__(
        self, t, amplitudes, ylms, phase_t, phase_coeff, l, m, k, n, *extra, **kwargs
    ):
        self.calls += 1
        if self.calls != 1 or kwargs.get("integrate_backwards", False):
            raise NotImplementedError("One forward unbatched summation required")
        if phase_t is None or amplitudes.ndim != 2:
            raise NotImplementedError("Sparse stock TD inputs required")
        if not isinstance(amplitudes, np.ndarray) or amplitudes.dtype != np.complex128:
            raise NotImplementedError("CPU FP64 prototype only")
        modes = amplitudes.shape[1]
        if (
            not modes
            or ylms.shape != (2 * modes,)
            or any(v.shape != (modes,) for v in (l, m, k, n))
        ):
            raise ValueError("Inconsistent selected mode/weight topology")
        if np.any(k != 0) or not all(
            np.all(np.isfinite(v)) for v in (t, amplitudes, ylms, phase_coeff)
        ):
            raise ValueError("Finite Kerr-equatorial inputs required")
        degree = int(np.max(np.abs(m)))
        if degree > 10:
            raise NotImplementedError("Stock Kerr truncation only")
        output = None
        for q in np.unique(np.abs(m)):
            keep = np.flatnonzero(np.abs(m) == q)
            amps = np.array(amplitudes[:, keep], copy=True, order="C")
            weights = np.concatenate((ylms[keep], ylms[modes + keep]))
            indices = [np.array(v[keep], copy=True, order="C") for v in (l, m, k, n)]

            def render(weight):
                self.group_calls += 1
                value = np.array(
                    self.renderer(
                        t,
                        amps,
                        weight,
                        phase_t,
                        phase_coeff,
                        *indices,
                        *extra,
                        **kwargs,
                    ),
                    copy=True,
                    order="C",
                )
                if (
                    value.ndim != 1
                    or value.dtype != np.complex128
                    or not np.all(np.isfinite(value))
                ):
                    raise AssertionError("Invalid native basis result")
                return value

            cosine = render(weights)
            if output is None:
                output = np.zeros((2 * degree + 1, cosine.size), dtype=np.complex128)
            output[int(q)] = cosine
            if q:
                sine_weights = weights.copy()
                sign = np.sign(m[keep])
                sine_weights[: len(keep)] *= -1j * sign
                sine_weights[len(keep) :] *= 1j * sign
                output[degree + int(q)] = render(sine_weights)
            self.group_sizes.append(
                dict(q=int(q), modes=len(keep), native_calls=1 if q == 0 else 2)
            )
        self.metadata = dict(
            m=degree,
            selected_modes=modes,
            sparse_knots=len(t),
            native_group_calls=self.group_calls,
            groups=self.group_sizes,
        )
        return output


class _CountProxy:
    def __init__(self, target):
        self.target, self.calls = target, 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.target(*args, **kwargs)


def build_ssb_basis(model, parameters, settings):
    """One original high-level call; basis phases are relative to parameters[11]."""
    from few.summation.interpolatedmodesum import InterpolatedModeSum
    from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

    if (
        type(model) is not GenerateEMRIWaveform
        or type(model.waveform_generator) is not FastKerrEccentricEquatorialFlux
    ):
        raise NotImplementedError("Stock Kerr wrapper only")
    if model.frame != "detector" or model.return_list or model.flip_output:
        raise NotImplementedError("Complex SSB forward output only")
    if settings.get("batch_size", -1) != -1:
        raise NotImplementedError("No time batching")
    inner = model.waveform_generator
    original_sum, original_trajectory = inner.create_waveform, inner.inspiral_generator
    if (
        type(original_sum) is not InterpolatedModeSum
        or original_trajectory.integrate_backwards
        or original_trajectory.dense_stepping
    ):
        raise NotImplementedError("Stock sparse forward TD renderer only")
    adapter = PairBasisSum(original_sum)
    trajectory = _CountProxy(original_trajectory)
    original_amplitude = inner.amplitude_generator.get_amplitudes
    amp_calls = 0

    def counted_amplitude(*args, **kwargs):
        nonlocal amp_calls
        amp_calls += 1
        return original_amplitude(*args, **kwargs)

    inner.create_waveform, inner.inspiral_generator = adapter, trajectory
    inner.amplitude_generator.get_amplitudes = counted_amplitude
    try:
        output = np.asarray(model(*parameters, **settings)).copy()
    finally:
        inner.create_waveform, inner.inspiral_generator = (
            original_sum,
            original_trajectory,
        )
        inner.amplitude_generator.get_amplitudes = original_amplitude
    if adapter.calls != 1 or trajectory.calls != 1 or amp_calls != 1:
        raise AssertionError(
            "Expected exactly one trajectory/amplitude/selection sum entry"
        )
    if output.shape[0] != 2 * adapter.metadata["m"] + 1 or not np.all(
        np.isfinite(output)
    ):
        raise AssertionError("Bad outer converted basis")
    return output, dict(
        adapter.metadata,
        trajectory_calls=trajectory.calls,
        amplitude_calls=amp_calls,
        outer_calls=1,
    )
