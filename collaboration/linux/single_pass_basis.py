"""Stock native m-pair basis adapter, independent of the FEW production tree."""

# 2026-09-11 17:27 CST (linux): Keep trajectory, amplitude selection and SSB
# conversion on the stock path. Only the TD summation's returned layout changes
# temporarily; every array is owned and the owner is restored even on failures.
import numpy as np
from contextlib import contextmanager


def phase_coefficients(phase, m):
    phase = np.asarray(phase)
    if phase.dtype.kind not in "fiu" or not np.all(np.isfinite(phase)):
        raise ValueError("Finite real phase required")
    with np.errstate(over="raise", invalid="raise"):
        angle = phase[..., None] * np.arange(1, m + 1)
    return np.concatenate(
        (np.ones(phase.shape + (1,)), np.cos(angle), np.sin(angle)), axis=-1
    )


class PairBasisSum:
    """Return [constant, cos(q), sin(q)] via the unchanged native renderer."""

    def __init__(self, renderer, audit_phases=False):
        if renderer.output_type != "td":
            raise NotImplementedError("TD only")
        self.renderer = renderer
        self.calls = 0
        self.group_calls = 0
        self.group_sizes = []
        self.output_type = "td"
        self.audit_phases = audit_phases

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
        if self.audit_phases:
            # 2026-09-11 17:36 CST (linux): Diagnostic only, outside all timed
            # preparation. Shift ONLY the azimuthal constant coefficient in the
            # actual DOP853 spline and call the unchanged full-mode native sum.
            if phase_coeff.ndim != 3 or phase_coeff.shape[1:] != (3, 8):
                raise AssertionError("Unexpected stock phase coefficient layout")
            checks = []
            for delta in (0.0, 0.71, -0.4):
                shifted = phase_coeff.copy()
                shifted[:, 0, 0] += delta
                direct = np.asarray(
                    self.renderer(
                        t,
                        amplitudes,
                        ylms,
                        phase_t,
                        shifted,
                        l,
                        m,
                        k,
                        n,
                        *extra,
                        **kwargs,
                    )
                ).copy()
                prediction = phase_coefficients(delta, degree) @ output
                relative = float(
                    np.linalg.norm(direct - prediction) / np.linalg.norm(direct)
                )
                checks.append(dict(delta=delta, relative_l2=relative))
                if not np.isfinite(relative) or relative > 1e-10:
                    raise AssertionError("Frozen native phase-shift oracle failed")
            self.metadata["frozen_native_checks"] = checks
        return output


class _CountProxy:
    def __init__(self, target):
        self.target, self.calls = target, 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.target(*args, **kwargs)


@contextmanager
def count_intrinsic(inner):
    # 2026-09-11 17:36 CST (linux): Count both A/B paths symmetrically and
    # exactly restore class-descriptor versus instance-override lookup state.
    original = inner.inspiral_generator
    amp = inner.amplitude_generator
    method = amp.get_amplitudes
    overridden = "get_amplitudes" in vars(amp)
    previous = vars(amp).get("get_amplitudes")
    proxy = _CountProxy(original)
    counts = dict(amplitude_calls=0)

    def counted(*args, **kwargs):
        counts["amplitude_calls"] += 1
        return method(*args, **kwargs)

    inner.inspiral_generator = proxy
    amp.get_amplitudes = counted
    try:
        yield counts
    finally:
        counts["trajectory_calls"] = proxy.calls
        inner.inspiral_generator = original
        if overridden:
            amp.get_amplitudes = previous
        else:
            delattr(amp, "get_amplitudes")


def build_ssb_basis(model, parameters, settings, audit_phases=False):
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
    adapter = PairBasisSum(original_sum, audit_phases=audit_phases)
    inner.create_waveform = adapter
    try:
        with count_intrinsic(inner) as counts:
            output = np.asarray(model(*parameters, **settings)).copy()
    finally:
        inner.create_waveform = original_sum
    if (
        adapter.calls != 1
        or counts["trajectory_calls"] != 1
        or counts["amplitude_calls"] != 1
    ):
        raise AssertionError(
            "Expected exactly one trajectory/amplitude/selection sum entry"
        )
    if output.shape[0] != 2 * adapter.metadata["m"] + 1 or not np.all(
        np.isfinite(output)
    ):
        raise AssertionError("Bad outer converted basis")
    return output, dict(
        adapter.metadata,
        **counts,
        outer_calls=1,
    )
