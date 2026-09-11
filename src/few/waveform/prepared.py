"""Explicit fixed-intrinsic snapshots for repeated source-observer projections."""

# 2026-09-11 13:29 CST (linux): Independent opt-in experiment. Reuse trajectory
# and full unweighted amplitudes, never a previous observer's selected modes.
from copy import deepcopy
from threading import Lock

import numpy as np

from ..amplitude.base import AmplitudeBase
from ..summation.interpolatedmodesum import InterpolatedModeSum
from ..trajectory.ode import KerrEccEqFlux
from ..utils.baseclasses import ParallelModuleBase, SphericalHarmonic
from ..utils.constants import Gpc, MRSUN_SI
from ..utils.modeselector import ModeSelector, get_mode_frequencies
from ..utils.ylm import GetYlms


def _own_array(xp, values, dtype=None):
    """Own the complete allocation: native wrappers must not see offset views."""
    result = xp.array(values, dtype=dtype, copy=True, order="C")
    if xp is np:
        result.flags.writeable = False
    return result


# 2026-09-11 13:43 CST (linux): Snapshot mode-selection sensitivity values,
# not an external mutable callback. The frequency vector and ordering are fixed
# by the intrinsic trajectory; +inf PSD is allowed (zero sensitivity weight).
class _SnapshotPSD:
    def __init__(self, xp, frequencies, sensitivity_fn):
        self.xp = xp
        self._frequencies = _own_array(xp, frequencies, xp.float64)
        values = xp.asarray(sensitivity_fn(self._frequencies.copy()))
        if values.shape != self._frequencies.shape or values.dtype.kind not in "fiu":
            raise ValueError(
                "Sensitivity must return real PSD values matching the frequency vector"
            )
        if bool(xp.any(xp.isnan(values))) or bool(xp.any(values <= 0)):
            raise ValueError(
                "Sensitivity PSD must be positive (positive infinity is allowed)"
            )
        self._values = _own_array(xp, values, xp.float64)

    def __call__(self, frequencies):
        if not bool(self.xp.array_equal(frequencies, self._frequencies)):
            raise ValueError("Snapshot PSD frequency vector/order changed")
        return self._values.copy()


def _check_threshold(value):
    if value is not None and (not np.isfinite(value) or not 0 <= value < 1):
        raise ValueError("Mode selection threshold must be finite and in [0, 1)")


class _SnapshotAmplitude(AmplitudeBase):
    """Private amplitude provider retaining the existing signed-mode rules."""

    def __init__(self, original, values):
        ParallelModuleBase.__init__(self, force_backend=original.backend)
        arrays = (
            "l_arr_no_mask",
            "m_arr_no_mask",
            "k_arr_no_mask",
            "n_arr_no_mask",
            "unique_l",
            "unique_m",
            "inverse_lm",
            "index_map_arr",
            "special_index_map_arr",
            "mode_indexes",
            "negative_mode_indexes",
            "l_arr",
            "k_arr",
        )
        for name in arrays:
            setattr(self, name, _own_array(self.xp, getattr(original, name)))
        self.num_teuk_modes = original.num_teuk_modes
        self.num_m_1_up = original.num_m_1_up
        if values.ndim != 2 or values.shape[1] != self.num_teuk_modes:
            raise ValueError("Snapshot amplitudes do not match full mode topology")
        # Full canonical amplitude columns follow mode_indexes. Verify rather
        # than assuming that custom mode permutations share native indices.
        if not bool(
            self.xp.array_equal(self.mode_indexes, self.xp.arange(self.num_teuk_modes))
        ):
            raise ValueError("Snapshot requires the stock full-mode ordering")
        self._values = _own_array(self.xp, values, self.xp.complex128)

    def get_amplitudes(self, a, p, e, xI, specific_modes=None):
        # Only PreparedKerrWaveform supplies the bound trajectory to this
        # private provider. Fancy indexing creates independent writable output,
        # allowing AmplitudeBase's existing negative-mode conjugation rules.
        return self._values[:, specific_modes].copy()


class PreparedKerrWaveform(ParallelModuleBase):
    """Own a stock Kerr TD trajectory and amplitudes for observer scans.

    Construct via ``model.prepare_observer_scan(...)``. Intrinsic parameters,
    initial phases, T and dt are fixed. Calling this object accepts only source
    angles, distance and mode-selection settings; new intrinsic parameters
    require a new snapshot. No hidden cache key or approximate matching is used.

    Preparation advances the owner's trajectory once, like a normal waveform
    call, but does not mutate its inspiral_kwargs dictionary. The snapshot has
    its own phase/frequency data and renderer and does not retain the owner's
    5 GB amplitude table. Later calls on the owner cannot overwrite this data.
    Mode-selection sensitivity is also frozen at preparation. Changing the
    original PSD callback cannot change an existing snapshot. Same-object
    concurrent rendering and nondefault CUDA streams/devices are rejected.
    This is source strain, not a LISA response or TDI/likelihood cache.
    """

    def __init__(
        self,
        model,
        m1,
        m2,
        a,
        p0,
        e0,
        xI0,
        *ode_args,
        Phi_phi0=0.0,
        Phi_theta0=0.0,
        Phi_r0=0.0,
        T=1.0,
        dt=10.0,
        inspiral_kwargs=None,
    ):
        ParallelModuleBase.__init__(self, force_backend=model.backend)
        # 2026-09-11 13:39 CST (linux): Bind the allocation device as well as
        # the stream; a different device also has a default stream with ptr=0.
        self._device_id = (
            self.xp.cuda.runtime.getDevice() if self.backend.uses_cuda else None
        )
        self._check_stream()
        if type(model.inspiral_generator.func) is not KerrEccEqFlux:
            raise NotImplementedError(
                "Prepared observer scans require the stock KerrEccEqFlux ODE"
            )
        if (
            type(model.mode_selector) is not ModeSelector
            or type(model.ylm_gen) is not GetYlms
        ):
            raise NotImplementedError(
                "Prepared observer scans require stock selection and angular modules"
            )
        if type(model.create_waveform) is not InterpolatedModeSum:
            raise NotImplementedError(
                "Prepared observer scans currently require time-domain interpolation"
            )
        _check_threshold(model.mode_selector.mode_selection_threshold)
        parameters = np.asarray(
            [m1, m2, a, p0, e0, xI0, Phi_phi0, Phi_theta0, Phi_r0, T, dt]
        )
        if (
            not np.all(np.isfinite(parameters))
            or m1 <= 0
            or m2 <= 0
            or T <= 0
            or dt <= 0
        ):
            raise ValueError("Require finite parameters, positive masses, T and dt")
        self._flip_observer = xI0 < 0.0
        if self._flip_observer:
            a, xI0 = -a, -xI0
        a, xI0 = model.sanity_check_init(m1, m2, a, p0, e0, xI0)
        effective = deepcopy(model.inspiral_kwargs)
        effective.update(deepcopy({} if inspiral_kwargs is None else inspiral_kwargs))
        effective.setdefault("err", 1e-11)
        if effective.get("DENSE_STEPPING", False):
            raise NotImplementedError(
                "Prepared observer scans require adaptive sparse trajectories"
            )
        trajectory = model.inspiral_generator
        result = trajectory(
            m1,
            m2,
            a,
            p0,
            e0,
            xI0,
            *ode_args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **effective,
        )
        t, p, e, xI, phi, theta, radial = result
        model.sanity_check_traj(a, p, e, xI)
        if trajectory.dense_stepping or not np.all(xI == 1.0):
            raise NotImplementedError(
                "Require sparse canonical equatorial output with xI=1"
            )
        if len(t) < 2 or not all(np.all(np.isfinite(v)) for v in result):
            raise ValueError("Invalid or empty intrinsic trajectory")
        self._a, self._M = float(a), float(m1 + m2)
        self._mu = float(m1 * m2 / (m1 + m2))
        self._T, self._dt = float(T), float(dt)
        self._backwards = bool(trajectory.integrate_backwards)
        self._trajectory = tuple(
            _own_array(self.xp, v, self.xp.float64) for v in result
        )
        # Keep derivative evaluation on host. This new API does not enter the
        # master's known CUDA waveform/base.py host/device boundary failure.
        frequencies = (
            trajectory.inspiral_generator.eval_integrator_derivative_spline(
                np.asarray(t), order=1
            )[:, 3:6]
            / 2
            / np.pi
        )
        self._frequencies = {
            key: _own_array(self.xp, frequencies[:, i], self.xp.float64)
            for i, key in enumerate(("f_phi", "f_theta", "f_r"))
        }
        phase = np.array(trajectory.integrator_spline_phase_coeff, copy=True, order="C")
        if a > 0:
            phase[:, 0] *= np.sign(xI0)
        if self._backwards:
            phase[:, :, 0] += np.array(
                [phi[-1] + phi[0], theta[-1] + theta[0], radial[-1] + radial[0]]
            )
        self._phase = _own_array(self.xp, phase, self.xp.float64)
        self._phase_t = _own_array(np, trajectory.integrator_spline_t, np.float64)
        td, pd, ed, xd = self._trajectory[:4]
        amplitudes = model.amplitude_generator(a, pd, ed, xd)
        if not bool(self.xp.all(self.xp.isfinite(amplitudes))):
            raise FloatingPointError("Non-finite intrinsic amplitude")
        cached_amplitude = _SnapshotAmplitude(model.amplitude_generator, amplitudes)
        selector = model.mode_selector
        sensitivity = None
        if selector.sensitivity_fn is not None:
            full_frequencies = self.xp.abs(
                get_mode_frequencies(
                    **self._frequencies,
                    m=selector.m_arr,
                    k=selector.k_arr,
                    n=selector.n_arr,
                )
            )
            sensitivity = _SnapshotPSD(
                self.xp, full_frequencies.flatten(), selector.sensitivity_fn
            )
        self._selector = ModeSelector(
            cached_amplitude,
            GetYlms(
                force_backend=self.backend,
                include_minus_m=model.ylm_gen.include_minus_m,
            ),
            mode_selection=deepcopy(selector.mode_selection),
            include_minus_mkn=selector.include_minus_mkn,
            mode_selection_threshold=selector.mode_selection_threshold,
            sensitivity_fn=sensitivity,
            modeinds_map=_own_array(self.xp, selector.modeinds_map),
            force_backend=self.backend,
        )
        original_sum = model.create_waveform
        self._sum = InterpolatedModeSum(
            force_backend=self.backend,
            pad_output=original_sum.pad_output,
            odd_len=original_sum.odd_len,
        )
        self._render_lock = Lock()
        self._selected = None
        self._metadata = {
            "parameters": tuple(float(x) for x in parameters),
            "canonical_spin": self._a,
            "backwards": self._backwards,
            "T_years": self._T,
            "dt_seconds": self._dt,
            "sparse_nodes": len(t),
            "full_modes": amplitudes.shape[1],
            "amplitude_bytes": cached_amplitude._values.nbytes,
            "backend": self.backend.name,
            "device_id": self._device_id,
            "ode_class": type(trajectory.func).__module__
            + "."
            + type(trajectory.func).__qualname__,
            "inspiral_kwargs": deepcopy(
                {k: v for k, v in effective.items() if k != "func"}
            ),
            "frozen_sensitivity": sensitivity is not None,
            "scope": "fixed intrinsic source strain, not TDI or inference acceptance",
        }
        if self.backend.uses_cuda:
            self.xp.cuda.runtime.deviceSynchronize()

    def _check_stream(self):
        if self.backend.uses_cuda:
            if self.xp.cuda.runtime.getDevice() != self._device_id:
                raise RuntimeError(
                    "Prepared observer scans must use their preparation CUDA device"
                )
            if self.xp.cuda.get_current_stream().ptr != 0:
                raise NotImplementedError(
                    "Prepared observer scans currently require the CUDA default stream"
                )

    @property
    def metadata(self):
        """Return independent descriptive metadata, not writable cache arrays."""
        return deepcopy(self._metadata)

    @property
    def selected_modes(self):
        """Return independent copies of the last render's (ell,m,k,n) arrays."""
        self._check_stream()
        return (
            None if self._selected is None else tuple(v.copy() for v in self._selected)
        )

    def __call__(
        self,
        theta,
        phi,
        *,
        dist=None,
        mode_selection=None,
        include_minus_mkn=None,
        mode_selection_threshold=1e-5,
    ):
        self._check_stream()
        if not np.isfinite(theta) or not np.isfinite(phi):
            raise ValueError("Observer angles must be finite")
        if dist is not None and (not np.isfinite(dist) or dist <= 0.0):
            raise ValueError("Distance must be finite and greater than zero")
        _check_threshold(mode_selection_threshold)
        if not self._render_lock.acquire(blocking=False):
            raise RuntimeError(
                "Concurrent use of one prepared renderer is not supported"
            )
        try:
            if self._flip_observer:
                theta, phi = np.pi - theta, -phi
            theta, phi = SphericalHarmonic.sanity_check_viewing_angles(self, theta, phi)
            td, pd, ed, xd = self._trajectory[:4]
            modes, ylms, *indices = self._selector(
                td,
                self._a,
                pd,
                ed,
                xd,
                theta,
                phi,
                online_mode_selection_args=self._frequencies,
                mode_selection=mode_selection,
                include_minus_mkn=include_minus_mkn,
                mode_selection_threshold=mode_selection_threshold,
            )
            # Copy before native handoff: no sliced offset pointer and no
            # mutable alias into the snapshot, even on the CPU backend.
            waveform = self._sum(
                td,
                modes,
                ylms,
                self._phase_t,
                self._phase.copy(),
                *indices,
                self._M,
                self._a,
                pd,
                ed,
                xd,
                dt=self._dt,
                T=self._T,
                integrate_backwards=self._backwards,
            )
            divisor = 1.0 if dist is None else (dist * Gpc) / (self._mu * MRSUN_SI)
            result = waveform / divisor
            if self.backend.uses_cuda:
                self.xp.cuda.runtime.deviceSynchronize()
            self._selected = tuple(v.copy() for v in indices)
            return result
        finally:
            self._render_lock.release()
