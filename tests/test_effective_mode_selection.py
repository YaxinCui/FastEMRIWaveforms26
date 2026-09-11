"""Actual frontend + stock selector with a small synthetic trajectory/sum."""

# 2026-09-11 14:13 CST (linux): Reproduce the configuration bug without the
# 5 GB table. The fake sum records every real selector output; not a physics test.
import unittest
from types import SimpleNamespace

import numpy as np

from few.amplitude.base import AmplitudeBase
from few.utils.baseclasses import ParallelModuleBase, SphericalHarmonic
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms
from few.waveform.base import SphericalHarmonicWaveformBase


class SmallAmplitude(AmplitudeBase, SphericalHarmonic):
    def __init__(self):
        SphericalHarmonic.__init__(self, lmax=3, nmax=2, force_backend="cpu")

    def get_amplitudes(self, a, p, e, xI, specific_modes=None):
        return (p[:, None] + 1j * e[:, None]) / (1 + specific_modes[None, :]) ** 2


class SmallTrajectory:
    dense_stepping = False
    integrate_backwards = False

    def __init__(self):
        self.derivative_times = []
        self.inspiral_generator = SimpleNamespace(
            eval_integrator_derivative_spline=self.derivatives
        )
        self.integrator_spline_t = np.arange(8, dtype=float)

    @property
    def integrator_spline_phase_coeff(self):
        return np.zeros((7, 3, 8))

    def derivatives(self, times, order):
        if order != 1:
            raise AssertionError("Unexpected derivative order")
        self.derivative_times.append(times.copy())
        return np.column_stack(
            [
                np.zeros((len(times), 3)),
                2 * np.pi * (1 + times[:, None]) * np.array([0.001, 0.002, 0.003]),
            ]
        )

    def __call__(self, *args, **kwargs):
        t = self.integrator_spline_t
        return (
            t.copy(),
            12 - 0.02 * t,
            0.3 - 0.01 * t,
            np.ones(8),
            0.01 * t,
            0.02 * t,
            0.03 * t,
        )


class SmallWaveform(SphericalHarmonicWaveformBase):
    allow_batching = True

    def __init__(self, selection):
        ParallelModuleBase.__init__(self, force_backend="cpu")
        self.inspiral_kwargs = {}
        self.inspiral_generator = SmallTrajectory()
        self.amplitude_generator = SmallAmplitude()
        self.ylm_gen = GetYlms(force_backend="cpu")
        self.psd_calls = []
        self.mode_selector = ModeSelector(
            self.amplitude_generator,
            self.ylm_gen,
            mode_selection=selection,
            sensitivity_fn=self.sensitivity,
            force_backend="cpu",
        )
        self.sums = []
        self.create_waveform = self.sum

    def sensitivity(self, frequencies):
        self.psd_calls.append(frequencies.copy())
        return 1 + 1000 * frequencies**2

    def sum(
        self,
        t,
        amplitudes,
        weights,
        phase_t,
        coefficients,
        ell,
        m,
        k,
        n,
        *args,
        **kwargs,
    ):
        self.sums.append(
            tuple(value.copy() for value in (t, amplitudes, weights, ell, m, k, n))
        )
        return np.sum(amplitudes * weights[: amplitudes.shape[1]], axis=1)

    def sanity_check_init(self, m1, m2, a, p, e, xI):
        return a, xI

    def sanity_check_traj(self, *args):
        pass

    def sanity_check_viewing_angles(self, theta, phi):
        return theta, phi

    def generate(self, selection=None, **kwargs):
        return self._generate_waveform(
            1e6,
            10.0,
            0.6,
            12.0,
            0.3,
            1.0,
            1.1,
            0.7,
            mode_selection=selection,
            mode_selection_threshold=0.1,
            **kwargs,
        )


class EffectiveModeSelectionTest(unittest.TestCase):
    def assert_full(self, model):
        self.assertEqual(model.num_modes_kept, model.amplitude_generator.num_teuk_modes)
        for actual, expected in zip(
            (model.ls, model.ms, model.ks, model.ns),
            (
                model.amplitude_generator.l_arr_no_mask,
                model.amplitude_generator.m_arr_no_mask,
                model.amplitude_generator.k_arr_no_mask,
                model.amplitude_generator.n_arr_no_mask,
            ),
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_all_threshold_default_sequence(self):
        model = SmallWaveform("all")
        before = model.generate()
        self.assert_full(model)
        self.assertEqual(len(model.inspiral_generator.derivative_times), 0)
        filtered = model.generate("threshold")
        self.assertLess(model.num_modes_kept, model.amplitude_generator.num_teuk_modes)
        self.assertEqual(len(model.psd_calls), 1)
        self.assertEqual(len(model.inspiral_generator.derivative_times), 1)
        direct = SmallWaveform("threshold")
        np.testing.assert_array_equal(filtered, direct.generate())
        self.assertEqual(model.mode_selector.mode_selection, "all")
        np.testing.assert_array_equal(model.generate(), before)
        self.assert_full(model)
        self.assertEqual(len(model.inspiral_generator.derivative_times), 1)

    def test_threshold_override_all_skips_derivatives(self):
        model = SmallWaveform("threshold")
        result = model.generate("all")
        self.assert_full(model)
        self.assertEqual(len(model.inspiral_generator.derivative_times), 0)
        self.assertEqual(len(model.psd_calls), 0)
        np.testing.assert_array_equal(result, SmallWaveform("all").generate())

    def test_list_default_remains_list(self):
        modes = [(2, 2, 0, 0), (3, 1, 0, -1)]
        model = SmallWaveform(modes)
        expected = model.generate()
        np.testing.assert_array_equal(
            np.column_stack((model.ls, model.ms, model.ks, model.ns)), modes
        )
        model.generate("all")
        self.assert_full(model)
        np.testing.assert_array_equal(model.generate(), expected)
        self.assertEqual(model.mode_selector.mode_selection, modes)

    def test_batch_sampling_obeys_override(self):
        model = SmallWaveform("all")
        result = model.generate("threshold", batch_size=4)
        self.assertEqual(result.shape, (8,))
        np.testing.assert_array_equal(
            model.inspiral_generator.derivative_times, np.arange(8).reshape(2, 4)
        )
        self.assertEqual(len(model.sums), 2)
        for entry in model.sums:
            self.assertEqual(entry[1].shape[0], 4)
            self.assertEqual(entry[1].shape[1], len(entry[3]))
        # 2026-09-11 14:32 CST (linux): Compare every actual selected batch,
        # not only its shape or the final model's last-batch mode fields.
        direct = SmallWaveform("threshold")
        np.testing.assert_array_equal(result, direct.generate(batch_size=4))
        for actual_batch, reference_batch in zip(model.sums, direct.sums):
            for actual, reference in zip(actual_batch, reference_batch):
                np.testing.assert_array_equal(actual, reference)
        model.generate("all", batch_size=4)
        self.assertEqual(len(model.inspiral_generator.derivative_times), 2)
        self.assertEqual(len(model.psd_calls), 2)
        for entry in model.sums[2:]:
            self.assertEqual(
                entry[1].shape[1], model.amplitude_generator.num_teuk_modes
            )
            for actual, expected in zip(
                entry[3:],
                (
                    model.amplitude_generator.l_arr_no_mask,
                    model.amplitude_generator.m_arr_no_mask,
                    model.amplitude_generator.k_arr_no_mask,
                    model.amplitude_generator.n_arr_no_mask,
                ),
            ):
                np.testing.assert_array_equal(actual, expected)
        self.assert_full(model)

    def test_none_constructor_uses_threshold(self):
        model = SmallWaveform(None)
        direct = SmallWaveform("threshold")
        expected = direct.generate()
        np.testing.assert_array_equal(model.generate(), expected)
        model.generate("all")
        self.assert_full(model)
        np.testing.assert_array_equal(model.generate(), expected)
        self.assertIsNone(model.mode_selector.mode_selection)
        self.assertEqual(len(model.psd_calls), 2)


if __name__ == "__main__":
    unittest.main()
