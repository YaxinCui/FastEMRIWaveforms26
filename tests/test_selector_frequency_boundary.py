"""Exercise host/array-backend boundaries without high-memory Kerr data."""

# 2026-09-11 13:56 CST (linux): Reproduce mixed NumPy/CuPy PSD input failure
# before changing master. Synthetic amplitudes isolate the consumer boundary.
import ast
import inspect
from types import SimpleNamespace
import unittest

import numpy as np

from few.amplitude.base import AmplitudeBase
from few.tests.base import FewBackendTest
from few.utils.baseclasses import SphericalHarmonic
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms
from few.waveform.base import SphericalHarmonicWaveformBase


class TestAmplitude(AmplitudeBase, SphericalHarmonic):
    def __init__(self, backend):
        SphericalHarmonic.__init__(self, lmax=3, nmax=2, force_backend=backend)

    def get_amplitudes(self, a, p, e, xI, specific_modes=None):
        return 1 + self.xp.arange(4)[:, None] / 10 + 1j / (1 + specific_modes[None, :])


class SelectorFrequencyBoundaryTest(FewBackendTest):
    @classmethod
    def name(cls):
        return "Selector frequency boundary"

    @classmethod
    def parallel_class(cls):
        return ModeSelector

    def test_host_mixed_and_backend_frequencies(self):
        xp = self.backend.xp
        seen = []

        def sensitivity(f):
            self.assertIsInstance(f, xp.ndarray)
            self.assertEqual(f.dtype, xp.float64)
            seen.append(f.copy())
            return 1 + 1e7 * f**2

        amplitude = TestAmplitude(self.backend)
        selector = ModeSelector(
            amplitude,
            GetYlms(force_backend=self.backend),
            sensitivity_fn=sensitivity,
            force_backend=self.backend,
        )
        host_frequencies = np.linspace(0.001, 0.03, 12).reshape(4, 3)
        keys = ("f_phi", "f_theta", "f_r")
        all_device = {
            key: xp.asarray(host_frequencies[:, i]) for i, key in enumerate(keys)
        }
        all_host = {key: host_frequencies[:, i] for i, key in enumerate(keys)}
        mixed_inputs = [dict(all_device, **{key: all_host[key]}) for key in keys]
        params = (
            xp.arange(4, dtype=xp.float64),
            0.6,
            xp.full(4, 12.0),
            xp.full(4, 0.3),
            xp.ones(4),
            1.1,
            0.7,
        )
        expected = selector(*params, online_mode_selection_args=all_device)
        for inputs in (all_host, *mixed_inputs):
            originals = tuple(inputs[key] for key in keys)
            actual = selector(*params, online_mode_selection_args=inputs)
            for left, right in zip(expected, actual):
                self.assertTrue(bool(xp.array_equal(left, right)))
            for key, original in zip(keys, originals):
                self.assertIs(inputs[key], original)
        self.assertEqual(len(seen), 5)
        # Independent Hz/order oracle, not a second call to the production
        # helper that would repeat a wrong factor of 2*pi in both variants.
        modes = [
            value.get() if hasattr(value, "get") else value
            for value in (selector.m_arr, selector.k_arr, selector.n_arr)
        ]
        oracle = np.abs(
            sum(host_frequencies[:, i, None] * modes[i][None, :] for i in range(3))
        ).flatten()
        for frequency_vector in seen:
            actual = (
                frequency_vector.get()
                if hasattr(frequency_vector, "get")
                else frequency_vector
            )
            np.testing.assert_allclose(actual, oracle, rtol=1e-15, atol=0)
        np.testing.assert_array_equal(
            host_frequencies, np.linspace(0.001, 0.03, 12).reshape(4, 3)
        )
        # Selection branches that do not need PSD frequencies must continue
        # accepting no online frequency dictionary.
        for selection in ("all", [(2, 2, 0, 0)]):
            selector(*params, online_mode_selection_args=None, mode_selection=selection)
        self.assertEqual(len(seen), 5)
        selector.sensitivity_fn = None
        selector(*params, online_mode_selection_args=None)
        self.assertEqual(len(seen), 5)


class TrajectoryHostBoundaryTest(unittest.TestCase):
    def test_actual_sampler_call_receives_host_times(self):
        tree = ast.parse(inspect.getsource(SphericalHarmonicWaveformBase))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "eval_integrator_derivative_spline"
        ]
        self.assertEqual(len(calls), 1)
        expression = compile(
            ast.fix_missing_locations(ast.Expression(calls[0])),
            "<actual source call>",
            "eval",
        )
        times = np.array([0.0, 1.0, 4.0, 9.0])
        received, copies = [], []

        def sampler(value, order):
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(order, 1)
            received.append(value)
            return np.ones((4, 6))

        def download():
            copies.append(True)
            return times.copy()

        owner = SimpleNamespace(
            backend=SimpleNamespace(uses_cupy=False),
            inspiral_generator=SimpleNamespace(
                inspiral_generator=SimpleNamespace(
                    eval_integrator_derivative_spline=sampler
                )
            ),
        )
        result = eval(expression, {"self": owner, "t_temp": times})
        self.assertIs(received[-1], times)
        self.assertEqual(result.shape, (4, 6))
        owner.backend.uses_cupy = True
        eval(expression, {"self": owner, "t_temp": SimpleNamespace(get=download)})
        np.testing.assert_array_equal(received[-1], times)
        self.assertEqual(len(copies), 1)


if __name__ == "__main__":
    unittest.main()
