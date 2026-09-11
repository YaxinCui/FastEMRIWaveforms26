"""Fast tests for the controlled model and the measurement harness."""

# 2026-09-11 13:06 CST (linux): No flux file or GPU needed. The analytic
# autonomous system checks the time/phase comparison independently of FEW.
import unittest
from types import SimpleNamespace

import numpy as np

from flux_scaling_probe import UniformRateScaledKerr, compare, evaluate


class FluxScalingProbeTest(unittest.TestCase):
    def test_hook_is_in_place_and_rejects_invalid_parameters(self):
        model = object.__new__(UniformRateScaledKerr)
        model.use_ELQ = False
        model.flux_output_convention = "pex"
        for invalid in [None, [], [0.0, 1.0], [np.nan], [np.inf], [-1.0], [-2.0]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                model.add_fixed_parameters(1e6, 10.0, 0.7, invalid)
        model.add_fixed_parameters(1e6, 10.0, 0.7, [1e-4])
        rates = np.array([-1.0, -2.0, 0.0, 3.0, 4.0, 5.0])
        original = rates.copy()
        self.assertIs(model.modify_rhs(rates, np.zeros(6)), rates)
        np.testing.assert_array_equal(rates[:2], original[:2] * 1.0001)
        np.testing.assert_array_equal(rates[2:], original[2:])
        model.use_ELQ = True
        with self.assertRaises(ValueError):
            model.modify_rhs(rates, np.zeros(6))

    def test_analytic_time_reparameterization_and_horizon(self):
        class Analytic:
            def __init__(self, scale):
                self.scale = scale
                self.integrator_spline_t = np.array([0.0, 10.0])
                self.inspiral_generator = SimpleNamespace(eval_integrator_spline=self.eval)

            def eval(self, t):
                q = np.exp(-self.scale * t)
                phi = t + (1.0 - q) / self.scale
                return np.column_stack((q, 0.2*q, np.ones_like(q), phi+0.3, 2*phi+0.4, 3*phi+0.5))

            def func(self, y):
                return np.array([-self.scale*y[0], -0.2*self.scale*y[0], 0.0,
                                 1+y[0], 2*(1+y[0]), 3*(1+y[0])])

        ref = Analytic(1.0)
        for delta in [0.0, -0.0001, 0.0001]:
            row = compare(ref, Analytic(1+delta), (1e6, 10, 0.7, 1, 0.2, 1),
                          delta, 101, 1.0)
            self.assertEqual(row["same_time_end_seconds"], 0.9)
            self.assertEqual(row["time_reparameterized_end_seconds"], 0.9)
            self.assertLess(max(row["max_abs_phase_invariant_residual_rad"]), 5e-15)
            self.assertLess(max(row["max_abs_orbit_invariant_residual"]), 5e-15)
        for invalid in [np.array([-1.0]), np.array([11.0]), np.array([np.nan])]:
            with self.assertRaises(ValueError):
                evaluate(ref, invalid)
        ref.inspiral_generator.eval_integrator_spline = lambda t: np.zeros((len(t), 3))
        with self.assertRaises(ValueError):
            evaluate(ref, np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
