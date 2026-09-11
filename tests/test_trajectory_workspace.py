"""Same-algorithm regression tests for reusable DOP853 stage arrays."""

# 2026-09-11 12:07 CST (linux): Analytic RHS and lifecycle regressions, no tables.
import unittest
import numpy as np
from few.trajectory.dopr853 import DOPR853


class WorkspaceTest(unittest.TestCase):
    def test_dense_output_survives_next_step(self):
        def rhs(t, y, out, args):
            out[0], out[1] = -y[1], y[0]
        steppers = [DOPR853(rhs), DOPR853(rhs, reuse_stage_workspace=True)]
        for stepper in steppers:
            stepper.abstol = 1e-12
        states = [np.array([[1.], [0.]]) for _ in steppers]
        times = [np.zeros(1) for _ in steppers]
        steps = [np.full(1, .01) for _ in steppers]
        saved = None
        accepted = 0
        for _ in range(30):
            coefficients = []
            for i, stepper in enumerate(steppers):
                y0, x0, h0 = states[i].copy(), times[i].copy(), steps[i].copy()
                flags = stepper.take_step(times[i], steps[i], states[i], np.zeros((0, 1)))
                if flags[0]:
                    coefficients.append(stepper.prep_evaluate(x0, y0, h0, times[i], states[i], np.zeros((0, 1))))
                    mid = (x0 + times[i]) / 2
                    got = stepper.eval(mid, np.array([x0[0], times[i][0]]), coefficients[-1])[0]
                    np.testing.assert_allclose(got, [np.cos(mid[0]), np.sin(mid[0])], atol=2e-11, rtol=2e-11)
            np.testing.assert_array_equal(states[0], states[1])
            np.testing.assert_array_equal(steps[0], steps[1])
            if coefficients:
                accepted += 1
                np.testing.assert_array_equal(coefficients[0], coefficients[1])
                if saved is None:
                    saved = (coefficients[1], coefficients[1].copy())
            if saved is not None:
                np.testing.assert_array_equal(*saved)
        self.assertGreater(accepted, 5)

    def test_fixed_steps_and_shapes(self):
        def rhs(t, y, out, args):
            out[:] = -.25 * y
        old = DOPR853(rhs)
        new = DOPR853(rhs, reuse_stage_workspace=True)
        old.fix_step = new.fix_step = True
        for shape in [(6, 1), (6, 4), (3, 2), (6, 1)]:
            states = [np.ones(shape), np.ones(shape)]
            times = [np.zeros(shape[1]), np.zeros(shape[1])]
            steps = [np.full(shape[1], .01), np.full(shape[1], .01)]
            for _ in range(20):
                for i, stepper in enumerate((old, new)):
                    stepper.take_step(times[i], steps[i], states[i], np.zeros((0, shape[1])))
                np.testing.assert_array_equal(states[0], states[1])
                np.testing.assert_array_equal(times[0], times[1])
            np.testing.assert_allclose(states[1], np.exp(-.25 * times[1])[None, :] * np.ones(shape), atol=1e-14, rtol=1e-14)
            self.assertEqual(new._stage_workspace.shape, (10, *shape))

    def test_reuse_zeros_and_rejection(self):
        def rhs(t, y, out, args):
            out[0] = -y[0]
        old = DOPR853(rhs)
        new = DOPR853(rhs, reuse_stage_workspace=True)
        states = [np.ones((2, 1)), np.ones((2, 1))]
        times = [np.zeros(1), np.zeros(1)]
        steps = [np.full(1, 10.), np.full(1, 10.)]
        rejected = False
        workspace_id = None
        for _ in range(20):
            flags = []
            for i, stepper in enumerate((old, new)):
                flags.append(stepper.take_step(times[i], steps[i], states[i], np.zeros((0, 1))))
            rejected |= not bool(flags[0][0])
            np.testing.assert_array_equal(flags[0], flags[1])
            for pair in (states, times, steps):
                np.testing.assert_array_equal(*pair)
            np.testing.assert_array_equal(old.errOldTemp, new.errOldTemp)
            np.testing.assert_array_equal(old.previousRejectTemp, new.previousRejectTemp)
            if workspace_id is not None:
                self.assertEqual(id(new._stage_workspace), workspace_id)
            workspace_id = id(new._stage_workspace)
            # All ten output arrays previously started zero, including unwritten
            # RHS rows. Poisoning the next workspace must not change that.
            new._stage_workspace.fill(999.)
        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
