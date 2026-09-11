"""Independent finite-series oracles, not FEW physics or noise acceptance."""

# 2026-09-11 17:04 CST (linux): No file writes/H5/GPU. Arbitrary real response
# matrices simulate mixing, windows and covariance without assuming orthogonality.
import unittest

import numpy as np

from phase_likelihood import PhaseLikelihood


class Tests(unittest.TestCase):
    # 2026-09-11 17:14 CST (linux): Nonzero tiny-phase analytic sentinel and
    # compact payload ownership. This does not guarantee high-SNR stationary
    # points retain relative accuracy after Fourier-coefficient cancellation.
    def test_sine_sign_and_tiny_phase(self):
        grid = np.arange(3) * 2 * np.pi / 3
        p = PhaseLikelihood.prepare(np.sin(grid)[:, None], np.asarray([0.3])).compact()
        for phase in (0.2, -0.4, 1e-8, 1e-12):
            expected = 0.3 * np.sin(phase) - 0.5 * np.sin(phase) ** 2
            actual = float(p(phase))
            self.assertNotEqual(actual, 0.0)
            self.assertLess(abs(actual - expected), 2e-13 * abs(expected))

    def test_compact_storage(self):
        import weakref
        import gc

        grid = np.arange(7) * 2 * np.pi / 7
        p = PhaseLikelihood.prepare(
            np.cos(grid)[:, None] * np.ones((1, 100)), np.ones(100)
        )
        expected = p(0.3)
        basis_ref = weakref.ref(p.basis)
        compact = p.compact()
        del p
        gc.collect()
        self.assertIsNone(basis_ref())
        self.assertEqual(compact(0.3), expected)
        self.assertEqual(compact.cosine.nbytes + compact.sine.nbytes, 4 * 3 * 8)
        self.assertFalse(compact.cosine.flags.writeable)

    def test_independent_longdouble_series(self):
        rng = np.random.default_rng(14014)
        for m in (0, 1, 3, 10):
            a, b = rng.normal(size=(2, m + 1, 19))
            mixing = rng.normal(size=(19, 11))

            def signal(phase):
                phase = np.longdouble(phase)
                result = a[0].astype(np.longdouble).copy()
                for k in range(1, m + 1):
                    result += a[k] * np.cos(k * phase) + b[k] * np.sin(k * phase)
                return result @ mixing.astype(np.longdouble)

            grid = 2 * np.pi * np.arange(2 * m + 1) / (2 * m + 1)
            samples = np.asarray([signal(v) for v in grid], dtype=np.float64)
            data = samples[0] + rng.normal(size=11)
            prepared = PhaseLikelihood.prepare(samples, data)
            for phase in list(rng.uniform(-4 * np.pi, 4 * np.pi, 30)) + [
                0.0,
                1e-8,
                1e-12,
                2 * np.pi,
            ]:
                dh = signal(phase) - signal(0.0)
                residual = data.astype(np.longdouble) - signal(0.0)
                expected = float(np.sum(dh * residual) - 0.5 * np.sum(dh * dh))
                for actual in (
                    prepared(phase),
                    prepared.gram_evaluate(phase),
                    prepared.direct(phase),
                ):
                    self.assertLessEqual(
                        abs(actual - expected), 2e-10 * max(1.0, abs(expected))
                    )

    def test_cross_terms_necessary(self):
        grid = np.arange(7) * 2 * np.pi / 7
        samples = np.asarray(
            [np.cos(x) + 0.8 * np.sin(2 * x) + 0.4 * np.cos(3 * x) for x in grid]
        )[:, None]
        p = PhaseLikelihood.prepare(samples, samples[0] + 0.2)
        from phase_likelihood import deltas

        x = deltas(0.71, 3)
        wrong = x @ p.projection - 0.5 * np.sum(x * x * np.diag(p.gram))
        self.assertGreater(abs(wrong - p(0.71)), 0.05)

    def test_aliasing_negative_control(self):
        # Undersampling an m=3 signal with only five phases is invalid.
        grid = np.arange(5) * 2 * np.pi / 5
        samples = np.cos(3 * grid)[:, None]
        p = PhaseLikelihood.prepare(samples, samples[0])
        expected = -0.5 * (np.cos(3 * 0.71) - 1) ** 2
        self.assertGreater(abs(p(0.71) - expected), 0.1)

    def test_invalid_and_ownership(self):
        for samples, data in (
            (np.ones((2, 3)), np.ones(3)),
            (np.ones((3, 3)), np.ones(2)),
            (np.full((3, 3), np.nan), np.ones(3)),
            (np.ones((3, 3)) * 1j, np.ones(3)),
        ):
            with self.assertRaises(ValueError):
                PhaseLikelihood.prepare(samples, data)
        samples, data = np.arange(9.0).reshape(3, 3), np.ones(3)
        p = PhaseLikelihood.prepare(samples, data)
        result = p(0.3)
        samples[:] = data[:] = 99
        self.assertEqual(result, p(0.3))
        with self.assertRaises(ValueError):
            p(np.nan)
        self.assertEqual(p(0.0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
