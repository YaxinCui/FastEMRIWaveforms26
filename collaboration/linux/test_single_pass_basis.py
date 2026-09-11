"""Independent signed-mode oracle for the temporary native basis adapter."""

# 2026-09-11 17:30 CST (linux): Pure NumPy small tests. Actual native/SSB/TDI
# validation is separate; this oracle deliberately includes signed negative m.
import unittest

import numpy as np

from single_pass_basis import PairBasisSum, phase_coefficients, count_intrinsic


def individual(amplitude, ylm, l, m, n, phi, radial, offset):
    result = np.zeros(len(phi), dtype=np.complex128)
    count = len(m)
    for j in range(count):
        for t in range(len(phi)):
            z = amplitude[t, j] * np.exp(
                -1j * (m[j] * (phi[t] + offset) + n[j] * radial[t])
            )
            result[t] += ylm[j] * z
            if m[j] != 0:
                result[t] += (-1.0) ** int(l[j]) * ylm[count + j] * np.conj(z)
    return result


class Renderer:
    output_type = "td"

    def __call__(self, t, amps, ylm, phase_t, phases, l, m, k, n, *extra, **kwargs):
        return individual(amps, ylm, l, m, n, phases[:, 0], phases[:, 2], 0.0)


class Tests(unittest.TestCase):
    # 2026-09-11 17:38 CST (linux): Verify exact lookup restoration both with
    # a class descriptor and a preexisting instance override, including failure.
    def test_counter_exception_restoration(self):
        from types import SimpleNamespace

        class Amplitude:
            def get_amplitudes(self):
                return 7

        for override in (False, True):
            amp = Amplitude()
            if override:
                amp.get_amplitudes = lambda: 9
            previous = vars(amp).get("get_amplitudes")
            trajectory = lambda: 3
            inner = SimpleNamespace(
                inspiral_generator=trajectory, amplitude_generator=amp
            )
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with count_intrinsic(inner) as counts:
                    self.assertEqual(inner.inspiral_generator(), 3)
                    self.assertEqual(amp.get_amplitudes(), 9 if override else 7)
                    raise RuntimeError("intentional")
            self.assertIs(inner.inspiral_generator, trajectory)
            self.assertEqual(counts, dict(trajectory_calls=1, amplitude_calls=1))
            self.assertEqual("get_amplitudes" in vars(amp), override)
            if override:
                self.assertIs(amp.get_amplitudes, previous)

    def test_signed_pair_basis(self):
        rng = np.random.default_rng(15015)
        l = np.array([2, 3, 3, 4, 4, 3, 2])
        m = np.array([0, 1, -1, 2, -2, 0, 2])
        n = np.array([1, -2, 2, 0, 1, -1, -3])
        t = np.arange(9.0)
        amplitude = rng.normal(size=(9, 7)) + 1j * rng.normal(size=(9, 7))
        ylm = rng.normal(size=14) + 1j * rng.normal(size=14)
        ylm[[2, 7 + 3]] = 0
        phase = rng.normal(size=(9, 3))
        inputs = (t, amplitude, ylm, t, phase, l, m, np.zeros(7, dtype=int), n)
        saved = [v.copy() for v in inputs]
        adapter = PairBasisSum(Renderer())
        basis = adapter(*inputs)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(adapter.group_calls, 5)
        for offset in (-1.2, 0.0, 0.71, np.pi / 2, 2 * np.pi + 0.2):
            expected = individual(
                amplitude, ylm, l, m, n, phase[:, 0], phase[:, 2], offset
            )
            np.testing.assert_allclose(
                phase_coefficients(offset, 2) @ basis, expected, rtol=2e-14, atol=2e-14
            )
        for actual, expected in zip(inputs, saved):
            np.testing.assert_array_equal(actual, expected)
        old = basis.copy()
        amplitude[:] = 0
        np.testing.assert_array_equal(basis, old)

    def test_missing_and_zero_groups(self):
        for m in (np.array([0]), np.array([2])):
            n = np.array([1])
            l = np.array([2])
            t = np.arange(4.0)
            phase = np.column_stack((t * 0.3, t * 0, t * 0.7))
            amplitudes = np.ones((4, 1), dtype=np.complex128) * (2 + 3j)
            ylms = np.array([1 + 2j, 2 - 1j])
            basis = PairBasisSum(Renderer())(
                t, amplitudes, ylms, t, phase, l, m, l * 0, n
            )
            expected = individual(
                amplitudes, ylms, l, m, n, phase[:, 0], phase[:, 2], 0.6
            )
            np.testing.assert_allclose(
                phase_coefficients(0.6, int(max(m))) @ basis, expected, atol=1e-13
            )

    def test_pair_global_phasor_is_wrong(self):
        alpha = np.pi / 2
        correct = np.exp(-1j * alpha) + np.exp(1j * alpha)
        wrong = 2 * np.exp(-1j * alpha)
        self.assertGreater(abs(correct - wrong), 1.9)

    def test_guard(self):
        class Wrong:
            output_type = "fd"

        with self.assertRaises(NotImplementedError):
            PairBasisSum(Wrong())
        with self.assertRaises(ValueError):
            phase_coefficients(np.nan, 2)
        with self.assertRaises(FloatingPointError):
            phase_coefficients(1e308, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
