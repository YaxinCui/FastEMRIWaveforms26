"""Adversarial checks for the saved-waveform evidence gate."""

# 2026-09-11 13:00 CST (linux): Test the validator independently of GPU/data.
import unittest
import numpy as np
from compare_saved import compare_arrays


class ComparisonEvidenceTest(unittest.TestCase):
    def test_reject_invalid_inputs(self):
        reference = np.ones(4, dtype=np.complex128)
        for invalid in [np.full(4, np.nan, dtype=np.complex128),
                        np.full(4, np.inf, dtype=np.complex128),
                        reference.astype(np.complex64), reference[:2]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare_arrays(reference, invalid)
        with self.assertRaises(ValueError):
            compare_arrays(reference[:0], reference[:0])

    def test_zero_and_finite_gates(self):
        zero = np.zeros(4, dtype=np.complex128)
        self.assertTrue(compare_arrays(zero, zero)["passes"])
        negative_zero = -zero
        self.assertTrue(compare_arrays(zero, negative_zero)["passes"])
        self.assertFalse(compare_arrays(zero, negative_zero)["bitwise_equal"])
        self.assertFalse(compare_arrays(zero, zero + 1e-30)["passes"])
        reference = np.ones(4, dtype=np.complex128) * 1e-22
        self.assertTrue(compare_arrays(reference, reference)["passes"])
        self.assertFalse(compare_arrays(reference, reference * 1.001)["passes"])


if __name__ == "__main__":
    unittest.main()
