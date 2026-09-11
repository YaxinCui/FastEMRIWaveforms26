"""Adversarial checks of the local experiment's comparison harness."""

# 2026-09-11 13:51 CST (linux): Separate research-harness tests from installed
# package tests; reject mismatched shape/dtype, NaN/Inf and strain-scale errors.
import unittest
from types import SimpleNamespace

import numpy as np

from collaboration.linux.observer_scan_probe import compare, to_host


class PreparedProbeGuardTest(unittest.TestCase):
    def test_mixed_host_device_conversion(self):
        value = np.array([1.0, 2.0])
        np.testing.assert_array_equal(to_host(value), value)
        np.testing.assert_array_equal(
            to_host(SimpleNamespace(get=lambda: value)), value
        )

    def test_waveform_comparison_guards(self):
        reference = np.array([1e-22 + 2e-22j, -3e-22 + 1e-22j], dtype=np.complex128)
        self.assertTrue(compare(reference, reference.copy())["raw_bytes_equal"])
        for candidate in (
            reference.astype(np.complex64),
            reference[:1],
            reference * np.nan,
            reference * 1.01,
            reference * np.inf,
            reference[None, :],
        ):
            with self.assertRaises((ValueError, AssertionError)):
                compare(reference, candidate)
        zeros = np.zeros(2, dtype=np.complex128)
        compare(zeros, zeros)
        self.assertFalse(compare(zeros, -zeros)["raw_bytes_equal"])
        with self.assertRaises(ValueError):
            compare(reference[None, :], reference[None, :])
        with self.assertRaises(AssertionError):
            compare(zeros, reference)


if __name__ == "__main__":
    unittest.main()
