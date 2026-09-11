"""Small owned-snapshot tests requiring no high-memory amplitude table."""

# 2026-09-11 13:39 CST (linux): Test signed-mode reuse, offset allocations and
# device/stream guards independently of the full-table end-to-end probe.
import unittest
from types import SimpleNamespace

import numpy as np

from few.amplitude.base import AmplitudeBase
from few.tests.base import FewBackendTest
from few.utils.baseclasses import SphericalHarmonic
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms
from few.waveform.prepared import (
    PreparedKerrWaveform,
    _own_array,
    _SnapshotAmplitude,
    _SnapshotPSD,
    _check_threshold,
)


class SyntheticAmplitude(AmplitudeBase, SphericalHarmonic):
    def __init__(self, backend):
        SphericalHarmonic.__init__(self, lmax=3, nmax=2, force_backend=backend)
        index = self.xp.arange(4 * self.num_teuk_modes).reshape(4, self.num_teuk_modes)
        self.values = (1.0 + index / 17.0 + 1j * (2.0 - index / 31.0)).astype(
            self.xp.complex128
        )

    def get_amplitudes(self, a, p, e, xI, specific_modes=None):
        return self.values[:, specific_modes].copy()


class PreparedSnapshotTest(FewBackendTest):
    @classmethod
    def name(cls):
        return "Prepared observer snapshot"

    @classmethod
    def parallel_class(cls):
        return ModeSelector

    def host(self, value):
        return value.get() if self.backend.uses_cupy else np.asarray(value)

    def test_owning_offset_and_strided_arrays(self):
        xp = self.backend.xp
        source = xp.arange(48, dtype=xp.float64).reshape(6, 8)
        for view in (source[1:5], source[1:5, ::2]):
            owned = _own_array(xp, view)
            expected = self.host(view).copy()
            self.assertTrue(owned.flags.c_contiguous)
            if self.backend.uses_cupy:
                self.assertEqual(owned.data.ptr, owned.data.mem.ptr)
                self.assertNotEqual(owned.data.ptr, source.data.mem.ptr)
            else:
                self.assertFalse(np.shares_memory(source, owned))
                self.assertFalse(owned.flags.writeable)
                with self.assertRaises(ValueError):
                    owned[0, 0] = 0
            view[:] = -5
            np.testing.assert_array_equal(self.host(owned), expected)

    def test_signed_modes_and_selector_match(self):
        xp = self.backend.xp
        original = SyntheticAmplitude(self.backend)
        cache = _SnapshotAmplitude(original, original.values)
        # Include odd ell, m=0, positive and negative m/n cases. Do not just
        # compare the positive-mode bulk path with itself.
        modes = [(2, 0, 0, 1), (3, 2, 0, -1), (3, -2, 0, 1), (2, -1, 0, -2)]
        intrinsic = (0.6, xp.full(4, 12.0), xp.full(4, 0.3), xp.ones(4))
        expected = original(*intrinsic, specific_modes=modes)
        actual = cache(*intrinsic, specific_modes=modes)
        for mode in modes:
            np.testing.assert_array_equal(
                self.host(actual[mode]), self.host(expected[mode])
            )
        t = xp.arange(4, dtype=xp.float64)
        for selection in ("all", "threshold", modes):
            kwargs = {"mode_selection": selection}
            if isinstance(selection, list):
                kwargs["include_minus_mkn"] = False
            first = ModeSelector(
                original,
                GetYlms(force_backend=self.backend),
                force_backend=self.backend,
            )
            second = ModeSelector(
                cache, GetYlms(force_backend=self.backend), force_backend=self.backend
            )
            expected = first(t, *intrinsic, 1.1, 0.7, **kwargs)
            actual = second(t, *intrinsic, 1.1, 0.7, **kwargs)
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(self.host(left), self.host(right))
        before = self.host(cache._values).copy()
        original.values[:] = 0
        original.l_arr_no_mask[:] = 9
        actual[0][:] = 0
        np.testing.assert_array_equal(self.host(cache._values), before)
        self.assertLess(int(xp.max(cache.l_arr_no_mask)), 9)

    def test_invalid_topology(self):
        original = SyntheticAmplitude(self.backend)
        with self.assertRaises(ValueError):
            _SnapshotAmplitude(original, original.values[:, :-1])
        original.mode_indexes = original.mode_indexes[::-1].copy()
        with self.assertRaises(ValueError):
            _SnapshotAmplitude(original, original.values)

    def test_frozen_sensitivity(self):
        xp = self.backend.xp
        frequencies = xp.array([0.0, 0.01, 0.04, 0.03])
        state = {"values": xp.array([xp.inf, 2.0, 4.0, 3.0]), "calls": 0}

        def sensitivity(f):
            state["calls"] += 1
            return state["values"]

        cached = _SnapshotPSD(xp, frequencies, sensitivity)
        before = self.host(state["values"]).copy()
        state["values"][:] = 1.0
        result = cached(frequencies)
        np.testing.assert_array_equal(self.host(result), before)
        result[:] = 0
        np.testing.assert_array_equal(self.host(cached(frequencies)), before)
        self.assertEqual(state["calls"], 1)
        with self.assertRaises(ValueError):
            cached(frequencies[::-1])
        for value in (xp.nan, -xp.inf, 0.0, -1.0):
            with self.assertRaises(ValueError):
                _SnapshotPSD(xp, frequencies, lambda f: xp.full(f.shape, value))


class PreparedDeviceGuardTest(unittest.TestCase):
    def test_threshold_guard(self):
        for value in (None, 0, 1e-5, 0.9):
            _check_threshold(value)
        for value in (np.nan, np.inf, -np.inf, -1, 1, 2):
            with self.assertRaises(ValueError):
                _check_threshold(value)

    def test_mock_device_and_stream_guard(self):
        # No second GPU is needed to prove the guard rejects a device switch.
        current = {"device": 0, "stream": 0}
        fake = SimpleNamespace(
            backend=SimpleNamespace(uses_cuda=True),
            _device_id=0,
            xp=SimpleNamespace(
                cuda=SimpleNamespace(
                    runtime=SimpleNamespace(getDevice=lambda: current["device"]),
                    get_current_stream=lambda: SimpleNamespace(ptr=current["stream"]),
                )
            ),
        )
        PreparedKerrWaveform._check_stream(fake)
        current["device"] = 1
        with self.assertRaisesRegex(RuntimeError, "device"):
            PreparedKerrWaveform._check_stream(fake)
        current["device"], current["stream"] = 0, 73
        with self.assertRaisesRegex(NotImplementedError, "default stream"):
            PreparedKerrWaveform._check_stream(fake)


# 2026-09-11 13:51 CST (linux): Probe-specific tests live beside the probe,
# so the packaged FEW tests never depend on an unshipped collaboration folder.
if __name__ == "__main__":
    unittest.main()
