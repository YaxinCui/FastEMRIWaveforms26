"""Public mode-domain checks with real base classes and a unique-label oracle."""

# 2026-09-11 14:57 CST (linux): Exercise indexing without the 5 GB table or
# unsafe invalid native amplitudes. No production mapping is used by the oracle.
import os
import unittest
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from few.amplitude.base import AmplitudeBase
from few.utils.baseclasses import SphericalHarmonic
from few.utils.modeselector import ModeSelector
from few.utils.ylm import GetYlms


class LabelAmplitude(AmplitudeBase, SphericalHarmonic):
    def __init__(self, backend):
        SphericalHarmonic.__init__(self, lmax=3, kmax=4, nmax=2, force_backend=backend)
        self.calls = 0

    def get_amplitudes(self, a, p, e, xI, specific_modes=None):
        self.calls += 1
        labels = self.xp.arange(self.num_teuk_modes, dtype=self.xp.float64)
        values = labels + 1 + 1j * (2 * labels + 3)
        return self.xp.tile(values[specific_modes], (len(p), 1))


def stored_index(ell, m, k, n):
    q, residue = 9 * 5, (k + 4) * 5 + n + 2
    return (
        (ell - 2) * q + residue
        if m == 0
        else 2 * q + (ell * (ell - 1) // 2 + m - 2) * q + residue
    )


def oracle(mode):
    ell, m, k, n = mode
    index = stored_index(ell, m, k, n) if m >= 0 else stored_index(ell, -m, -k, -n)
    value = index + 1 + 1j * (2 * index + 3)
    return value if m >= 0 else (1 if (ell + k) % 2 == 0 else -1) * value.conjugate()


class ModeDomainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("FEW_MODE_DOMAIN_TEST_BACKEND", "cpu")
        cls.amp = LabelAmplitude(cls.backend)
        cls.xp = cls.amp.xp
        cls.ylm = GetYlms(force_backend=cls.backend)
        # 2026-09-11 15:08 CST (linux): Report files actually imported; the same
        # test can run against a separate clean-master source overlay.
        paths = [Path(__file__).resolve()] + [
            Path(inspect.getsourcefile(obj)).resolve()
            for obj in (AmplitudeBase, SphericalHarmonic, ModeSelector)
        ]
        helper = importlib.util.find_spec("few.utils.modeindices")
        if helper is not None:
            paths.append(Path(helper.origin).resolve())
        native = Path(
            importlib.import_module(f"few_backend_{cls.backend}.pyinterp").__file__
        ).resolve()
        paths.append(native)
        print(
            "DOMAIN_META="
            + json.dumps(
                dict(
                    backend=cls.backend,
                    xp=cls.xp.__name__,
                    source_sha256={
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths
                    },
                )
            ),
            flush=True,
        )

    def setUp(self):
        self.amp.calls = 0

    def host(self, value):
        return value.get() if hasattr(value, "get") else np.asarray(value)

    def selector(self, selection=None, mapping=None):
        return ModeSelector(
            self.amp,
            self.ylm,
            mode_selection=selection,
            include_minus_mkn=False,
            modeinds_map=mapping,
            force_backend=self.backend,
        )

    def select(self, selector, selection=None):
        t = self.xp.asarray([0.0, 1.0])
        return selector(
            t,
            0.6,
            t + 10.0,
            t * 0 + 0.3,
            t * 0 + 1.0,
            1.1,
            0.7,
            mode_selection=selection,
        )

    def amplitude(self, selection):
        return self.amp(
            0.6,
            self.xp.asarray([10.0, 9.0]),
            self.xp.asarray([0.3, 0.2]),
            1.0,
            specific_modes=selection,
        )

    def test_alias_and_invalid_tuples_rejected_before_amplitude(self):
        invalid = [[(2, 2, 0, n)] for n in (-100, -5, -3, 3, 5, 100)]
        invalid += [[(2, 2, k, 0)] for k in (-100, -9, -5, 5, 9, 100)]
        invalid += [
            [(ell, m, 0, 0)] for ell, m in ((1, 0), (4, 2), (2, 3), (2, -3), (-1, 0))
        ]
        invalid += [
            [],
            [(2, 2, 0)],
            [(2, 2, 0, 0, 1)],
            [(2.0, 2.0, 0.0, 0.5)],
            [(2, 2, 0, 2**63)],
            [(2, 2, 0, float("inf"))],
            [(2, 2, 0, 0j)],
            [(2, True, 0, 0)],
            [(2, 2, np.bool_(False), 0)],
            [(2, 2, 0, True)],
            "bad",
        ]
        for request in invalid:
            for entry in ("constructor", "selector_call", "amplitude"):
                with self.subTest(request=request, entry=entry):
                    self.amp.calls = 0
                    with self.assertRaises(ValueError):
                        if entry == "constructor":
                            self.selector(request)
                        elif entry == "selector_call":
                            self.select(self.selector(), request)
                        else:
                            self.amplitude(request)
                    self.assertEqual(self.amp.calls, 0)

    def test_flat_indices_rejected_before_amplitude(self):
        limit = len(self.amp.l_arr)
        requests = [
            self.xp.asarray([index]) for index in (-1, -100, limit, limit + 1, 100000)
        ]
        requests += [
            self.xp.asarray(0),
            self.xp.asarray([2**63], dtype=self.xp.uint64),
            self.xp.asarray([[0, 1]]),
            self.xp.asarray([0.5]),
            self.xp.asarray([True]),
            self.xp.asarray([np.nan]),
            self.xp.asarray([], dtype=int),
        ]
        for request in requests:
            with self.subTest(request=self.host(request).tolist()):
                self.amp.calls = 0
                with self.assertRaises(ValueError):
                    self.amplitude(request)
                self.assertEqual(self.amp.calls, 0)

    def test_bad_map_entries_rejected(self):
        for bad in (-1, len(self.amp.l_arr), len(self.amp.l_arr) + 37):
            with self.subTest(bad=bad):
                mapping = self.amp.index_map_arr.copy()
                mapping[2, 2, 0, 0] = bad
                self.amp.calls = 0
                with self.assertRaises(ValueError):
                    self.select(self.selector(mapping=mapping), [(2, 2, 0, 0)])
                self.assertEqual(self.amp.calls, 0)

    def test_valid_signed_modes_and_parity(self):
        modes = [
            (ell, m, k, n)
            for ell in (2, 3)
            for m in (-ell, -1, 0, 1, ell)
            for k in (-4, -3, -1, 0, 1, 3, 4)
            for n in (-2, 0, 2)
        ]
        actual = self.amplitude(modes)
        for mode in modes:
            np.testing.assert_array_equal(
                self.host(actual[mode]), np.full(2, oracle(mode))
            )
        selected = self.select(self.selector(), modes)
        expected = np.asarray([oracle(mode) for mode in modes])
        np.testing.assert_array_equal(self.host(selected[0]), np.tile(expected, (2, 1)))
        np.testing.assert_array_equal(
            np.stack([self.host(x) for x in selected[2:]]).T, modes
        )

    def test_custom_map_coverage_and_native_holes(self):
        # 2026-09-11 15:18 CST (linux): A custom map can narrow native coverage,
        # not expand it; legacy custom-only adapters retain explicit ownership.
        for shape, request in (
            ((4, 7, 9, 7), [(2, 2, 0, 3)]),
            ((5, 9, 9, 5), [(4, 2, 0, 0)]),
            ((4, 7, 9, 3), [(2, 2, 0, 2)]),
            ((4, 7, 9, 4), [(2, 2, 0, 0)]),
        ):
            with self.subTest(shape=shape):
                mapping = self.xp.zeros(shape, dtype=self.xp.int64)
                with self.assertRaises(ValueError):
                    self.selector(request, mapping)
                with self.assertRaises(ValueError):
                    self.select(self.selector(mapping=mapping), request)
                self.assertEqual(self.amp.calls, 0)
        native = self.amp.index_map_arr
        custom = native.copy()
        try:
            self.amp.index_map_arr = native.copy()
            self.amp.index_map_arr[2, 2, 0, 0] = -1
            with self.assertRaises(ValueError):
                self.selector([(2, 2, 0, 0)], custom)
            self.assertEqual(self.amp.calls, 0)
            del self.amp.index_map_arr
            result = self.select(self.selector([(2, 2, 0, 0)], custom))
            np.testing.assert_array_equal(
                self.host(result[0]), np.full((2, 1), oracle((2, 2, 0, 0)))
            )
        finally:
            self.amp.index_map_arr = native

    def test_constructor_defaults_own_input(self):
        modes = [(2, 2, 0, 0), (3, 1, -3, 1)]
        expected = self.select(self.selector(modes))
        for request in (
            [list(row) for row in modes],
            np.asarray(modes),
            self.xp.asarray(modes),
        ):
            selector = self.selector(request)
            if isinstance(request, list):
                request[0][1] = 1
            else:
                request[0, 1] = 1
            actual = self.select(selector)
            for got, want in zip(actual, expected):
                np.testing.assert_array_equal(self.host(got), self.host(want))

    def test_flat_signed_ids_and_input_ownership(self):
        positive = [(2, 0, -4, -2), (2, 1, -4, 2), (3, 3, 4, -2)]
        negative = [(2, -1, 4, -2), (3, -3, -4, 2), (2, -1, 3, -2), (3, -3, 4, 2)]
        modes = positive + negative
        plus_count = (3 * 4 // 2 - 1) * 9 * 5
        ids = [
            stored_index(ell, abs(m), k, n) + (plus_count if m < 0 else 0)
            for ell, m, k, n in modes
        ]
        host_ids = np.asarray(ids, dtype=np.int64)
        host_ids.flags.writeable = False
        for request in (
            host_ids,
            self.xp.asarray(ids),
            self.xp.asarray(ids + ids)[::2],
        ):
            original = self.host(request).copy()
            # Non-contiguous request may have a different order; enumerate its
            # exact full-mode tuple independently from the known ID list.
            lookup = dict(zip(ids, modes))
            expected = np.asarray([oracle(lookup[int(index)]) for index in original])
            np.testing.assert_array_equal(
                self.host(self.amplitude(request)), np.tile(expected, (2, 1))
            )
            np.testing.assert_array_equal(self.host(request), original)

    def test_array_mode_selection_defaults_and_noncontiguous(self):
        modes = [(2, 0, 0, -2), (3, -2, 0, 2), (2, 1, 4, 0)]
        expected = self.select(self.selector(), modes)
        storage = np.asarray(modes + modes, dtype=np.int64)
        for array in (np.asarray(modes), self.xp.asarray(modes), storage[::2]):
            original = self.host(array).copy()
            reference = self.select(self.selector(), [tuple(row) for row in original])
            for entry in ("constructor", "call"):
                with self.subTest(entry=entry, backend=self.backend):
                    selector = (
                        self.selector(array)
                        if entry == "constructor"
                        else self.selector()
                    )
                    result = (
                        self.select(selector)
                        if entry == "constructor"
                        else self.select(selector, array)
                    )
                    for actual, wanted in zip(result, reference):
                        np.testing.assert_array_equal(
                            self.host(actual), self.host(wanted)
                        )
                    np.testing.assert_array_equal(self.host(array), original)
        self.assertEqual(expected[0].shape, (2, 3))

    def test_default_all_threshold_unchanged_and_duplicates_rejected(self):
        def default_selector(selection):
            return ModeSelector(
                self.amp,
                self.ylm,
                mode_selection=selection,
                include_minus_mkn=True,
                force_backend=self.backend,
            )

        all_modes = self.select(default_selector("all"))
        np.testing.assert_array_equal(
            self.host(all_modes[0]), self.host(self.amplitude(None))
        )
        # 2026-09-11 15:03 CST (linux): Non-list controls use supported default
        # include-minus=True; the old warning-only override is a separate bug.
        first = self.select(default_selector("threshold"))
        second = self.select(default_selector(None))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(self.host(a), self.host(b))
        with self.assertRaises(ValueError):
            self.select(self.selector(), [(2, 2, 0, 0), (2, 2, 0, 0)])


if __name__ == "__main__":
    unittest.main()
