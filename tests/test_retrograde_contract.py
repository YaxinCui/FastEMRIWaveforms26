"""Independent labelled amplitudes and finite spin-weighted harmonic oracles."""

# 2026-09-11 16:37 CST (linux): No H5, downloads or FEW numerical solvers.
# Backend arrays exercise real selection/conjugation, not a copied selector.
import cmath
import hashlib
import inspect
import json
import math
import os
import unittest
from pathlib import Path

import numpy as np

from few.amplitude.base import AmplitudeBase
from few.utils.baseclasses import KerrEccentricEquatorial, SphericalHarmonic


def labels(lmax=3, nmax=2):
    positive = [
        (ell, m, 0, n)
        for ell in range(2, lmax + 1)
        for m in range(ell + 1)
        for n in range(-nmax, nmax + 1)
    ]
    positive = [mode for mode in positive if mode[1] == 0] + [
        mode for mode in positive if mode[1] > 0
    ]
    return positive, positive + [(ell, -m, k, n) for ell, m, k, n in positive if m > 0]


def coefficient(ell, m, n, row):
    value = (10 * ell + 2 * m + 3 * n) + 1j * (7 * ell - 3 * m + 2 * n)
    if (ell, m, n) == (3, 1, 2):
        value = 2 + 3j
    if (ell, m, n) == (3, 1, -2):
        value = 5 + 7j
    return value + row * (1 + 2j)


def expected(mode, count, reverse=False):
    ell, m, k, n = mode
    wanted_m = -m if reverse else m
    result = np.asarray(
        [
            coefficient(ell, abs(wanted_m), -n if wanted_m < 0 else n, row)
            for row in range(count)
        ],
        dtype=np.complex128,
    )
    if wanted_m < 0:
        result = (-1) ** ell * result.conj()
    if reverse:
        result = (-1) ** ell * result
    return result


class LabelAmplitude(AmplitudeBase, KerrEccentricEquatorial):
    def __init__(self, backend):
        KerrEccentricEquatorial.__init__(self, lmax=3, nmax=2, force_backend=backend)
        self.positive, self.catalogue = labels()
        self.received = None

    def get_amplitudes(self, a, p, e, xI, *args, specific_modes=None, **kwargs):
        count = max(len(a), len(p), len(e), len(xI))
        self.received = (a.copy(), xI.copy(), args, kwargs)
        table = np.asarray(
            [
                [coefficient(ell, m, n, row) for ell, m, k, n in self.positive]
                for row in range(count)
            ],
            dtype=np.complex128,
        )
        return self.xp.asarray(table)[:, specific_modes]


class GenericLabelAmplitude(AmplitudeBase, SphericalHarmonic):
    # Not the known Kerr-equatorial capability, even with a kmax=0 truncation.
    def __init__(self, backend):
        SphericalHarmonic.__init__(self, lmax=3, nmax=2, force_backend=backend)
        self.positive, self.catalogue = labels()

    get_amplitudes = LabelAmplitude.get_amplitudes


def swsh(ell, m, theta, phi):
    """Independent finite Wigner-d sum for spin -2, used only at ell=2/3."""
    f = math.factorial
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    pref = math.sqrt(f(ell + 2) * f(ell - 2) * f(ell + m) * f(ell - m))
    terms = []
    for r in range(max(0, 2 - m), min(ell + 2, ell - m) + 1):
        denominator = f(ell + 2 - r) * f(r) * f(m - 2 + r) * f(ell - m - r)
        terms.append(
            (-1) ** (m - 2 + r)
            * pref
            * c ** (2 * ell + 2 - m - 2 * r)
            * s ** (m - 2 + 2 * r)
            / denominator
        )
    return (
        math.sqrt((2 * ell + 1) / (4 * math.pi))
        * math.fsum(terms)
        * cmath.exp(1j * m * phi)
    )


class RetrogradeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("FEW_RETROGRADE_TEST_BACKEND", "cpu")
        paths = [
            Path(__file__).resolve(),
            Path(inspect.getsourcefile(AmplitudeBase)),
            Path(inspect.getsourcefile(SphericalHarmonic)),
        ]
        print(
            "RETROGRADE_META="
            + json.dumps(
                dict(
                    backend=cls.backend,
                    sources={
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths
                    },
                )
            ),
            flush=True,
        )

    def setUp(self):
        self.amp = LabelAmplitude(self.backend)

    def host(self, value):
        return value.get() if hasattr(value, "get") else np.asarray(value)

    def equal(self, value, target):
        value = self.host(value)
        # 2026-09-11 16:39 CST (linux): Either side may be a device result;
        # do not mistake NumPy's rejected implicit CuPy conversion for a bug.
        target = self.host(target)
        self.assertEqual(value.dtype, np.complex128)
        self.assertTrue(np.all(np.isfinite(value)))
        np.testing.assert_array_equal(value, target)

    def call(self, selection=None, xi=-1.0, a=0.6, count=3):
        return self.amp(
            a,
            np.arange(count) + 12.0,
            np.full(count, 0.3),
            xi,
            specific_modes=selection,
        )

    def test_distinct_n_and_minus_n_counterexample(self):
        mode = (3, 1, 0, 2)
        index = self.amp.catalogue.index(mode)
        named = self.call([mode], count=1)[mode]
        self.equal(named, np.asarray([5 - 7j]))
        self.equal(self.call(self.amp.xp.asarray([index]), count=1)[:, 0], named)
        self.equal(self.call(count=1)[:, index], named)

    def test_default_all_matches_named_mode_and_independent_labels(self):
        actual = self.call()
        named = self.call(self.amp.positive)
        self.assertEqual(actual.shape, (3, len(self.amp.positive)))
        for index, mode in enumerate(self.amp.positive):
            with self.subTest(mode=mode):
                self.equal(actual[:, index], expected(mode, 3, True))
                self.equal(actual[:, index], named[mode])

    def test_raw_full_signed_catalogue_and_permutation_duplicates(self):
        count = len(self.amp.catalogue)
        for ids in (
            np.arange(count),
            np.asarray([count - 1, 35, 34, 0, 35, 1]),
            np.arange(count)[::-2],
        ):
            requested = self.amp.xp.asarray(ids)
            frozen = self.host(requested).copy()
            if self.backend == "cpu":
                requested.flags.writeable = False
            actual = self.call(requested)
            self.assertEqual(actual.shape, (3, len(ids)))
            np.testing.assert_array_equal(self.host(requested), frozen)
            for column, index in enumerate(ids):
                with self.subTest(index=int(index)):
                    self.equal(
                        actual[:, column], expected(self.amp.catalogue[index], 3, True)
                    )

    def test_tuple_prograde_and_canonical_contract_unchanged(self):
        ids = self.amp.xp.arange(len(self.amp.catalogue))
        for a in (-0.6, 0.0, 0.6):
            for xi in (-1.0, 1.0):
                named = self.call(self.amp.catalogue, xi, a)
                for mode in self.amp.catalogue:
                    self.equal(named[mode], expected(mode, 3, xi < 0))
            for selection in (None, ids):
                actual = self.call(selection, 1.0, a)
                modes = self.amp.positive if selection is None else self.amp.catalogue
                for column, mode in enumerate(modes):
                    self.equal(actual[:, column], expected(mode, 3))

    def test_scalar_arrays_state_and_forwarded_input(self):
        amp = self.amp
        for a in (-0.6, 0.0, 0.6):
            scalar = amp(a, 12.0, 0.3, -1.0)
            array = amp(
                np.asarray([a]),
                np.asarray([12.0]),
                np.asarray([0.3]),
                np.asarray([-1.0]),
            )
            self.equal(scalar, array)
            for index, mode in enumerate(amp.positive):
                self.equal(scalar[:, index], expected(mode, 1, True))
        first = self.host(self.call()).copy()
        self.call(amp.xp.asarray([0, 36, 59]), xi=1.0)
        self.call(amp.xp.asarray([33, 2]), xi=-1.0)
        self.equal(self.call(), first)
        amp(
            0.6,
            12.0,
            0.3,
            -1.0,
            "sentinel",
            specific_modes=amp.xp.asarray([3]),
            extra="keep",
        )
        self.assertEqual(self.host(amp.received[0]).tolist(), [0.6])
        self.assertEqual(self.host(amp.received[1]).tolist(), [-1.0])
        self.assertEqual(amp.received[2:], (("sentinel",), {"extra": "keep"}))

    def test_generic_provider_keeps_legacy_route(self):
        amp = GenericLabelAmplitude(self.backend)
        mode = (3, 1, 0, 2)
        index = amp.catalogue.index(mode)
        for xi in (-1.0, -0.5):
            raw = amp(0.6, 12.0, 0.3, xi, specific_modes=amp.xp.asarray([index]))
            self.equal(raw[:, 0], np.asarray([2 - 3j]))
            self.equal(amp(0.6, 12.0, 0.3, xi)[:, index], np.asarray([-5 - 7j]))

    def test_independent_harmonic_rotation(self):
        for theta, phi in ((1.1, 0.7), (0.4, -1.2), (2.4, 2.1)):
            for ell in (2, 3):
                for m in range(-ell, ell + 1):
                    self.assertLess(
                        abs(
                            swsh(ell, m, math.pi - theta, -phi)
                            - (-1) ** ell * swsh(ell, -m, theta, phi)
                        ),
                        1e-14,
                    )
            self.assertLess(
                abs(
                    swsh(2, 2, theta, phi)
                    - math.sqrt(5 / (64 * math.pi))
                    * (1 + math.cos(theta)) ** 2
                    * cmath.exp(2j * phi)
                ),
                1e-14,
            )
            self.assertLess(
                abs(
                    swsh(2, -2, theta, phi)
                    - math.sqrt(5 / (64 * math.pi))
                    * (1 - math.cos(theta)) ** 2
                    * cmath.exp(-2j * phi)
                ),
                1e-14,
            )

    def test_finite_mode_rotation_and_nonzero_initial_phase(self):
        modes = [
            (ell, m, n)
            for ell in (2, 3)
            for m in range(-ell, ell + 1)
            for n in (-1, 0, 1)
        ]
        canonical = {
            (ell, m, n): complex(ell + 2 * m + n, 3 * ell - m + 2 * n) / 17
            for ell, m, n in modes
        }
        direct = {
            (ell, m, n): (-1) ** ell * canonical[(ell, -m, n)] for ell, m, n in modes
        }

        def total(coeff, theta, phi, azimuth, radial):
            return sum(
                coeff[(ell, m, n)]
                * swsh(ell, m, theta, phi)
                * cmath.exp(-1j * (m * azimuth + n * radial))
                for ell, m, n in modes
            )

        for theta, phi, azimuth, radial in (
            (1.1, 0.7, 0.3, 0.5),
            (0.6, -0.4, 0.8, 0.2),
            (2.0, 1.0, -0.5, 0.9),
        ):
            h = total(direct, theta, phi, azimuth, radial)
            self.assertLess(
                abs(h - total(canonical, math.pi - theta, -phi, -azimuth, radial)),
                1e-13,
            )
            self.assertGreater(
                abs(h - total(canonical, math.pi - theta, -phi, azimuth, radial)), 1e-3
            )
            delta = 0.37
            shifted = {
                (ell, m, n): direct[(ell, m, n)] * cmath.exp(1j * m * delta)
                for ell, m, n in modes
            }
            self.assertLess(
                abs(
                    total(shifted, theta, phi, azimuth, radial)
                    - total(canonical, math.pi - theta, -phi, -azimuth + delta, radial)
                ),
                1e-13,
            )


if __name__ == "__main__":
    unittest.main()
