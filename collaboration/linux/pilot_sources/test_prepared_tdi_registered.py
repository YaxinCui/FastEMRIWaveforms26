"""Independent stencil arithmetic and actual-native contract regression tests."""

# 2026-09-11 21:15 CST (linux): CPU-only tests; no FEW/data/environment writes.
import gc
import unittest
import weakref

import numpy as np
from fastlisaresponse.response import pyResponseTDI
from lisatools.detector import ESAOrbits

from prepared_tdi import PreparedTDI, make_response, interpolation_weights


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orbit = ESAOrbits(force_backend="cpu")
        cls.orbit.configure(linear_interp_setup=True)
        cls.response = cls.fresh()
        cls.plan = PreparedTDI(cls.response)

    @classmethod
    def fresh(cls, order=25, n=3500):
        r = make_response(
            1 / 15.0,
            n,
            order=order,
            tdi="2nd generation",
            tdi_chan="AET",
            orbits=cls.orbit,
            force_backend="cpu",
        )
        r.get_projections(np.zeros(n, dtype=np.complex128), 0.7, 0.47, t0=18000.0)
        return r

    def test_scalar_stencil(self):
        rng = np.random.default_rng(1626)
        y = rng.normal(size=500)
        for order in (15, 25, 35):
            r = self.fresh(order)
            x = np.array(
                [
                    100.0,
                    100.2,
                    100.999999,
                    100.000001,
                    np.nextafter(100.0, 0),
                    np.nextafter(100.0, 200),
                ]
            )
            ind, w = interpolation_weights(x, order, r.A_in, r.deps, r.E_in)
            for i, xx in enumerate(x):
                d = int(np.ceil(xx)) - 1
                e = 1.0 + d - xx
                k = int(e / r.deps)
                f = (e - k * r.deps) / r.deps
                aa = np.longdouble(r.A_in[k] * (1.0 - f) + r.A_in[k + 1] * f)
                ee = np.longdouble(e)
                acc = np.longdouble(0)
                for j in range(1, (order + 1) // 2):
                    acc += np.longdouble(r.E_in[j - 1]) * (
                        np.longdouble(y[d + 1 + j]) / (j + ee)
                        + np.longdouble(y[d - j]) / (j + 1 - ee)
                    )
                expected = aa * ((1 - ee) * y[d + 1] + ee * y[d] + ee * (1 - ee) * acc)
                self.assertLess(abs(float(w[i] @ y[ind[i]]) - float(expected)), 3e-14)
            self.assertEqual(float(w[0] @ y[ind[0]]), r.A_in[0] * y[100])

    def test_native_random_pulse_zero(self):
        r, p = self.response, self.plan
        rng = np.random.default_rng(1631)
        cases = [np.zeros((6, 3500)), rng.normal(size=(6, 3500))]
        impulse = np.zeros((6, 3500))
        impulse[np.arange(6), [1200, 1201, 1211, 1800, 2287, 2299]] = 1
        cases.append(impulse)
        for y in cases:
            before = y.tobytes()
            # 2026-09-11 21:17 CST (linux): Stock explicit-y API references a
            # missing attribute. Use the unchanged normal native buffer path.
            r.y_gw_flat = y.ravel().copy()
            r.y_gw_length = y.shape[1]
            r.get_tdi_delays()
            truth = r.XYZ[:, p.cut : -p.cut].copy()
            prediction = p.xyz(y)
            self.assertLess(
                np.linalg.norm(prediction - truth), 3e-13 * max(np.linalg.norm(y), 1)
            )
            self.assertEqual(y.tobytes(), before)
        y = cases[1]
        self.assertLess(
            np.linalg.norm(
                p(0.31 * y - 0.7 * impulse) - (0.31 * p(y) - 0.7 * p(impulse))
            ),
            1e-12,
        )

    def test_owned_plan_and_outputs(self):
        r = self.fresh()
        p = PreparedTDI(r)
        ref = weakref.ref(r)
        y = np.random.default_rng(1641).normal(size=(6, 3500))
        out = p(y)
        saved = out.copy()
        y[:] = 0
        self.assertTrue(np.array_equal(p(y), np.zeros_like(out)))
        self.assertTrue(np.array_equal(out, saved))
        self.assertFalse(np.shares_memory(out, p._matrix.data))
        del r
        gc.collect()
        self.assertIsNone(ref())
        self.assertTrue(np.array_equal(p(np.zeros((6, 3500))), np.zeros_like(out)))

    def test_unregistered_and_changed_config(self):
        raw = pyResponseTDI(
            1 / 15.0,
            3500,
            tdi="2nd generation",
            tdi_chan="AET",
            orbits=self.orbit,
            force_backend="cpu",
        )
        with self.assertRaises(ValueError):
            PreparedTDI(raw)
        for key, value in [
            ("dt", 16.0),
            ("order", 15),
            ("num_pts", 3499),
            ("tdi_chan", "XYZ"),
        ]:
            r = self.fresh()
            setattr(r, key, value)
            with self.assertRaises(ValueError):
                PreparedTDI(r)
        r = self.fresh()
        r.response_orbits = self.orbit
        with self.assertRaises(ValueError):
            PreparedTDI(r)
        r = self.fresh()
        r.tdi_operation_index = np.array([0, 1, 0, 1], dtype=np.int32)
        with self.assertRaises(ValueError):
            PreparedTDI(r)

    def test_mutated_geometry_and_tables(self):
        for name in ("geometry", "A_in", "tdi_signs"):
            r = self.fresh()
            a = (
                r.response_orbits.pycppdetector_args[3]
                if name == "geometry"
                else getattr(r, name)
            )
            a.flags.writeable = True
            a[0] += 1.0
            a.flags.writeable = False
            with self.assertRaises(ValueError):
                PreparedTDI(r)

    def test_domain_and_budget(self):
        for y in [
            np.zeros((6, 3499)),
            np.zeros((6, 3500), dtype=np.float32),
            np.full((6, 3500), np.nan),
            np.zeros((6, 3500), dtype=complex),
        ]:
            with self.assertRaises(ValueError):
                self.plan(y)
        with self.assertRaises(MemoryError):
            PreparedTDI(self.response, max_entries=1)
        with self.assertRaises(ValueError):
            PreparedTDI(self.response, max_entries=True)
        r = self.fresh()
        r.tdi_start_ind = 1
        with self.assertRaises(ValueError):
            PreparedTDI(r)
        for x in [[], [np.inf], [2**40], [[1.0]]]:
            with self.assertRaises(ValueError):
                interpolation_weights(x, 25, r.A_in, r.deps, r.E_in)
        with self.assertRaises(ValueError):
            interpolation_weights([100.2], 25, r.A_in, 0.0, r.E_in)


if __name__ == "__main__":
    unittest.main(verbosity=2)
