"""CPU native regression, explicit input contract and recoverable-failure tests."""

# 2026-09-11 22:06 CST (linux): Root-owned tests, ordinary CPU arrays only.
# The original explicit API is broken; arbitrary-link oracle uses its unchanged
# resident buffer/native path. Actual projections are tested separately too.
import ast
from datetime import datetime
import inspect
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from contract_support import HERE, identity, load, orbit, response

Original = load("original")
Candidate = load("candidate")


class Counted(Candidate):
    native_calls = 0
    fail_native = False

    @property
    def tdi_gen(self):
        native = super().tdi_gen

        def call(*args):
            self.native_calls += 1
            if args[4].dtype != np.dtype(np.float64) or not args[4].flags.c_contiguous:
                raise AssertionError("Native time pointer is not contiguous FP64")
            if self.fail_native:
                args[0][:] = 777.
                raise RuntimeError("injected recoverable native-wrapper exception")
            return native(*args)

        return call


def native_reference(r, y, cut):
    r.y_gw_flat = y.flatten()
    r.y_gw_length = r.num_pts
    r.nlinks = 6
    r.tdi_start_ind = cut
    return np.stack(r.get_tdi_delays())


def state(r):
    keys = ("y_gw_flat", "y_gw_length", "nlinks", "tdi_start_ind", "delayed_links_flat")
    return {k: (id(v), v.tobytes()) if isinstance(v, np.ndarray) else v
            for k in keys if (v := getattr(r, k, None)) is not None}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = orbit()

    def fresh(self, cls=Counted, n=257, **kwargs):
        return response(cls, self.geometry, n=n, **kwargs)

    def equal(self, left, right):
        left, right = np.asarray(left), np.asarray(right)
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(np.all(np.isfinite(left)))
        self.assertEqual(left.tobytes(), right.tobytes())

    def rejects(self, r, action, exception=(TypeError, ValueError, NotImplementedError)):
        before, count = state(r), r.native_calls
        with self.assertRaises(exception):
            action()
        self.assertEqual(state(r), before)
        self.assertEqual(r.native_calls, count)

    def test_single_method_patch_scope(self):
        stripped, outside = [], []
        for kind in ("original", "candidate"):
            source = (HERE / kind / "response.py").read_text()
            tree = ast.parse(source)
            classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "pyResponseTDI"]
            self.assertEqual(len(classes), 1)
            methods = [x for x in classes[0].body if isinstance(x, ast.FunctionDef) and x.name == "get_tdi_delays"]
            self.assertEqual(len(methods), 1)
            method = methods[0]
            classes[0].body.remove(method)
            stripped.append(ast.dump(tree, include_attributes=False))
            lines = source.splitlines(keepends=True)
            outside.append(("".join(lines[:method.lineno-1]), "".join(lines[method.end_lineno:])))
        self.assertEqual(*stripped)
        self.assertEqual(*outside)

    def test_fresh_initialization_and_crop(self):
        r = self.fresh()
        y = np.random.default_rng(17001).normal(size=(6, 257))
        self.rejects(r, lambda: r.get_tdi_delays())
        self.rejects(r, lambda: r.get_tdi_delays(y))
        for t0 in (True, np.bool_(False), "960", 1j, np.array(960.), -1., 0., 15.,
                   np.nan, np.inf, -np.inf, 1935., 1e308):
            with self.subTest(t0=repr(t0)):
                self.rejects(r, lambda: r.get_tdi_delays(y, t0=t0))
        out = r.get_tdi_delays(y, t0=960.)
        self.equal(out, native_reference(self.fresh(Original), y, 64))
        self.equal(r.y_gw, y)
        self.equal(r.get_tdi_delays(), out)
        self.rejects(r, lambda: r.get_tdi_delays(t0=960.))
        # Odd N: int(128.75)=128 leaves one sample, consistent floor semantics.
        out = r.get_tdi_delays(y, t0=128.75 * 15.)
        self.equal(out, native_reference(self.fresh(Original), y, 128))
        self.assertEqual(r.tdi_start_ind, 128)

    def test_channels_orders_topologies_against_original(self):
        rng = np.random.default_rng(17002)
        pulse = np.zeros((6, 257))
        pulse[np.arange(6), [64, 65, 81, 120, 190, 192]] = np.arange(1., 7.)
        signals = (np.zeros((6, 257)), pulse, rng.normal(size=(6, 257)))
        for order in (1, 2, 15, 25, 35, 40):
            for tdi in ("1st generation", "2nd generation"):
                for channel in ("XYZ", "AET", "AE"):
                    with self.subTest(order=order, tdi=tdi, channel=channel):
                        r = self.fresh(order=order, tdi=tdi, channel=channel)
                        ref = self.fresh(Original, order=order, tdi=tdi, channel=channel)
                        for y in signals:
                            truth = native_reference(ref, y, 64)
                            self.equal(r.get_tdi_delays(y, t0=960.), truth)
                            self.equal(r.get_tdi_delays(), truth)

    def test_projection_and_replacement_not_ignored(self):
        r = self.fresh(n=3500)
        ref = self.fresh(Original, n=3500)
        time = np.arange(3500) * 15.
        waves = (np.exp(.001j*time), .43*np.exp(.003j*time+.2j))
        saved = None
        for wave in waves:
            for obj in (r, ref):
                obj.get_projections(wave.copy(), .7, .47, t0=18000.)
            self.equal(r.y_gw, ref.y_gw)
            truth = np.stack(ref.get_tdi_delays())
            self.equal(r.get_tdi_delays(), truth)
            # Cache A, then explicitly inject DIFFERENT B without projection.
            if saved is not None:
                r.get_tdi_delays(saved[0], t0=18000.)
                out = r.get_tdi_delays(ref.y_gw.copy())
                self.equal(out, truth)
                self.assertFalse(np.array_equal(np.stack(out), saved[1]))
                self.equal(r.get_tdi_delays(), truth)
            saved = (ref.y_gw.copy(), truth.copy())

    def test_independent_single_link_order(self):
        time = np.arange(257)
        y = np.stack([(i + 1) * np.sin((i + 2)*time*.013) for i in range(6)])
        for base, rows in ((12, [0, 1, 2]), (13, [3, 5, 4])):
            r = self.fresh(tdi=[dict(link=base, links_for_delay=[], sign=1)])
            out = np.stack(r.get_tdi_delays(y, t0=960.))
            expected = np.zeros_like(out)
            # Native's A table at integer nodes is retained (may be 1 ulp off).
            expected[:, 64:-64] = r.A_in[0] * y[rows, 64:-64]
            self.equal(out, expected)

    def test_owned_strided_readonly_inputs_and_output_lifetime(self):
        r = self.fresh()
        rng = np.random.default_rng(17003)
        y = rng.normal(size=(6, 257))
        offset = np.empty((6, 516))
        offset[:, 1:515:2] = y
        readonly = y.copy()
        readonly.flags.writeable = False
        for value in (y.copy(), np.asfortranarray(y), offset[:, 1:515:2],
                      y[:, ::-1].copy()[:, ::-1], readonly):
            before = value.tobytes()
            result = r.get_tdi_delays(value, t0=960.)
            self.equal(result, native_reference(self.fresh(Original), y, 64))
            self.assertEqual(value.tobytes(), before)
            self.assertFalse(np.shares_memory(value, r.y_gw))
            self.assertTrue(np.shares_memory(result[0], r.XYZ))
            old = np.stack(result).copy()
            r.get_tdi_delays(.71*y, t0=960.)
            self.equal(result, old)
        external = y.copy()
        r.get_tdi_delays(external, t0=960.)
        external[:] = 0.
        self.equal(r.y_gw, y)
        r.y_gw[:] = 123.
        self.equal(external, np.zeros_like(y))

    def test_invalid_input_before_native_and_state_unchanged(self):
        r = self.fresh()
        y = np.zeros((6, 257))
        r.get_tdi_delays(y, t0=960.)
        masked = np.ma.array(y.copy(), mask=np.zeros_like(y, dtype=bool))
        masked.data[0, 64] = np.nan
        masked.mask[0, 64] = True
        invalid = [y.astype(np.float32), y.astype(np.complex128), y.astype(np.int64),
                   y.astype(object), y.astype(">f8"), y.tolist(), y.ravel(), y.T,
                   y[:5], y[:, :-1], np.array(y, ndmin=3), masked, np.matrix(y)]
        for value in (np.nan, np.inf, -np.inf):
            bad = y.copy()
            bad[2, 100] = value
            invalid.append(bad)
        class Alien:
            shape = (6, 257)
            dtype = np.dtype(np.float64)
        invalid.append(Alien())
        for bad in invalid:
            with self.subTest(dtype=getattr(bad, "dtype", None), shape=getattr(bad, "shape", None)):
                self.rejects(r, lambda: r.get_tdi_delays(bad))
        for attr, value in (("tdi_chan", "BAD"), ("num_pts", True), ("num_pts", 0),
                            ("num_pts", 2**28), ("order", 0), ("order", 41),
                            ("tdi_start_ind", 0), ("dt", np.nan)):
            previous = getattr(r, attr)
            setattr(r, attr, value)
            self.rejects(r, lambda: r.get_tdi_delays(y))
            setattr(r, attr, previous)

    def test_scalar_normalization_and_topology_gates(self):
        r = self.fresh()
        y = np.zeros((6, 257))
        r.dt, r.sampling_frequency = np.longdouble(15.), np.longdouble(1/15.)
        self.equal(r.get_tdi_delays(y, t0=960.), np.zeros((3, 257)))
        r.dt, r.sampling_frequency = np.float32(15.), np.float32(1/15.)
        self.rejects(r, lambda: r.get_tdi_delays(y, t0=960.))
        r.dt, r.sampling_frequency = 15., 1/15.
        for attr, value in (("channels", np.full_like(r.channels, 3)),
                            ("tdi_base_links", np.full_like(r.tdi_base_links, 11)),
                            ("tdi_operation_index", r.tdi_operation_index[::-1].copy()),
                            ("tdi_link_combinations", r.tdi_link_combinations[:-1]),
                            ("A_in", r.A_in[:-1]), ("deps", .01)):
            old = getattr(r, attr)
            setattr(r, attr, value)
            self.rejects(r, lambda: r.get_tdi_delays(y))
            setattr(r, attr, old)
        r = self.fresh(tdi=[dict(link=12, links_for_delay=[12]*65, sign=1)])
        self.rejects(r, lambda: r.get_tdi_delays(y, t0=960.))

    def test_python_native_and_conversion_failure_do_not_commit(self):
        r = self.fresh(channel="AET")
        y = np.random.default_rng(17004).normal(size=(6, 257))
        saved = np.stack(r.get_tdi_delays(y, t0=960.))
        before = state(r)
        r.fail_native = True
        with self.assertRaises(RuntimeError):
            r.get_tdi_delays(2*y, t0=990.)
        self.assertEqual(state(r), before)
        r.fail_native = False
        module = sys.modules[Candidate.__module__]
        with patch.object(module, "AET", side_effect=RuntimeError("injected conversion")):
            with self.assertRaises(RuntimeError):
                r.get_tdi_delays(2*y, t0=990.)
        self.assertEqual(state(r), before)
        with patch.object(module, "AET", return_value=(np.full(257, np.inf),)*3):
            with self.assertRaises(FloatingPointError):
                r.get_tdi_delays(2*y, t0=990.)
        self.assertEqual(state(r), before)
        self.equal(r.get_tdi_delays(), saved)

    def test_orbit_endpoint_and_failed_projection(self):
        r = self.fresh(n=3500)
        with self.assertRaises(ValueError):
            r.get_projections(np.zeros(3500, dtype=np.complex128), .7, .47, t0=0.)
        self.rejects(r, lambda: r.get_tdi_delays())
        self.rejects(r, lambda: r.get_tdi_delays(np.zeros((6, 3500))))
        # Replace only Python domain metadata to prove pre-native gate. This
        # is NOT a supported changed-orbit workflow or native rebind test.
        r = self.fresh()
        args = r.response_orbits.pycppdetector_args
        small = list(args)
        small[1] = 2
        r.response_orbits.pycppdetector_args = small
        self.rejects(r, lambda: r.get_tdi_delays(np.zeros((6, 257)), t0=960.))
        r.response_orbits.pycppdetector_args = args


def main():
    output = HERE / sys.argv[1]
    if output.exists():
        raise FileExistsError(output)
    capture = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    report = dict(timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  terminal=True, passes=result.wasSuccessful(), tests=result.testsRun,
                  optimized_python=bool(sys.flags.optimize), log=capture.getvalue(),
                  sources=[identity(__file__), identity(HERE / "contract_support.py"),
                           identity(inspect.getsourcefile(Original)), identity(inspect.getsourcefile(Candidate))])
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(capture.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
