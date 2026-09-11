"""Small HDF5 + analytic tensor-spline oracle for opt-in spin residency."""

# 2026-09-11 15:42 CST (linux): Only generated tiny fixtures are written. The
# controlled mapping isolates interpolation/cache correctness from geodesics.
import gc
import hashlib
import inspect
import importlib
import json
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from few.amplitude._spincache import SpinSliceCache
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq


def polynomial(region, z, mode, w, u):
    return (
        100 * region
        + 10 * z
        + mode
        + 1
        + 2 * w
        + 3 * u
        + 4 * w * u
        + 1j * (-30 * region + 7 * z - mode / 2 + 5 * w - 2 * u - 3 * w * u)
    )


def controlled_map(a, p, e, x, **kwargs):
    # p carries u; e carries w. Region selection is an independent fixture rule.
    return p.copy(), e.copy(), np.ones_like(a), (a + 1) / 2, p < 0.5


class SpinCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("FEW_SPIN_CACHE_TEST_BACKEND", "cpu")
        cls.tmp = tempfile.TemporaryDirectory(prefix="few-exp12-tiny-")
        cls.path = Path(cls.tmp.name) / "tiny.h5"
        with h5py.File(cls.path, "w") as file:
            for region, middle in (("A", []), ("B", [0.5])):
                knots = np.asarray([0.0] * 4 + middle + [1.0] * 4)
                nb = len(knots) - 4
                greville = np.asarray(
                    [np.mean(knots[i + 1 : i + 4]) for i in range(nb)]
                )
                coeff = np.empty((5, 9, 2, nb * nb))
                for iz, z in enumerate(np.linspace(0, 1, 5)):
                    for mode in range(9):
                        value = polynomial(
                            1 if region == "A" else 2,
                            z,
                            mode,
                            greville[:, None],
                            greville[None, :],
                        ).ravel()
                        coeff[iz, mode, 0], coeff[iz, mode, 1] = value.real, value.imag
                group = file.create_group(f"region{region}")
                group.create_dataset(f"CoeffsRegion{region}", data=coeff)
                for axis in ("w", "u"):
                    group.create_dataset(f"{axis}_knots", data=knots)
                group.create_dataset("z_knots", data=np.linspace(0, 1, 5))
        cls.fm_patch = patch(
            "few.get_file_manager",
            return_value=SimpleNamespace(get_file=lambda _: cls.path),
        )
        cls.map_patch = patch(
            "few.amplitude.ampinterp2d.kerrecceq_forward_map", controlled_map
        )
        cls.fm_patch.start()
        cls.map_patch.start()
        paths = [
            Path(__file__).resolve(),
            Path(inspect.getsourcefile(AmpInterpKerrEccEq)),
            Path(inspect.getsourcefile(SpinSliceCache)),
        ]
        paths.append(
            Path(
                importlib.import_module(
                    f"few_backend_{cls.backend}.pyAmpInterp2D"
                ).__file__
            )
        )
        print(
            "SPIN_CACHE_META="
            + json.dumps(
                dict(
                    backend=cls.backend,
                    sources={
                        str(path.resolve()): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in paths
                    },
                )
            ),
            flush=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.map_patch.stop()
        cls.fm_patch.stop()
        cls.tmp.cleanup()

    def build(self, cache=2, stride=1):
        return AmpInterpKerrEccEq(
            filename="tiny.h5",
            lmax=2,
            nmax=1,
            force_backend=self.backend,
            spin_cache_size=cache,
            downsample_Z=stride,
        )

    def call(self, amp, z, regions="mixed", ids=None, xi=1):
        p = np.asarray(
            {"A": [0.1, 0.3], "B": [0.7, 0.9], "mixed": [0.1, 0.7, 0.3, 0.9]}[regions]
        )
        w = np.linspace(0.2, 0.7, len(p))
        raw = None if ids is None else amp.xp.asarray(ids, dtype=amp.xp.int64)
        value = amp(
            (2 * z - 1) / xi,
            amp.xp.asarray(p),
            amp.xp.asarray(w),
            xi,
            specific_modes=raw,
        )
        host = value.get() if hasattr(value, "get") else np.asarray(value)
        if xi == 1:
            wanted = polynomial(
                np.where(p < 0.5, 1, 2)[:, None],
                z,
                np.arange(9)[None, :] if ids is None else np.asarray(ids)[None, :],
                w[:, None],
                p[:, None],
            )
            np.testing.assert_allclose(host, wanted, rtol=2e-15, atol=2e-13)
        return host.copy()

    def test_no_coefficient_reads_at_construction_and_one_two_four_slices(self):
        accesses = []
        original = h5py.Dataset.__getitem__

        def read(dataset, key):
            if "Coeffs" in dataset.name:
                accesses.append((dataset.name, key))
            return original(dataset, key)

        with patch.object(h5py.Dataset, "__getitem__", read):
            amp = self.build()
            self.assertEqual(accesses, [])
            self.call(amp, 0.5, "A")
            self.assertEqual(len(accesses), 1)
            self.assertEqual(amp.spin_cache_info()["B"]["loads"], 0)
            amp.clear_spin_cache()
            accesses.clear()
            self.call(amp, 0.375, "mixed")
            self.assertEqual(len(accesses), 4)
            self.call(amp, 0.4, "mixed", [0, 4, 8, 4])
            self.assertEqual(len(accesses), 4)
            self.assertTrue(
                all(
                    key != () and isinstance(key, tuple) and isinstance(key[0], int)
                    for _, key in accesses
                )
            )
            self.assertEqual(
                [row["resident_indices"] for row in amp.spin_cache_info().values()],
                [[2, 1], [2, 1]],
            )

    def test_analytic_oracle_and_eager_bytes(self):
        for stride in (1, 2, 3):
            lazy, eager = self.build(stride=stride), self.build(None, stride)
            valid_z = (
                [0.0, 0.125, 0.5, 0.7]
                if stride == 3
                else [0.0, 0.125, 0.25, 0.375, 0.5, 0.875, 1.0]
            )
            for z in valid_z:
                for region in ("A", "B", "mixed"):
                    for ids in (None, [8, 0, 4, 4]):
                        with self.subTest(stride=stride, z=z, region=region, ids=ids):
                            a, b = (
                                self.call(lazy, z, region, ids),
                                self.call(eager, z, region, ids),
                            )
                            self.assertEqual(a.tobytes(), b.tobytes())
            if stride == 3:
                # Master does not automatically re-add the discarded last knot.
                for obj in (lazy, eager):
                    with self.assertRaises(IndexError):
                        self.call(obj, 1.0)

    def test_lru_changes_spin_without_stale_outputs(self):
        amp = self.build()
        first = self.call(amp, 0.125)
        frozen = first.copy()
        self.call(amp, 0.375)
        self.assertEqual(
            [row["loads"] for row in amp.spin_cache_info().values()], [3, 3]
        )
        self.call(amp, 0.875)
        self.assertEqual(
            [row["loads"] for row in amp.spin_cache_info().values()], [5, 5]
        )
        replay = self.call(amp, 0.125)
        self.assertEqual(
            [row["loads"] for row in amp.spin_cache_info().values()], [7, 7]
        )
        self.assertEqual(first.tobytes(), frozen.tobytes())
        self.assertEqual(replay.tobytes(), frozen.tobytes())
        self.assertTrue(
            all(
                len(row["resident_indices"]) <= 2
                for row in amp.spin_cache_info().values()
            )
        )

    def test_clear_and_parent_release_ownership(self):
        amp = self.build()
        self.call(amp, 0.375)
        holder = amp.spin_information_holder_A
        node = holder[1]
        weak_node, weak_coeff = weakref.ref(node), weakref.ref(node.coeff)
        if self.backend == "cpu":
            self.assertTrue(node.coeff.flags.owndata)
            self.assertIsNone(node.coeff.base)
        del node
        amp.clear_spin_cache()
        gc.collect()
        self.assertIsNone(weak_node())
        self.assertIsNone(weak_coeff())
        self.call(amp, 0.375)
        weak_parent, weak_holder = weakref.ref(amp), weakref.ref(holder)
        del holder, amp
        gc.collect()
        self.assertIsNone(weak_parent())
        self.assertIsNone(weak_holder())

    def test_changed_file_and_failed_load_are_not_cached(self):
        amp = self.build()
        holder = amp.spin_information_holder_A
        original_factory = holder.factory
        with patch.object(
            holder, "factory", side_effect=RuntimeError("fixture factory failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture factory failure"):
                holder[0]
        self.assertEqual(holder.cache_info()["loads"], 0)
        self.assertIs(holder.factory, original_factory)
        holder[0]
        self.assertEqual(holder.cache_info()["loads"], 1)
        with patch.object(holder, "_stat", return_value=(0, 0, 0, 0)):
            with self.assertRaisesRegex(RuntimeError, "table changed"):
                holder[1]
        self.assertEqual(holder.cache_info()["resident_indices"], [0])

    def test_parameter_validation_and_downsample_physical_indices(self):
        for invalid in (0, 1, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.build(invalid)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.build(stride=invalid)
        amp = self.build(stride=2)
        self.call(amp, 0.375)
        self.assertEqual(
            [row["resident_indices"] for row in amp.spin_cache_info().values()],
            [[2, 0], [2, 0]],
        )
        for bad in (5, -6):
            with self.assertRaises(IndexError):
                amp.spin_information_holder_A[bad]
        eager = self.build(None)
        self.assertIsNone(eager.spin_cache_info())
        eager.clear_spin_cache()

    def test_metadata_safety_and_initialization_identity(self):
        original = h5py.Dataset.__getitem__
        for malformed in ("float32", "nan", "order", "short"):

            def read(dataset, key):
                value = original(dataset, key)
                if dataset.name == "/regionA/w_knots":
                    if malformed == "float32":
                        value = value.astype(np.float32)
                    elif malformed == "nan":
                        value[0] = np.nan
                    elif malformed == "order":
                        value = value[::-1].copy()
                    else:
                        value = value[:6]
                if "Coeffs" in dataset.name:
                    self.fail("Invalid metadata must fail before coefficient loading")
                return value

            with (
                self.subTest(malformed=malformed),
                patch.object(h5py.Dataset, "__getitem__", read),
            ):
                with self.assertRaisesRegex(ValueError, "spline knots"):
                    self.build()
        with patch(
            "few.amplitude._spincache.file_identity",
            side_effect=[(1, 2, 3, 4), (1, 2, 3, 5)],
        ):
            with self.assertRaisesRegex(RuntimeError, "initialization"):
                self.build()

    def test_nondefault_stream_and_device_guard(self):
        if self.backend == "cpu":
            # The GPU-only condition is explicit; the CPU test still checks
            # repeated use of outputs after switching the working set.
            amp = self.build()
            expected = self.call(amp, 0.375)
            self.call(amp, 0.875)
            np.testing.assert_allclose(expected, self.call(amp, 0.375), rtol=0, atol=0)
            return
        amp = self.build()
        xp = amp.xp
        reference = self.call(amp, 0.375)
        stream = xp.cuda.Stream(non_blocking=True)
        # 2026-09-11 16:10 CST (linux): Keep results on device until AFTER
        # delayed consumption and same-sized null-stream allocation pressure.
        # self.call() fetches to the host, so it cannot test this lifetime.
        delayed_copy = xp.RawKernel(
            r"""
            extern "C" __global__ void delayed_copy(
                const double2* input, double2* output, int count) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned long long start = clock64();
                while (clock64() - start < 20000000ULL) {}
                if (i < count) output[i] = input[i];
            }
            """,
            "delayed_copy",
        )
        delayed_copy.compile()
        p_host = np.asarray([0.1, 0.7, 0.3, 0.9])
        w_host = np.linspace(0.2, 0.7, 4)
        p, w = xp.asarray(p_host), xp.asarray(w_host)
        mask, ids = p < 0.5, xp.arange(9)
        xp.cuda.Stream.null.synchronize()
        for index in (2, 4, 0, 2):
            with stream:
                result = amp._evaluate_interpolant_at_index(index, mask, w, p, ids)
                observed = xp.empty_like(result)
                delayed_copy((1,), (64,), (result, observed, np.int32(result.size)))
                # The queued native kernel does not retain the Python input.
                del result
            with xp.cuda.Stream.null:
                pressure = [xp.full((4, 9), -999.0 + 777.0j) for _ in range(32)]
            stream.synchronize()
            xp.cuda.Stream.null.synchronize()
            expected = polynomial(
                np.where(p_host < 0.5, 1, 2)[:, None],
                index / 4,
                np.arange(9)[None, :],
                w_host[:, None],
                p_host[:, None],
            )
            np.testing.assert_allclose(observed.get(), expected, rtol=2e-15, atol=2e-13)
            del pressure, observed
        for z in (0.375, 0.875, 0.125, 0.375):
            with stream:
                result = amp(2 * z - 1, p, w, 1.0)
                observed = xp.empty_like(result)
                delayed_copy((1,), (64,), (result, observed, np.int32(result.size)))
                del result
            with xp.cuda.Stream.null:
                pressure = [xp.full((4, 9), -999.0 + 777.0j) for _ in range(32)]
            stream.synchronize()
            xp.cuda.Stream.null.synchronize()
            result_host = observed.get()
            expected = polynomial(
                np.where(p_host < 0.5, 1, 2)[:, None],
                z,
                np.arange(9)[None, :],
                w_host[:, None],
                p_host[:, None],
            )
            np.testing.assert_allclose(result_host, expected, rtol=2e-15, atol=2e-13)
            del pressure, observed
        self.assertEqual(result_host.tobytes(), reference.tobytes())
        with patch.object(
            xp.cuda.runtime, "getDevice", return_value=amp._spin_cache_device + 1
        ):
            with self.assertRaisesRegex(ValueError, "construction CUDA device"):
                self.call(amp, 0.375)


if __name__ == "__main__":
    unittest.main()
