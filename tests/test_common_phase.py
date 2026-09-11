"""Independent synthetic oracles for opt-in, same-FP64 phase grouping."""

# 2026-09-11 12:13 CST (linux): Exercise both branches, true phase keys and
# interpolation boundaries; no external amplitude tables are required.
import os
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from few.summation.commonphase import aggregate_common_phase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.constants import YRSID_SI


def fixture(count=11, phase_offset=0.3):
    rng = np.random.default_rng(917)
    t = np.array([0., .75, 2.4, 6.7, 12.2, 21.3, 31.125])
    ell = np.array([2, 3, 4, 2, 3, 4, 2, 4, 5, 2, 3], dtype=np.int32)
    m = np.array([2, 2, 2, 0, 0, 0, -2, -2, 2, 2, 3], dtype=np.int32)
    k = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int32)
    n = np.array([1, 1, 1, 1, 1, -1, 3, 3, 1, -1, 1], dtype=np.int32)
    a = rng.normal(size=(t.size, count)) + 1j * rng.normal(size=(t.size, count))
    y = rng.normal(size=2 * count) + 1j * rng.normal(size=2 * count)
    if count:
        y[0] = 0
        y[count - 1] = 0
        y[count] = 0
    omega = np.array([.37, .11, .19])
    phase0 = np.array([phase_offset, -.2, .7])
    coeff = np.zeros((t.size - 1, 3, 8))
    coeff[:, :, 0] = t[:-1, None] * omega + phase0
    coeff[:, :, 1] = np.diff(t)[:, None] * omega
    return t, a, y, tuple(x[:count] for x in (ell, m, k, n)), coeff, omega, phase0


def oracle(t, a, y, modes, times, omega, phase0):
    ell, m, k, n = modes
    sampled = CubicSpline(t, a, axis=0)(times)
    phase = times[:, None] * omega + phase0
    phasor = np.exp(-1j * (phase @ np.stack((m, k, n))))
    positive = sampled * phasor * y[:len(m)]
    negative = sampled.conj() * phasor.conj() * y[len(m):] * (-1.)**ell * (m != 0)
    return (positive + negative).sum(axis=1)


class CommonPhaseTest(unittest.TestCase):
    def backends(self):
        # Explicit GPU selection must fail if unavailable rather than silently
        # skipping the very backend this branch is intended to validate.
        return os.environ.get("FEW_TEST_BACKENDS", "cpu").split(",")

    def test_grouped_native_and_independent_oracle(self):
        for backend in self.backends():
            for backwards in (False, True):
                with self.subTest(backend=backend, backwards=backwards):
                    t, a, y, modes, coeff, omega, phase0 = fixture()
                    old = InterpolatedModeSum(force_backend=backend, pad_output=True, odd_len=True)
                    new = InterpolatedModeSum(force_backend=backend, common_phase=True, pad_output=True, odd_len=True)
                    xp = new.xp
                    inputs = [xp.asarray(t), xp.asarray(a), xp.asarray(y), t, coeff,
                              *(xp.asarray(x) for x in modes)]
                    kwargs = dict(T=40/YRSID_SI, dt=.25, integrate_backwards=backwards)
                    ref = old(*inputs, **kwargs)
                    got = new(*inputs, **kwargs)
                    ref = ref.get() if xp is not np else ref
                    got = got.get() if xp is not np else got
                    offset = t[-1] % .25 if backwards else 0.
                    times = np.arange(new.num_pts) * .25 + offset
                    truth = oracle(t, a, y, modes, times, omega, phase0)
                    np.testing.assert_allclose(got[:new.num_pts], truth, atol=2e-12, rtol=2e-12)
                    np.testing.assert_allclose(got, ref, atol=2e-12, rtol=2e-12)
                    self.assertTrue(np.all(got[new.num_pts:] == 0))
                    self.assertEqual(new.common_phase_counts, (11, 7))
                    # Same object, different topology: no stale cached grouping.
                    changed = tuple(x[:1] for x in modes)
                    single_y = y[[0, len(modes[0])]]
                    single_y[:] = 1 + .5j
                    again = new(xp.asarray(t), xp.asarray(a[:, :1]), xp.asarray(single_y),
                                t, coeff, *(xp.asarray(x) for x in changed), **kwargs)
                    again = again.get() if xp is not np else again
                    np.testing.assert_allclose(again[:new.num_pts], oracle(t, a[:, :1], single_y, changed, times, omega, phase0), atol=2e-12, rtol=2e-12)
                    self.assertEqual(new.common_phase_counts, (1, 1))

    def test_large_phase_and_nondefault_stream(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                t, a, y, modes, coeff, _, _ = fixture(phase_offset=1e8)
                old = InterpolatedModeSum(force_backend=backend)
                new = InterpolatedModeSum(force_backend=backend, common_phase=True)
                xp = new.xp
                args = [xp.asarray(t), xp.asarray(a), xp.asarray(y), t, coeff,
                        *(xp.asarray(x) for x in modes)]
                ref = old(*args, T=31/YRSID_SI, dt=.25)
                if xp is np:
                    got = new(*args, T=31/YRSID_SI, dt=.25)
                else:
                    with xp.cuda.Stream(non_blocking=True) as stream:
                        # Actually produce fresh amplitudes on this stream.
                        args[1] = args[1] * 1.0
                        got = new(*args, T=31/YRSID_SI, dt=.25)
                        stream.synchronize()
                    got, ref = got.get(), ref.get()
                np.testing.assert_allclose(got, ref, atol=2e-12, rtol=2e-12)

    def test_empty_and_invalid_inputs(self):
        t, a, y, modes, _, _, _ = fixture(count=0)
        out, keys = aggregate_common_phase(a, y, *modes)
        self.assertEqual(out.shape, (0, len(t)))
        self.assertEqual(len(keys[0]), 0)
        with self.assertRaises(ValueError):
            aggregate_common_phase(np.zeros((3, 2)), np.ones(3), *(np.ones(2, dtype=int) for _ in range(4)))
        t, a, y, modes, _, _, _ = fixture()
        out_int, keys = aggregate_common_phase(a, y, *modes)
        out_float, _ = aggregate_common_phase(a, y, *(x.astype(float) for x in modes))
        np.testing.assert_array_equal(out_int, out_float)
        bad = modes[0].astype(float)
        bad[0] = 2.5
        with self.assertRaises(ValueError):
            aggregate_common_phase(a, y, bad, *modes[1:])

    def test_shared_memory_tile_boundaries(self):
        for backend in self.backends():
            for groups in (225, 226, 450, 451):
                with self.subTest(backend=backend, groups=groups):
                    t, _, _, _, coeff, omega, phase0 = fixture()
                    rng = np.random.default_rng(groups)
                    nm = 2 * groups
                    a = rng.normal(size=(len(t), nm)) + 1j * rng.normal(size=(len(t), nm))
                    y = rng.normal(size=2 * nm) + 1j * rng.normal(size=2 * nm)
                    modes = (np.repeat([2, 3], groups), np.full(nm, 2),
                             np.zeros(nm, dtype=int), np.tile(np.arange(groups), 2))
                    old = InterpolatedModeSum(force_backend=backend)
                    new = InterpolatedModeSum(force_backend=backend, common_phase=True)
                    xp = new.xp
                    args = [xp.asarray(t), xp.asarray(a), xp.asarray(y), t, coeff,
                            *(xp.asarray(x) for x in modes)]
                    ref = old(*args, T=31/YRSID_SI, dt=.25)
                    got = new(*args, T=31/YRSID_SI, dt=.25)
                    if xp is not np:
                        got, ref = got.get(), ref.get()
                    scale = np.max(np.abs(ref))
                    self.assertLess(np.max(np.abs(got-ref))/scale, 5e-13)
                    truth = oracle(t, a, y, modes, np.arange(new.num_pts)*.25, omega, phase0)
                    self.assertLess(np.max(np.abs(got-truth))/scale, 2e-11)
                    self.assertEqual(new.common_phase_counts, (nm, groups))


if __name__ == "__main__":
    unittest.main()
