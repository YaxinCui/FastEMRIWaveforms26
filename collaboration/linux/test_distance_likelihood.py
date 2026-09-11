"""Independent Decimal residual oracle and time-domain FFT normalization."""

# 2026-09-11 14:29 CST (linux): Predeclare numerical engineering gates. These
# tests are not physical-model, LISA-noise, prior or posterior acceptance.
import argparse
import hashlib
import io
import json
import unittest
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from distance_likelihood import (
    DistanceConditionalLikelihood,
    one_sided_weights,
    weighted_inner,
)

ROWS = []


def decimal_residual_oracle(distance0, distance, data, signal, weights):
    """Subtract two direct residual norms, not the implementation quadratic."""
    with localcontext() as ctx:
        ctx.prec = 100
        alpha = Decimal.from_float(float(distance0)) / Decimal.from_float(
            float(distance)
        )
        current, reference = Decimal(0), Decimal(0)
        for d, s, w in zip(
            data.flat, signal.flat, np.broadcast_to(weights, data.shape).flat
        ):
            weight = Decimal.from_float(float(w))
            for dr, sr in ((d.real, s.real), (d.imag, s.imag)):
                dr, sr = Decimal.from_float(float(dr)), Decimal.from_float(float(sr))
                current += weight * (dr - alpha * sr) ** 2
                reference += weight * (dr - sr) ** 2
        return float(-(current - reference) / 2)


class DistanceMathTest(unittest.TestCase):
    def test_decimal_matrix(self):
        rng = np.random.default_rng(93011)
        weights = np.linspace(0.4, 2.0, 10).reshape(2, 5)
        seed = rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5))
        for rho in (20.0, 100.0, 1000.0):
            signal = seed * (rho / np.sqrt(weighted_inner(seed, seed, weights)))
            gaussian = rng.normal(size=signal.shape) + 1j * rng.normal(
                size=signal.shape
            )
            orthogonal = gaussian - signal * (
                weighted_inner(gaussian, signal, weights)
                / weighted_inner(signal, signal, weights)
            )
            for residual_name, residual in {
                "zero": np.zeros_like(signal),
                "parallel": 0.03 * signal,
                "orthogonal": orthogonal,
                "gaussian": gaussian,
            }.items():
                data = signal + residual
                model = DistanceConditionalLikelihood.from_data(
                    1.0, data, signal, weights
                )
                distances = sorted(
                    set(
                        [1.0, np.nextafter(1.0, 0.0), np.nextafter(1.0, np.inf)]
                        + [
                            1.0 + sign * change
                            for sign in (-1, 1)
                            for change in (1e-14, 1e-10, 1e-6)
                        ]
                        + [1.0 / (1.0 + j / rho) for j in (-10, -3, -1, 0, 1, 3, 10)]
                        + [1e-6, 1e-3, 0.1, 10.0, 1e3, 1e6]
                    )
                )
                vector = model(np.asarray(distances))
                for index, distance in enumerate(distances):
                    with self.subTest(
                        rho=rho, residual=residual_name, distance=distance
                    ):
                        expected = decimal_residual_oracle(
                            1.0, distance, data, signal, weights
                        )
                        actual = model(distance)
                        self.assertEqual(actual, vector[index])
                        delta = (1.0 - distance) / distance
                        scale = (
                            abs(delta * model.residual_projection)
                            + 0.5 * delta * delta * model.signal_norm2
                        )
                        error = abs(actual - expected)
                        # 2026-09-11 14:29 CST (linux): Correct the initial
                        # harness's misclassification of D/D0=.1 at rho=1000
                        # as posterior-local (it is 9000 sigma in amplitude).
                        tail = abs(delta) * rho > 10.000000001
                        gate = 1e-12 * max(1.0, scale) if tail else 1e-8
                        relative = error / abs(expected) if expected else 0.0
                        ROWS.append(
                            dict(
                                kind="decimal",
                                rho=rho,
                                residual=residual_name,
                                distance=distance,
                                expected=expected,
                                actual=actual,
                                absolute_error=error,
                                relative_error=relative,
                                gate=gate,
                                posterior_local=not tail,
                                passes=error <= gate,
                            )
                        )
                        self.assertLessEqual(error, gate)
                        if (
                            residual_name == "zero"
                            and 0 < abs(distance - 1.0) <= 1e-6
                            and expected != 0
                        ):
                            self.assertNotEqual(actual, 0.0)
                            self.assertLessEqual(relative, 1e-12)

    def test_parseval_endpoints(self):
        rng = np.random.default_rng(93112)
        for n in (16, 17, 128, 129):
            dt, sigma = 15.0, 2.3
            t = np.arange(n)
            cases = {
                "dc": np.ones(n),
                "sine": np.sin(2 * np.pi * 3 * t / n),
                "cosine": np.cos(2 * np.pi * 3 * t / n),
                "last_bin": np.cos(2 * np.pi * (n // 2) * t / n),
                "random": rng.normal(size=n),
            }
            weights = one_sided_weights(
                np.full(n // 2 + 1, 2 * sigma * sigma * dt), dt, n
            )
            for name, x in cases.items():
                for cross in (False, True):
                    y = rng.normal(size=n) if cross else x
                    expected = float(np.dot(x, y) / sigma**2)
                    actual = weighted_inner(
                        dt * np.fft.rfft(x), dt * np.fft.rfft(y), weights
                    )
                    scale = float(np.linalg.norm(x) * np.linalg.norm(y) / sigma**2)
                    error = abs(actual - expected) / scale
                    self.assertLessEqual(error, 1e-12)
                    ROWS.append(
                        dict(
                            kind="parseval",
                            n=n,
                            case=name,
                            cross=cross,
                            normalized_error=error,
                            passes=True,
                        )
                    )

    def test_ownership_and_weight_mask(self):
        signal = np.asarray([1 + 2j, 4 - 1j, 2 + 3j])
        data = signal + 0.2
        weights = np.asarray([1.0, 0.0, 3.0])
        model = DistanceConditionalLikelihood.from_data(1.0, data, signal, weights)
        before = model(0.7)
        data[:] = 90
        signal[:] = 50
        weights[:] = 20
        self.assertEqual(model(0.7), before)
        psd = np.asarray([2.0, np.inf, 3.0])
        self.assertEqual(one_sided_weights(psd, 15.0, 4)[1], 0.0)
        self.assertEqual(weighted_inner([1e300], [1e300], [0.0]), 0.0)

    def test_invalid_inputs(self):
        for value in (0.0, -1.0, np.nan, np.inf, True, 1 + 0j, "1", [1.0]):
            with self.subTest(reference=value), self.assertRaises(ValueError):
                DistanceConditionalLikelihood.from_data(value, [1.0], [1.0], [1.0])
        model = DistanceConditionalLikelihood.from_data(1.0, [1.0], [1.0], [1.0])
        for value in (0.0, -1.0, np.nan, np.inf, True, 1 + 0j, "1"):
            with self.subTest(distance=value), self.assertRaises(ValueError):
                model(value)
        for value in (1 + 1j, True, [1.0], np.nan, np.inf):
            for name in ("residual_projection", "signal_norm2"):
                kw = dict(distance0=1.0, residual_projection=0.0, signal_norm2=1.0)
                kw[name] = value
                with (
                    self.subTest(coefficient=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    DistanceConditionalLikelihood(**kw)
        for psd in (
            [0.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [np.nan, 1.0, 1.0],
            [1.0, 1.0],
            [1j, 1.0, 1.0],
        ):
            with self.assertRaises(ValueError):
                one_sided_weights(psd, 1.0, 4)
        for n in (True, 1, 4.0, -4):
            with self.assertRaises(ValueError):
                one_sided_weights([1.0, 1.0, 1.0], 1.0, n)
        for first, second, weights in (
            ([np.nan], [1.0], [1.0]),
            ([1.0], [np.inf], [1.0]),
            ([1.0], [1.0], [-1.0]),
            ([1.0], [1.0], [np.inf]),
            ([], [], []),
            ([1.0, 2.0], [1.0], [1.0]),
        ):
            with self.assertRaises(ValueError):
                weighted_inner(first, second, weights)
        with self.assertRaises(FloatingPointError):
            weighted_inner([1e300], [1e300], [1.0])

    def test_zero_signal_and_underflow(self):
        model = DistanceConditionalLikelihood.from_data(1.0, [3.0], [0.0], [1.0])
        values = model([np.nextafter(0.0, 1.0), 1.0, 1e300])
        np.testing.assert_array_equal(values, np.zeros(3))
        # This nonzero signal cannot be represented by FP64 norm coefficients;
        # reject explicitly instead of silently reporting a zero likelihood.
        with self.assertRaises(FloatingPointError):
            DistanceConditionalLikelihood.from_data(1.0, [0.0], [1e-200], [1.0])

    def test_extreme_weights(self):
        if np.finfo(np.longdouble).tiny < np.finfo(np.float64).tiny:
            tiny = np.longdouble("1e-400")
            with self.assertRaises(FloatingPointError):
                DistanceConditionalLikelihood(1.0, 0.0, tiny)
            with self.assertRaises(FloatingPointError):
                weighted_inner([1.0], [1.0], np.asarray([tiny]))
        np.testing.assert_allclose(
            one_sided_weights([np.inf, 1e308], 1e-309, 2),
            [0.0, 10.0],
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            one_sided_weights(np.ones(3), 1e308, 4),
            [5e-309, 1e-308, 5e-309],
            rtol=1e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            one_sided_weights(np.full(2, 1e308), 1e-308, 2),
            [1.0, 1.0],
            rtol=1e-14,
            atol=0.0,
        )
        with self.assertRaises(FloatingPointError):
            weighted_inner([1e-200], [1e200], [1e-300])

    def test_nonunit_reference_distance(self):
        # 2026-09-11 14:40 CST (linux): Check unit independence against direct
        # Decimal residuals, including adjacent representable distances.
        signal = np.asarray([2.0 + 3j, -1.0 + 0.4j])
        data = signal + np.asarray([0.3 - 0.2j, 0.1 + 0.5j])
        weights = np.asarray([0.7, 1.2])
        d0 = 2.3
        model = DistanceConditionalLikelihood.from_data(d0, data, signal, weights)
        for distance in (
            d0,
            np.nextafter(d0, 0.0),
            np.nextafter(d0, np.inf),
            d0 * 0.5,
            d0 * 1.3,
            d0 * 2.0,
        ):
            expected = decimal_residual_oracle(d0, distance, data, signal, weights)
            error = abs(model(distance) - expected)
            self.assertLessEqual(error, 1e-12 * max(abs(expected), 1e-25))
            ROWS.append(
                dict(
                    kind="nonunit_decimal",
                    distance0=d0,
                    distance=distance,
                    expected=expected,
                    actual=model(distance),
                    absolute_error=error,
                    passes=True,
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DistanceMathTest)
    )
    root = Path(__file__).parent
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        numpy=np.__version__,
        decimal_digits=100,
        passes=result.wasSuccessful(),
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        rows=ROWS,
        log=log.getvalue(),
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("distance_likelihood.py", Path(__file__).name)
        },
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(log.getvalue(), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
