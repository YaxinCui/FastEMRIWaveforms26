"""Small regression for the waveform-to-CPU-trajectory array boundary."""

# 2026-09-11 12:49 CST (linux): AST isolation executes the actual boundary block
# without requiring a 5 GB table or accepting a duplicated helper as the oracle.
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from few.waveform.base import SphericalHarmonicWaveformBase


class BoundaryTest(unittest.TestCase):
    def test_actual_waveform_boundary_block(self):
        # 2026-09-11 13:00 CST (linux): Inspect the actually imported source so
        # this test also works after wheel installation, outside the checkout.
        source = Path(inspect.getsourcefile(SphericalHarmonicWaveformBase))
        tree = ast.parse(source.read_text())
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and ast.unparse(node.test) == "self.mode_selector.mode_selection != 'all'")
        class DeviceTimes:
            def __init__(self, host):
                self.host, self.copies = host, 0
            def get(self):
                self.copies += 1
                return self.host.copy()
            def __array__(self, *args, **kwargs):
                raise TypeError("implicit GPU-to-host conversion forbidden")
        times = np.array([0., 1., 4.])
        calls = []
        def derivative(t, order):
            self.assertIsInstance(t, np.ndarray)
            np.testing.assert_array_equal(t,times)
            self.assertEqual(order,1)
            calls.append(t)
            return np.tile(np.arange(6,dtype=float),(len(t),1))
        for uses_cupy in (False,True):
            obj = SimpleNamespace(backend=SimpleNamespace(uses_cupy=uses_cupy),
                mode_selector=SimpleNamespace(mode_selection="threshold"),
                inspiral_generator=SimpleNamespace(inspiral_generator=SimpleNamespace(eval_integrator_derivative_spline=derivative)))
            t_temp = DeviceTimes(times) if uses_cupy else times
            namespace = {"self":obj,"np":np,"t_temp":t_temp}
            program = ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[]))
            exec(compile(program,str(source),"exec"),namespace)
            self.assertEqual(namespace["freqs"].shape,(3,3))
            np.testing.assert_array_equal(namespace["freqs"],
                np.tile(np.arange(3,6)/(2*np.pi),(len(times),1)))
            if uses_cupy:
                self.assertEqual(t_temp.copies,1)
        self.assertEqual(len(calls),2)
        obj.mode_selector.mode_selection = "all"
        namespace = {"self":obj,"np":np,"t_temp":DeviceTimes(times)}
        exec(compile(program,str(source),"exec"),namespace)
        self.assertEqual(len(calls),2)
        self.assertEqual(namespace["t_temp"].copies,0)


if __name__ == "__main__":
    unittest.main()
