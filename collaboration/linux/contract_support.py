"""Local source loading and evidence identities, not installed-package mutation."""

# 2026-09-11 21:49 CST (linux): Shared read-only fixtures for API experiment.
import hashlib
import importlib.util
from pathlib import Path
import sys

from lisatools.detector import ESAOrbits

HERE = Path(__file__).resolve().parent
EXPECTED_SOURCE = "7a40c9ffd992bf09b2e84d201ce5e9f5da0c2c19ea33efd821f7afa4d027af4a"


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(kind):
    name = f"fastlisaresponse._exp17_{kind}"
    spec = importlib.util.spec_from_file_location(name, HERE / kind / "response.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.pyResponseTDI


def orbit():
    value = ESAOrbits(force_backend="cpu")
    value.configure(linear_interp_setup=True)
    return value


def response(cls, geometry, n=3500, order=25, tdi="2nd generation", channel="XYZ", dt=15.0):
    return cls(1 / dt, n, order=order, tdi=tdi, tdi_chan=channel,
               orbits=geometry, force_backend="cpu")
