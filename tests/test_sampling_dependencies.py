"""Distribution names, not import names, belong in the sampling extra."""

# 2026-09-11 15:05 CST (linux): Pure metadata regression; no FEW imports,
# numerical backend, network or sampling installation required.
import tomllib
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


class SamplingDependencyTest(unittest.TestCase):
    def test_sampling_distribution_names(self):
        project = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        )["project"]
        requirements = [
            Requirement(entry) for entry in project["optional-dependencies"]["sampling"]
        ]
        names = {canonicalize_name(requirement.name) for requirement in requirements}
        self.assertTrue({"eryn", "fastlisaresponse", "lisaanalysistools"} <= names)
        self.assertNotIn(
            "lisatools",
            names,
            "PyPI lisatools is an unrelated portfolio tool sharing the import namespace",
        )


if __name__ == "__main__":
    unittest.main()
