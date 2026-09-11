"""Record the original public API failure before the experimental repair."""

# 2026-09-11 21:49 CST (linux): Generate new local report only; no native-invalid
# pointers/dtypes are ever passed. Preserve original failure evidence.
from datetime import datetime
import hashlib
import inspect
import json
from zoneinfo import ZoneInfo

import numpy as np
from fastlisaresponse.response import pyResponseTDI

from contract_support import EXPECTED_SOURCE, HERE, identity, load, orbit, response


def main():
    output = HERE / "original_failure.json"
    if output.exists():
        raise FileExistsError(output)
    cls = load("original")
    report = dict(timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  terminal=False, passes=False, sources=[identity(__file__),
                  identity(HERE / "contract_support.py"), identity(inspect.getsourcefile(cls)),
                  identity(inspect.getsourcefile(pyResponseTDI))], cases=[])
    try:
        if any(x["sha256"] != EXPECTED_SOURCE for x in report["sources"][-2:]):
            raise RuntimeError("Original/installed source not byte-identical")
        geometry = orbit()
        y = np.zeros((6, 3500), dtype=np.float64)
        for initialized, explicit in ((False, True), (True, True), (False, False)):
            r = response(cls, geometry)
            if initialized:
                t = np.arange(3500) * 15.0
                r.get_projections(np.exp(0.001j * t), .7, .47, t0=18000.)
                good = np.stack(r.get_tdi_delays())
                report["healthy_implicit_sha256"] = hashlib.sha256(good.tobytes()).hexdigest()
            try:
                r.get_tdi_delays(y) if explicit else r.get_tdi_delays()
            except AttributeError as error:
                report["cases"].append(dict(initialized=initialized, explicit=explicit,
                                            exception=type(error).__name__, message=str(error)))
            else:
                raise AssertionError("Expected original missing-attribute failure")
        report["passes"] = len(report["cases"]) == 3
    finally:
        report["terminal"] = True
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
