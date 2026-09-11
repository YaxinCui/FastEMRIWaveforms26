"""Fresh, source-bound master/candidate real-table contract and wave regression."""

# 2026-09-11 17:00 CST (linux): Read the existing registered H5 files without
# downloading/modifying them. Only write new Linux reports/archives. The oracle
# enumerates physical labels independently; it is not an independent Teukolsky
# solve. CUDA uses the same explicit host-time adapter on BOTH source variants.
import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.waveform import FastKerrEccentricEquatorialFlux

BASE = "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
DATA = {
    "ZNAmps_l10_m10_n55_DS2Outer.h5": "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834",
    "KerrEccEqFluxData.h5": "db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014",
}


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def identical(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda12x"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("master", "fixed"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    archive = args.output.with_suffix(".npz")
    if args.output.exists() or archive.exists():
        raise FileExistsError("Refuse to overwrite evidence")
    if (args.reference is None) != (args.variant == "master"):
        raise ValueError("Fixed run requires fresh same-backend master report")
    root = args.source_root.resolve()
    if Path(few.__file__).resolve() != root / "src/few/__init__.py":
        raise AssertionError("Unexpected source overlay")
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    backend = few.get_backend(args.backend)
    xp = backend.xp
    host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    arrays, comparisons, checks = {}, [], []
    report = dict(timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  terminal=False, passes=False, variant=args.variant, backend=args.backend,
                  base_commit=BASE, source_root=str(root), producer=identity(__file__),
                  python=platform.python_version(), numpy=np.__version__, xp_version=xp.__version__,
                  scope="Actual FP64 amplitudes and short source waveforms; API contract, not independent physics truth or speed",
                  cuda_host_time_adapter=backend.uses_cuda, checks=checks, comparisons=comparisons)
    baseline, ref = None, None

    def record(key, value):
        arr = host(value).copy()
        if not arr.size or not np.all(np.isfinite(arr)):
            raise AssertionError("Invalid output " + key)
        arrays[key] = arr
        if ref is not None:
            same = identical(arr, ref[key])
            allowed = key.startswith("direct_") and key.endswith(("_all", "_raw"))
            comparisons.append(dict(key=key, byte_equal=same, allowed_contract_change=allowed))
            if not same and not allowed:
                raise AssertionError("Unintended change " + key)
        return arr

    def check(label, actual, expected):
        same = bool(actual.shape == expected.shape and np.array_equal(actual, expected))
        checks.append(dict(label=label, passes=same,
                           max_abs=float(np.max(np.abs(actual - expected)))))
        return same

    try:
        if args.reference:
            baseline = json.loads(args.reference.read_text())
            if not baseline["terminal"] or not baseline["passes"] or baseline["variant"] != "master" or baseline["backend"] != args.backend:
                raise AssertionError("Invalid master report")
            if baseline["producer"] != report["producer"]:
                raise AssertionError("Producer differs")
            report["reference"] = identity(args.reference)
            if identity(baseline["archive"]["path"]) != baseline["archive"]:
                raise AssertionError("Master archive changed")
            ref = np.load(baseline["archive"]["path"], allow_pickle=False)
        model = FastKerrEccentricEquatorialFlux(force_backend=args.backend)
        if backend.uses_cuda:
            trajectory = model.inspiral_generator.inspiral_generator
            original = trajectory.eval_integrator_derivative_spline
            def derivative(times, *pos, **kw):
                return original(host(times), *pos, **kw)
            trajectory.eval_integrator_derivative_spline = derivative
        amp = model.amplitude_generator
        labels = [(l, m, 0, n) for l in range(2, 11) for m in range(l + 1) for n in range(-55, 56)]
        labels = [v for v in labels if v[1] == 0] + [v for v in labels if v[1] > 0]
        indices = {label: i for i, label in enumerate(labels)}
        layout = host(xp.stack((amp.l_arr, amp.m_arr, amp.k_arr, amp.n_arr))).T
        if not np.array_equal(layout[:len(labels)], np.asarray(labels)):
            raise AssertionError("Unexpected canonical mode ordering")
        signed = [tuple(map(int, row)) for row in layout]
        # All signed IDs, repeated and reordered, exercise negative-mode routes.
        ids = np.concatenate((np.arange(len(signed))[::-1], [0, len(signed) - 1, 0]))
        modes = [(2, 2, 0, 0), (3, 1, 0, 2), (3, -1, 0, -2),
                 (3, 0, 0, -1), (3, 0, 0, 1), (2, 0, 0, 0), (10, 10, 0, 55)]
        report["amplitude_parameters"] = dict(spins=[0.6, -0.6, 0.0, 0.999, -0.999],
                                              p=[12., 14., 40.], e=[0.2, 0.3, 0.4], modes=modes)
        p, e = xp.asarray([12., 14., 40.]), xp.asarray([0.2, 0.3, 0.4])
        for i, a in enumerate(report["amplitude_parameters"]["spins"]):
            canonical = record(f"canonical_{i}_all", amp(-a, p, e, 1.0))
            direct = record(f"direct_{i}_all", amp(a, p, e, -1.0))
            raw = record(f"direct_{i}_raw", amp(a, p, e, -1.0, specific_modes=xp.asarray(ids)))
            named = amp(a, p, e, -1.0, specific_modes=modes)
            named = record(f"direct_{i}_tuple", xp.stack([named[m] for m in modes], axis=1))
            # Coordinate rotation flips m only. The distinct negative-mode
            # conjugation identity flips BOTH m and n; k is zero here.
            def oracle(mode):
                l, m, k, n = mode
                if m > 0:
                    return canonical[:, indices[(l, m, k, -n)]].conj()
                return (-1)**l * canonical[:, indices[(l, -m, k, n)]]
            expected_all = np.stack([oracle(m) for m in labels], axis=1)
            expected_raw = np.stack([oracle(signed[j]) for j in ids], axis=1)
            expected_tuple = np.stack([oracle(m) for m in modes], axis=1)
            check(f"spin{i}_all_rotation", direct, expected_all)
            check(f"spin{i}_raw_rotation", raw, expected_raw)
            if not check(f"spin{i}_tuple_rotation", named, expected_tuple):
                raise AssertionError("Existing tuple convention disagrees with oracle")
        settings = dict(T=0.003, dt=15., Phi_phi0=0.3, Phi_theta0=0.4, Phi_r0=0.5)
        cases = [(0.6, 12., 1., None), (0.6, 12., -1., None),
                 (-0.6, 12., 1., None), (0., 12., -1., None),
                 (0.999, 12., 1., None), (0.6, 40., 1., None),
                 (0.6, 12., -1., [(2, 2, 0, 0), (3, 1, 0, 2)])]
        report["waveform_parameters"] = dict(settings=settings, cases=cases,
                                             fixed=[1e6, 10., 0.3, 1.1, 0.7, 1.])
        for i, (a, p0, xi, selection) in enumerate(cases):
            wave = model(1e6, 10., a, p0, 0.3, xi, 1.1, 0.7, dist=1.,
                         mode_selection=selection, **settings)
            record(f"wave_{i}", wave)
            record(f"wave_{i}_modes", xp.stack((model.ls, model.ms, model.ks, model.ns)))
        report["data"] = []
        for name, digest in DATA.items():
            entry = identity(few.get_file_manager().get_file(name))
            if entry["sha256"] != digest:
                raise AssertionError("Data identity differs")
            report["data"].append(entry)
        native = Path(importlib.import_module(f"few_backend_{args.backend}.pyinterp").__file__).parent
        report["native"] = [identity(p) for p in sorted(native.glob("*.so"))]
        # Authenticate every actually imported FEW Python file against pinned
        # master, allowing exactly one changed production file on the candidate.
        sources = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("few.") and path and path.endswith(".py"):
                relative = Path(path).resolve().relative_to(root)
                # 2026-09-11 17:06 CST (linux): setuptools-scm generates this
                # untracked metadata file. Record it, but do not git-show it.
                # Preserve the first failed provenance pilot as evidence.
                if str(relative) == "src/few/_version.py":
                    report["generated_version"] = identity(path)
                    continue
                expected = subprocess.check_output(["git", "show", BASE + ":" + str(relative)], cwd=root)
                entry = identity(path)
                entry["master_equal"] = entry["sha256"] == hashlib.sha256(expected).hexdigest()
                sources[str(relative)] = entry
        report["sources"] = sources
        changed = sorted(k for k, v in sources.items() if not v["master_equal"])
        if changed != ([] if args.variant == "master" else ["src/few/amplitude/base.py"]):
            raise AssertionError("Unexpected production modifications " + str(changed))
        if baseline and (report["native"] != baseline["native"] or report["data"] != baseline["data"]):
            raise AssertionError("Native/data A/B identity differs")
        if ref is not None and set(ref.files) != set(arrays):
            raise AssertionError("Reference output coverage differs")
        report["contract_passes"] = all(c["passes"] for c in checks)
        report["bug_reproduced"] = not report["contract_passes"]
        if report["contract_passes"] != (args.variant == "fixed"):
            raise AssertionError("Unexpected contract outcome")
        np.savez_compressed(archive, **arrays)
        report["archive"] = identity(archive)
        report["array_manifest"] = {k: dict(shape=v.shape, dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if ref is not None:
            ref.close()
        report["terminal"] = True
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(passes=report["passes"], backend=args.backend, variant=args.variant,
                              checks=len(checks), arrays=len(arrays))), flush=True)


if __name__ == "__main__":
    main()
