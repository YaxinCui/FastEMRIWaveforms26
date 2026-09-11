"""Actual Kerr table regression, each public entry compared to its own master."""

# 2026-09-11 15:23 CST (linux): Reuse one table per process. Never send unsafe
# raw invalid IDs to native code. Baseline n=56 is a known in-array alias only.
import argparse
import hashlib
import importlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.amplitude.base import AmplitudeBase
from few.utils.modeselector import ModeSelector
from few.waveform import FastKerrEccentricEquatorialFlux
from few.waveform.base import SphericalHarmonicWaveformBase


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda12x"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--guarded", action="store_true")
    args = parser.parse_args()
    archive = args.output.with_suffix(".npz")
    if args.output.exists() or archive.exists():
        raise FileExistsError("Refuse to overwrite previous evidence")
    os.environ["FEW_FILE_EXTRA_PATHS"] = str(args.data_dir.resolve())
    few.get_config_setter().disable_file_download().set_file_integrity_check("once")
    report = dict(
        timestamp_cst=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        base_commit="47e4fea4bb3e5fbe1ba34e5e399ed37c97814191",
        backend=args.backend,
        guarded=args.guarded,
        passes=False,
        terminal=False,
        scope="Actual table amplitude/selector same-entry regression; CPU short source waveforms. Not independent physics or speed certification.",
        cuda_full_waveform="Not tested: independent master host/CuPy derivative boundary (exp04) is outside this branch",
        sources=[identity(__file__)]
        + [
            identity(inspect.getsourcefile(cls))
            for cls in (
                AmplitudeBase,
                ModeSelector,
                SphericalHarmonicWaveformBase,
                FastKerrEccentricEquatorialFlux,
            )
        ],
        arrays=[],
        compatibility=[],
        invalid_requests=[],
    )
    actual, reference = {}, None

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def same(left, right):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.tobytes() == right.tobytes()
        )

    def record(key, value):
        array = host(value).copy()
        if (
            not array.size
            or array.dtype.kind not in "iufc"
            or not np.all(np.isfinite(array))
        ):
            raise AssertionError(f"Nonfinite/empty/invalid output {key}")
        actual[key] = array
        row = dict(
            key=key,
            shape=array.shape,
            dtype=str(array.dtype),
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        )
        if reference is not None:
            if key not in reference.files or not same(array, reference[key]):
                raise AssertionError(
                    f"Valid-input output changed from clean master: {key}"
                )
            row["master_byte_equal"] = True
        report["arrays"].append(row)

    def compatible(label, value, wanted):
        if not same(host(value), host(wanted)):
            raise AssertionError(
                f"New array interface differs from legacy list: {label}"
            )
        report["compatibility"].append(dict(label=label, byte_equal=True))

    try:
        helper = importlib.util.find_spec("few.utils.modeindices")
        if (helper is not None) != args.guarded:
            raise AssertionError(
                "Actual source overlay does not match guarded/baseline flag"
            )
        if helper is not None:
            report["sources"].append(identity(helper.origin))
        # 2026-09-11 15:26 CST (linux): Backend top-level is a namespace package
        # with __file__=None; anchor identity in an actually imported extension.
        native = importlib.import_module(f"few_backend_{args.backend}.pyinterp")
        report["native"] = [
            identity(path) for path in sorted(Path(native.__file__).parent.glob("*.so"))
        ]
        if not report["native"]:
            raise AssertionError("No authenticated native backend")
        report["data"] = [
            identity(few.get_file_manager().get_file(name))
            for name in ("ZNAmps_l10_m10_n55_DS2Outer.h5", "KerrEccEqFluxData.h5")
        ]
        if args.reference:
            report["reference"] = identity(args.reference)
            reference = np.load(args.reference, allow_pickle=False)
            ref_report = json.loads(args.reference.with_suffix(".json").read_text())
            if (
                not ref_report["passes"]
                or not ref_report["terminal"]
                or ref_report["guarded"]
                or ref_report["backend"] != args.backend
            ):
                raise AssertionError(
                    "Reference is not a terminal successful same-backend baseline"
                )
            if (
                ref_report["data"] != report["data"]
                or ref_report["native"] != report["native"]
            ):
                raise AssertionError("Data/native identity changed between variants")
            if ref_report["archive"]["sha256"] != report["reference"]["sha256"]:
                raise AssertionError("Reference archive does not match its manifest")
        model = FastKerrEccentricEquatorialFlux(force_backend=args.backend)
        amp, xp = model.amplitude_generator, model.xp
        report["xp"] = xp.__name__
        if report["xp"] != ("numpy" if args.backend == "cpu" else "cupy"):
            raise AssertionError("Backend mismatch")
        report["sources"].append(identity(inspect.getsourcefile(type(amp))))
        modes = [
            (2, 2, 0, 0),
            (3, 1, 0, 2),
            (3, -1, 0, -2),
            (3, 1, 0, -2),
            (3, -1, 0, 2),
            (2, 0, 0, -1),
            (2, 0, 0, 1),
            (10, 10, 0, 55),
            (10, -10, 0, -55),
        ]
        ids = xp.asarray(
            [int(amp.index_map_arr[mode]) for mode in modes], dtype=xp.int64
        )
        report["modes"] = modes
        report["full_ids"] = host(ids).tolist()
        p, e = xp.asarray([12.0, 13.0, 14.0]), xp.asarray([0.3, 0.2, 0.1])
        t = xp.asarray([0.0, 15.0, 30.0])
        cases = (
            ("prograde", 0.6, 1.0),
            ("canonical_retrograde", -0.6, 1.0),
            ("direct_retrograde", 0.6, -1.0),
        )
        report["amplitude_cases"] = cases
        for label, a, x in cases:
            inputs = (a, p, e, x)
            record(f"{label}_all", amp(*inputs))
            result = amp(*inputs, specific_modes=modes)
            record(
                f"{label}_tuples", xp.stack([result[mode] for mode in modes], axis=1)
            )
            raw = amp(*inputs, specific_modes=ids)
            record(f"{label}_raw_ids", raw)
            selected = model.mode_selector(
                t,
                a,
                p,
                e,
                xp.full(3, x),
                1.1,
                0.7,
                mode_selection=modes,
                include_minus_mkn=False,
            )
            for i, value in enumerate(selected):
                record(f"{label}_selector_{i}", value)
            if args.guarded:
                compatible(
                    label + "_host_raw_ids", amp(*inputs, specific_modes=host(ids)), raw
                )
                array_result = model.mode_selector(
                    t,
                    a,
                    p,
                    e,
                    xp.full(3, x),
                    1.1,
                    0.7,
                    mode_selection=xp.asarray(modes),
                    include_minus_mkn=False,
                )
                for i, (value, wanted) in enumerate(zip(array_result, selected)):
                    compatible(f"{label}_selector_array_{i}", value, wanted)
            print(
                f"{label}: amplitude entries and selector outputs recorded", flush=True
            )
            save()
        invalid, legal = (2, 2, 0, 56), (2, 2, 0, -55)
        for entry in ("amplitude", "selector"):

            def request(mode):
                if entry == "amplitude":
                    return amp(0.6, p, e, 1.0, specific_modes=[mode])[mode]
                return model.mode_selector(
                    t,
                    0.6,
                    p,
                    e,
                    xp.ones(3),
                    1.1,
                    0.7,
                    mode_selection=[mode],
                    include_minus_mkn=False,
                )[0]

            if args.guarded:
                try:
                    request(invalid)
                except ValueError as exc:
                    report["invalid_requests"].append(
                        dict(entry=entry, rejected=True, error=str(exc))
                    )
                else:
                    raise AssertionError("Invalid n=56 was accepted")
            else:
                bad_value, legal_value = host(request(invalid)), host(request(legal))
                if not same(bad_value, legal_value) or int(
                    amp.index_map_arr[invalid]
                ) != int(amp.index_map_arr[legal]):
                    raise AssertionError(
                        "Expected safe in-array alias was not reproduced"
                    )
                report["invalid_requests"].append(
                    dict(
                        entry=entry,
                        rejected=False,
                        aliases=list(legal),
                        byte_equal=True,
                        max_amplitude=float(np.max(np.abs(bad_value))),
                    )
                )
        if args.backend == "cpu":
            settings = dict(T=0.0001, dt=15.0, Phi_phi0=0.3, Phi_theta0=0.4, Phi_r0=0.5)
            report["waveform_settings"] = settings
            positive = [(2, 2, 0, 0), (3, 1, 0, 2), (2, 0, 0, -1)]
            signed = [(2, 2, 0, 0), (3, -1, 0, 2), (2, 0, 0, -1)]
            stock = model.mode_selector
            for label, x in (("prograde", 1.0), ("retrograde", -1.0)):

                def waveform(selection, include=True):
                    return model(
                        1e6,
                        10.0,
                        0.6,
                        12.0,
                        0.3,
                        x,
                        1.1,
                        0.7,
                        dist=1.0,
                        mode_selection=selection,
                        include_minus_mkn=include,
                        **settings,
                    ).copy()

                for name, selection, include in (
                    ("threshold", "threshold", True),
                    ("all", "all", True),
                    ("positive", positive, True),
                    ("signed", signed, False),
                ):
                    model.mode_selector = stock
                    value = waveform(selection, include)
                    if value.dtype != np.complex128 or np.max(np.abs(value)) == 0:
                        raise AssertionError("Invalid actual waveform")
                    record(f"wave_{label}_{name}", value)
                    record(
                        f"wave_{label}_{name}_modes",
                        np.stack([model.ls, model.ms, model.ks, model.ns]),
                    )
                    if args.guarded and name in ("positive", "signed"):
                        compatible(
                            f"wave_{label}_{name}_array_call",
                            waveform(np.asarray(selection), include),
                            value,
                        )
                        model.mode_selector = ModeSelector(
                            amp,
                            model.ylm_gen,
                            mode_selection=np.asarray(selection),
                            include_minus_mkn=include,
                            force_backend=args.backend,
                        )
                        compatible(
                            f"wave_{label}_{name}_array_constructor_default",
                            waveform(None, include),
                            value,
                        )
                model.mode_selector = stock
                print(
                    f"{label}: actual CPU waveform list/default/array checks recorded",
                    flush=True,
                )
                save()
        if reference is not None and set(actual) != set(reference.files):
            raise AssertionError(
                "Reference and candidate valid-case inventories differ"
            )
        np.savez_compressed(archive, **actual)
        report["archive"] = identity(archive)
        report["passes"] = True
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if reference is not None:
            reference.close()
        report["terminal"] = True
        save()
        print(
            json.dumps(
                dict(passes=report["passes"], terminal=True, arrays=len(actual))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
