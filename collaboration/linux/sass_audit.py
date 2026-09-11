"""Compare sm75 legacy kernel instructions, not whole-program semantics."""

# 2026-09-11 13:07 CST (linux): Keep machine-code identity evidence separate
# from waveform accuracy and measured performance. Reject empty extractions.
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SYMBOLS = {
    "reference": "_Z13make_waveformPN6gcmplx7complexIdEEPdS3_PiS4_S4_iS2_ddiiiidd",
    "candidate": "_Z13make_waveformILb0EEvPN6gcmplx7complexIdEEPdS4_PiS5_S5_iS3_ddiiiidd",
}


def extract(tool, path, symbol):
    result = subprocess.run([str(tool), "--dump-sass", "--function", symbol, str(path)],
                            capture_output=True, text=True, check=True).stdout
    architectures = set(re.findall(r"arch = (sm_\d+)", result))
    functions = re.findall(r"Function\s*:\s*(\S+)", result)
    if architectures != {"sm_75"} or functions != [symbol]:
        raise ValueError(f"Unexpected architecture/functions: {architectures}, {functions}")
    instructions = [line.strip() for line in result.splitlines()
                    if re.search(r"/\*[0-9a-f]{4,}\*/", line)]
    words = re.findall(r"/\* (0x[0-9a-f]{16}) \*/", result)
    if not instructions or len(words) != 2 * len(instructions):
        raise ValueError("Incomplete sm75 instruction extraction")
    return {"module": str(path.resolve()),
            "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "symbol": symbol, "architecture": "sm_75",
            "instructions": len(instructions), "machine_words": len(words),
            "instruction_sha256": hashlib.sha256("\n".join(instructions).encode()).hexdigest(),
            "machine_words_sha256": hashlib.sha256("\n".join(words).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Legacy kernel SASS only; excludes host wrappers, constants, helper code, PTX JIT and performance"}
    for label in SYMBOLS:
        report[label] = extract(args.cuobjdump, getattr(args, label), SYMBOLS[label])
    report["passes"] = all(report["reference"][key] == report["candidate"][key]
                           for key in ("instructions", "machine_words", "instruction_sha256", "machine_words_sha256"))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passes"]:
        raise AssertionError("Legacy kernel SASS differs")


if __name__ == "__main__":
    main()
