#!/usr/bin/env python3
"""Measure Fused power cycles for explicit knob settings, bypassing BayesOpt.

Why this exists
---------------
BayesOpt is a poor way to evaluate a small, discrete knob space, and it can
silently fail to evaluate a setting at all. A single binary knob has nothing to
search: on pcam-iclib's `use_fixed_point` knob every candidate BO drew landed
on 0, so the run reported "no improvement found" having never once compiled the
integer path. That reads exactly like a negative result and is not one.

This script compiles and simulates the settings it is told to, so a reported
cycle count always corresponds to a configuration that was actually built.

Run inside the CHAI Docker image, which carries the MSP430 toolchain and the
Fused binary:

    docker run --rm -v "$PWD:/workspace" -w /workspace chai-dev:local \\
      python3 -m biomedical.measure_power_cycles \\
        --kernel results_archive/<run>/target/pcam.c \\
        --header benchmark_applications/pcam-iclib/pcam_weights.h \\
        --app pcam-iclib --source-name pcam.c \\
        --setting identity --setting approximation_level=1
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# buildObjects and runFused report their failures through Dprint, which is
# disabled by default, so a failed build otherwise surfaces only as None.
import config.config as _chai_config
_chai_config.DEBUG = 1
import utils.utils as _chai_utils
_chai_utils.DEBUG = 1

from utils.checkpoints import checkpointOrchestration
from utils.fused_trace import prepare_fused_trace


def apply_knobs(source: str, assignments: str) -> str:
    if assignments == "identity":
        return source
    for pair in assignments.split(","):
        name, value = pair.split("=")
        source, count = re.subn(
            rf"int {re.escape(name.strip())} = [0-9]+;",
            f"int {name.strip()} = {value.strip()};",
            source,
        )
        if not count:
            raise SystemExit(f"knob {name!r} not found in the kernel")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--app", required=True, help="e.g. pcam-iclib")
    parser.add_argument("--source-name", required=True,
                        help="filename the app's CMakeLists compiles, e.g. pcam.c")
    parser.add_argument("--setting", action="append", default=[],
                        help="'identity' or 'knob=value[,knob=value...]'; "
                             "repeat for each configuration to measure")
    parser.add_argument("--work", type=Path, default=Path("target"),
                        help="directory checkpointOrchestration compiles from")
    parser.add_argument("--trace", default="traces/RF_2.csv")
    parser.add_argument("--capacitor", default="68e-6")
    args = parser.parse_args()

    # Fused reads fusedBin/config.yaml, which the pipeline renders per
    # (trace, capacitor) from a template. Omitting this step leaves the last
    # run's config -- or none at all -- and Fused silently produces no cycles.
    shutil.copy2("fusedBin/fusedConfig/config.yaml.in", "fusedBin/config.yaml")
    normalized = prepare_fused_trace(args.trace)
    with open("fusedBin/config.yaml", "a") as handle:
        handle.write(f'VoltageTraceFile: "./{normalized.name}"\n')
        handle.write(f"CapacitorValue: {args.capacitor}\n")
    print(f"Fused configured: trace {args.trace}, capacitor {args.capacitor}")

    settings = args.setting or ["identity"]
    source = args.kernel.read_text()
    args.work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.header, args.work / args.header.name)

    results = {}
    for setting in settings:
        (args.work / args.source_name).write_text(apply_knobs(source, setting))
        cycles = checkpointOrchestration(str(args.work), args.app)
        results[setting] = cycles
        print(f"  {setting:<28} -> "
              + ("build or simulation FAILED" if cycles is None
                 else f"{cycles} power cycles"))

    baseline = results.get("identity")
    if baseline:
        print()
        for setting, cycles in results.items():
            if setting == "identity" or cycles is None:
                continue
            reduction = 1.0 - cycles / baseline
            print(f"  {setting:<28} {cycles} vs {baseline} baseline "
                  f"-> {reduction * 100:+.1f}%")


if __name__ == "__main__":
    main()
