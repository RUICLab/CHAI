#!/usr/bin/env python3
"""Compile and measure every endpoint combination in the manual Sobel plan."""

import argparse
import copy
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity as ssim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.lsp import lsp_patcher
from utils.knob_metadata import normalize_knob_metadata, validate_plan_entries
from utils.validator import updateKnobValue


def run_checked(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({' '.join(command)}):\n{result.stdout}{result.stderr}"
        )
    return result


def build_and_run(directory):
    run_checked(["make", "clean"], directory)
    run_checked(["make", "main"], directory)
    run_checked(["./main"], directory)
    output = directory / "img.pgm"
    if not output.exists():
        raise RuntimeError(f"benchmark produced no image: {output}")
    return output


def image_error(reference_path, candidate_path):
    reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    candidate = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
    if reference is None or candidate is None or reference.shape != candidate.shape:
        raise RuntimeError("reference and candidate images are missing or incompatible")
    return float((1.0 - ssim(reference, candidate)) / 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="manual_plans/sobel-iclib_t2_t7.json"
    )
    parser.add_argument(
        "--source", default="benchmark_applications/sobel-iclib"
    )
    args = parser.parse_args()

    plan_path = (REPO_ROOT / args.plan).resolve()
    source_path = (REPO_ROOT / args.source).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan_entries(plan)
    if len(plan) != 1:
        raise RuntimeError("the manual Sobel evaluator expects exactly one plan entry")

    variables, ranges, _ = normalize_knob_metadata(plan[0])
    endpoints = [sorted(set(bounds[next(iter(bounds))])) for bounds in ranges]
    report = {"plan": str(plan_path), "evaluations": []}

    with tempfile.TemporaryDirectory(prefix="chai_manual_sobel_") as temporary:
        temporary_path = Path(temporary)
        baseline_dir = temporary_path / "baseline"
        shutil.copytree(source_path, baseline_dir)
        reference_path = build_and_run(baseline_dir)
        reference_bytes = reference_path.read_bytes()

        for values in itertools.product(*endpoints):
            candidate_dir = temporary_path / ("candidate_" + "_".join(map(str, values)))
            shutil.copytree(source_path, candidate_dir)
            candidate_plan = copy.deepcopy(plan)
            for variable, value in zip(variables, values):
                updateKnobValue(candidate_plan[0], value, variable)
            (candidate_dir / "apx_all.json").write_text(
                json.dumps(candidate_plan, indent=2), encoding="utf-8"
            )
            lsp_patcher(str(candidate_dir), "apx_all.json")
            candidate_path = build_and_run(candidate_dir)
            report["evaluations"].append(
                {
                    "knobs": dict(zip(variables, values)),
                    "error": image_error(reference_path, candidate_path),
                    "byte_identical": candidate_path.read_bytes() == reference_bytes,
                }
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
