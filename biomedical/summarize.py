#!/usr/bin/env python3
"""Aggregate the three-seed biomedical pilots into JSON and Markdown."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biomedical.common import write_json


REPORT_DIRECTORY = Path("reports/biomedical")


def load_runs(pattern: str) -> list[dict]:
    paths = sorted(REPORT_DIRECTORY.glob(pattern))
    if len(paths) != 3:
        raise RuntimeError(f"expected three runs for {pattern}, found {len(paths)}")
    return [json.loads(path.read_text()) for path in paths]


def aggregate(runs: list[dict], key: str) -> list[dict]:
    identifiers = [level[key] for level in runs[0]["levels"]]
    rows = []
    for identifier in identifiers:
        levels = [next(level for level in run["levels"] if level[key] == identifier) for run in runs]
        row = {key: identifier}
        for metric in (
            "balanced_accuracy",
            "auroc",
            "sensitivity",
            "specificity",
            "prediction_disagreement_vs_l0",
        ):
            values = np.asarray([level[metric] for level in levels])
            row[metric] = {
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        reduction_name = "frame_reduction_vs_l0" if key == "hop_multiplier" else "mac_reduction_vs_l0"
        row[reduction_name] = float(np.median([level[reduction_name] for level in levels]))
        rows.append(row)
    return rows


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def interval(metric: dict) -> str:
    return f"{percentage(metric['median'])} ({percentage(metric['min'])}–{percentage(metric['max'])})"


def main() -> None:
    cough_runs = load_runs("coughvid_pilot_seed*.json")
    pcam_runs = load_runs("pcam_pilot_seed*.json")
    cough = aggregate(cough_runs, "hop_multiplier")
    pcam = aggregate(pcam_runs, "resolution")
    payload = {
        "seeds": [2026, 2027, 2028],
        "coughvid": cough,
        "pcam": pcam,
        "scope": "host-side dataset pilot; no Fused power-cycle measurements",
    }
    write_json(REPORT_DIRECTORY / "biomedical_pilot_summary.json", payload)

    lines = [
        "# Biomedical dataset pilot results",
        "",
        "Three seeds (2026–2028). Values are medians with the seed range in parentheses. ",
        "Decision thresholds are selected only on validation data. These are host-side ",
        "classification/compute results; neither model has yet been exported to C or measured ",
        "for MSP430 power cycles in Fused.",
        "",
        "## COUGHVID — self-reported COVID-19 vs healthy/symptomatic",
        "",
        "| STFT hop | frame reduction | balanced accuracy | AUROC | disagreement vs L0 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in cough:
        lines.append(
            f"| {row['hop_multiplier']}× | {percentage(row['frame_reduction_vs_l0'])} | "
            f"{interval(row['balanced_accuracy'])} | {interval(row['auroc'])} | "
            f"{interval(row['prediction_disagreement_vs_l0'])} |"
        )
    lines += [
        "",
        "The exact-level classifier is not stable across seeds, so this is not yet a defensible ",
        "medical-classification result. The 2× hop level is only a compute-preservation signal: ",
        "it halves STFT frames, but a stronger frozen base classifier is required before CHAI ",
        "optimization or power measurements are meaningful.",
        "",
        "## PatchCamelyon — tumor vs non-tumor",
        "",
        "| center resolution | MAC reduction | balanced accuracy | AUROC | disagreement vs L0 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pcam:
        lines.append(
            f"| {row['resolution']}×{row['resolution']} | {percentage(row['mac_reduction_vs_l0'])} | "
            f"{interval(row['balanced_accuracy'])} | {interval(row['auroc'])} | "
            f"{interval(row['prediction_disagreement_vs_l0'])} |"
        )
    lines += [
        "",
        "PCam is the promising workload. The useful operating point is the smallest resolution ",
        "whose seed range retains the 64×64 reference accuracy; that decision must be based on ",
        "the complete table rather than a single seed.",
        "",
        "## Protocol qualifications",
        "",
        "- COUGHVID uses 500 high-confidence cough recordings per class and a 60/20/20 ",
        "  stratified split per seed. Most COVID status labels are self-reported.",
        "- PCam uses 24,576 images from the official validation file for development, holds out ",
        "  the remaining 8,192 for model/threshold selection, and evaluates all 32,768 images ",
        "  in the separately distributed test file. This is a pilot split because the official ",
        "  6.1 GB training file was rate-limited.",
        "- The official PCam repository documents 327,680 balanced 96×96 patches and labels ",
        "  based on tumor in the center 32×32 region: https://github.com/basveeling/pcam",
        "- COUGHVID source and metadata description: https://zenodo.org/records/4048312",
        "",
    ]
    (REPORT_DIRECTORY / "biomedical_pilot_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

