#!/usr/bin/env python3
"""Figure: false operator provenance survives a model-scale change.

    python3 scripts/make_scale_figure.py

The obvious objection to the plan-quality result is that it only measures a
small model being sloppy. This figure answers that directly: a 7B and a
frontier model, same benchmark, same arms, same repeats. The unenforced arms
mislabel their operators at both scales; the enforced arms reach zero at both.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

COL_OFF, COL_ON = "#D9541F", "#0F7A5A"
INK, INK_2, RULE = "#101619", "#4A5A61", "#C8D2D5"

ARMS = [
    ("checkmate_baseline",   "A", "CheckMate T1–T20",  COL_OFF, "///"),
    ("operator_only_plus10", "B", "CheckMate+10",      COL_OFF, "\\\\\\"),
    ("chai_plus10",          "C", "CHAI T1–T30",       COL_ON,  ""),
    ("chai_baseline_ops",    "D", "CHAI T1–T20",       COL_ON,  "..."),
]


def rates(path):
    data = json.load(open(path))
    app = data["apps"].get("ar-iclib", {})
    out = {}
    for key, letter, *_ in ARMS:
        entry = app.get(key) or {}
        plans = sum(entry.get("plans", []) or [])
        false = sum(entry.get("false_claims", []) or [])
        out[letter] = (false / plans, plans, false) if plans else None
    return out


def main():
    seven = rates("/tmp/pq_7b.json")
    sonnet = rates("/tmp/pq_s5.json")
    outdir = "reports/figures"
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    n = len(ARMS)
    x = np.arange(n)
    width = 0.36

    for offset, (label, data) in enumerate(
        [("qwen2.5-coder 7B", seven), ("Claude Sonnet 5", sonnet)]
    ):
        pos = x + (offset - 0.5) * width
        for i, (_key, letter, _name, color, hatch) in enumerate(ARMS):
            rec = data.get(letter)
            if rec is None:
                # Absent data must not read as a zero: label it.
                ax.text(pos[i], 0.02, "no data", rotation=90, fontsize=6.5,
                        ha="center", va="bottom", color=INK_2)
                continue
            rate, plans, false = rec
            alpha = 1.0 if offset == 0 else 0.62
            bar = ax.bar(pos[i], rate, width, color=color, edgecolor="white",
                         linewidth=0.7, alpha=alpha)[0]
            bar.set_hatch(hatch)
            if rate == 0:
                ax.plot([pos[i] - width / 2.4, pos[i] + width / 2.4], [0.006, 0.006],
                        color=COL_ON, lw=2.6, solid_capstyle="butt")
            ax.text(pos[i], rate + 0.018, f"{rate:.2f}", ha="center",
                    fontsize=7.2, fontweight="bold", color=INK)
            ax.text(pos[i], rate + 0.052, f"{false}/{plans}", ha="center",
                    fontsize=6.3, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a[1]}\n{a[2]}" for a in ARMS], fontsize=7.4)
    ax.set_ylabel("False operator labels\nper accepted plan")
    ax.set_ylim(0, 0.63)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(RULE)

    # Left group = 7B, right group = Sonnet 5, marked by alpha and the legend.
    ax.legend(handles=[
        Patch(facecolor=COL_OFF, hatch="///", edgecolor="white", label="control plane off"),
        Patch(facecolor=COL_ON, edgecolor="white", label="control plane on"),
        Patch(facecolor="#8A8A8A", edgecolor="white", alpha=1.0, label="qwen2.5-coder 7B (left bar)"),
        Patch(facecolor="#8A8A8A", edgecolor="white", alpha=0.62, label="Claude Sonnet 5 (right bar)"),
    ], loc="upper right", frameon=False, ncol=2, handlelength=1.4,
        borderpad=0.2, columnspacing=1.2)

    ax.set_title(
        "ar-iclib · 3 repeats per arm per model · identical frozen target list",
        loc="left", fontsize=7.6, color=INK_2, pad=6,
    )

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig5_model_scale.{ext}"))
    plt.close(fig)
    print(f"  {outdir}/fig5_model_scale.pdf / .png")


if __name__ == "__main__":
    main()
