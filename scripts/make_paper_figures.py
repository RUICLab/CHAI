#!/usr/bin/env python3
"""Render publication-quality figures from a plan-quality sweep.

    python3 scripts/make_paper_figures.py reports/plan_quality_data.json

Writes vector PDF and 400 dpi PNG for each figure into reports/figures/, plus a
single combined PDF. Figures are sized for a two-column paper: 3.4 in for a
single column, 7.0 in for a full-width span.

Print conventions this follows, which screen figures usually get wrong:

* No dark mode and no theme tokens; one palette that survives a laser printer.
* Every colour-coded series also carries a hatch, so the figure still reads in
  greyscale and for colour-vision deficiency. Colour alone is never the encoding.
* Type is set at real point sizes relative to the printed figure width, not
  scaled from a screen layout, so 8 pt in the file is 8 pt on the page.
* Text is embedded as TrueType (Type 42) rather than outlines, which keeps the
  PDF searchable and satisfies most publisher checks.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np

# --- print style -----------------------------------------------------------
matplotlib.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType, keep text selectable
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL_OFF = "#D9541F"     # control plane OFF
COL_ON = "#0F7A5A"      # control plane ON
COL_SEQ = "#0F6E68"
INK = "#101619"
INK_2 = "#4A5A61"
RULE = "#C8D2D5"

ARMS = [
    ("checkmate_baseline", "A", "CheckMate\nT1–T20", COL_OFF, "///"),
    ("operator_only_plus10", "B", "CheckMate+10\nT1–T30", COL_OFF, "\\\\\\"),
    ("chai_plus10", "C", "CHAI+10\nT1–T30", COL_ON, ""),
    ("chai_baseline_ops", "D", "CHAI\nT1–T20", COL_ON, "..."),
]


def _outdir(path="reports/figures"):
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, name, out, pdf_pages):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    pdf_pages.savefig(fig)
    plt.close(fig)
    print(f"  {name}.pdf / {name}.png")


def fig_rate_and_repeats(data, out, pdf_pages):
    """Full-width two-panel: pooled rate, and the per-repeat values behind it.

    This accepted-artifact measure tests whether enforcement prevents a bad
    label from shipping. It does not measure how often the underlying model
    originated bad labels before control-plane pruning.
    """
    pooled = data["pooled"]
    ar = data["apps"]["ar-iclib"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # ---- (a) pooled rate --------------------------------------------------
    rows = []
    for key, letter, name, color, hatch in ARMS:
        t = pooled[key]
        rate = t["false"] / t["plans"] if t["plans"] else 0.0
        rows.append((f"{letter}  {name}", rate, color, hatch,
                     f"{t['false']}/{t['plans']}"))

    y = np.arange(len(rows))[::-1]
    bars = ax1.barh(y, [r[1] for r in rows], height=0.58,
                    color=[r[2] for r in rows], edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, rows):
        bar.set_hatch(row[3])

    for yi, row in zip(y, rows):
        rate = row[1]
        if rate == 0:
            ax1.plot([0, 0.006], [yi, yi], color=COL_ON, lw=4,
                     solid_capstyle="butt")
        # One combined label: a separate count string collided with the value.
        ax1.text(rate + 0.022, yi, f"{rate:.2f}   ({row[4]})", va="center",
                 ha="left", fontsize=7.5, color=INK)

    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax1.set_xlim(0, 0.80)
    ax1.set_xticks([0, 0.2, 0.4, 0.6])
    ax1.set_xlabel("Unsupported labels shipped per accepted plan")
    ax1.set_title("(a)  pooled over 6 benchmarks", loc="left", fontsize=8,
                  color=INK_2, pad=6)
    ax1.grid(axis="x", color=RULE, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.spines["left"].set_color(RULE)
    ax1.tick_params(axis="y", length=0)

    ax1.legend(handles=[
        Patch(facecolor=COL_OFF, hatch="///", edgecolor="white",
              label="control plane off"),
        Patch(facecolor=COL_ON, edgecolor="white", label="control plane on"),
    ], loc="lower right", frameon=False, handlelength=1.5,
        borderpad=0.1, labelspacing=0.35)

    # ---- (b) per-repeat ---------------------------------------------------
    for row, (key, letter, _name, color, _hatch) in enumerate(ARMS):
        entry = ar.get(key) or {}
        values = entry.get("false_claims", [])
        if not values:
            continue
        yi = len(ARMS) - 1 - row
        if len(set(values)) > 1:
            ax2.plot([min(values), max(values)], [yi, yi], color=color, lw=1.3,
                     alpha=0.5, zorder=1, solid_capstyle="round")
        seen = {}
        for value in values:
            n_same = values.count(value)
            offset = seen.get(value, 0)
            seen[value] = offset + 1
            ax2.scatter(value, yi + (offset - (n_same - 1) / 2) * 0.15,
                        s=24, color=color, edgecolor="white", linewidth=0.7,
                        zorder=3)

    ax2.set_yticks(range(len(ARMS)))
    ax2.set_yticklabels([a[1] for a in ARMS][::-1], fontsize=7.5,
                        fontweight="bold")
    ax2.set_xlim(-0.4, 5.0)
    ax2.set_xticks(range(0, 6))
    ax2.set_xlabel("Unsupported labels shipped per repeat")
    ax2.set_title("(b)  ar-iclib, each of 3 repeats", loc="left", fontsize=8,
                  color=INK_2, pad=6)
    ax2.grid(axis="x", color=RULE, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.spines["left"].set_color(RULE)
    ax2.tick_params(axis="y", length=0)

    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig1_false_provenance", out, pdf_pages)


def fig_coverage(data, out, pdf_pages):
    """Coverage matrix: repeats that produced any accepted plan."""
    apps = ["ar-iclib", "fft-iclib", "bc-iclib", "sobel-iclib",
            "lqi-iclib", "stringsearch-iclib"]
    apps = [a for a in apps if a in data["apps"]]
    matrix = np.zeros((len(apps), len(ARMS)))
    for i, app in enumerate(apps):
        for j, (key, *_rest) in enumerate(ARMS):
            entry = data["apps"][app].get(key) or {}
            matrix[i, j] = entry.get("repeats", 0)

    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=3, aspect="auto")

    for i in range(len(apps)):
        for j in range(len(ARMS)):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                    color="white" if value >= 2 else INK,
                    fontweight="bold" if value == 0 else "normal")

    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[1] for a in ARMS], fontweight="bold")
    ax.set_yticks(range(len(apps)))
    ax.set_yticklabels(apps, fontsize=7.2)
    ax.set_xlabel("Repeats yielding an accepted plan (of 3)")
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-.5, len(ARMS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(apps), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _save(fig, "fig3_coverage", out, pdf_pages)


def fig_specialisation(data, out, pdf_pages):
    """Distinct per-function operator menus."""
    ar = data["apps"]["ar-iclib"]
    fig, ax = plt.subplots(figsize=(3.4, 1.7))

    letters, menus, colors, hatches = [], [], [], []
    for key, letter, name, color, hatch in ARMS:
        entry = ar.get(key) or {}
        values = entry.get("distinct_menus", [])
        letters.append(letter)
        menus.append(max(values) if values else 0)
        colors.append(color)
        hatches.append(hatch)

    x = np.arange(len(letters))
    bars = ax.bar(x, menus, width=0.6, color=colors, edgecolor="white", linewidth=0.7)
    for bar, hatch, value in zip(bars, hatches, menus):
        bar.set_hatch(hatch)
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, str(value),
                ha="center", fontsize=8, fontweight="bold", color=INK)

    ax.axhline(1, color=INK_2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.text(3.42, 1.35, "1 = same menu for every function", fontsize=6.8,
            color=INK_2, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(letters, fontweight="bold")
    ax.set_ylim(0, 8.6)
    ax.set_ylabel("Distinct operator menus")
    ax.set_xlabel("Per-function specialisation (ar-iclib, 9 functions)")
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(RULE)

    _save(fig, "fig4_specialisation", out, pdf_pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", default="reports/plan_quality_data.json")
    parser.add_argument("--outdir", default="reports/figures")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as handle:
        data = json.load(handle)

    out = _outdir(args.outdir)
    combined = os.path.join(out, "all_figures.pdf")
    print(f"Writing figures to {out}/")
    with PdfPages(combined) as pdf_pages:
        fig_rate_and_repeats(data, out, pdf_pages)
        fig_coverage(data, out, pdf_pages)
        fig_specialisation(data, out, pdf_pages)
    print(f"  all_figures.pdf  (combined)")


if __name__ == "__main__":
    main()
