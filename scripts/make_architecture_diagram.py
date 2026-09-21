"""Architecture figure: CHAI offline, level dispatcher on the device.

Replaces scripts/make_adaptive_diagram.py, which drew the reactive controller
("banked vs needed energy, degrade if behind"). That design was withdrawn: it
had no way to hold a level and missed every other deadline by construction.
What is validated, and what this draws, is a table lookup on measured capacitor
voltage.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "reports/figures"
INK = "#22201E"
MUTED = "#6B6660"
OFF = "#8C6A4A"        # offline stage
ON = "#1F6F7A"         # on-device stage
WARN = "#C25E3A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "figure.dpi": 300,
})


def box(ax, x, y, w, h, title, sub, edge, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color=edge)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=7.4, color=MUTED, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=1.0, color=color,
                                 shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(10.2, 7.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.2)
ax.axis("off")

# ---------------------------------------------------------------- offline band
ax.add_patch(FancyBboxPatch((0.15, 4.95), 9.7, 1.95,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            fc="#FBF7F2", ec="#E3D9CC", lw=1.0))
ax.text(0.38, 6.68, "OFFLINE  —  once, on a workstation", fontsize=8.6,
        fontweight="bold", color=OFF)

box(ax, 0.45, 5.20, 1.75, 1.15, "Benchmark C", "sobel, fft\nunmodified source", OFF)
box(ax, 2.45, 5.20, 2.35, 1.15, "CHAI control plane",
    "gate by hardware -> rank ->\nLLM -> validate -> prune\nfalse operator labels", OFF)
box(ax, 5.05, 5.20, 2.15, 1.15, "Knob search",
    "BayesOpt + Fused\nmeasure error and\npower cycles", OFF)
box(ax, 7.45, 5.20, 2.25, 1.15, "LEVEL TABLE",
    "knobs + energy + error\none binary, all levels", OFF, fill="#FDF3E7")

arrow(ax, 2.20, 5.78, 2.45, 5.78, OFF)
arrow(ax, 4.80, 5.78, 5.05, 5.78, OFF)
arrow(ax, 7.20, 5.78, 7.45, 5.78, OFF)

# ---------------------------------------------------------------- the table
ax.add_patch(FancyBboxPatch((3.15, 3.20), 3.7, 1.28,
                            boxstyle="round,pad=0.015,rounding_size=0.02",
                            fc="#FDF3E7", ec=OFF, lw=1.1))
ax.text(5.0, 4.34, "characterised table  (sobel)", ha="center",
        fontsize=7.8, style="italic", color=MUTED)
rows = [("level", "knobs", "energy", "error"),
        ("L0", "[1,1]", "14.84 mJ", "0.000"),
        ("L1", "[5,2]", "3.11 mJ", "0.019"),
        ("L2", "[4,4]", "1.91 mJ", "0.280")]
for r, row in enumerate(rows):
    yy = 4.11 - r * 0.215
    for c, val in enumerate(row):
        xx = 3.38 + c * 0.92
        ax.text(xx, yy, val, fontsize=7.2, color=INK if r else MUTED,
                fontweight="bold" if r == 0 else "normal", ha="left")

arrow(ax, 8.55, 5.20, 6.90, 4.50, OFF)
ax.text(8.75, 4.74, "flashed once", fontsize=7.2, color=MUTED,
        ha="left", style="italic")

# ---------------------------------------------------------------- device band
ax.add_patch(FancyBboxPatch((0.15, 0.30), 9.7, 2.42,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            fc="#F2F8F8", ec="#CFE3E5", lw=1.0))
ax.text(0.38, 2.50, "ON THE DEVICE  —  at every deadline boundary", fontsize=8.6,
        fontweight="bold", color=ON)

box(ax, 0.45, 1.02, 1.70, 1.15, "Harvester\n+ capacitor",
    "RF or solar\n68 uF - 4700 uF", ON)
box(ax, 2.40, 1.02, 1.95, 1.15, "Measure V",
    "E = 1/2 C (V^2 - Vrel^2)\nmeasured, not\nforecast", ON)
box(ax, 4.60, 1.02, 2.05, 1.15, "Look up level",
    "least approximate\nwith E_req x 1.1\n<= E_available", ON)
box(ax, 6.90, 1.02, 1.50, 1.15, "Execute",
    "set knobs,\nrun one task", ON)
box(ax, 8.65, 1.02, 1.05, 1.15, "DEFER", "nothing fits:\nwait", WARN)

arrow(ax, 2.15, 1.60, 2.40, 1.60, ON)
arrow(ax, 4.35, 1.60, 4.60, 1.60, ON)
arrow(ax, 6.65, 1.60, 6.90, 1.60, ON)
arrow(ax, 8.40, 1.45, 8.65, 1.45, WARN)

# table feeds the lookup
arrow(ax, 4.80, 3.20, 5.30, 2.17, OFF)

# loop back for the next period
ax.add_patch(FancyArrowPatch((7.65, 1.02), (1.30, 1.02),
                             connectionstyle="arc3,rad=-0.16",
                             arrowstyle="-|>", mutation_scale=11, lw=1.0,
                             color=ON, ls=(0, (4, 3))))
# white bbox so the label masks the dashed arc it sits on
ax.text(4.50, 0.60, "next deadline period", fontsize=7.4, color=ON,
        ha="center", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.28", fc="#F2F8F8", ec="none"))

ax.text(0.0, 7.02,
        "no forecast, no radio, no model on the device: "
        "two comparisons and a table index",
        ha="left", fontsize=8.8, style="italic", color=ON)

ax.set_title("CHAI generates the quality-energy ladder offline; "
             "the device dispatches on measured energy",
             fontsize=11.5, fontweight="bold", loc="left", color=INK, pad=14)

os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "architecture." + ext),
                bbox_inches="tight", facecolor="white")
print("wrote " + OUT + "/architecture.png and .pdf")
