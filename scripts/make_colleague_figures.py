"""Two figures isolating CHAI's control plane from its operator catalogue.

Arm B (CheckMate + the same 10 added operators, no control plane) versus
arm C (CHAI: identical 30-operator catalogue, control plane ON). Because both
arms draw from the same menu, any difference between them is attributable to
the control plane and not to the added operators.

Source: reports/plan_quality_data.json (6 benchmarks x 3 repeats, 7B planner)
plus operator-conformance detection counts recomputed from the campaign's
experiment_metrics_*.json files.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "reports/figures"

C_BASE = "#C25E3A"   # arm B - CheckMate + 10 operators
C_CHAI = "#1F6F7A"   # arm C - CHAI
C_RAW = "#B8B3AC"    # pre-control-plane (model's own output)
INK = "#22201E"
MUTED = "#6B6660"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

# ---- measured values ----------------------------------------------------
DETECTED_B, SHIPPED_B = 35, 10      # arm B: mislabels found / mislabels delivered
DETECTED_C, SHIPPED_C = 37, 0       # arm C: mislabels found / mislabels delivered
PLANS_B, PLANS_C = 20, 13
OFFERED_B, OFFERED_C = 32, 16       # operators exposed to planner (C: median, 15-17)
MENUS_B, MENUS_C = 1, 7             # distinct per-function menus, ar-iclib
INFEAS_B, INFEAS_C = 1, 0           # hardware-impossible operators delivered


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, name + "." + ext),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote " + OUT + "/" + name + ".png and .pdf")


def figure_provenance():
    """Same mislabelling rate, opposite delivered outcome."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    xs = [0, 1]
    w = 0.30
    det = [DETECTED_B, DETECTED_C]
    shp = [SHIPPED_B, SHIPPED_C]
    xd = [x - w / 2 - 0.02 for x in xs]
    xp = [x + w / 2 + 0.02 for x in xs]

    ax.bar(xd, det, w, color=C_RAW, edgecolor=INK, linewidth=0.7,
           label="Detected during generation  (the planner's own mislabels)")
    ax.bar(xp, shp, w, color=[C_BASE, C_CHAI], edgecolor=INK, linewidth=0.7,
           label="Shipped in the delivered plan  (what a user receives)")

    for x, v in zip(xd, det):
        ax.text(x, v + 1.0, str(v), ha="center", fontsize=11, color=MUTED)
    for x, v, c in zip(xp, shp, [C_BASE, C_CHAI]):
        ax.text(x, v + 1.2, str(v), ha="center", fontsize=15,
                fontweight="bold", color=c)

    # a measured zero, not a missing bar
    ax.plot([xp[1] - w / 2, xp[1] + w / 2], [0, 0], color=C_CHAI,
            linewidth=2.6, solid_capstyle="butt")

    top = max(det)
    y = top + 6.0
    ax.plot([xd[0], xd[0], xd[1], xd[1]], [y - 1.3, y, y, y - 1.3],
            color="#8A847C", linewidth=0.9)
    ax.text(0.5, y + 0.9, "same mislabelling rate   35  vs  37", ha="center",
            fontsize=9.5, style="italic", color=MUTED)

    ax.text(xp[0] + 0.19, 6.0,
            "1 in 2 plans\ncarries a false\noperator label",
            fontsize=9, color=C_BASE, ha="left", va="center", linespacing=1.5)
    ax.text(xp[1] + 0.19, 4.0,
            "all 37 caught\nand removed\nbefore delivery",
            fontsize=9, color=C_CHAI, ha="left", va="center", linespacing=1.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([
        "B   CheckMate + 10 operators\nno control plane   (" + str(PLANS_B) + " plans)",
        "C   CHAI, same 30 operators\ncontrol plane ON   (" + str(PLANS_C) + " plans)",
    ], fontsize=10)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_ylabel("Unsupported operator labels")
    ax.set_xlim(-0.52, 1.74)
    ax.set_ylim(0, top + 11)
    ax.legend(frameon=False, fontsize=8.8, loc="upper center",
              bbox_to_anchor=(0.46, -0.22), ncol=1, handlelength=1.4,
              labelspacing=0.5)
    ax.set_title("Both arms have the same operators. Only one delivers truthful plans.",
                 fontsize=12, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.045,
            "MSP430   |   6 benchmarks x 3 repeats   |   identical 30-operator catalogue in both arms",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _save(fig, "chai_vs_checkmate_provenance")


def figure_hardware():
    """Where the control plane intervenes: before, during and after the prompt."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2))
    panels = [
        (OFFERED_B, OFFERED_C, "Operators offered\nto the planner",
         "BEFORE the prompt", "out of 32", "lower = hardware-filtered"),
        (MENUS_B, MENUS_C, "Distinct per-function\noperator menus",
         "DURING planning", "ar-iclib", "higher = specialised"),
        (INFEAS_B, INFEAS_C, "Hardware-impossible\noperators delivered",
         "AFTER generation", "all benchmarks", "lower = better"),
    ]
    for ax, (vb, vc, title, stage, note, hint) in zip(axes, panels):
        bars = ax.bar([0, 1], [vb, vc], 0.56, color=[C_BASE, C_CHAI],
                      edgecolor=INK, linewidth=0.7)
        top = max(vb, vc)
        for b, v, c in zip(bars, [vb, vc], [C_BASE, C_CHAI]):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.05, str(v),
                    ha="center", fontsize=14, fontweight="bold", color=c)
        if vc == 0:
            ax.plot([1 - 0.28, 1 + 0.28], [0, 0], color=C_CHAI, linewidth=2.6,
                    solid_capstyle="butt")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["B", "C"], fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_ylim(0, top * 1.34 if top else 1)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=19)
        ax.text(0.5, 1.045, stage, transform=ax.transAxes, ha="center",
                fontsize=8, color=MUTED, style="italic")
        ax.text(0.5, -0.19, note, transform=ax.transAxes, ha="center",
                fontsize=8, color=MUTED)
        ax.text(0.5, -0.33, hint, transform=ax.transAxes, ha="center",
                fontsize=7.5, color="#9A948C", style="italic")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    fig.suptitle("The control plane acts at three points the operator catalogue cannot reach",
                 fontsize=11.5, fontweight="bold", x=0.045, ha="left", y=1.30)
    fig.text(0.045, 1.17,
             "B = CheckMate + 10 operators (no control plane)        "
             "C = CHAI, identical catalogue, control plane ON",
             fontsize=8.5, color=MUTED, ha="left")
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "chai_vs_checkmate_hardware")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_provenance()
    figure_hardware()
    print("done")
