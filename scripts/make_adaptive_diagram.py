"""Flow diagram for the energy-adaptive approximation system."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#22201E", "#6B6660"
OFF, ON, ACC = "#C25E3A", "#1F6F7A", "#F4F1ED"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "text.color": INK, "figure.dpi": 300})


def box(ax, x, y, w, h, title, sub, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=edge, facecolor="white"))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
            fontsize=7.8, color=MUTED, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=INK, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 linewidth=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

# ---------------- offline row ----------------
ax.add_patch(FancyBboxPatch((0.15, 4.05), 9.7, 1.95, boxstyle="round,pad=0.02,rounding_size=0.03",
                            linewidth=0, facecolor=ACC))
ax.text(0.38, 5.78, "OFFLINE  —  done once, on a workstation", fontsize=8.6,
        fontweight="bold", color=OFF)

box(ax, 0.45, 4.35, 1.95, 1.15, "Benchmark C", "sobel (image)\nfft (spectrum)", OFF)
box(ax, 2.85, 4.35, 2.25, 1.15, "Approximation plan", "integer knobs:\nstride, terms", OFF)
box(ax, 5.55, 4.35, 2.05, 1.15, "BayesOpt +\nFused simulator", "measure error &\npower cycles", OFF)
box(ax, 8.05, 4.35, 1.65, 1.15, "LEVEL TABLE", "L0 / L1 / L2\nknob + cost", OFF)
for x1, x2 in ((2.40, 2.85), (5.10, 5.55), (7.60, 8.05)):
    arrow(ax, x1, 4.93, x2, 4.93, OFF)

# flashed once
arrow(ax, 8.88, 4.35, 8.88, 3.30, MUTED, rad=0.0)
ax.text(9.0, 3.78, "all levels ship in\none binary", fontsize=7.5, color=MUTED, ha="left",
        va="center", linespacing=1.3)

# ---------------- online row ----------------
ax.add_patch(FancyBboxPatch((0.15, 0.35), 9.7, 2.6, boxstyle="round,pad=0.02,rounding_size=0.03",
                            linewidth=0, facecolor="#EDF3F4"))
ax.text(0.38, 2.72, "ON THE DEVICE  —  every deadline period", fontsize=8.6,
        fontweight="bold", color=ON)

box(ax, 0.45, 1.45, 1.75, 1.05, "RF harvester", "V / 10 kΩ", ON)
box(ax, 2.60, 1.45, 1.75, 1.05, "Capacitor", "68 µF, boots\nat 3.5 V", ON)
box(ax, 4.75, 1.45, 2.25, 1.05, "Reactive controller", "banked vs needed\nenergy", ON)
box(ax, 7.40, 1.45, 2.30, 1.05, "Run task at level", "exact, degrade\nif behind", ON)

arrow(ax, 2.20, 1.97, 2.60, 1.97, ON)
arrow(ax, 4.35, 1.97, 4.75, 1.97, ON)
arrow(ax, 7.00, 1.97, 7.40, 1.97, ON)

# feedback loop: measured progress returns to the controller
arrow(ax, 8.55, 1.42, 5.88, 1.42, ON, rad=-0.45)
ax.text(7.2, 0.66, "measured progress, no forecast", fontsize=7.8, color=ON,
        ha="center", style="italic")

# the alternative we built and measured: a server decides instead
ax.add_patch(FancyBboxPatch((4.75, 2.98), 2.25, 0.62,
                            boxstyle="round,pad=0.012,rounding_size=0.02",
                            linewidth=1.1, edgecolor=MUTED, facecolor="white",
                            linestyle="--"))
ax.text(5.88, 3.29, "server predicts instead", ha="center", va="center",
        fontsize=8.2, color=MUTED)
arrow(ax, 5.88, 2.98, 5.88, 2.52, MUTED, style="-|>")
ax.text(7.15, 3.29, "measured: worse", fontsize=7.8, color=MUTED,
        ha="left", va="center", style="italic")

# No title inside the figure: the document captions it.
fig.savefig("reports/adaptive/flow.png", bbox_inches="tight", facecolor="white")
print("wrote reports/adaptive/flow.png")
