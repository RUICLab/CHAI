"""Build an optimization-level table from one wide-bound BayesOpt run.

Re-running BayesOpt at a tighter error bound to find each level does not work:
on ar-iclib a bound-0.40 run found nothing even though 89 of a bound-0.50 run's
159 accepted points were under 0.40. Tightening the bound changes the rejection
penalty, reshapes the surrogate, and the search never revisits the feasible
region. So run the search ONCE at a wide bound and choose levels from the
points it already sampled.

A level is a knob vector, not a separate program: all levels must come from the
same plan so one binary can switch between them at run time from a few bits in
a radio packet.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Level:
    name: str
    knobs: list
    error: float
    power_cycles: int
    original_power_cycles: int

    @property
    def reduction(self) -> float:
        return 1.0 - self.power_cycles / self.original_power_cycles


def _parse_knobs(text: str) -> list:
    unwrapped = re.sub(r"np\.int64\(\s*(-?\d+)\s*\)", r"\1", text)
    return [int(x) for x in re.findall(r"-?\d+", unwrapped)]


def read_points(run_dir: str):
    """Return every accepted (error, cycles, knobs) the run logged."""
    files = glob.glob(f"{run_dir}/**/logs/*_68_RF_*.csv", recursive=True)
    if not files:
        raise FileNotFoundError(f"no BayesOpt csv under {run_dir}")
    points = []
    for line in open(files[0]):
        line = line.rstrip("\n")
        if not line or line.startswith("knobs_list"):
            continue
        knobs, error, cycles = line.rsplit(",", 2)
        try:
            points.append((float(error), int(cycles), _parse_knobs(knobs)))
        except ValueError:
            continue
    return points


def original_cycles(run_dir: str, fallback: int | None = None) -> int:
    for path in glob.glob(f"{run_dir}/**/logs/best_knobs_*.csv", recursive=True):
        rows = open(path).read().strip().split("\n")
        if len(rows) >= 2:
            header = rows[0].split(",")
            if "original_power_cycles" in header:
                idx = header.index("original_power_cycles") - len(header)
                try:
                    return int(rows[1].rsplit(",", -idx)[-(-idx)])
                except (ValueError, IndexError):
                    pass
    for path in glob.glob(f"{run_dir}/**/logs/original_checkpoints_*.txt", recursive=True):
        text = open(path).read().strip()
        if text.isdigit():
            return int(text)
    if fallback is None:
        raise ValueError("could not determine original_power_cycles")
    return fallback


def pareto(points):
    """Keep points not beaten on both error and cycles by another point."""
    front = []
    for error, cycles, knobs in sorted(points, key=lambda p: (p[1], p[0])):
        if any(e <= error and c <= cycles for e, c, _ in front):
            continue          # dominated, or an exact duplicate already kept
        front.append((error, cycles, knobs))
    return front


def choose_levels(points, original, n_levels=3, bound=None):
    """Pick evenly spread levels along the energy axis of the Pareto front.

    L0 is always the identity point: the plan at its declared defaults, which
    reproduces the original output exactly. Remaining levels are spread across
    the range of power-cycle counts so the table is a genuine energy dial
    rather than several configurations of near-identical cost.
    """
    usable = [p for p in points if bound is None or p[0] <= bound]
    if not usable:
        raise ValueError("no points within the bound")
    front = pareto(usable)

    identity = min(usable, key=lambda p: (p[0], p[1]))
    levels = [Level("L0", identity[2], identity[0], identity[1], original)]

    candidates = [p for p in front if p[1] < identity[1]]
    if candidates:
        cheapest = min(c[1] for c in candidates)
        dearest = max(c[1] for c in candidates)
        for i in range(1, n_levels):
            frac = i / (n_levels - 1) if n_levels > 1 else 1.0
            target = dearest - frac * (dearest - cheapest)
            pick = min(candidates, key=lambda p: (abs(p[1] - target), p[0]))
            if pick[1] not in [l.power_cycles for l in levels]:
                levels.append(Level(f"L{i}", pick[2], pick[0], pick[1], original))
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--bound", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    points = read_points(args.run_dir)
    original = original_cycles(args.run_dir)
    errors = sorted({round(p[0], 3) for p in points})
    print(f"{len(points)} accepted points, original = {original} power cycles")
    print(f"distinct errors: {errors[:8]}{' ...' if len(errors) > 8 else ''}")

    front = pareto(points)
    print(f"\nPareto front ({len(front)} points):")
    for error, cycles, _ in front:
        print(f"   error {error:6.3f}   cycles {cycles:>3}/{original}   "
              f"reduction {100 * (1 - cycles / original):5.1f}%")

    levels = choose_levels(points, original, args.levels, args.bound)
    print(f"\nLEVEL TABLE ({len(levels)} levels)")
    print(f"{'level':<7}{'error':>8}{'cycles':>9}{'reduction':>11}   knobs")
    print("-" * 64)
    for lv in levels:
        print(f"{lv.name:<7}{lv.error:>8.3f}{str(lv.power_cycles) + '/' + str(original):>9}"
              f"{lv.reduction * 100:>10.1f}%   {lv.knobs}")

    if len(levels) < args.levels:
        print(f"\nWARNING: only {len(levels)} distinct levels exist in this run. "
              "The plan does not expose a usable energy dial.")
    spread = max(l.error for l in levels) - min(l.error for l in levels)
    approx = [l for l in levels if l.error > 1e-9]
    if len(approx) > 1:
        gap = max(l.error for l in approx) - min(l.error for l in approx)
        if gap < 0.1 * spread:
            print(f"WARNING: the approximate levels span only {gap:.3f} error "
                  f"({', '.join(f'{l.error:.3f}' for l in approx)}). This is an "
                  "energy dial with no meaningful quality trade-off, so a "
                  "quality-aware selector has nothing to choose between.")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"run_dir": args.run_dir, "original_power_cycles": original,
                       "levels": [asdict(l) for l in levels]}, handle, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
