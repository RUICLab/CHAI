"""Aggressive when energy is scarce: pick the level from RECENT HARVEST RATE.

    PYTHONPATH=. python3 -m adaptive.rate_table_study

    rate >= r_hi        run L0, the exact level
    r_lo <= rate < r_hi run L1
    rate <  r_lo        run L2, the cheapest

Why the input is the rate and not the stored energy. The obvious reading of "if I
have X energy use L0" cannot be implemented on this device: a full 68 uF
capacitor holds 261 uJ, while the cheapest task in the suite needs 1436 uJ and
sobel-L0 needs 14836 uJ. Every task costs between 6 and 57 capacitor-fulls, so
"do I have enough energy for L0" is always no. The device never banks a task's
worth of energy; it harvests and burns through repeated brown-outs. Measuring the
capacitor instead of the rate is what the voltage-table study did, and it failed
for exactly this reason -- the voltage read 3.583 V at every single decision
instant, a constant.

Recent harvest rate does vary: 154.7-199.1 uA across decision instants on the
held-out solar half. It is also a MEASUREMENT of the past, not a forecast of the
future, and the mapping from rate to level is learned from which level actually
worked -- so unlike the forecasters there is no energy model and no calibration
in between to be biased.

PROTOCOL. Thresholds grid-searched on the FIRST half of the solar recording,
scored on the SECOND. The fixed-level comparator is chosen the same way on the
same half. Both bounds reported: under 0.05 only L0 and L1 qualify and L1 costs
5x less for identical credit, so a constant is already optimal; under the
project's own 0.30 all three qualify and there is a real choice to make.
"""

from __future__ import annotations

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, build_levels, forecast_persistence,
                               policy_fixed, simulate)

P = PowerParams()
SOLAR = load_trace("traces/Solar_Indoor_Moving.csv")
HALF = len(SOLAR) // 2
SELECT, EVALUATE = SOLAR[:HALF], SOLAR[HALF:]
DURATION = 145.0
DEADLINES = [8.0, 15.0, 25.0, 40.0]
BOUNDS = [0.05, 0.30]

# Thresholds in amperes, spanning the observed range with headroom either side.
GRID = np.round(np.arange(120e-6, 260e-6, 10e-6), 9)


def table(r_hi, r_lo):
    def choose(levels, rate):
        if rate >= r_hi:
            return 0
        if rate >= r_lo:
            return min(1, len(levels) - 1)
        return len(levels) - 1
    return choose


def score(trace, levels, deadline, bound, rate_policy=None, fixed=None):
    res = simulate(trace, levels, P, DURATION, deadline,
                   policy_fixed(fixed or levels[1].name),
                   forecast_persistence, rate_policy=rate_policy)
    tasks = sum(c for c, l in zip(res["level_mix"], levels)
                if l.error <= bound + 1e-12)
    offered = res["energy_harvested_mJ"] + res["energy_spilled_mJ"]
    return {"per_offered": tasks / offered if offered else 0.0,
            "in_bound": tasks, "mix": res["level_mix"]}


def main():
    print(__doc__.strip())
    print("=" * 104)

    wins = 0
    total = 0
    for bench in ("sobel", "fft"):
        levels = build_levels(P, BENCHMARKS[bench])
        for bound in BOUNDS:
            ok = ", ".join(l.name for l in levels if l.error <= bound + 1e-12)
            print(f"\n\n### {bench}   bound {bound}   (in bound: {ok})")
            print(f"  {'deadline':<9}{'method':<20}{'chosen on select':<26}"
                  f"{'HELD-OUT per_off':>18}{'in-bnd':>8}")
            for deadline in DEADLINES:
                fixed_sel = {l.name: score(SELECT, levels, deadline, bound,
                                           fixed=l.name)["per_offered"]
                             for l in levels}
                best_fixed = max(fixed_sel, key=lambda k: fixed_sel[k])
                f_ev = score(EVALUATE, levels, deadline, bound, fixed=best_fixed)

                best, best_v = None, -1.0
                for r_hi in GRID:
                    for r_lo in GRID[GRID <= r_hi]:
                        s = score(SELECT, levels, deadline, bound,
                                  rate_policy=table(r_hi, r_lo))
                        if s["per_offered"] > best_v:
                            best_v, best = s["per_offered"], (r_hi, r_lo)
                t_ev = score(EVALUATE, levels, deadline, bound,
                             rate_policy=table(*best))

                d = ((t_ev["per_offered"] / f_ev["per_offered"] - 1) * 100
                     if f_ev["per_offered"] else float("nan"))
                total += 1
                if d > 1e-9:
                    wins += 1
                print(f"  {str(int(deadline)) + 's':<9}{'best fixed':<20}"
                      f"{best_fixed:<26}{f_ev['per_offered']:>18.4f}"
                      f"{f_ev['in_bound']:>8}")
                print(f"  {'':<9}{'rate table':<20}"
                      f"{f'hi={best[0] * 1e6:.0f}uA lo={best[1] * 1e6:.0f}uA':<26}"
                      f"{t_ev['per_offered']:>18.4f}{t_ev['in_bound']:>8}"
                      f"   [{d:+6.1f}%]  mix={t_ev['mix']}")

    print("\n" + "=" * 104)
    print(f"rate table beat the best fixed level in {wins} of {total} cells")
    print("One solar recording, thresholds and comparator both chosen on its")
    print("first half. Shows whether the idea is worth pursuing, not that it")
    print("generalises to RF or to other harvesters.")


if __name__ == "__main__":
    main()
