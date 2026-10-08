"""Pick the level from measured capacitor voltage. No forecast, no radio.

    PYTHONPATH=. python3 -m adaptive.voltage_table_study

Everything tried so far predicted the future and lost to a constant. This asks
whether the device can adapt using only a quantity it already knows exactly: the
voltage on its own capacitor at the moment it must choose.

The policy is a lookup table, two thresholds over three levels:

    V >= v_hi           run L0, the exact level
    v_lo <= V < v_hi    run L1
    V < v_lo            run L2, the cheapest

That is a comparison and a table index on the MSP430. No model, no history
window, no server, and nothing to be wrong about -- the voltage is measured, not
forecast.

WHY THIS MIGHT MATTER AT ONE BOUND AND NOT THE OTHER. Under a 0.05 error bound
only L0 and L1 qualify and L1 costs 5x less for the same credit, so "always L1"
is already optimal and no table can beat it. Under the project's own
CHAI_ERROR_BOUND of 0.30 all three levels qualify, the best fixed level differs
by scenario and deadline, and there is a real choice for a table to make. Both
bounds are reported.

PROTOCOL. Thresholds are grid-searched on the FIRST half of the solar recording
and scored on the SECOND half, which the search never sees. The fixed-level
comparator is chosen the same way on the same half, so neither side gets to pick
its answer with knowledge of the test data.
"""

from __future__ import annotations

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, DEFER, V_RELEASE, build_levels,
                               forecast_persistence, policy_fixed, simulate)

P = PowerParams()
SOLAR = load_trace("traces/Solar_Indoor_Moving.csv")
HALF = len(SOLAR) // 2
SELECT, EVALUATE = SOLAR[:HALF], SOLAR[HALF:]
DURATION = 145.0                      # each half is 150 s
DEADLINES = [8.0, 15.0, 25.0, 40.0]
BOUNDS = [0.05, 0.30]

# Candidate thresholds across the operating window (2.283 V release to 3.59 V clamp).
GRID = np.round(np.arange(2.4, 3.60, 0.1), 2)


def table_policy(v_hi, v_lo):
    def choose(levels, v):
        if v >= v_hi:
            return 0
        if v >= v_lo:
            return min(1, len(levels) - 1)
        return len(levels) - 1
    return choose


def score(trace, levels, deadline, bound, voltage_policy=None, fixed=None):
    """In-bound tasks per offered joule, and in-bound completion."""
    res = simulate(trace, levels, P, DURATION, deadline,
                   policy_fixed(fixed) if fixed else policy_fixed(levels[1].name),
                   forecast_persistence, voltage_policy=voltage_policy)
    tasks = sum(c for c, l in zip(res["level_mix"], levels)
                if l.error <= bound + 1e-12)
    offered = res["energy_harvested_mJ"] + res["energy_spilled_mJ"]
    return {"per_offered": tasks / offered if offered else 0.0,
            "in_bound": tasks,
            "completion": res["completion_rate"],
            "mix": res["level_mix"]}


def main():
    print(__doc__.strip())
    print("=" * 100)

    for bench in ("sobel", "fft"):
        levels = build_levels(P, BENCHMARKS[bench])
        for bound in BOUNDS:
            ok = [l.name for l in levels if l.error <= bound + 1e-12]
            print(f"\n\n### {bench}   error bound {bound}   "
                  f"(levels in bound: {', '.join(ok)})")
            print(f"  {'deadline':<10}{'method':<26}{'chosen on select half':<26}"
                  f"{'HELD-OUT per_offered':>21}{'in-bnd':>8}")
            for deadline in DEADLINES:
                # --- comparator: best FIXED level, chosen on the select half ---
                fixed_sel = {l.name: score(SELECT, levels, deadline, bound,
                                           fixed=l.name)["per_offered"]
                             for l in levels}
                best_fixed = max(fixed_sel, key=lambda k: fixed_sel[k])
                f_eval = score(EVALUATE, levels, deadline, bound, fixed=best_fixed)

                # --- table: thresholds grid-searched on the select half ---
                best, best_v = None, -1.0
                for v_hi in GRID:
                    for v_lo in GRID[GRID <= v_hi]:
                        s = score(SELECT, levels, deadline, bound,
                                  voltage_policy=table_policy(v_hi, v_lo))
                        if s["per_offered"] > best_v:
                            best_v, best = s["per_offered"], (v_hi, v_lo)
                t_eval = score(EVALUATE, levels, deadline, bound,
                               voltage_policy=table_policy(*best))

                delta = ((t_eval["per_offered"] / f_eval["per_offered"] - 1) * 100
                         if f_eval["per_offered"] else float("nan"))
                print(f"  {str(int(deadline)) + 's':<10}{'best fixed level':<26}"
                      f"{best_fixed:<26}{f_eval['per_offered']:>21.4f}"
                      f"{f_eval['in_bound']:>8}")
                print(f"  {'':<10}{'voltage table':<26}"
                      f"{f'v_hi={best[0]}  v_lo={best[1]}':<26}"
                      f"{t_eval['per_offered']:>21.4f}{t_eval['in_bound']:>8}"
                      f"   [{delta:+6.1f}% vs fixed]  mix={t_eval['mix']}")

    print("\n" + "=" * 100)
    print("Thresholds and comparator both chosen on the first half of one solar")
    print("recording and scored on the second. One recording, one harvester type:")
    print("this shows whether the idea is worth pursuing, not that it generalises.")


if __name__ == "__main__":
    main()
