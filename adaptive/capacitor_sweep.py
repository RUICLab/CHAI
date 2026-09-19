"""Direct voltage->energy->level lookup, swept over capacitor size.

    PYTHONPATH=. python3 -m adaptive.capacitor_sweep

The architecture under test, with no forecasting anywhere in it:

    E_available = 0.5 * C * (V^2 - V_release^2)        measured, not predicted
    pick the least-approximate level with
        E_required(Li) * (1 + margin) <= E_available
    else defer

At 68 uF this cannot work, and the reason is arithmetic rather than control: a
full capacitor holds 261 uJ while the cheapest sobel level needs 1914 uJ. Holding
a whole task takes 499 uF for L2, 811 uF for L1, and 3866 uF for L0.

But a larger tank also takes longer to fill. On solar at about 440 uW net, 500 uF
needs 4.4 s from empty, 1000 uF needs 8.7 s, and 4700 uF needs 41 s -- against
deadlines of 8 to 40 s. So the lookup becomes feasible and the device becomes
sluggish at the same time, and the sweep is what settles which wins where.

TASK ENERGY IS HELD FIXED at its 68 uF measurement. Only the buffer changes. An
earlier version of this sweep let build_levels rescale E_cycle with C, which made
tasks proportionally more expensive in bigger capacitors and drove completion to
zero -- an artifact, not a result.

Compared against the best fixed level and against the oracle tier (which reads
the true future and is a ceiling, not a policy).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, V_RELEASE, build_levels,
                               forecast_persistence, policy_fixed, simulate,
                               DEFER)

BASE = PowerParams()                       # 68 uF: where the cycle counts were measured
LEVELS = {b: build_levels(BASE, BENCHMARKS[b]) for b in ("sobel", "fft")}
CAPS_UF = [68, 250, 500, 1000, 2200, 4700]
DEADLINES = [8.0, 15.0, 25.0, 40.0]
MARGIN = 0.10                              # require 10% headroom over the table value


def lookup_policy(levels, capacitance, margin=MARGIN):
    """The device's whole decision: read V, compute stored energy, index the table."""
    def choose(_levels, v):
        available = 0.5 * capacitance * (max(v, V_RELEASE) ** 2 - V_RELEASE ** 2)
        for i, level in enumerate(levels):          # least approximate first
            if level.energy * (1.0 + margin) <= available:
                return i
        return DEFER
    return choose


def evaluate(res, levels):
    errs = np.array(res["errors"], dtype=float)
    periods = res["periods"]
    quality = float(np.sum(1.0 - errs)) if errs.size else 0.0
    return {"completion": res["completed"] / periods if periods else 0.0,
            "mean_error": float(errs.mean()) if errs.size else float("nan"),
            "quality": quality,
            "mix": res["level_mix"], "deferred": res.get("deferred", 0)}


def main():
    print(__doc__.strip())
    print("=" * 104)
    trace = load_trace("traces/Solar_Indoor_Moving.csv")

    for bench in ("sobel", "fft"):
        levels = LEVELS[bench]
        need = "  ".join(f"{l.name} needs "
                         f"{l.energy / (0.5 * (BASE.supply_voltage_limit ** 2 - V_RELEASE ** 2)) * 1e6:.0f}uF"
                         for l in levels)
        print(f"\n\n### {bench}   to hold a whole task: {need}")
        for deadline in DEADLINES:
            print(f"\n  deadline {deadline:.0f}s")
            print(f"    {'cap':>7}{'policy':<16}{'compl':>7}{'mean err':>10}"
                  f"{'quality':>9}{'defer':>7}   mix")
            for cap_uF in CAPS_UF:
                params = replace(BASE, capacitance=cap_uF * 1e-6)
                # best fixed level at THIS capacitance
                best = None
                for l in levels:
                    r = simulate(trace, levels, params, 295.0, deadline,
                                 policy_fixed(l.name), forecast_persistence)
                    m = evaluate(r, levels)
                    if best is None or m["quality"] > best[1]["quality"]:
                        best = (l.name, m)
                r = simulate(trace, levels, params, 295.0, deadline,
                             policy_fixed(levels[0].name), forecast_persistence,
                             voltage_policy=lookup_policy(levels, cap_uF * 1e-6))
                lk = evaluate(r, levels)
                r = simulate(trace, levels, params, 295.0, deadline,
                             policy_fixed(levels[0].name), forecast_persistence,
                             hindsight=True)
                orc = evaluate(r, levels)
                mark = "  <-- lookup wins" if lk["quality"] > best[1]["quality"] + 1e-9 else ""
                print(f"    {cap_uF:>5}uF{'best fixed=' + best[0]:<16}"
                      f"{best[1]['completion'] * 100:6.0f}%{best[1]['mean_error']:>10.3f}"
                      f"{best[1]['quality']:>9.1f}{0:>7}   {best[1]['mix']}")
                print(f"    {'':>7}{'V-lookup':<16}{lk['completion'] * 100:6.0f}%"
                      f"{lk['mean_error']:>10.3f}{lk['quality']:>9.1f}"
                      f"{lk['deferred']:>7}   {lk['mix']}{mark}")
                print(f"    {'':>7}{'oracle (ceiling)':<16}{orc['completion'] * 100:6.0f}%"
                      f"{orc['mean_error']:>10.3f}{orc['quality']:>9.1f}"
                      f"{orc['deferred']:>7}   {orc['mix']}")

    print("\n" + "=" * 104)
    print("Solar only, unlooped, 295 s. Task energies pinned at their 68 uF")
    print("measurement. Error per level comes from ONE image, so the table's")
    print("quality column is a point estimate, not a mean or a P95 over inputs.")


if __name__ == "__main__":
    main()
