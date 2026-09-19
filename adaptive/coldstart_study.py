"""Cold versus warm start: how much of the big-capacitor penalty is one-off?

    PYTHONPATH=. python3 -m adaptive.coldstart_study

The capacitor sweep started every run at 0 V, so a 4700 uF device spent its first
104 s of a 295 s run merely filling the tank before it could execute anything.
That cost is paid ONCE. A device deployed for hours amortises it to nothing, so
attributing the large-capacitor loss to the architecture rather than to the cold
start would be wrong.

    cold start   initial_v = 0 V        includes filling the capacitor
    warm start   initial_v = 3.59 V     steady state, tank already full

Startup latency is reported separately from post-boot behaviour so the two are
never conflated again.

Also widens the sweep beyond solar. The RF recordings are 25-52 s and the
simulator tiles them to fill a run, so RF rows carry the usual looping caveat --
they are indicative of capacitor dynamics, not evidence about RF predictability.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, V_RELEASE, build_levels, DEFER,
                               forecast_persistence, policy_fixed, simulate)

BASE = PowerParams()
LEVELS = {b: build_levels(BASE, BENCHMARKS[b]) for b in ("sobel", "fft")}
CAPS_UF = [500, 1000, 2200, 4700]
DEADLINES = [8.0, 15.0]
TRACES = [("solar", "Solar_Indoor_Moving", False), ("RF_2", "RF_2", True),
          ("RF_6", "RF_6", True), ("RF_7", "RF_7", True), ("RF_9", "RF_9", True)]
MARGIN = 0.10
DURATION = 295.0


def lookup_policy(levels, capacitance, margin=MARGIN):
    def choose(_levels, v):
        available = 0.5 * capacitance * (max(v, V_RELEASE) ** 2 - V_RELEASE ** 2)
        for i, level in enumerate(levels):
            if level.energy * (1.0 + margin) <= available:
                return i
        return DEFER
    return choose


def ev(res, levels):
    errs = np.array(res["errors"], dtype=float)
    p = res["periods"]
    return {"completion": res["completed"] / p if p else 0.0,
            "mean_error": float(errs.mean()) if errs.size else float("nan"),
            "quality": float(np.sum(1.0 - errs)) if errs.size else 0.0,
            "startup": res["startup_s"], "mix": res["level_mix"],
            "deferred": res.get("deferred", 0)}


def main():
    print(__doc__.strip())
    print("=" * 108)
    wins = {"cold": 0, "warm": 0}
    cells = {"cold": 0, "warm": 0}

    for bench in ("sobel", "fft"):
        levels = LEVELS[bench]
        print(f"\n\n{'#' * 108}\n### {bench}\n{'#' * 108}")
        for tname, fname, looped in TRACES:
            trace = load_trace(f"traces/{fname}.csv")
            tag = "  [LOOPED]" if looped else "  [unlooped]"
            for deadline in DEADLINES:
                print(f"\n  {tname}{tag}   deadline {deadline:.0f}s")
                print(f"    {'cap':>7}{'start':>7}{'startup':>9}"
                      f"{'best fixed':>12}{'q':>7}{'V-lookup':>10}{'q':>7}{'defer':>7}   verdict")
                for cap_uF in CAPS_UF:
                    params = replace(BASE, capacitance=cap_uF * 1e-6)
                    for label, v0 in (("cold", 0.0), ("warm", 3.59)):
                        best = None
                        for l in levels:
                            m = ev(simulate(trace, levels, params, DURATION, deadline,
                                            policy_fixed(l.name), forecast_persistence,
                                            initial_v=v0), levels)
                            if best is None or m["quality"] > best[1]["quality"]:
                                best = (l.name, m)
                        lk = ev(simulate(trace, levels, params, DURATION, deadline,
                                         policy_fixed(levels[0].name),
                                         forecast_persistence, initial_v=v0,
                                         voltage_policy=lookup_policy(levels, cap_uF * 1e-6)),
                                levels)
                        cells[label] += 1
                        won = lk["quality"] > best[1]["quality"] + 1e-9
                        if won:
                            wins[label] += 1
                        su = lk["startup"]
                        su_s = "never" if su is None else f"{su:.1f}s"
                        print(f"    {cap_uF:>5}uF{label:>7}{su_s:>9}"
                              f"{best[0]:>12}{best[1]['quality']:>7.1f}"
                              f"{'':>10}{lk['quality']:>7.1f}{lk['deferred']:>7}"
                              f"   {'LOOKUP WINS' if won else ''}")

    print("\n" + "=" * 108)
    for k in ("cold", "warm"):
        print(f"  {k}-start: lookup beat the best fixed level in "
              f"{wins[k]} of {cells[k]} cells")
    print("\nTask energy is still pinned at its 68 uF measurement. Larger")
    print("capacitors change brown-out frequency and checkpoint overhead, so a")
    print("hardware claim needs task energy re-measured per capacitor, or useful")
    print("progress and checkpoint cost modelled explicitly.")


if __name__ == "__main__":
    main()
