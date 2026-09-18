"""Phase B: does a slower forecaster change the CONTROLLER's behaviour?

    PYTHONPATH=. python3 -m adaptive.forecast_policy_study

Phase A (adaptive/forecast_study.py) showed slow-memory averaging cuts forecast
error on held-out solar. Lower forecast error is not the deliverable. This asks
the question that decides whether the controller changes:

    false deferrals   declined a task that would in fact have finished
    false runs        started a task that then missed its deadline, burning the
                      period's harvest for nothing
    per-offered       in-bound tasks per mJ the environment actually offered
                      (harvested + spilled), so declining work is charged for
                      the harvest it spills
    completion        reported alongside, so nothing looks efficient by idling

Run on the 300 s solar recording ONLY, for 295 s, so the trace is never looped.
The RF recordings are 25-52 s and `simulate()` tiles them to fill a run; a
long-window average would then be reading repeated copies of its own past. Phase
A also showed RF supports 11 independent windows at 8 s and none at 40 s, so RF
cannot settle this either way.

DECISION RULE, fixed before the numbers:

  * improves per_offered AND does not materially cut completion  -> adopt
  * improves forecast error but not policy outcome               -> do not adopt,
                                                                    do not rerun
                                                                    the campaign
  * solar and RF disagree                                        -> domain-specific
                                                                    policy, not a
                                                                    global switch
"""

from __future__ import annotations

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, ERROR_BOUND, build_levels,
                               forecast_ewma, forecast_ewma_slow,
                               forecast_mean_window, forecast_persistence,
                               policy_bound_aware_defer, policy_fixed, simulate)

DURATION = 295.0            # < 300 s recording: no looping
DEADLINES = [8.0, 15.0, 25.0, 40.0]

CANDIDATES = [
    ("persistence", forecast_persistence),
    ("ewma a=0.3 (current)", forecast_ewma),
    ("ewma a=0.05", lambda h, f, b: forecast_ewma_slow(h, f, b, alpha=0.05)),
    ("ewma a=0.02", forecast_ewma_slow),
    ("mean 30s", forecast_mean_window(30)),
    ("mean 60s", forecast_mean_window(60)),
]


def efficiency(result, levels, bound=ERROR_BOUND):
    tasks = sum(c for c, l in zip(result["level_mix"], levels)
                if l.error <= bound + 1e-12)
    harvested = result["energy_harvested_mJ"]
    spilled = result["energy_spilled_mJ"]
    offered = harvested + spilled
    return tasks, (tasks / offered if offered else 0.0)


def main():
    params = PowerParams()
    trace = load_trace("traces/Solar_Indoor_Moving.csv")
    print(__doc__.split("DECISION RULE")[0].strip())
    print("=" * 104)

    for bench in ("sobel", "fft"):
        levels = build_levels(params, BENCHMARKS[bench])
        print(f"\n### {bench}   (solar, {DURATION:.0f}s, not looped)")
        for deadline in DEADLINES:
            print(f"\n  deadline {deadline:.0f}s")
            print(f"    {'forecaster':<24}{'compl':>7}{'defer':>7}{'false_def':>11}"
                  f"{'missed':>8}{'spill mJ':>10}{'per_offered':>13}")

            # Control: never defers, never forecasts.
            ref = simulate(trace, levels, params, DURATION, deadline,
                           policy_fixed(levels[1].name), forecast_persistence)
            r_tasks, r_off = efficiency(ref, levels)
            print(f"    {'fixed-L1 (no forecast)':<24}"
                  f"{ref['completion_rate'] * 100:6.0f}%{0:>7}{0:>11}"
                  f"{ref['missed']:>8}{ref['energy_spilled_mJ']:>10.1f}"
                  f"{r_off:>13.3f}")

            for label, fc in CANDIDATES:
                res = simulate(trace, levels, params, DURATION, deadline,
                               policy_bound_aware_defer(), fc,
                               state_aware_defer=True)
                tasks, off = efficiency(res, levels)
                delta = (off / r_off - 1) * 100 if r_off else float("nan")
                print(f"    {label:<24}{res['completion_rate'] * 100:6.0f}%"
                      f"{res['deferred']:>7}{res['false_defers']:>11}"
                      f"{res['missed']:>8}{res['energy_spilled_mJ']:>10.1f}"
                      f"{off:>13.3f}  [{delta:+5.1f}% vs fixed]")

    print("\n" + "=" * 104)
    print("Read per_offered against fixed-L1, not against the other forecasters:")
    print("the question is whether ANY of them earns the deferral machinery, not")
    print("which of them is least bad.")


if __name__ == "__main__":
    main()
