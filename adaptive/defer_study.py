"""Targeted study: does state-aware deferral help, and under which denominator?

    PYTHONPATH=. python3 -m adaptive.defer_study

Not a campaign. Two deadlines, two benchmarks, three forecasters, three
policies, so the question is answered without another 448-cell run.

What is being tested
--------------------
The affordability test used by every predicting policy asks only whether the
INCOMING harvest over the next period covers a task. It ignores the charge
already in the capacitor, so it can refuse to start work while the energy to do
that work is sitting in the buffer. At 68 uF a full capacitor holds 261 uJ
against an L1 task's 3111 uJ -- about 8% -- so the effect is small here but it
is a correctness defect, not a tuning choice, and it grows with capacitance.

`policy_state_aware_defer` replaces the scalar comparison with a replay: copy
the live PowerState, run the period forward, and defer only if the task really
does not finish. That inherits stored charge, boot thresholds, the supervisor's
constant draw, the voltage clamp and spill from the model rather than
approximating any of them.

Decision rule, fixed BEFORE the numbers were seen
-------------------------------------------------
* realistic (ewma/persistence) state-aware defer improves per_load AND
  per_offered without materially reducing completion  -> expand to a
  confirmatory campaign;
* improves per_load only, per_offered still negative  -> report as a
  compute-energy optimisation that does not improve whole-system utilisation;
* only the perfect forecast wins                      -> stop; the mechanism is
  real but not deployable with the forecasts we have.

per_load is a legitimate PRIMARY compute-efficiency metric here, because the
paper's original cost metric is MCU power cycles. per_offered and spill stay
visible so whole-system claims cannot be made on per_load alone.
"""

from __future__ import annotations

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, ERROR_BOUND, build_levels,
                               cheapest_within_bound, forecast_ewma,
                               forecast_oracle, forecast_persistence,
                               policy_bound_aware_defer, policy_fixed, simulate)

DEADLINES = [8.0, 15.0]
FORECASTS = [("perfect", forecast_oracle),
             ("persistence", forecast_persistence),
             ("ewma", forecast_ewma)]
TRACES = ["RF_2", "RF_6"]
DURATION = 360.0


def efficiency(result, levels, bound=ERROR_BOUND):
    """In-bound tasks under the three denominators that disagree."""
    tasks = sum(c for c, l in zip(result["level_mix"], levels)
                if l.error <= bound + 1e-12)
    harvested = result["energy_harvested_mJ"]
    spilled = result["energy_spilled_mJ"]
    load = result["energy_to_load_mJ"]
    return {
        "tasks": tasks,
        "per_harvest": tasks / harvested if harvested else 0.0,
        "per_load": tasks / load if load else 0.0,
        "per_offered": tasks / (harvested + spilled) if (harvested + spilled) else 0.0,
    }


def main():
    params = PowerParams()
    print(__doc__.split("Decision rule")[0].strip())
    print("\n" + "=" * 118)

    verdicts = []
    for bench in ("sobel", "fft"):
        levels = build_levels(params, BENCHMARKS[bench])
        idx = cheapest_within_bound(levels, ERROR_BOUND)
        print(f"\n{bench}: cheapest in-bound level is {levels[idx].name} "
              f"(error {levels[idx].error}, {levels[idx].energy * 1e3:.2f} mJ)")

        for deadline in DEADLINES:
            print(f"\n  deadline {deadline:.0f}s")
            print(f"    {'policy':<34}{'compl':>7}{'defer':>7}{'false':>7}"
                  f"{'spill':>8}{'/harvest':>10}{'/load':>9}{'/offered':>10}")

            base = None
            for trace_name in TRACES:
                trace = load_trace(f"traces/{trace_name}.csv")

                runs = [("fixed-L1", policy_fixed(levels[idx].name),
                         forecast_oracle, False)]
                for fname, fn in FORECASTS:
                    runs.append((f"defer (incoming-only) / {fname}",
                                 policy_bound_aware_defer(), fn, False))
                for fname, fn in FORECASTS:
                    runs.append((f"defer (state-aware) / {fname}",
                                 policy_bound_aware_defer(), fn, True))

                print(f"    -- trace {trace_name}")
                base = None
                for label, policy, forecast, state_aware in runs:
                    result = simulate(trace, levels, params, DURATION, deadline,
                                      policy, forecast,
                                      state_aware_defer=state_aware)
                    eff = efficiency(result, levels)
                    if base is None:
                        base = eff
                    tail = ""
                    if label != "fixed-L1":
                        def pct(key):
                            b = base[key]
                            return f"{(eff[key] / b - 1) * 100:+6.1f}%" if b else "   n/a"
                        tail = f"  [{pct('per_harvest')}{pct('per_load')}{pct('per_offered')}]"
                        verdicts.append((bench, deadline, trace_name, label,
                                         result, eff, base))
                    print(f"    {label:<34}{result['completion_rate'] * 100:6.0f}%"
                          f"{result['deferred']:>7}{result.get('false_defers', 0):>7}"
                          f"{result['energy_spilled_mJ']:>8.1f}"
                          f"{eff['per_harvest']:>10.3f}{eff['per_load']:>9.3f}"
                          f"{eff['per_offered']:>10.3f}{tail}")

    print("\n" + "=" * 118)
    print("DECISION RULE (fixed in advance)")
    realistic = [v for v in verdicts
                 if "state-aware" in v[3] and "perfect" not in v[3]]
    gained_load = [v for v in realistic if v[5]["per_load"] > v[6]["per_load"]]
    gained_off = [v for v in realistic if v[5]["per_offered"] > v[6]["per_offered"]]
    print(f"  realistic state-aware runs: {len(realistic)}")
    print(f"    improved per_load:    {len(gained_load)}")
    print(f"    improved per_offered: {len(gained_off)}")
    if gained_load and gained_off and len(gained_off) >= len(realistic) / 2:
        print("  -> EXPAND: realistic defer improves both. Run the confirmatory campaign.")
    elif gained_load:
        print("  -> COMPUTE-ONLY: improves energy the MCU spends, not whole-system")
        print("     utilisation. Report as a compute-energy optimisation.")
    else:
        print("  -> STOP: only a perfect forecast wins. The mechanism is real but")
        print("     not deployable with the forecasts available.")


if __name__ == "__main__":
    main()
