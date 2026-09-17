"""Full experiment matrix for the energy-adaptive selector.

Sweeps scenario x deadline x radio cost across every policy, writes tidy JSON
plus a readable table. The simulation is deterministic, so no repeats are needed.
"""

from __future__ import annotations

import json
import time

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, FORECASTERS, Server, build_levels, forecast_ewma,
                               forecast_persistence, mixed_trace, policy_affordable,
                               policy_fixed, simulate, varying_trace)

TRACES = ["RF_1", "RF_2", "RF_6", "RF_7", "RF_9"]
# Tasks got ~3x cheaper with the two-knob level table (L2 is 1.91 mJ, was 5.50),
# so shorter deadlines are now reachable -- which matters because the measured RF
# traces only vary at short horizons (RF_6 swings 2.72x at 15 s, 1.72x at 25 s).
DEADLINES = [8.0, 15.0, 25.0, 40.0]
COMMS_UJ = [0.0, 200.0]
DURATION = 360.0


def policy_set(levels):
    rows = [(f"fixed-{l.name}", policy_fixed(l.name), forecast_persistence, None)
            for l in levels]
    rows += [(f"adaptive-{n}", policy_affordable(), fn, None)
             for n, fn in FORECASTERS.items()]
    rows += [("reactive (on-device)", policy_fixed(levels[0].name),
              forecast_persistence, None),
             # Boundary hindsight oracle: the best a policy choosing once per
             # period could have done, judged after the fact.
             ("hindsight (boundary oracle)", policy_fixed(levels[0].name),
              forecast_persistence, None)]
    # server+reactive was dropped: with mid-task correction gone it takes the
    # server branch at every boundary and is identical to server-learned.
    rows += [("server-forecast", policy_affordable(), forecast_ewma, False),
             ("server-learned", policy_affordable(), forecast_ewma, True)]
    return rows


def scenarios():
    # "solar" is the only MEASURED scenario with real variation at the decision
    # timescale: 300 s of indoor solar with a moving source, 22 windows of 25 s
    # spanning 155-211 uA with no scaling. The RF traces are 25-52 s, so at a
    # 25 s deadline a window wraps the whole looped trace and flattens; at 15 s
    # they do vary (RF_6 swings 2.72x, RF_2 2.01x).
    yield "measured-rf", [(n, load_trace(f"traces/{n}.csv")) for n in TRACES]
    yield "measured-solar", [("solar", load_trace("traces/Solar_Indoor_Moving.csv"))]
    yield "varying", [("varying", varying_trace(TRACES))]
    yield "mixed", [("mixed", mixed_trace(TRACES))]


def main():
    import sys
    bench = sys.argv[1] if len(sys.argv) > 1 else "sobel"
    params = PowerParams()
    levels = build_levels(params, BENCHMARKS[bench])
    results = []
    started = time.time()
    print(f"benchmark: {bench}")
    for l in levels:
        print(f"   {l.name}  error {l.error:.3f}  {l.energy * 1e3:6.2f} mJ/task")

    for scen_name, scen in scenarios():
        for deadline in DEADLINES:
            for comms in COMMS_UJ:
                for label, choose, forecast, learn in policy_set(levels):
                    is_server = learn is not None
                    if comms > 0 and not is_server:
                        continue          # radio cost only applies to server policies
                    agg = []
                    for _n, trace in scen:
                        srv = Server(forecast_ewma, learn=learn) if is_server else None
                        agg.append(simulate(
                            trace, levels, params, DURATION, deadline,
                            choose, forecast,
                            hindsight=label.startswith("hindsight"),
                            reactive=label.startswith("reactive"),
                            server=srv,
                            comms_energy=(comms * 1e-6 if srv else 0.0)))
                    results.append({
                        "benchmark": bench,
                        "scenario": scen_name, "deadline_s": deadline,
                        "comms_uJ": comms if is_server else 0.0, "policy": label,
                        "completion": float(np.mean([a["completion_rate"] for a in agg])),
                        "mean_error": float(np.nanmean([a["mean_error"] for a in agg])),
                        "tasks_per_mJ": float(np.mean([a["tasks_per_mJ"] for a in agg])),
                        "spill_mJ": float(np.mean([a["energy_spilled_mJ"] for a in agg])),
                        "level_mix": np.sum([a["level_mix"] for a in agg], axis=0).tolist(),
                    })
                    print(f"  [{time.time() - started:6.0f}s] {scen_name:<9} "
                          f"d={deadline:>4.0f}s c={comms:>5.0f}uJ  {label:<22}"
                          f"{results[-1]['completion'] * 100:5.0f}%  "
                          f"err {results[-1]['mean_error']:.3f}", flush=True)

    out = f"reports/adaptive/results_{bench}.json"
    with open(out, "w") as handle:
        json.dump({"levels": [l.__dict__ for l in levels],
                   "duration_s": DURATION, "traces": TRACES,
                   "results": results}, handle, indent=1)
    print(f"\nwrote {out}  ({len(results)} rows, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
