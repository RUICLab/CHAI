"""Chronos ON THE SERVER, paying the radio cost it would really require.

The campaign never ran this: adaptive-chronos got Chronos for free on a device
that cannot hold it, while server-* paid radio for plain ewma.
Unlooped solar, 295 s of a 300 s recording.
"""
import numpy as np
from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, ERROR_BOUND, Server, build_levels,
                               forecast_chronos, forecast_ewma, forecast_ewma_slow,
                               forecast_persistence, policy_affordable,
                               policy_fixed, simulate)

P = PowerParams()
tr = load_trace("traces/Solar_Indoor_Moving.csv")
DUR = 295.0
COMMS = 200e-6          # J per server round trip, charged to the device

def eff(res, levels, bound=ERROR_BOUND):
    tasks = sum(c for c,l in zip(res["level_mix"], levels) if l.error <= bound+1e-12)
    off = res["energy_harvested_mJ"] + res["energy_spilled_mJ"]
    return tasks, (tasks/off if off else 0.0)

for bench in ("sobel","fft"):
    L = build_levels(P, BENCHMARKS[bench])
    print(f"\n### {bench}  (unlooped solar, radio {COMMS*1e6:.0f} uJ/period for server rows)")
    for dl in (8.0, 15.0, 25.0, 40.0):
        print(f"\n  deadline {dl:.0f}s")
        print(f"    {'policy':<34}{'radio':>7}{'compl':>7}{'per_offered':>13}")
        rows = []
        r = simulate(tr, L, P, DUR, dl, policy_fixed(L[1].name), forecast_persistence)
        rows.append(("fixed-L1 (no radio, no forecast)", "no", r))
        r = simulate(tr, L, P, DUR, dl, policy_affordable(),
                     lambda h,f,b: forecast_ewma_slow(h,f,b,alpha=0.02))
        rows.append(("on-device slow ewma a=0.02", "no", r))
        for label, fc in (("SERVER + ewma (current)", forecast_ewma),
                          ("SERVER + CHRONOS", forecast_chronos)):
            srv = Server(fc, learn=True)
            r = simulate(tr, L, P, DUR, dl, policy_affordable(), fc,
                         server=srv, comms_energy=COMMS)
            rows.append((label, "YES", r))
        base = None
        for label, radio, res in rows:
            tasks, off = eff(res, L)
            if base is None: base = off
            d = (off/base - 1)*100 if base else float("nan")
            print(f"    {label:<34}{radio:>7}{res['completion_rate']*100:6.0f}%"
                  f"{off:>13.3f}  [{d:+6.1f}%]")
