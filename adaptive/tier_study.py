"""Stage 1: what is dynamic-quality tiering worth, with a perfect budget?

    PYTHONPATH=. python3 -m adaptive.tier_study

THE OBJECTIVE, corrected. Earlier studies scored against a FIXED error bound of
0.05. That bound excludes sobel-L2 (0.280) outright, so the controller could only
ever choose L1, and every "adaptation" study was really asking which constant to
run. The intended policy has a DYNAMIC quality allowance:

    B = energy available over this task's window
    if   B >= energy(L0): run L0     exact
    elif B >= energy(L1): run L1     moderate
    elif B >= energy(L2): run L2     aggressive
    else:                 defer      drop rather than start something doomed

Plenty of energy buys quality; scarcity buys a completed-but-rougher result
instead of a missed one.

WHAT "AVAILABLE" MEANS. Not the charge in the capacitor: 68 uF holds 261 uJ while
the cheapest sobel task needs 1914 uJ, so every task spans 7-57 charge cycles.
The budget is the energy delivered ACROSS THE WINDOW, which is why this stage
uses an oracle -- it replays the real trace for the period and tiers on the
energy that genuinely arrives.

This measures the CEILING. It is not deployable: no device knows its next 40 s of
harvest. Stage 2 replaces the oracle with a causal estimator and measures how
much of this ceiling survives.

METRICS. Completion alone rewards always-L2; mean error alone rewards always-L0.
The headline is quality-adjusted timely results, sum(1 - error) over tasks that
finished inside their deadline, which prices both. p95 error is reported because
a policy can look good on the mean while occasionally shipping junk.
"""

from __future__ import annotations

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, build_levels, forecast_persistence,
                               policy_fixed, simulate)

P = PowerParams()
DEADLINES = [8.0, 15.0, 25.0, 40.0]

# Solar is 300 s and used unlooped. The RF recordings are 25-52 s and the
# simulator tiles them; for an ORACLE that is acceptable (it reads the true
# future either way and learns nothing), but it is flagged, and no deployable
# claim may rest on the looped rows.
TRACES = [("solar", "Solar_Indoor_Moving", 295.0, False),
          ("RF_2", "RF_2", 295.0, True),
          ("RF_6", "RF_6", 295.0, True)]


def evaluate(res, levels):
    errs = np.array(res["errors"], dtype=float)
    periods = res["periods"]
    offered = res["energy_harvested_mJ"] + res["energy_spilled_mJ"]
    quality = float(np.sum(1.0 - errs)) if errs.size else 0.0
    return {
        "completion": res["completed"] / periods if periods else 0.0,
        "mean_error": float(errs.mean()) if errs.size else float("nan"),
        "p95_error": float(np.percentile(errs, 95)) if errs.size else float("nan"),
        "quality": quality,                       # sum(1 - error), on-time only
        "quality_per_period": quality / periods if periods else 0.0,
        "quality_per_mJ": quality / offered if offered else 0.0,
        "mix": res["level_mix"],
        "deferred": res.get("deferred", 0),
    }


def main():
    print(__doc__.strip())
    print("=" * 112)

    for bench in ("sobel", "fft"):
        levels = build_levels(P, BENCHMARKS[bench])
        tiers = "   ".join(f"{l.name} err {l.error:.3f} @ {l.energy * 1e3:.2f} mJ"
                           for l in levels)
        print(f"\n\n{'#' * 112}\n### {bench}    {tiers}\n{'#' * 112}")

        for tname, fname, dur, looped in TRACES:
            trace = load_trace(f"traces/{fname}.csv")
            print(f"\n  trace {tname}" + ("   [LOOPED - oracle only, not deployable evidence]"
                                          if looped else "   [unlooped]"))
            for deadline in DEADLINES:
                print(f"\n    deadline {deadline:.0f}s")
                print(f"      {'policy':<22}{'compl':>7}{'mean err':>10}{'p95 err':>9}"
                      f"{'quality':>9}{'q/period':>10}{'q/mJ':>8}{'defer':>7}   mix")
                rows = []
                for l in levels:
                    rows.append((f"fixed-{l.name}",
                                 simulate(trace, levels, P, dur, deadline,
                                          policy_fixed(l.name), forecast_persistence)))
                rows.append(("ORACLE tier",
                             simulate(trace, levels, P, dur, deadline,
                                      policy_fixed(levels[0].name),
                                      forecast_persistence, hindsight=True)))
                rows.append(("ORACLE tier + defer",
                             simulate(trace, levels, P, dur, deadline,
                                      policy_fixed(levels[0].name),
                                      forecast_persistence, hindsight=True,
                                      hindsight_defer=True)))
                best_q = max(evaluate(r, levels)["quality"] for _, r in rows)
                for label, res in rows:
                    m = evaluate(res, levels)
                    star = "  <-- best quality" if m["quality"] >= best_q - 1e-9 else ""
                    print(f"      {label:<22}{m['completion'] * 100:6.0f}%"
                          f"{m['mean_error']:>10.3f}{m['p95_error']:>9.3f}"
                          f"{m['quality']:>9.1f}{m['quality_per_period']:>10.3f}"
                          f"{m['quality_per_mJ']:>8.3f}{m['deferred']:>7}   "
                          f"{m['mix']}{star}")

    print("\n" + "=" * 112)
    print("quality = sum(1 - error) over tasks finished inside their deadline.")
    print("An oracle result is a CEILING, not a policy: it reads the real future.")
    print("Stage 2 must replace it with a causal estimator before any of this is")
    print("deployable. Looped RF rows are oracle-only and carry no such claim.")


if __name__ == "__main__":
    main()
