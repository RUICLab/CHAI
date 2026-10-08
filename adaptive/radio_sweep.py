"""How cheap must a server query be before server forecasting pays for itself?

    PYTHONPATH=. python3 -m adaptive.radio_sweep

Sensitivity analysis on ONE solar recording, 295 s of a 300 s capture so nothing
loops. Not a campaign, and not proof that any particular radio will work: real
link energy, latency and retries need hardware measurement.

SCOPE. This measures one architecture -- a server query per decision period.
Batching several periods into one query, querying only on unusual conditions, or
piggybacking on traffic the device already sends would amortise the cost
differently and are untested here.

LATENCY IS ZERO, with no packet loss and no retransmission, so every number is
the server's BEST case. Real latency and retries move break-even lower.

THREE DEFECTS THIS VERSION FIXES, each of which made an earlier sweep unusable:

  * `learn=True` was unsound. The server calibrated on energy CONSUMED, which is
    censored: deferring, finishing early, or picking a cheap level all lower it.
    The learner read "I chose not to spend energy" as "no energy was there" and
    spiralled. On sobel/8s at free communication it hit the 0.05 floor and
    produced 2 in-bound tasks where the same server unlearned produced 13. Every
    run here uses learn=False, so radio cost is isolated from calibration.
  * A "defer" reply was free. Radio was charged only while a task ran, so at
    200 uJ with 34 deferrals the device paid 0.40 mJ instead of ~7.2 mJ. The
    query is now owed for the period whatever the server replies.
  * Zero-cost queries were counted as zero queries.

Because calibration used to react to radio spending, per-offered efficiency ROSE
with radio cost in some cells -- which silently broke the monotonicity that
break-even search assumes. Monotonicity is now checked and reported, not assumed.

PROTOCOL:
  Stage 1  every benchmark/deadline at 0 uJ. A cell that loses with FREE
           communication has no break-even, and is not swept further.
  Stage 2  only cells that beat a baseline at 0 uJ get the cost sweep and a
           binary-search break-even.

PREDECLARED READING OF BREAK-EVEN:
    above 200 uJ   conventional low-power radio viable
    10 - 200 uJ    needs a cheaper or amortised communication design
    below 10 uJ    needs specialised backscatter / piggybacking
    none           server forecasting offers no efficiency value in that cell
"""

from __future__ import annotations

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, ERROR_BOUND, Server, build_levels,
                               forecast_chronos, forecast_ewma,
                               forecast_ewma_slow, forecast_persistence,
                               policy_affordable, policy_fixed, simulate)

DURATION = 295.0
DEADLINES = [8.0, 15.0, 25.0, 40.0]
COSTS_UJ = [0, 1, 2, 5, 10, 20, 50, 100, 200]

P = PowerParams()
TRACE = load_trace("traces/Solar_Indoor_Moving.csv")


def slow_ewma(h, f, b):
    return forecast_ewma_slow(h, f, b, alpha=0.02)


# label, forecaster, bound (None = quality-first: most exact AFFORDABLE level,
# which falls through to the cheapest -- and the cheapest is out of bound).
VARIANTS = [
    ("server+ewma quality-first", forecast_ewma, None),
    ("server+ewma bound-aware", forecast_ewma, ERROR_BOUND),
    ("server+chronos quality-first", forecast_chronos, None),
    ("server+chronos bound-aware", forecast_chronos, ERROR_BOUND),
]


def metrics(res, levels, bound=ERROR_BOUND):
    tasks = sum(c for c, l in zip(res["level_mix"], levels)
                if l.error <= bound + 1e-12)
    out_of_bound = res["completed"] - tasks
    offered = res["energy_harvested_mJ"] + res["energy_spilled_mJ"]
    return {
        "per_offered": tasks / offered if offered else 0.0,
        "in_bound_completion": tasks / res["periods"] if res["periods"] else 0.0,
        "in_bound": tasks,
        "out_of_bound": out_of_bound,
        "radio_mJ": res.get("comms_paid_mJ", 0.0),
        "queries": res.get("queries", 0),
        "deferred": res.get("deferred", 0),
    }


def run_server(levels, deadline, forecast, cost_uJ, bound):
    """learn=False throughout: the calibrator's feedback signal is censored."""
    srv = Server(forecast, learn=False, bound=bound)
    return metrics(simulate(TRACE, levels, P, DURATION, deadline,
                            policy_affordable(), forecast,
                            server=srv, comms_energy=cost_uJ * 1e-6), levels)


def baselines(levels, deadline):
    fixed = metrics(simulate(TRACE, levels, P, DURATION, deadline,
                             policy_fixed(levels[1].name),
                             forecast_persistence), levels)
    slow = metrics(simulate(TRACE, levels, P, DURATION, deadline,
                            policy_affordable(), slow_ewma), levels)
    return fixed, slow


def verdict(cost):
    if cost is None:
        return "NO BREAK-EVEN"
    if cost > 200:
        return "conventional low-power radio viable"
    if cost >= 10:
        return "needs cheaper or amortised communication"
    return "needs specialised backscatter / piggybacking"


def break_even(levels, deadline, forecast, bound, target, lo, hi, tol=0.25):
    for _ in range(20):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if run_server(levels, deadline, forecast, mid, bound)["per_offered"] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print(__doc__.strip())
    print("=" * 112)

    survivors = []

    # ---------------------------------------------------------------- stage 1
    print("\nSTAGE 1 -- FREE COMMUNICATION (0 uJ). A cell that loses here has no")
    print("break-even at any price, so it is not swept.\n")
    for bench in ("sobel", "fft"):
        levels = build_levels(P, BENCHMARKS[bench])
        for deadline in DEADLINES:
            fixed, slow = baselines(levels, deadline)
            print(f"  {bench} / {deadline:.0f}s      "
                  f"fixed-L1 per_offered={fixed['per_offered']:.4f} "
                  f"in-bound={fixed['in_bound']}/{int(fixed['in_bound'] / max(fixed['in_bound_completion'], 1e-9)):d}"
                  f"   |   slow-ewma per_offered={slow['per_offered']:.4f} "
                  f"in-bound={slow['in_bound']}")
            print(f"    {'variant':<32}{'per_off':>9}{'in-bnd':>8}{'out-bnd':>9}"
                  f"{'defer':>7}{'queries':>9}   vs fixed / vs slow")
            for label, fc, bnd in VARIANTS:
                m = run_server(levels, deadline, fc, 0, bnd)
                d_f = (m["per_offered"] / fixed["per_offered"] - 1) * 100 \
                    if fixed["per_offered"] else float("nan")
                d_s = (m["per_offered"] / slow["per_offered"] - 1) * 100 \
                    if slow["per_offered"] else float("nan")
                beats = (m["per_offered"] > fixed["per_offered"]
                         or m["per_offered"] > slow["per_offered"])
                if beats:
                    survivors.append((bench, levels, deadline, label, fc, bnd,
                                      fixed, slow))
                print(f"    {label:<32}{m['per_offered']:>9.4f}{m['in_bound']:>8}"
                      f"{m['out_of_bound']:>9}{m['deferred']:>7}{m['queries']:>9}"
                      f"   {d_f:+7.1f}% / {d_s:+7.1f}%"
                      + ("   <-- SWEEP" if beats else ""))
            print()

    # ---------------------------------------------------------------- stage 2
    print("=" * 112)
    if not survivors:
        print("\nSTAGE 2 -- nothing to sweep.")
        print("No server variant beat either baseline in any cell with FREE")
        print("communication, so no cell has a radio-cost break-even. Radio price")
        print("is not the binding constraint in the cells tested.")
    else:
        print(f"\nSTAGE 2 -- cost sweep for the {len(survivors)} cell(s) that won at 0 uJ\n")
        for bench, levels, deadline, label, fc, bnd, fixed, slow in survivors:
            print(f"  {bench} / {deadline:.0f}s / {label}")
            print(f"    {'cost uJ':>8}{'per_off':>9}{'in-bnd':>8}{'radio mJ':>10}"
                  f"{'queries':>9}   vs fixed / vs slow")
            curve = []
            for c in COSTS_UJ:
                m = run_server(levels, deadline, fc, c, bnd)
                curve.append((c, m["per_offered"]))
                d_f = (m["per_offered"] / fixed["per_offered"] - 1) * 100
                d_s = (m["per_offered"] / slow["per_offered"] - 1) * 100
                print(f"    {c:>8}{m['per_offered']:>9.4f}{m['in_bound']:>8}"
                      f"{m['radio_mJ']:>10.2f}{m['queries']:>9}"
                      f"   {d_f:+7.1f}% / {d_s:+7.1f}%")
            rising = [(a, b) for (a, va), (b, vb) in zip(curve, curve[1:]) if vb > va + 1e-9]
            if rising:
                print(f"    NOT MONOTONE between {rising}: break-even search assumes")
                print("    efficiency falls as radio gets dearer. Treat the value below "
                      "as unreliable.")
            for bname, ref in (("fixed-L1", fixed["per_offered"]),
                               ("slow-ewma", slow["per_offered"])):
                above = [c for c, v in curve if v > ref]
                below = [c for c, v in curve if v <= ref]
                if not above:
                    print(f"    break-even vs {bname:<10}: {verdict(None)}")
                elif not below or min(below) < max(above):
                    print(f"    break-even vs {bname:<10}: > {COSTS_UJ[-1]} uJ "
                          f"({verdict(201)})")
                else:
                    be = break_even(levels, deadline, fc, bnd, ref,
                                    max(above), min(c for c in below if c > max(above)))
                    print(f"    break-even vs {bname:<10}: {be:6.1f} uJ ({verdict(be)})")
            print()

    print("=" * 112)
    print("learn=False throughout. Zero latency, no loss, no retransmission.")
    print("One solar recording. In-bound completion counts only levels inside the")
    print("error bound; out-of-bound results score zero and are shown separately.")


if __name__ == "__main__":
    main()
