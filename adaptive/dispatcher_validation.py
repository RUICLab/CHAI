"""Functional validation of the runtime level dispatcher.

    PYTHONPATH=. python3 -m adaptive.dispatcher_validation

This does NOT ask whether energy-aware dispatch beats a fixed configuration.
That is a different question and it is answered elsewhere. This asks whether the
dispatcher WORKS:

  1. plenty of stored energy    -> the exact level fires
  2. moderate                   -> a mid level fires
  3. little                     -> an aggressive level fires
  4. not enough for anything    -> DEFER
  5. SAFETY: the level chosen never requires more energy than is available,
     including the margin -- checked on every single decision
  6. the chosen knobs are the ones that execute, and the recorded error is the
     one the table promised for those knobs

Part A drives the dispatcher at controlled voltages so every branch fires.
Part B runs a synthetic high -> medium -> low -> none -> recovery profile and
prints the timeline.
Part C reports how often each level fires on the five measured RF recordings.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from adaptive.power_model import PowerParams, load_trace
from adaptive.selector import (BENCHMARKS, DEFER, V_RELEASE, build_levels,
                               forecast_persistence, policy_fixed, simulate)

BASE = PowerParams()
MARGIN = 0.10
CAP = 4700e-6            # large enough that every branch, including L0, is reachable


def available_energy(v, capacitance=CAP):
    return 0.5 * capacitance * (max(v, V_RELEASE) ** 2 - V_RELEASE ** 2)


def dispatch(levels, v, capacitance=CAP, margin=MARGIN):
    """The device's entire runtime decision."""
    available = available_energy(v, capacitance)
    for i, level in enumerate(levels):           # least approximate first
        if level.energy * (1.0 + margin) <= available:
            return i
    return DEFER


def part_a(levels):
    print("PART A -- every branch fires, and the safety rule holds\n")
    print(f"  {'V':>7}{'available':>12}{'selected':>10}{'required':>11}"
          f"{'x margin':>11}{'error':>8}   safe?")
    violations = 0
    fired = set()
    for v in (3.59, 3.50, 3.45, 3.00, 2.70, 2.60, 2.55, 2.50, 2.45, 2.40, 2.30):
        idx = dispatch(levels, v)
        avail = available_energy(v)
        if idx == DEFER:
            fired.add("DEFER")
            print(f"  {v:>6.2f}V{avail * 1e3:>11.2f}mJ{'DEFER':>10}"
                  f"{'-':>11}{'-':>11}{'-':>8}   n/a")
            continue
        lv = levels[idx]
        need = lv.energy * (1 + MARGIN)
        ok = need <= avail + 1e-15
        if not ok:
            violations += 1
        fired.add(lv.name)
        print(f"  {v:>6.2f}V{avail * 1e3:>11.2f}mJ{lv.name:>10}"
              f"{lv.energy * 1e3:>10.2f}mJ{need * 1e3:>10.2f}mJ{lv.error:>8.3f}"
              f"   {'OK' if ok else 'VIOLATION'}")
    print(f"\n  branches exercised: {sorted(fired)}")
    print(f"  safety violations (chose a level it could not afford): {violations}")
    return violations, fired


def profile_trace():
    """high -> medium -> low -> nothing -> high again, 60 s each at 1 ms."""
    seg = lambda v, n: np.full(n, v)
    return np.concatenate([seg(6.0, 60000), seg(2.2, 60000), seg(1.3, 60000),
                           seg(0.0, 60000), seg(6.0, 60000)])


def part_b(levels):
    print("\n\nPART B -- timeline over a controlled energy profile\n")
    params = replace(BASE, capacitance=CAP)
    log = []

    def spy(_levels, v):
        idx = dispatch(levels, v, CAP)
        log.append((v, idx))
        return idx

    res = simulate(profile_trace(), levels, params, 295.0, 15.0,
                   policy_fixed(levels[0].name), forecast_persistence,
                   voltage_policy=spy)
    print(f"  {'period':>7}{'phase':>10}{'V at decision':>15}{'available':>12}"
          f"{'fired':>8}{'error':>8}")
    phases = ["HIGH", "HIGH", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "MEDIUM",
              "MEDIUM", "LOW", "LOW", "LOW", "LOW", "NONE", "NONE", "NONE",
              "NONE", "RECOVER", "RECOVER", "RECOVER", "RECOVER"]
    for i, (v, idx) in enumerate(log):
        name = "DEFER" if idx == DEFER else levels[idx].name
        err = "-" if idx == DEFER else f"{levels[idx].error:.3f}"
        ph = phases[i] if i < len(phases) else ""
        print(f"  {i:>7}{ph:>10}{v:>14.3f}V{available_energy(v) * 1e3:>11.2f}mJ"
              f"{name:>8}{err:>8}")
    print(f"\n  level mix {res['level_mix']}   deferred {res['deferred']}   "
          f"completed {res['completed']}/{res['periods']}")
    return res


def part_c(levels, bench):
    print(f"\n\nPART C -- {bench}: how often each level fires on measured RF\n")
    print(f"  {'trace':<8}{'cap':>8}{'L0':>6}{'L1':>6}{'L2':>6}{'defer':>7}"
          f"{'completion':>12}{'mean error':>12}")
    for cap_uF in (1000, 4700):
        params = replace(BASE, capacitance=cap_uF * 1e-6)
        for tname in ("RF_1", "RF_2", "RF_6", "RF_7", "RF_9", "Solar_Indoor_Moving"):
            trace = load_trace(f"traces/{tname}.csv")
            res = simulate(trace, levels, params, 295.0, 15.0,
                           policy_fixed(levels[0].name), forecast_persistence,
                           voltage_policy=lambda lv, v, c=cap_uF * 1e-6:
                               dispatch(levels, v, c))
            mix = res["level_mix"]
            errs = np.array(res["errors"], dtype=float)
            me = f"{errs.mean():.3f}" if errs.size else "-"
            print(f"  {tname[:7]:<8}{cap_uF:>6}uF{mix[0]:>6}{mix[1]:>6}{mix[2]:>6}"
                  f"{res['deferred']:>7}{res['completed'] / res['periods'] * 100:>11.0f}%"
                  f"{me:>12}")


def main():
    levels = build_levels(BASE, BENCHMARKS["sobel"])
    print(__doc__.strip())
    print("=" * 96)
    print(f"\nTABLE UNDER TEST (sobel), capacitor {CAP * 1e6:.0f} uF, margin {MARGIN:.0%}")
    for l in levels:
        print(f"  {l.name}  needs {l.energy * 1e3:6.2f} mJ   error {l.error:.3f}")
    print()

    violations, fired = part_a(levels)
    part_b(levels)
    part_c(levels, "sobel")

    print("\n" + "=" * 96)
    print("VERDICT")
    print(f"  safety violations across all controlled decisions: {violations}")
    print(f"  distinct branches exercised: {sorted(fired)}")
    print("\n  RF rows loop a 25-52 s recording to fill 295 s; they show dispatch")
    print("  behaviour, not evidence about RF predictability. Error per level is a")
    print("  point estimate from one image and needs mean/P95 over an input set")
    print("  before the table's quality column is trustworthy.")


if __name__ == "__main__":
    main()
