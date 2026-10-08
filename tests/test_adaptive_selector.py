"""Regressions for the defects that invalidated the first 480 adaptive runs.

Each test drives the SELECTOR and inspects what it actually did. Earlier
versions of these tests exercised PowerState directly, or asserted identities
that held regardless of the behaviour they claimed to check.
"""

from dataclasses import replace

import numpy as np

from adaptive.power_model import PowerParams, PowerState, hold_until, simulate as power_simulate
from adaptive.selector import (I_ACTIVE, SOBEL_LEVELS, V_RELEASE, build_levels,
                               forecast_persistence, hindsight_level,
                               policy_fixed, simulate)

P = PowerParams()
LEVELS = build_levels(P, SOBEL_LEVELS)


def _trace(n=4000):
    return np.tile(np.array([2.0, 3.2, 1.4, 4.1, 0.9]), n // 5)


def _run(choose=None, deadline_s=30.0, **kw):
    return simulate(_trace(), LEVELS, P, duration_s=240.0, deadline_s=deadline_s,
                    choose=choose or policy_fixed("L1"),
                    forecast=forecast_persistence, **kw)


def test_selector_runs_the_device_below_the_supervisor_threshold():
    """The old selector re-derived the power loop without KEEP_ALIVE and cut off
    at 3.4 V, while tasks were priced assuming discharge to 2.283 V."""
    result = _run()
    assert result["min_powered_v"] is not None, "device never powered on"
    assert result["min_powered_v"] < P.v_off - 0.5, (
        f"device never ran below {P.v_off} V (min {result['min_powered_v']:.3f} V); "
        "KEEP_ALIVE is not being applied")
    assert result["min_powered_v"] >= P.pmm_off - 1e-6


def test_no_level_changes_inside_a_period():
    """Mid-task switching kept work accumulated under one level and recorded
    another level's error. Counted directly rather than inferred."""
    for reactive in (False, True):
        result = _run(reactive=reactive)
        assert result["mid_task_switches"] == 0, (
            f"level changed mid-period {result['mid_task_switches']} times "
            f"(reactive={reactive})")
        assert len(result["decisions"]) == result["periods"]


def test_selector_power_dynamics_match_the_standalone_power_model():
    """Same input, same dynamics. Comparing two PowerState objects to each other
    proved nothing, so drive the selector and an independent power_model run.

    The comparison only holds while both draw continuously: the selector idles
    once a period's task is done, which lets the capacitor recover and suppresses
    boots (49 against 469 in one run). So pick a level that cannot finish inside
    the deadline, leaving the selector drawing without pause.
    """
    trace = _trace()
    duration = 120.0
    sel = simulate(trace, LEVELS, P, duration_s=duration, deadline_s=5.0,
                   choose=policy_fixed("L0"),          # 14.84 mJ: never completes
                   forecast=forecast_persistence)
    assert sel["completed"] == 0, "level was affordable; pick a costlier one"

    state = PowerState(P)
    keep = hold_until(V_RELEASE)
    for k in range(int(duration / P.timestep)):
        idx = ((k + 1) // P.steps_per_sample) % len(trace)
        state.step(trace[idx] / P.load_resistance, k,
                   lambda powered: I_ACTIVE, keep_alive=keep)

    assert sel["boots"] == state.boots, f"{sel['boots']} vs {state.boots} boots"
    assert sel["final_v"] == state.v, (
        f"final voltage {sel['final_v']!r} vs {state.v!r}")
    assert np.isclose(sel["energy_to_load_mJ"], state.energy_to_load * 1e3,
                      rtol=0, atol=1e-12)
    assert np.isclose(sel["energy_harvested_mJ"], state.energy_harvested * 1e3,
                      rtol=0, atol=1e-12)


def test_hindsight_picks_the_best_level_the_period_could_afford():
    """The oracle's contract is per-period: given the energy this period will
    actually deliver, choose the highest-quality level that fits, or the
    cheapest if none does.

    That is ALL it guarantees. An earlier version of this test asserted the
    global property -- that no fixed level ever beats hindsight on completion --
    which contradicts the oracle's own docstring and is false: on
    fft/mixed at an 8 s deadline the cheapest fixed level completes 91.1%
    against the oracle's 88.9%, because per-period greed spends charge a later
    period needed. Do not reinstate that assertion.
    """
    # These (period, current) pairs were chosen so the three cases below land on
    # three DIFFERENT levels. The first version of this test used a 0.3 s period
    # and delivered 0.00 mJ every time, so every case fell through to the
    # cheapest level and the assertion held no matter what the function did.
    cases = [(40000, 1.5e-4),      # 4 s, 150 uA ->  0.83 mJ, affords nothing
             (80000, 2.0e-4),      # 8 s, 200 uA ->  3.51 mJ, affords L1
             (150000, 4.0e-4)]     # 15 s, 400 uA -> 15.65 mJ, affords L0
    seen = set()
    for n_period, current in cases:
        state = PowerState(P)
        avail = np.full(n_period + 10, current)

        # Energy this period delivers, measured with the shared power model.
        clone = replace(state)
        delivered = 0.0
        for j in range(n_period):
            before = clone.energy_to_load
            clone.step(avail[j], j, lambda powered: I_ACTIVE,
                       keep_alive=hold_until(V_RELEASE))
            delivered += clone.energy_to_load - before

        idx = hindsight_level(LEVELS, state, avail, 0, n_period, P,
                              hold_until(V_RELEASE), 0.0)
        seen.add(idx)

        affordable = [i for i, l in enumerate(LEVELS) if l.energy <= delivered]
        expected = min(affordable) if affordable else len(LEVELS) - 1
        assert idx == expected, (
            f"{n_period * P.timestep:.0f}s at {current * 1e6:.0f}uA delivered "
            f"{delivered * 1e3:.2f} mJ: chose {LEVELS[idx].name}, best "
            f"affordable was {LEVELS[expected].name} "
            f"(costs {[round(l.energy * 1e3, 2) for l in LEVELS]} mJ)")

    assert seen == {0, 1, 2}, (
        f"cases only exercised levels {sorted(seen)}; the test is vacuous "
        "unless all three are reached")


def test_level_zero_is_the_declared_identity():
    """L0 was [5,1], zero error on ONE image, reported as the exact level."""
    assert SOBEL_LEVELS[0]["stride"] == "1,1"
    assert SOBEL_LEVELS[0]["cycles"] == 62
    assert SOBEL_LEVELS[0]["error"] == 0.0
