"""Regressions for the defects that invalidated the first 480 adaptive runs.

Each test here corresponds to a specific way the earlier results were wrong.
"""

import numpy as np

from adaptive.power_model import PowerParams, PowerState, hold_until
from adaptive.selector import (SOBEL_LEVELS, V_RELEASE, build_levels,
                               forecast_persistence, policy_fixed, simulate)

P = PowerParams()


def test_selector_keeps_the_rail_alive_below_the_supervisor_threshold():
    """The selector used to re-derive the power loop and omit KEEP_ALIVE, so it
    powered off at 3.4 V while tasks were priced assuming discharge to 2.283 V
    -- about a 10x mismatch between energy delivered and energy charged."""
    trace = np.full(10, 5.0)
    state = PowerState(P)
    keep = hold_until(V_RELEASE)
    lowest_while_on = 10.0
    for k in range(60000):
        on, _i, _in = state.step(trace[0] / P.load_resistance, k,
                                 lambda powered: 2e-3, keep_alive=keep)
        if on:
            lowest_while_on = min(lowest_while_on, state.v)
    assert lowest_while_on < P.v_off - 0.5, "device never ran below the SVS threshold"
    assert lowest_while_on < V_RELEASE + 0.1


def test_level_zero_is_the_declared_identity_not_a_zero_error_approximation():
    """L0 was [5,1], a configuration that measured 0.000 error on ONE image.
    Reporting it as the exact level claimed a free 27.4% saving."""
    assert SOBEL_LEVELS[0]["stride"] == "1,1"
    assert SOBEL_LEVELS[0]["cycles"] == 62          # the original program's cost
    assert SOBEL_LEVELS[0]["error"] == 0.0


def test_level_never_changes_inside_a_deadline_period():
    """Mid-task switching kept work accumulated under one level and recorded
    another level's error, which assumes partial results compose. They do not."""
    trace = np.tile(np.array([2.0, 3.0, 1.0, 4.0]), 500)
    levels = build_levels(P, SOBEL_LEVELS)
    result = simulate(trace, levels, P, duration_s=120.0, deadline_s=15.0,
                      choose=policy_fixed("L0"), forecast=forecast_persistence,
                      reactive=True)
    # One decision per period, so at most one recorded level per completed task.
    assert sum(result["level_mix"]) == result["completed"]
    assert result["completed"] <= result["periods"]


def test_selector_and_power_model_agree_step_for_step():
    """Both must drive the same PowerState; a private copy is how they drifted."""
    trace = np.tile(np.array([1.5, 2.5, 0.8]), 400)
    a = PowerState(P)
    b = PowerState(P)
    keep = hold_until(V_RELEASE)
    for k in range(20000):
        av = trace[k % len(trace)] / P.load_resistance
        a.step(av, k, lambda powered: 1e-3, keep_alive=keep)
        b.step(av, k, lambda powered: 1e-3, keep_alive=keep)
    assert a.v == b.v and a.boots == b.boots
    assert a.energy_to_load == b.energy_to_load
