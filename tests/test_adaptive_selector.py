"""Regressions for the defects that invalidated the first 480 adaptive runs.

Each test drives the SELECTOR and inspects what it actually did. Earlier
versions of these tests exercised PowerState directly, or asserted identities
that held regardless of the behaviour they claimed to check.
"""

import numpy as np

from adaptive.power_model import PowerParams, PowerState, hold_until, simulate as power_simulate
from adaptive.selector import (I_ACTIVE, SOBEL_LEVELS, V_RELEASE, build_levels,
                               forecast_persistence, policy_fixed, simulate)

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


def test_hindsight_beats_every_fixed_level_at_each_deadline():
    """A boundary hindsight oracle must never be beaten, at any deadline, by a
    policy that picks one level and keeps it.

    Completion first; when completion ties, error decides. The earlier version
    checked completion on a single deadline, so a hindsight run that met the
    same deadlines with worse output would have passed.
    """
    for deadline in (10.0, 30.0, 60.0):
        hind = _run(deadline_s=deadline, hindsight=True)
        for name in ("L0", "L1", "L2"):
            fixed = _run(deadline_s=deadline, choose=policy_fixed(name))
            assert hind["completion_rate"] >= fixed["completion_rate"] - 1e-9, (
                f"deadline {deadline}s: hindsight {hind['completion_rate']:.3f} "
                f"< fixed-{name} {fixed['completion_rate']:.3f}")
            if abs(hind["completion_rate"] - fixed["completion_rate"]) < 1e-9:
                he = hind["mean_error"]
                fe = fixed["mean_error"]
                if not (np.isnan(he) or np.isnan(fe)):
                    assert he <= fe + 1e-9, (
                        f"deadline {deadline}s: completion ties but hindsight "
                        f"error {he:.4f} > fixed-{name} {fe:.4f}")


def test_level_zero_is_the_declared_identity():
    """L0 was [5,1], zero error on ONE image, reported as the exact level."""
    assert SOBEL_LEVELS[0]["stride"] == "1,1"
    assert SOBEL_LEVELS[0]["cycles"] == 62
    assert SOBEL_LEVELS[0]["error"] == 0.0
