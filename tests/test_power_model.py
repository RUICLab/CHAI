"""The Python power model must reproduce Fused's harvesting semantics."""

import numpy as np

from adaptive.power_model import PowerParams, simulate, trace_index

P = PowerParams()


def test_constant_trace_charges_linearly_after_supervisor_draw():
    trace = np.full(50, 2.0)                     # 2 V / 10 kOhm = 200 uA
    steps = 400
    result = simulate(trace, 0.0, steps, P)
    net = 200e-6 - P.external_icc
    expected = net * P.timestep * (steps - 1) / P.capacitance
    assert np.isclose(result.v_cap[-1], expected, rtol=1e-9)


def test_trace_index_advances_one_step_before_naive_division():
    assert [trace_index(k, 1000, P) for k in (0, 8, 9, 18, 19)] == [0, 0, 1, 1, 2]


def test_supervisor_uses_svs_thresholds_with_hysteresis():
    trace = np.full(10, 5.0)                     # 500 uA, charges quickly
    load = 2e-3                                  # heavier than the harvest once on
    result = simulate(trace, load, 60000, P)
    on_idx = np.argmax(result.powered)
    assert result.v_cap[on_idx - 1] > P.v_on - 1e-3
    off_after = on_idx + np.argmax(~result.powered[on_idx:])
    assert P.v_off - 0.01 < result.v_cap[off_after] <= P.v_off + 1e-3
    assert result.boots >= 2


def test_harvesting_clamps_at_supply_limit_and_counts_spill():
    trace = np.full(10, 5.28)
    result = simulate(trace, 0.0, 400000, P)     # boots, but the MCU draws no load
    assert result.v_cap.max() <= P.supply_voltage_limit + P.clamp_margin
    assert result.energy_spilled > 0


def test_voltage_never_goes_negative_when_harvest_below_supervisor_draw():
    trace = np.full(10, 0.1)                     # 10 uA < 25 uA supervisor
    result = simulate(trace, 0.0, 5000, P)
    assert result.v_cap.min() == 0.0


def test_usable_energy_per_cycle_matches_svs_window():
    assert np.isclose(P.usable_energy_per_cycle, 0.5 * 68e-6 * (3.5**2 - 3.4**2))


def test_keep_alive_holds_the_rail_below_svs_off_until_release_voltage():
    from adaptive.power_model import hold_until
    trace = np.full(10, 5.0)
    held = simulate(trace, 2e-3, 60000, P, keep_alive=hold_until(P.voltage_warning))
    on_idx = np.argmax(held.powered)
    off = on_idx + np.argmax(~held.powered[on_idx:])
    assert held.v_cap[off] < P.v_off - 0.5              # well below 3.4 V
    assert abs(held.v_cap[off] - P.voltage_warning) < 0.05


def test_keep_alive_cannot_defeat_the_mcu_brownout():
    from adaptive.power_model import hold_until
    trace = np.full(10, 5.0)
    held = simulate(trace, 2e-3, 60000, P, keep_alive=hold_until(0.0))
    on_idx = np.argmax(held.powered)
    off = on_idx + np.argmax(~held.powered[on_idx:])
    assert abs(held.v_cap[off] - P.pmm_off) < 0.05


def test_keep_alive_multiplies_energy_per_cycle_versus_svs_window_alone():
    from adaptive.power_model import hold_until
    trace = np.full(10, 5.0)
    bare = simulate(trace, 2e-3, 60000, P)
    held = simulate(trace, 2e-3, 60000, P, keep_alive=hold_until(P.voltage_warning))
    assert held.energy_to_load / max(held.boots, 1) > 8 * bare.energy_to_load / max(bare.boots, 1)
