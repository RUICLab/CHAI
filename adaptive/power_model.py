"""Python replica of the Fused simulator's energy-harvesting power model.

Every constant below is taken from ``fusedBin/fusedConfig/config.yaml.in`` or the
Fused SystemC source (``fused-checkmate/ps/ExternalCircuitry.hpp``), and the
model is validated against Fused's own log (``fusedBin/fusedLogs/ext.tab``) by
``adaptive/validate_against_fused.py``: over the charging phase of an RF_2 run,
harvester current and capacitor voltage match Fused to floating-point precision.

Three details are easy to get wrong, and each was wrong in a first attempt:

* The device boots on the external supply-voltage supervisor (SVSVon/SVSVoff =
  3.5 V / 3.4 V), not on the MCU's internal PMM thresholds (1.9 V / 1.8 V).
* The supervisor draws a constant 25 uA (ExtIcc) from the capacitor even while
  the MCU is off. It is a current, not a resistance.
* The trace advances one sample per 1 ms, but its index changes one power-model
  step earlier than naive ``step // 10`` because of how Fused accumulates time.
* Firmware drives a KEEP_ALIVE GPIO that forces the supervisor on below SVSVoff.
  In an observed RF_2 run it was asserted for 97% of powered time, so a power
  cycle discharges far below 3.4 V: toward VoltageWarning (2.0 V) or the MCU's
  own PMM brown-out (1.8 V). Ignoring it understates energy per cycle ~12x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PowerParams:
    capacitance: float = 68e-6          # CapacitorValue [F] (campaign default)
    load_resistance: float = 10_000.0   # LoadResistance [ohm]: i_in = V_trace / R
    timestep: float = 1e-4              # PowerModelTimestep [s]
    steps_per_sample: int = 10          # trace advances every 1 ms
    supply_voltage_limit: float = 3.59  # SupplyVoltageLimit [V]: harvesting clamps here
    supply_current_limit: float = 5e-3  # SupplyCurrentLimit [A]: sets the clamp margin
    v_on: float = 3.5                   # SVSVon [V]
    v_off: float = 3.4                  # SVSVoff [V]
    external_icc: float = 25e-6         # ExtIcc [A]: supervisor draw, always present
    voltage_warning: float = 2.0        # VoltageWarning [V]: v_warn GPIO asserted below
    pmm_off: float = 1.8                # PMMOff [V]: MCU brown-out even when held on

    @property
    def clamp_margin(self) -> float:
        return self.timestep * (self.supply_current_limit / self.capacitance)

    @property
    def usable_energy_per_cycle(self) -> float:
        """Energy released between the on and off thresholds [J]."""
        return 0.5 * self.capacitance * (self.v_on ** 2 - self.v_off ** 2)


def load_trace(path: str | Path) -> np.ndarray:
    """Load a single-column voltage trace, tolerating a UTF-8 byte-order mark."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return np.array([float(line) for line in text.split() if line.strip()])


def trace_index(step: int, n_samples: int, params: PowerParams) -> int:
    return ((step + 1) // params.steps_per_sample) % n_samples


@dataclass
class SimResult:
    v_cap: np.ndarray
    i_harvest: np.ndarray
    powered: np.ndarray
    boots: int
    energy_harvested: float           # delivered into the capacitor [J]
    energy_spilled: float             # available but clamped away at the limit [J]
    energy_to_load: float             # drawn by the MCU while powered [J]
    energy_to_supervisor: float       # drawn by the 25 uA supervisor [J]
    extra: dict = field(default_factory=dict)


def hold_until(v_release: float):
    """keepAlive policy: once booted, hold the rail while v >= *v_release*."""
    def policy(step: int, v_prev: float, running: bool) -> bool:
        return running and v_prev >= v_release
    return policy


def simulate(trace: np.ndarray, load_current, n_steps: int,
             params: PowerParams = PowerParams(), v0: float = 0.0,
             start_step: int = 0, keep_alive=None) -> SimResult:
    """Step the harvester, capacitor and supervisor exactly as Fused does.

    ``load_current`` is either a constant MCU current in amperes or a callable
    ``f(step, powered) -> amperes``. The MCU draws current only while the
    supervisor reports the rail as powered.

    ``keep_alive`` is ``None`` (firmware never asserts KEEP_ALIVE) or a callable
    ``f(step, v_prev, running) -> bool`` modelling the firmware's GPIO, e.g.
    ``hold_until(params.voltage_warning)``. Holding cannot defeat the MCU's own
    PMM brown-out at ``params.pmm_off``.
    """
    constant = not callable(load_current)
    dt, cap = params.timestep, params.capacitance
    v = v_prev = v0
    on = False
    boots = 0
    v_cap = np.empty(n_steps)
    i_harv = np.empty(n_steps)
    powered = np.zeros(n_steps, dtype=bool)
    e_harv = e_spill = e_load = e_sup = 0.0
    n = len(trace)

    for j in range(n_steps):
        k = start_step + j
        available = trace[trace_index(k, n, params)] / params.load_resistance

        if (v_prev + params.clamp_margin) < params.supply_voltage_limit:
            i_in = available
        else:
            i_in = 0.0
            e_spill += available * v_prev * dt

        was_on = on
        forced = bool(keep_alive and keep_alive(k, v_prev, was_on))
        on = forced or (v_prev > params.v_on) or (v_prev > params.v_off and on)
        if on and v_prev < params.pmm_off:
            on = False                       # MCU brown-out overrides keepAlive
        if on and not was_on:
            boots += 1

        i_mcu = (load_current if constant else load_current(k, on)) if on else 0.0
        i_out = i_mcu + params.external_icc

        e_harv += i_in * v_prev * dt
        e_load += i_mcu * v_prev * dt
        e_sup += params.external_icc * v_prev * dt

        v_cap[j] = v
        i_harv[j] = i_in
        powered[j] = on
        v_prev = v
        v = max(0.0, v + dt * (i_in - i_out) / cap)

    return SimResult(v_cap, i_harv, powered, boots, e_harv, e_spill, e_load, e_sup)
