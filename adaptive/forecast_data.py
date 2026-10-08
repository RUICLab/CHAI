"""Turn RF voltage traces into a forecasting dataset for the energy selector.

The selector must answer one question at the start of each charge cycle: how
much energy will arrive during the next cycle? Measured on an RF_2 run, a cycle
is ~653 ms (98.8 ms active, 558.5 ms recharging), so the forecast horizon is set
to one cycle by default.

Raw traces are 1 ms voltage samples. Harvested current is ``V / LoadResistance``
and harvested power into the capacitor is ``V_trace * V_cap / R``; since V_cap
varies, the trace-side quantity we forecast is the *available current*, which
depends only on the trace. Samples are averaged into bins (default 10 ms) so a
one-cycle horizon is a manageable number of steps for a forecasting model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adaptive.power_model import PowerParams, load_trace

RF_TRACES = ["RF_1", "RF_2", "RF_6", "RF_7", "RF_9"]
SAMPLE_MS = 1.0          # one trace sample per millisecond
CYCLE_MS = 653.0         # measured median charge-cycle period on RF_2


@dataclass(frozen=True)
class BinnedTrace:
    name: str
    current: np.ndarray   # mean available harvest current per bin [A]
    bin_ms: float

    @property
    def duration_s(self) -> float:
        return len(self.current) * self.bin_ms / 1000.0

    def energy_upper_bound(self, start: int, stop: int, v_cap: float) -> float:
        """Energy the harvester could deliver over [start, stop) at ``v_cap``."""
        return float(self.current[start:stop].sum() * self.bin_ms / 1000.0 * v_cap)


def load_binned(name: str, bin_ms: float = 10.0,
                traces_dir: str | Path = "traces",
                params: PowerParams = PowerParams()) -> BinnedTrace:
    volts = load_trace(Path(traces_dir) / f"{name}.csv")
    current = volts / params.load_resistance
    per_bin = int(round(bin_ms / SAMPLE_MS))
    usable = (len(current) // per_bin) * per_bin
    binned = current[:usable].reshape(-1, per_bin).mean(axis=1)
    return BinnedTrace(name, binned, bin_ms)


def load_all(bin_ms: float = 10.0, names=None, traces_dir="traces") -> list[BinnedTrace]:
    return [load_binned(n, bin_ms, traces_dir) for n in (names or RF_TRACES)]


def horizon_bins(bin_ms: float = 10.0, cycle_ms: float = CYCLE_MS) -> int:
    return max(1, int(round(cycle_ms / bin_ms)))


def windows(trace: BinnedTrace, context: int, horizon: int, stride: int = 1):
    """Yield (context_slice, target_slice) pairs that stay inside the trace."""
    n = len(trace.current)
    for start in range(0, n - context - horizon + 1, stride):
        ctx = trace.current[start:start + context]
        tgt = trace.current[start + context:start + context + horizon]
        yield ctx, tgt


def leave_one_out(traces: list[BinnedTrace]):
    """Yield (held_out_trace, remaining_traces) for every trace."""
    for i, held in enumerate(traces):
        yield held, [t for j, t in enumerate(traces) if j != i]
