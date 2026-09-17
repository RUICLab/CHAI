"""Energy-adaptive level selection, simulated end to end.

The device must deliver one result per deadline period. At the start of each
period it picks an approximation level for that task; it cannot switch mid-task,
so the forecast horizon is the deadline period, not a single charge cycle. When
harvested energy is plentiful the exact level should be chosen, and only when it
is scarce should the task degrade, so that a result still lands before the
deadline.

Levels come from `adaptive/levels.py` run over a canonical Sobel T22 measurement
(see levels_sobel.json). A level is one integer knob on one binary, which is
what a few-bit radio packet can switch at run time.

Energy per task is derived from the measured power-cycle count:

    E_task = cycles * E_cycle,  E_cycle = 0.5 * C * (v_on^2 - v_release^2)

E_cycle is ~239 uJ at 68 uF between the 3.5 V boot threshold and the 2.283 V at
which firmware releases KEEP_ALIVE. Cross-check: at the measured active draw of
968 uA and 2.96 V rail, 239 uJ sustains ~83 ms, against a measured 98.8 ms
active period -- the same order, so the accounting is consistent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from adaptive.power_model import PowerParams, load_trace

V_RELEASE = 2.283          # measured KEEP_ALIVE release voltage
REACTIVE_CHECK_STEPS = 10_000      # re-assess every 1 s of simulated time
REACTIVE_MIN_PROGRESS = 0.30       # ignore the charging-dominated start of a period
REACTIVE_UPGRADE_MARGIN = 1.20     # hysteresis: only climb back with 20% headroom
I_ACTIVE = 968.5e-6        # measured mean MCU current while powered [A]

# Canonical Sobel T22 with TWO knobs (min/max scan stride, output stride),
# RF_2, 68 uF: adaptive/levels.py on sobel_t22_wide.
#
# Sampling pass 1 buys savings at exactly zero error, because subsampling still
# finds the same min/max extremes -- for THIS image. That is image-dependent and
# should not be assumed to hold generally. The second knob also drops the floor
# from 23 to 6 cycles, which widens the L1/L2 gap from 22% to 63%: wider than
# the ~30% forecast error, so prediction can finally tell them apart.
SOBEL_LEVELS = [
    {"name": "L0", "stride": "5,1", "error": 0.000, "cycles": 45},
    {"name": "L1", "stride": "5,2", "error": 0.019, "cycles": 13},
    {"name": "L2", "stride": "4,4", "error": 0.280, "cycles": 8},
]


# Canonical FFT T7 (adaptive/levels.py on fft_taylor): truncate the sin/cos
# Taylor series that produce the twiddle factors. FFT's output is complex
# spectral bins, so error degrades continuously -- unlike ar-iclib's classify,
# whose discrete two-class decision flips unpredictably when its model is
# sampled (intermediate strides gave 0.62-1.0 error while stride 8 gave 0.398).
# Continuous outputs give graded dials; discrete decisions give cliffs.
FFT_LEVELS = [
    {"name": "L0", "stride": "7,7", "error": 0.000, "cycles": 12},
    {"name": "L1", "stride": "4,4", "error": 0.006, "cycles": 8},
    {"name": "L2", "stride": "2,3", "error": 0.158, "cycles": 6},
]

BENCHMARKS = {"sobel": SOBEL_LEVELS, "fft": FFT_LEVELS}


@dataclass(frozen=True)
class Level:
    name: str
    error: float
    energy: float          # joules of MCU energy to finish one task


def build_levels(params: PowerParams, levels=None) -> list[Level]:
    e_cycle = 0.5 * params.capacitance * (params.v_on ** 2 - V_RELEASE ** 2)
    return [Level(l["name"], l["error"], l["cycles"] * e_cycle)
            for l in (levels or SOBEL_LEVELS)]


def harvest_series(trace: np.ndarray, n_steps: int, params: PowerParams) -> np.ndarray:
    """Available harvest current per power-model step (trace loops, as Fused does)."""
    idx = ((np.arange(n_steps) + 1) // params.steps_per_sample) % len(trace)
    return trace[idx] / params.load_resistance


# ---- level-choosing policies ------------------------------------------------

def policy_fixed(name):
    def choose(levels, predicted_energy, level_index_hint=None):
        return next(i for i, l in enumerate(levels) if l.name == name)
    return choose


def policy_affordable(safety=1.0):
    """Least-approximate level whose task energy fits the predicted budget."""
    def choose(levels, predicted_energy, level_index_hint=None):
        for i, level in enumerate(levels):              # L0 first = most exact
            if level.energy <= predicted_energy * safety:
                return i
        return len(levels) - 1                           # nothing fits: cheapest
    return choose


# ---- forecasters: predict mean available current over the next period -------

def forecast_oracle(history, future, bins_ahead):
    return float(future[:bins_ahead].mean()) if len(future) else 0.0


def forecast_persistence(history, future, bins_ahead):
    return float(history[-1]) if len(history) else 0.0


def forecast_ewma(history, future, bins_ahead, alpha=0.3):
    if not len(history):
        return 0.0
    w = (1 - alpha) ** np.arange(len(history))[::-1]
    return float((w * history).sum() / w.sum())


_CHRONOS = {}


def _chronos_pipeline(model="amazon/chronos-bolt-small"):
    if "pipe" not in _CHRONOS:
        import torch
        from chronos import BaseChronosPipeline
        _CHRONOS["torch"] = torch
        _CHRONOS["pipe"] = BaseChronosPipeline.from_pretrained(
            model, device_map="cpu", torch_dtype=torch.float32)
    return _CHRONOS["pipe"], _CHRONOS["torch"]


def _chronos_forecast(history, bins_ahead, quantile):
    if len(history) < 8:
        return forecast_persistence(history, None, bins_ahead)
    pipe, torch = _chronos_pipeline()
    # Chronos is scale-free; microamps keep the values in a sane numeric range.
    ctx = torch.tensor(np.asarray(history)[None, :] * 1e6, dtype=torch.float32)
    q, _ = pipe.predict_quantiles(ctx, prediction_length=bins_ahead,
                                  quantile_levels=[0.1, 0.5])
    col = 0 if quantile == 0.1 else 1
    return float(q[0, :, col].numpy().mean() / 1e6)


def forecast_chronos(history, future, bins_ahead):
    return _chronos_forecast(history, bins_ahead, 0.5)


def forecast_chronos_q10(history, future, bins_ahead):
    """Pessimistic: plan against the 10th percentile of predicted harvest.

    Browning out mid-task wastes every joule already spent on it, so under
    uncertainty the cheaper level is the safer error.
    """
    return _chronos_forecast(history, bins_ahead, 0.1)


FORECASTERS = {
    "oracle": forecast_oracle,
    "persistence": forecast_persistence,
    "ewma": forecast_ewma,
    "chronos": forecast_chronos,
    "chronos-q10": forecast_chronos_q10,
}


class Server:
    """Server-in-the-loop selection: the device reports, the server decides.

    Each period the device sends its recent harvest, the server picks a level
    and sends it back, and the device reports the outcome. Both messages cost
    the device radio energy, charged to that period's budget.

    ``learn`` turns on the only thing a server can do that an autonomous device
    cannot: calibrate against outcomes. A raw harvest forecast systematically
    overestimates the energy that reaches the MCU, because the supervisor's
    constant 25 uA, charging losses and brown-outs all take a cut first. The
    server watches how much energy each period actually delivered against what
    it predicted, and scales later predictions by the running ratio. This
    corrects bias without needing a better forecast -- which matters, because
    forecast error (~30%) exceeds the L1/L2 cost gap (22%), so accuracy alone
    cannot separate the two levels it must choose between.
    """

    def __init__(self, forecast, learn=True, alpha=0.3):
        self.forecast = forecast
        self.learn = learn
        self.alpha = alpha
        self.ratio = 1.0
        self.pending = None
        self.last_level = 0

    def decide(self, levels, history, future, bins_ahead, deadline_s, params):
        mean_current = self.forecast(history, future, bins_ahead)
        usable = max(0.0, mean_current - params.external_icc)
        raw = usable * params.v_on * deadline_s
        predicted = raw * self.ratio
        for i, level in enumerate(levels):
            if level.energy <= predicted:
                self.last_level = i
                break
        else:
            self.last_level = len(levels) - 1
        self.pending = raw
        return self.last_level

    def observe(self, delivered):
        """Feedback from the device: energy that actually reached the MCU."""
        if not self.learn or not self.pending or self.pending <= 0:
            return
        observed = delivered / self.pending
        self.ratio = (1 - self.alpha) * self.ratio + self.alpha * observed
        self.ratio = min(max(self.ratio, 0.05), 3.0)


def simulate(trace, levels, params, duration_s, deadline_s, choose, forecast,
             bin_s=1.0, context_bins=64, reactive=False, server=None,
             comms_energy=0.0):
    """Run the selector over a trace and report what the device achieved."""
    dt = params.timestep
    steps_per_period_pre = int(deadline_s / dt)
    # Simulate a whole number of deadline periods. A trailing partial period
    # still starts a task and can finish it, which made completion exceed 100%
    # of the periods counted.
    n_steps = (int(duration_s / dt) // steps_per_period_pre) * steps_per_period_pre
    avail = harvest_series(trace, n_steps, params)

    steps_per_bin = int(bin_s / dt)
    binned = avail[:(n_steps // steps_per_bin) * steps_per_bin]
    binned = binned.reshape(-1, steps_per_bin).mean(axis=1)
    bins_ahead = max(1, int(deadline_s / bin_s))

    steps_per_period = int(deadline_s / dt)
    v = v_prev = 0.0
    on = False
    task_energy = 0.0
    done_this_period = False
    current = 0
    completed, missed, errors = 0, 0, []
    executed = []
    period_delivered = 0.0
    comms_paid = 0.0
    warm_level = 0
    last_completed = True
    e_harvested = e_spilled = 0.0
    chosen = []

    for k in range(n_steps):
        if k % steps_per_period == 0:                    # decision point
            last_completed = done_this_period
            if k > 0 and not done_this_period:
                missed += 1                              # deadline passed unfinished
            if k > 0 and server is not None:
                server.observe(period_delivered)     # device reports the outcome
            period_delivered = 0.0
            comms_paid = 0.0
            done_this_period = False
            bin_now = k // steps_per_bin
            history = binned[max(0, bin_now - context_bins):bin_now]
            future = binned[bin_now:]
            if reactive and server is None and k > 0:
                # Start from the last period's outcome, not from a fixed level.
                # Always restarting at the exact level wastes a tight period
                # rediscovering that it is unaffordable (45% vs 85% completion
                # at an 8 s deadline); always restarting where the last period
                # ended is too pessimistic when energy is plentiful (6x worse
                # error at 40 s). So: climb one level after a success, back off
                # one after a miss. The mid-period rule still corrects within
                # the period.
                current = (max(0, warm_level - 1) if last_completed
                           else min(len(levels) - 1, warm_level + 1))
            elif server is not None:
                current = server.decide(levels, history, future, bins_ahead,
                                        deadline_s, params)
            # Energy reaching the MCU over the period: the supervisor's constant
            # draw is paid first, and only the remainder does useful work.
            else:
                mean_current = forecast(history, future, bins_ahead)
                usable = max(0.0, mean_current - params.external_icc)
                predicted = usable * params.v_on * deadline_s
                current = choose(levels, predicted)
            chosen.append(current)
            task_energy = 0.0
            warm_level = current

        i_in = avail[k]
        if (v_prev + params.clamp_margin) >= params.supply_voltage_limit:
            e_spilled += i_in * v_prev * dt
            i_in = 0.0

        was_on = on
        on = (v_prev > params.v_on) or (v_prev > params.v_off and on)
        if on and v_prev < V_RELEASE:                     # firmware releases hold
            on = False
        if on and not was_on:
            pass

        i_mcu = I_ACTIVE if (on and not done_this_period) else 0.0
        e_harvested += i_in * v_prev * dt

        # Reactive control: no prediction. Partway through the period, compare
        # the energy actually banked against what this level still needs. If the
        # task cannot finish at the current pace, drop to a cheaper level and
        # keep the work already done -- the levels share one binary, so only the
        # stride changes. This sidesteps forecasting entirely, which matters
        # because forecast error (~30%) is larger than the L1/L2 energy gap
        # (22%), so no forecaster can reliably tell those two levels apart.
        if reactive and on and not done_this_period:
            elapsed = k % steps_per_period
            progress = elapsed / steps_per_period
            # Judging too early is wrong: the device spends the start of a
            # period charging, so the observed rate looks pessimistic and the
            # task degrades needlessly. Wait until enough of the period has run
            # for the rate estimate to mean something.
            if (progress >= REACTIVE_MIN_PROGRESS
                    and elapsed % REACTIVE_CHECK_STEPS == 0):
                rate = task_energy / (elapsed * dt)            # J/s observed so far
                projected = task_energy + rate * (1.0 - progress) * deadline_s
                while (current + 1 < len(levels)
                       and projected < levels[current].energy):
                    current += 1                               # cannot finish: degrade
                # If energy recovered, climb back toward the exact level. The
                # margin stops it oscillating around a threshold.
                while (current > 0
                       and projected > levels[current - 1].energy * REACTIVE_UPGRADE_MARGIN):
                    current -= 1
                warm_level = current

        # One result per deadline period: once this period's task is done the
        # device stops working rather than racing ahead into the next one.
        if on:
            period_delivered += i_mcu * v_prev * dt
        if on and not done_this_period:
            step_energy = i_mcu * v_prev * dt
            if comms_paid < comms_energy:             # radio round trip first
                comms_paid += step_energy
                step_energy = 0.0
            task_energy += step_energy
            if task_energy >= levels[current].energy:
                completed += 1
                errors.append(levels[current].error)
                executed.append(current)
                task_energy = 0.0
                done_this_period = True

        v_prev = v
        v = max(0.0, v + dt * (i_in - (i_mcu + params.external_icc)) / params.capacitance)

    periods = max(1, n_steps // steps_per_period)
    assert completed <= periods, f"{completed} completions in {periods} periods"
    return {
        "completed": completed,
        "periods": periods,
        "missed": missed,
        "completion_rate": completed / periods,
        "mean_error": float(np.mean(errors)) if errors else float("nan"),
        "energy_harvested_mJ": e_harvested * 1e3,
        "energy_spilled_mJ": e_spilled * 1e3,
        "tasks_per_mJ": completed / (e_harvested * 1e3) if e_harvested > 0 else 0.0,
        "level_mix": np.bincount(executed, minlength=len(levels)).tolist(),
    }


def mixed_trace(names, traces_dir="traces"):
    """Concatenate traces to model a device moving between RF environments.

    Each RF trace is 25-52 s and Fused loops it, so a single trace has almost no
    variation left at a 25 s decision horizon (three of the five measure a 1.00x
    swing). The energy scale forbids shortening the deadline: harvesting is
    ~0.5 mW, so even the cheapest level needs ~11 s of charging. Concatenating
    different environments restores variation at the timescale the selector
    actually decides on. This is a synthetic mobility scenario, not measured
    data, and must be labelled as such.
    """
    return np.concatenate([load_trace(f"{traces_dir}/{n}.csv") for n in names])


# Harvest scaling models distance to the RF source. All five measured traces sit
# within a 1.7x band (127-215 uA), which is far narrower than a real deployment
# where moving toward or away from a transmitter changes harvested power by an
# order of magnitude. VARYING_GAINS spans scarcity to abundance so that no single
# fixed level is best throughout.
VARYING_GAINS = [0.45, 1.5, 0.35, 2.2, 0.8]


def varying_trace(names, gains=None, traces_dir="traces"):
    """Concatenate traces at different harvest scales (changing distance)."""
    gains = gains or VARYING_GAINS
    parts = [load_trace(f"{traces_dir}/{n}.csv") * g
             for n, g in zip(names, gains)]
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--deadline", type=float, default=25.0)
    ap.add_argument("--traces", default="RF_1,RF_2,RF_6,RF_7,RF_9")
    ap.add_argument("--comms", type=float, default=200.0,
                    help="device energy per server round trip [uJ]; a short packet "
                         "on a low-power radio is ~200 uJ, backscatter far less")
    ap.add_argument("--scenario", choices=["per-trace", "mixed", "varying"], default="per-trace",
                    help="per-trace: each trace separately (little variation); "
                         "mixed: one concatenated moving-environment trace")
    args = ap.parse_args()

    params = PowerParams()
    levels = build_levels(params)
    print("LEVEL TABLE (canonical Sobel T22)")
    for l in levels:
        print(f"   {l.name}  error {l.error:.3f}   energy/task {l.energy * 1e3:6.2f} mJ")
    print(f"\ndeadline {args.deadline:.0f} s, {args.duration:.0f} s per trace, "
          f"{int(args.duration / args.deadline)} tasks per trace\n")

    policies = [(f"fixed-{l.name}", policy_fixed(l.name), forecast_persistence)
                for l in levels]
    policies += [(f"adaptive-{name}", policy_affordable(), fn)
                 for name, fn in FORECASTERS.items()]
    policies += [("reactive (on-device)", policy_fixed(levels[0].name),
                  forecast_persistence)]

    names = args.traces.split(",")
    if args.scenario == "varying":
        trace = varying_trace(names)
        scenarios = [("varying", trace)]
        gains = ", ".join(f"{n}x{g}" for n, g in zip(names, VARYING_GAINS))
        print(f"scenario: varying distance ({gains}), "
              f"{len(trace) / 1000:.0f} s before looping\n")
    elif args.scenario == "mixed":
        scenarios = [("mixed", mixed_trace(names))]
        print(f"scenario: mixed  ({' -> '.join(names)}, "
              f"{len(scenarios[0][1]) / 1000:.0f} s before looping)\n")
    else:
        scenarios = [(n, load_trace(f"traces/{n}.csv")) for n in names]

    runs = [(label, choose, fn, None) for label, choose, fn in policies]
    runs += [("server-forecast", policy_affordable(), forecast_ewma, False),
             ("server-learned", policy_affordable(), forecast_ewma, True),
             # Hybrid: the server picks the starting level from what it predicts,
             # then the device still corrects from measured progress. This keeps
             # the server's wider view without trusting the forecast outright.
             ("server+reactive", policy_affordable(), forecast_ewma, "hybrid")]

    rows = []
    for label, choose, forecast, learn in runs:
        agg = []
        for _name, trace in scenarios:
            hybrid = (learn == "hybrid")
            srv = (Server(forecast_ewma, learn=True) if hybrid
                   else Server(forecast_ewma, learn=learn) if learn is not None
                   else None)
            agg.append(simulate(trace, levels, params, args.duration,
                                args.deadline, choose, forecast,
                                reactive=label.startswith("reactive") or hybrid,
                                server=srv,
                                comms_energy=(args.comms * 1e-6 if srv else 0.0)))
        rows.append((label,
                     float(np.mean([a["completion_rate"] for a in agg])),
                     float(np.nanmean([a["mean_error"] for a in agg])),
                     float(np.mean([a["tasks_per_mJ"] for a in agg])),
                     float(np.mean([a["energy_spilled_mJ"] for a in agg])),
                     np.sum([a["level_mix"] for a in agg], axis=0).tolist()))

    print(f"{'policy':<22}{'done':>7}{'mean err':>10}{'tasks/mJ':>11}"
          f"{'spill mJ':>10}   levels run [L0,L1,L2]")
    print("-" * 78)
    for label, rate, err, tpj, spill, mix in rows:
        print(f"{label:<22}{rate * 100:>6.0f}%{err:>10.3f}{tpj:>11.3f}"
              f"{spill:>10.2f}   {mix}")


if __name__ == "__main__":
    main()
