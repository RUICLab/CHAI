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
from dataclasses import dataclass, replace

import numpy as np

from adaptive.power_model import PowerParams, PowerState, hold_until, load_trace

V_RELEASE = 2.283          # measured KEEP_ALIVE release voltage

def mean_rail_voltage(params: PowerParams) -> float:
    """Average voltage at which energy actually reaches the MCU.

    Energy is delivered across the whole discharge, from the 3.5 V boot
    threshold down to the 2.283 V KEEP_ALIVE release -- not at 3.5 V. Valuing it
    at v_on overestimated the period's budget by ~25% (predicted 17.93 mJ where
    14.38 mJ arrived), which made L0 look affordable when it was not. Every
    predicting policy, including the oracle, then over-committed and missed
    deadlines: the oracle scored 72% where a fixed level scored 100%.
    """
    return 0.5 * (params.v_on + V_RELEASE)
I_ACTIVE = 968.5e-6        # measured mean MCU current while powered [A]

# Canonical Sobel T22 with TWO knobs (min/max scan stride, output stride),
# RF_2, 68 uF: adaptive/levels.py on sobel_t22_wide.
#
# The second knob drops the floor
# from 23 to 6 cycles, which widens the L1/L2 gap from 22% to 63%: wider than
# the ~30% forecast error, so prediction can finally tell them apart.
SOBEL_LEVELS = [
    # L0 is the plan's DECLARED IDENTITY [1,1] = the original program, 62 cycles.
    # It is not [5,1]: that configuration also measured 0.000 error, but only on
    # one image, and sampling the min/max scan is an approximation whose error is
    # image-dependent. Calling it "exact" claimed a free 27.4% saving that the
    # data does not support.
    {"name": "L0", "stride": "1,1", "error": 0.000, "cycles": 62},
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


# Returned by a policy that declines to start a task at all this period.
DEFER = -1

ERROR_BOUND = 0.05


def cheapest_within_bound(levels, bound=ERROR_BOUND):
    """Index of the least expensive level that still satisfies the error bound.

    This is the level the grading metric actually wants. Under "results per
    joule subject to error <= bound", accuracy beyond the bound earns nothing,
    so a more exact level is strictly wasted energy.
    """
    ok = [i for i, l in enumerate(levels) if l.error <= bound + 1e-12]
    if not ok:
        return None
    return min(ok, key=lambda i: levels[i].energy)


def policy_bound_aware(bound=ERROR_BOUND):
    """Always run the cheapest level meeting the bound.

    Note what this is: because the level table is static, the answer never
    changes, so this policy is DETERMINISTICALLY IDENTICAL to fixed-L1 on both
    benchmarks. It is included as a labelled control -- it isolates "optimise
    the right objective" from "be able to wait" -- and its agreement with
    fixed-L1 is a check on the implementation, not a finding.
    """
    def choose(levels, predicted_energy, level_index_hint=None):
        idx = cheapest_within_bound(levels, bound)
        return len(levels) - 1 if idx is None else idx
    return choose


def can_finish(level, state, window, k_start, n_period, params, keep_alive,
               comms_energy=0.0):
    """Replay one period from the LIVE power state and report whether it lands.

    The scalar test this replaces compared a task's cost against predicted
    INCOMING harvest alone, so it could refuse to start work while the energy to
    do that work was already in the capacitor -- at 68 uF a full buffer holds
    261 uJ against an L1 task's 3111 uJ. Replaying inherits stored charge, the
    boot thresholds, the supervisor's constant draw, the voltage clamp and spill
    from the model instead of approximating any of them.

    `window` is the harvest current to assume for the period: the real trace
    slice for a perfect forecast, a constant at the forecast mean otherwise.
    """
    clone = replace(state)
    task_energy = 0.0
    paid = 0.0
    for j in range(min(n_period, len(window))):
        before = clone.energy_to_load
        clone.step(window[j], k_start + j, lambda powered: I_ACTIVE,
                   keep_alive=keep_alive)
        gained = clone.energy_to_load - before
        if paid < comms_energy:                  # radio round trip first
            paid += gained
            gained = 0.0
        task_energy += gained
        if task_energy >= level.energy:
            return True
    return False


def policy_bound_aware_defer(bound=ERROR_BOUND):
    """Run the cheapest in-bound level when it is affordable; otherwise wait.

    Starting a task the period cannot finish burns the whole period's harvest
    and produces nothing. Waiting keeps that charge in the capacitor for the
    next period. Deferring is counted as a missed deadline, so this cannot make
    throughput look better by simply doing less.
    """
    def choose(levels, predicted_energy, level_index_hint=None):
        idx = cheapest_within_bound(levels, bound)
        if idx is None:
            return len(levels) - 1
        return idx if levels[idx].energy <= predicted_energy else DEFER
    return choose


# ---- forecasters: predict mean available current over the next period -------

def forecast_oracle(history, future, bins_ahead):
    """Perfect knowledge of the future MEAN CURRENT, not a policy ceiling.

    This is a clairvoyant forecaster: it still passes its perfect input through
    the same approximate affordability arithmetic as every other predictor, so a
    fixed policy can legitimately beat it. For an actual upper bound use
    ``hindsight_level`` below.
    """
    return float(future[:bins_ahead].mean()) if len(future) else 0.0


def hindsight_level(levels, state, avail, k_start, n_period, params,
                    keep_alive, comms_energy):
    """The best level this period could have run, judged after the fact.

    Replays the coming period through a copy of the CURRENT capacitor state --
    so carried-over charge counts -- and returns the most exact level whose task
    energy actually arrives. One replay settles every level: the device draws
    the same current until a task completes, so the energy trajectory is shared
    up to each level's completion point.

    This is a BOUNDARY hindsight oracle, not a global ceiling. It maximises each
    period in isolation, so it can spend charge now that a later period needed
    and end up completing fewer tasks overall than a more frugal policy.
    """
    clone = replace(state)
    delivered = 0.0
    paid = 0.0
    for j in range(n_period):
        before = clone.energy_to_load
        clone.step(avail[k_start + j], k_start + j,
                   lambda powered: I_ACTIVE, keep_alive=keep_alive)
        gained = clone.energy_to_load - before
        if paid < comms_energy:
            paid += gained
            gained = 0.0
        delivered += gained
    for i, level in enumerate(levels):          # most exact first
        if level.energy <= delivered:
            return i
    return len(levels) - 1


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


def forecast_ewma_slow(history, future, bins_ahead, alpha=0.02):
    """Long-memory EWMA. The deployed alpha=0.3 has an effective memory of a few
    seconds while 64 s of history is available, so it chases fluctuations that do
    not persist to the 8-40 s horizon the controller actually decides over."""
    return forecast_ewma(history, future, bins_ahead, alpha=alpha)


def forecast_mean_window(seconds):
    """Flat average of the last `seconds` of harvest. Cheap enough for the MCU:
    a running sum and a divide, no model."""
    def fn(history, future, bins_ahead):
        if not len(history):
            return 0.0
        return float(np.asarray(history)[-seconds:].mean())
    return fn


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

    def __init__(self, forecast, learn=True, alpha=0.3, bound=None):
        self.forecast = forecast
        self.learn = learn
        self.alpha = alpha
        self.ratio = 1.0
        self.pending = None
        self.last_level = 0
        # When set, the server only ever selects levels inside the error bound,
        # and DEFERS rather than shipping output the metric scores as zero. The
        # default (None) keeps the original quality-first behaviour: pick the
        # most exact AFFORDABLE level, which falls through to the cheapest one
        # when energy is tight -- and the cheapest level is out of bound, so it
        # earns nothing. That is why the server looked catastrophic even at zero
        # radio cost: 32 of its 35 completed tasks were out-of-bound L2.
        self.bound = bound

    def decide(self, levels, history, future, bins_ahead, deadline_s, params):
        mean_current = self.forecast(history, future, bins_ahead)
        usable = max(0.0, mean_current - params.external_icc)
        raw = usable * mean_rail_voltage(params) * deadline_s
        predicted = raw * self.ratio
        self.pending = raw
        if self.bound is not None:
            allowed = [i for i, l in enumerate(levels)
                       if l.error <= self.bound + 1e-12]
            if allowed:
                for i in allowed:                       # most exact first
                    if levels[i].energy <= predicted:
                        self.last_level = i
                        return i
                return DEFER            # nothing in bound is affordable: wait
        for i, level in enumerate(levels):
            if level.energy <= predicted:
                self.last_level = i
                break
        else:
            self.last_level = len(levels) - 1
        return self.last_level

    def observe(self, delivered):
        """Feedback from the device: energy that actually reached the MCU.

        WARNING -- this signal is CENSORED, and `learn=True` is unsound because
        of it. What the device reports is energy CONSUMED, not energy AVAILABLE:
        finishing early, deferring, or picking a cheap level all lower it, while
        radio traffic raises it. The learner therefore reads "I chose not to
        spend energy" as "no energy was there", drives `ratio` down, and defers
        or drops to the cheapest level even more -- a self-reinforcing loop. On
        sobel/8s with free communication it collapsed to the 0.05 floor and
        produced 2 in-bound tasks where the same server unlearned produced 13.

        Measuring availability needs an uncensored signal, e.g. capacitor
        voltage or harvest current sampled while idle. Until then use
        learn=False."""
        if not self.learn or not self.pending or self.pending <= 0:
            return
        observed = delivered / self.pending
        self.ratio = (1 - self.alpha) * self.ratio + self.alpha * observed
        self.ratio = min(max(self.ratio, 0.05), 3.0)


def simulate(trace, levels, params, duration_s, deadline_s, choose, forecast,
             bin_s=1.0, context_bins=64, reactive=False, server=None,
             comms_energy=0.0, hindsight=False, state_aware_defer=False,
             voltage_policy=None, rate_policy=None):
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
    # Share the validated transition logic rather than re-deriving it. The old
    # copy omitted KEEP_ALIVE, so the device powered off at 3.4 V while
    # build_levels priced tasks assuming discharge to 2.283 V -- roughly a 10x
    # mismatch that invalidated every run built on it.
    state = PowerState(params)
    keep_alive = hold_until(V_RELEASE)
    task_energy = 0.0
    done_this_period = False
    current = 0
    completed, missed, errors = 0, 0, []
    executed = []
    period_delivered = 0.0
    comms_paid = 0.0
    comms_paid_total = 0.0    # radio energy the device actually spent
    queries = 0               # server round trips attempted (one per period)
    warm_level = 0
    decisions = []            # level chosen at each period boundary
    period_level = 0
    mid_task_switches = 0     # must stay 0: levels change only at boundaries
    min_powered_v = float("inf")
    last_completed = True
    last_delivered = 0.0      # energy the previous period actually delivered
    deferred = 0              # periods the policy declined to start a task
    false_defers = 0          # deferrals where the task would in fact have landed
    chosen = []

    for k in range(n_steps):
        if k % steps_per_period == 0:                    # decision point
            last_completed = done_this_period
            last_delivered = period_delivered
            if k > 0 and not done_this_period:
                missed += 1                              # deadline passed unfinished
            if k > 0 and server is not None:
                server.observe(period_delivered)     # device reports the outcome
            period_delivered = 0.0
            comms_paid_total += comms_paid
            comms_paid = 0.0
            if server is not None:      # count it even when modelled as free
                queries += 1
            done_this_period = False
            bin_now = k // steps_per_bin
            history = binned[max(0, bin_now - context_bins):bin_now]
            future = binned[bin_now:]
            if hindsight:
                current = hindsight_level(levels, state, avail, k,
                                          steps_per_period, params, keep_alive,
                                          comms_energy)
            elif reactive and server is None and k > 0:
                # Start from the last period's outcome, not from a fixed level.
                # Mid-task correction no longer exists, so this boundary
                # decision is the only adaptation reactive control has.
                #
                # Three outcomes, not two. An earlier version climbed after
                # EVERY success and backed off after every miss, which has no
                # way to stay put: it succeeded at L1, climbed to L0, could not
                # finish it, backed off, succeeded, climbed again, producing
                # decisions [0,1,0,1,...] and missing every other deadline by
                # construction. That is where reactive's flat ~50% completion
                # across the whole campaign came from.
                #
                # Climbing now needs evidence, not just a success: the energy
                # actually delivered last period must cover the next level up.
                # That quantity is the accumulated harvest the device already
                # measures, so this stays implementable on-device.
                if not last_completed:
                    current = min(len(levels) - 1, warm_level + 1)
                elif (warm_level > 0
                      and last_delivered >= levels[warm_level - 1].energy):
                    current = warm_level - 1
                else:
                    current = warm_level
            elif server is not None:
                current = server.decide(levels, history, future, bins_ahead,
                                        deadline_s, params)
            # Energy reaching the MCU over the period: the supervisor's constant
            # draw is paid first, and only the remainder does useful work.
            elif rate_policy is not None:
                # "Energy coming in fast -> exact; slow -> approximate." The
                # input is a MEASUREMENT of recent harvest, not a prediction of
                # future harvest, and no energy model or calibration sits
                # between it and the decision -- the table is learned directly
                # from which level actually worked at that rate.
                recent = float(np.mean(history[-30:])) if len(history) else 0.0
                current = rate_policy(levels, recent)
            elif voltage_policy is not None:
                # No forecast, no radio, no history: read the capacitor and look
                # the answer up. The device already knows this voltage.
                current = voltage_policy(levels, state.v)
            else:
                mean_current = forecast(history, future, bins_ahead)
                usable = max(0.0, mean_current - params.external_icc)
                predicted = usable * mean_rail_voltage(params) * deadline_s
                current = choose(levels, predicted)
                if state_aware_defer:
                    idx = cheapest_within_bound(levels)
                    if idx is not None:
                        # Replay the period from the live state rather than
                        # comparing scalars, so stored charge counts.
                        window = (avail[k:k + steps_per_period]
                                  if forecast is forecast_oracle
                                  else np.full(steps_per_period, mean_current))
                        current = idx if can_finish(
                            levels[idx], state, window, k, steps_per_period,
                            params, keep_alive, comms_energy) else DEFER

                # Was a deferral actually necessary? Judged against the REAL
                # trace, so it is a diagnostic and never feeds the decision.
                if current == DEFER:
                    truth = cheapest_within_bound(levels)
                    if truth is not None and can_finish(
                            levels[truth], state, avail[k:k + steps_per_period],
                            k, steps_per_period, params, keep_alive,
                            comms_energy):
                        false_defers += 1
            chosen.append(current)
            decisions.append(current)
            period_level = current
            task_energy = 0.0
            if current != DEFER:
                warm_level = current      # DEFER is not a level to warm-start from
            else:
                deferred += 1

        # A deferred period draws nothing: the device stays asleep and the
        # harvest accumulates in the capacitor instead of being spent on a task
        # that will not finish. MSP430 LPM draw (<1 uA) is negligible beside the
        # supervisor's constant 25 uA, which PowerState charges regardless, so
        # modelling the sleeping MCU as 0 A costs nothing in fidelity.
        # The radio round trip is owed for the period regardless of what the
        # server replied. Paying it only while a task ran made a "defer" reply
        # free: at 200 uJ with 34 deferrals the device was charged 0.40 mJ
        # instead of ~7.2 mJ, so deferral looked far cheaper than it is.
        owes_comms = comms_energy > 0.0 and comms_paid < comms_energy
        idle = (done_this_period or current == DEFER) and not owes_comms
        before_load = state.energy_to_load
        on, i_mcu, _i_in = state.step(
            avail[k], k,
            lambda powered: (0.0 if idle else I_ACTIVE),
            keep_alive=keep_alive)
        step_energy = state.energy_to_load - before_load
        period_delivered += step_energy
        if on:
            min_powered_v = min(min_powered_v, state.v)
        if current != period_level:
            mid_task_switches += 1

        # Mid-task switching is DISABLED. It changed stride or term count
        # partway through a task, kept the work already done, and recorded the
        # final level's error -- which assumes the two levels' partial results
        # compose. For a Sobel convolution they do not: the output would be a
        # hybrid image belonging to neither level, with an error neither was
        # measured at. Level choice therefore happens only at task boundaries,
        # where the reactive policy still adapts from the previous period's
        # outcome. Restart-on-switch is a separate design worth studying.

        # One result per deadline period: once this period's task is done the
        # device stops working rather than racing ahead into the next one.
        if on:
            if comms_paid < comms_energy:             # radio round trip first
                comms_paid += step_energy
                step_energy = 0.0
        if on and not idle and current != DEFER and not done_this_period:
            task_energy += step_energy
            if task_energy >= levels[current].energy:
                completed += 1
                errors.append(levels[current].error)
                executed.append(current)
                task_energy = 0.0
                done_this_period = True


    periods = max(1, n_steps // steps_per_period)
    assert completed <= periods, f"{completed} completions in {periods} periods"
    return {
        "completed": completed,
        "periods": periods,
        "missed": missed,
        "completion_rate": completed / periods,
        "mean_error": float(np.mean(errors)) if errors else float("nan"),
        "energy_harvested_mJ": state.energy_harvested * 1e3,
        "energy_spilled_mJ": state.energy_spilled * 1e3,
        "tasks_per_mJ": (completed / (state.energy_harvested * 1e3)
                         if state.energy_harvested > 0 else 0.0),
        "level_mix": np.bincount(executed, minlength=len(levels)).tolist(),
        "decisions": decisions,
        "boots": state.boots,
        "final_v": state.v,
        "energy_to_load_mJ": state.energy_to_load * 1e3,
        "deferred": deferred,
        "comms_paid_mJ": (comms_paid_total + comms_paid) * 1e3,
        "queries": queries,
        "false_defers": false_defers,
        "mid_task_switches": mid_task_switches,
        "min_powered_v": (None if min_powered_v == float("inf") else min_powered_v),
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
                  forecast_persistence),
                 ("hindsight (boundary oracle)", policy_fixed(levels[0].name),
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
    # server+reactive was removed: with mid-task correction gone, a hybrid run
    # takes the server branch at every boundary and is byte-identical to
    # server-learned. It measured as a separate policy only while mid-task
    # switching existed.
    runs += [("server-forecast", policy_affordable(), forecast_ewma, False),
             ("server-learned", policy_affordable(), forecast_ewma, True)]

    rows = []
    for label, choose, forecast, learn in runs:
        agg = []
        for _name, trace in scenarios:
            srv = Server(forecast_ewma, learn=learn) if learn is not None else None
            agg.append(simulate(trace, levels, params, args.duration,
                                args.deadline, choose, forecast,
                                hindsight=label.startswith("hindsight"),
                                reactive=label.startswith("reactive"),
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
