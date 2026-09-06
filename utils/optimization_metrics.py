"""Shared definitions for power-cycle and optimization metrics.

Terminology
-----------
Fused reports ``getPowerOnResetCount()`` (see ``fused-checkmate/main.cpp``).
That is a count of **simulated power-on resets / power cycles** under an
intermittent-power trace.  It is *not* a checkpoint count, and this module
therefore names it ``power_cycles``.  The legacy ``checkpoint_*`` aliases are
retained so older callers and CSV headers keep working.

Objective
---------
The campaign objective is a *constrained* one and must be frozen before
results are inspected:

    reject   error > ERROR_BOUND                        (hard accuracy gate)
    reject   power_cycle_ratio > 1.0                    (hard dominance gate)
    minimize w_err * (error / ERROR_BOUND) + w_cyc * power_cycle_ratio

Normalising the error by the accuracy bound is the point.  The previous
formulation summed a raw error in ``[0, 0.30]`` with a ratio in ``[0, inf)``,
which silently weighted the hardware term about 3x more than accuracy while
claiming the two were equal.  Dividing by the bound puts both terms on a
comparable ``[0, 1]`` scale inside the feasible region, so ``w_err`` and
``w_cyc`` mean what they say.
"""

import os


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    return float(raw)


def _env_flag(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in ("0", "false", "no", "off")


#: Weight applied to the bound-normalised accuracy term.
ERROR_WEIGHT = _env_float("CHAI_OBJECTIVE_ERROR_WEIGHT", 1.0)
#: Weight applied to the remaining-power-cycle ratio.
CYCLE_WEIGHT = _env_float("CHAI_OBJECTIVE_CYCLE_WEIGHT", 1.0)
#: Largest error accepted as evidence that a knob setting is the identity point.
IDENTITY_TOLERANCE = _env_float("CHAI_IDENTITY_TOLERANCE", 1e-6)
#: Reject candidates that need at least as many power cycles as the original.
REQUIRE_IMPROVEMENT = _env_flag("CHAI_REQUIRE_IMPROVEMENT", True)
#: Slack allowed on the dominance gate (absorbs simulator quantisation only).
DOMINANCE_TOLERANCE = _env_float("CHAI_DOMINANCE_TOLERANCE", 0.0)


def power_cycle_metrics(power_cycles, original_power_cycles):
    """Return ``(remaining_ratio, reduction_fraction)`` for power cycles."""
    current = float(power_cycles)
    original = float(original_power_cycles)
    if original <= 0:
        raise ValueError("original_power_cycles must be greater than zero")
    remaining_ratio = current / original
    reduction_fraction = 1.0 - remaining_ratio
    return remaining_ratio, reduction_fraction


def is_identity_error(error, tolerance=None):
    """True when *error* is small enough to call a knob setting the identity."""
    limit = IDENTITY_TOLERANCE if tolerance is None else float(tolerance)
    return float(error) <= limit


def dominates_original(power_cycles, original_power_cycles):
    """True when a candidate is no worse than the unapproximated program."""
    remaining_ratio, _ = power_cycle_metrics(power_cycles, original_power_cycles)
    return remaining_ratio <= 1.0 + DOMINANCE_TOLERANCE


def constrained_optimization_metric(
    error, power_cycles, original_power_cycles, error_bound=None
):
    """Objective minimized by BO after the hard accuracy constraint passes."""
    remaining_ratio, _ = power_cycle_metrics(power_cycles, original_power_cycles)
    if error_bound is None:
        from config.config import ERROR_BOUND as _bound

        error_bound = _bound
    bound = float(error_bound)
    normalized_error = (float(error) / bound) if bound > 0 else float(error)
    return ERROR_WEIGHT * normalized_error + CYCLE_WEIGHT * remaining_ratio


def objective_description(error_bound=None):
    """One-line, log-friendly statement of the frozen objective."""
    if error_bound is None:
        from config.config import ERROR_BOUND as _bound

        error_bound = _bound
    return (
        f"minimize {ERROR_WEIGHT} * (error / {error_bound}) "
        f"+ {CYCLE_WEIGHT} * (power_cycles / original_power_cycles); "
        f"reject error > {error_bound}"
        + ("; reject power_cycle_ratio > 1.0" if REQUIRE_IMPROVEMENT else "")
        + f"; identity tolerance {IDENTITY_TOLERANCE}"
    )


# ---------------------------------------------------------------------------
#  Backwards-compatible aliases (the metric was previously called "checkpoints")
# ---------------------------------------------------------------------------

def checkpoint_metrics(checkpoints, original_checkpoints):
    """Deprecated alias for :func:`power_cycle_metrics`."""
    return power_cycle_metrics(checkpoints, original_checkpoints)
