"""Validation for Bayesian-optimization campaign settings."""

import os

#: Model-guided evaluations needed before a GP surrogate says anything useful.
#: With ``CHAI_BO_CALLS=8`` the campaign spent 1 call on the identity baseline
#: and 5 on random initial points, leaving *two* model-guided evaluations for
#: search spaces as large as 16 dimensions. Results from that budget are
#: dominated by which random points happened to be drawn, not by the planner.
MIN_MODEL_GUIDED_CALLS = int(os.getenv("CHAI_MIN_MODEL_GUIDED_CALLS", "10"))


def recommended_call_budget(dimensions, initial_points=5, seeded_points=1):
    """Return a defensible call budget for a search space of *dimensions*."""
    return initial_points + seeded_points + max(
        MIN_MODEL_GUIDED_CALLS, 4 * max(dimensions, 1)
    )


def validate_bo_call_budget(call_count, initial_points=5, seeded_points=1):
    """Validate scikit-optimize's total-call requirement.

    Enforces only the hard floor that ``gp_minimize`` needs. Budgets that are
    technically valid but too small to be informative are warned about rather
    than rejected, so a deliberate smoke run still works.
    """
    minimum = initial_points + seeded_points
    if call_count < minimum:
        raise ValueError(
            f"CHAI_BO_CALLS must be at least {minimum} "
            f"({initial_points} initial points + {seeded_points} seeded baseline)"
        )

    model_guided = call_count - minimum
    if model_guided < MIN_MODEL_GUIDED_CALLS:
        print(
            f"[BayesOpt] WARNING: CHAI_BO_CALLS={call_count} leaves only "
            f"{model_guided} model-guided evaluation(s) after "
            f"{initial_points} random initial points and the identity baseline. "
            f"Below {MIN_MODEL_GUIDED_CALLS} the result is driven by the random "
            "draw rather than by the optimizer; treat it as a smoke run, not "
            "as a comparison."
        )
    return call_count
