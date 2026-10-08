"""Simple forecasters the pretrained model has to beat.

If a foundation model cannot beat persistence and an exponential moving average
on this data, it does not belong in the system.
"""

from __future__ import annotations

import numpy as np


def persistence(context: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    return np.full(horizon, context[-1], dtype=float)


def moving_average(context: np.ndarray, horizon: int, window: int = 32) -> np.ndarray:
    return np.full(horizon, float(context[-window:].mean()), dtype=float)


def ewma(context: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    weights = (1 - alpha) ** np.arange(len(context))[::-1]
    value = float(np.sum(weights * context) / np.sum(weights))
    return np.full(horizon, value, dtype=float)


def drift(context: np.ndarray, horizon: int) -> np.ndarray:
    """Linear continuation of the context's overall slope, clipped at zero."""
    if len(context) < 2:
        return persistence(context, horizon)
    slope = (context[-1] - context[0]) / (len(context) - 1)
    return np.clip(context[-1] + slope * np.arange(1, horizon + 1), 0.0, None)


BASELINES = {
    "persistence": persistence,
    "moving_average": moving_average,
    "ewma": ewma,
    "drift": drift,
}
