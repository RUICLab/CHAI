"""Shared utilities for the biomedical pilot experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_metrics(
    labels: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(labels if isinstance(labels, np.ndarray) else list(labels), dtype=np.int64)
    p = np.asarray(
        probabilities if isinstance(probabilities, np.ndarray) else list(probabilities), dtype=np.float64
    )
    predictions = (p >= threshold).astype(np.int64)
    return {
        "accuracy": float((predictions == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "auroc": float(roc_auc_score(y, p)),
        "sensitivity": float(((predictions == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
        "specificity": float(((predictions == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
    }


def bootstrap_ci(
    labels: Iterable[int],
    probabilities: Iterable[float],
    seed: int,
    threshold: float = 0.5,
    repetitions: int = 1_000,
) -> dict[str, list[float]]:
    """Percentile 95% CIs for the two primary classification metrics."""
    y = np.asarray(labels if isinstance(labels, np.ndarray) else list(labels), dtype=np.int64)
    p = np.asarray(
        probabilities if isinstance(probabilities, np.ndarray) else list(probabilities), dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    values = {"balanced_accuracy": [], "auroc": []}
    for _ in range(repetitions):
        sample = generator.integers(0, len(y), size=len(y))
        if len(np.unique(y[sample])) != 2:
            continue
        metrics = binary_metrics(y[sample], p[sample], threshold)
        for name in values:
            values[name].append(metrics[name])
    return {
        name: [float(np.quantile(metric_values, 0.025)), float(np.quantile(metric_values, 0.975))]
        for name, metric_values in values.items()
    }


def youden_threshold(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    """Select a decision threshold on validation data, never on test data."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        np.asarray(labels if isinstance(labels, np.ndarray) else list(labels), dtype=np.int64),
        np.asarray(
            probabilities if isinstance(probabilities, np.ndarray) else list(probabilities),
            dtype=np.float64,
        ),
    )
    finite = np.isfinite(thresholds)
    scores = np.where(finite, true_positive_rate - false_positive_rate, -np.inf)
    return float(thresholds[int(np.argmax(scores))])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
