import pytest

from utils.optimization_metrics import (
    checkpoint_metrics,
    constrained_optimization_metric,
    dominates_original,
    is_identity_error,
    power_cycle_metrics,
)


def test_power_cycle_ratio_is_not_mislabeled_as_reduction():
    remaining_ratio, reduction = power_cycle_metrics(11, 12)

    assert remaining_ratio == pytest.approx(11 / 12)
    assert reduction == pytest.approx(1 / 12)


def test_checkpoint_metrics_alias_still_works():
    """Legacy callers and CSV headers keep working after the rename."""
    assert checkpoint_metrics(11, 12) == power_cycle_metrics(11, 12)


def test_objective_normalises_error_by_the_accuracy_bound():
    """Both terms must live on a comparable scale inside the feasible region.

    Summing a raw error in [0, 0.30] with a ratio in [0, inf) weighted the
    hardware term about 3x more than accuracy while claiming they were equal.
    """
    assert constrained_optimization_metric(
        0.134, 11, 12, error_bound=0.30
    ) == pytest.approx(0.134 / 0.30 + 11 / 12)


def test_objective_at_the_bound_contributes_exactly_one():
    assert constrained_optimization_metric(
        0.30, 12, 12, error_bound=0.30
    ) == pytest.approx(2.0)


def test_power_cycle_metrics_reject_zero_baseline():
    with pytest.raises(ValueError, match="greater than zero"):
        power_cycle_metrics(0, 0)


def test_identity_tolerance_rejects_a_visibly_wrong_baseline():
    """An 18% error was previously accepted as the 'safe baseline'."""
    assert is_identity_error(0.0)
    assert is_identity_error(1e-9)
    assert not is_identity_error(0.1827)


def test_dominance_gate_rejects_candidates_worse_than_doing_nothing():
    """Selected plans previously ran at 1.89x and 2.21x the original."""
    assert dominates_original(11, 12)
    assert dominates_original(12, 12)
    assert not dominates_original(36, 19)
    assert not dominates_original(42, 19)
