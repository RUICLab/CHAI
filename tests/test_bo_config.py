import pytest

from utils.bo_config import validate_bo_call_budget


def test_bo_budget_includes_seeded_baseline():
    assert validate_bo_call_budget(6, initial_points=5, seeded_points=1) == 6


def test_bo_budget_rejects_five_calls_before_expensive_work():
    with pytest.raises(ValueError, match="at least 6"):
        validate_bo_call_budget(5, initial_points=5, seeded_points=1)


