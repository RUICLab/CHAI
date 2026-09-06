from utils.retry_context import (
    generation_attempt_context,
    json_conversion_context,
)


def test_first_attempt_receives_full_planning_context():
    assert generation_attempt_context([("system", "plan")], 1) == [
        ("system", "plan")
    ]


def test_repair_attempt_drops_rejected_long_context():
    assert generation_attempt_context([("ai", "old broken candidate")], 2) == []
    assert json_conversion_context(
        [("ai", "old broken candidate")],
        [("ai", "new repaired candidate")],
        2,
    ) == [("ai", "new repaired candidate")]


