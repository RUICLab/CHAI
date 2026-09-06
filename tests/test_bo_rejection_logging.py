import json

import numpy as np

from lib.bo import _record_evaluation_rejection, _safe_error_evaluation


def test_rejection_log_serializes_optimizer_numpy_scalars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    _record_evaluation_rejection(
        "fft-iclib",
        [np.int64(3), np.float64(0.25)],
        "candidate build failed",
    )

    payload = json.loads(
        (tmp_path / "logs" / "optimization_rejections_fft-iclib.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert payload["knobs"] == [3, 0.25]
    assert payload["reason"] == "candidate build failed"


def test_evaluator_failure_is_logged_at_campaign_root(tmp_path, monkeypatch):
    tuning = tmp_path / "knob_tuning"
    tuning.mkdir()
    monkeypatch.chdir(tmp_path)

    def failing_evaluator():
        import os

        os.chdir(tuning)
        raise TimeoutError("candidate timed out")

    result = _safe_error_evaluation("test-app", [2], failing_evaluator)

    assert result is None
    root_log = tmp_path / "logs" / "optimization_rejections_test-app.jsonl"
    assert root_log.exists()
    assert "candidate timed out" in root_log.read_text(encoding="utf-8")
    assert not (tuning / "logs" / "optimization_rejections_test-app.jsonl").exists()

