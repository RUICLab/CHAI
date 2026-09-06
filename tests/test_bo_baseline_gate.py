import json
from types import SimpleNamespace

import lib.bo as bo


def test_bayes_opt_discards_plan_when_identity_point_fails(monkeypatch):
    monkeypatch.setattr(bo, "createSearchSpace", lambda _: [object()])
    monkeypatch.setattr(bo, "getBaselineKnobs", lambda _: [1])
    monkeypatch.setattr(bo, "evaluateKnobs", lambda _: 100.0)
    monkeypatch.setattr(bo, "getAppName", lambda: "test-app")
    recorded = []
    monkeypatch.setattr(
        bo, "_record_evaluation_rejection", lambda *args: recorded.append(args)
    )
    monkeypatch.setattr(
        bo,
        "gp_minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("optimizer must not run after a failed identity point")
        ),
    )

    score, knobs = bo.runBayesOpt()

    assert score == float("inf")
    assert knobs == []
    assert "declared safe baseline failed" in recorded[0][2]


def test_bayes_opt_counts_identity_point_inside_total_budget(monkeypatch):
    monkeypatch.setattr(bo, "createSearchSpace", lambda _: [object()])
    monkeypatch.setattr(bo, "getBaselineKnobs", lambda _: [1])
    monkeypatch.setattr(bo, "evaluateKnobs", lambda _: 1.0)
    monkeypatch.setattr(bo, "n_calls", 6)
    # Pin the fixed budget: this test is about the identity point being counted
    # inside it, not about dimension-based budget scaling.
    monkeypatch.setattr(bo, "AUTO_BUDGET", False)
    observed = {}

    def fake_minimize(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(fun=2.0, x=[2])

    monkeypatch.setattr(bo, "gp_minimize", fake_minimize)

    score, knobs = bo.runBayesOpt()

    assert observed["n_calls"] == 5
    assert score == 1.0
    assert knobs == [1]


def test_auto_budget_scales_the_search_with_plan_dimensionality(monkeypatch):
    """A 3-knob plan and a 16-knob plan cannot share one adequate budget."""
    from skopt.space import Integer

    observed = {}

    def fake_minimize(func, space, n_calls, n_initial_points, random_state):
        observed["n_calls"] = n_calls

        class Result:
            fun = 0.5
            x = [1] * len(space)

        return Result()

    monkeypatch.setattr(
        bo, "createSearchSpace",
        lambda _: [Integer(0, 10, name=f"k{i}") for i in range(8)],
    )
    monkeypatch.setattr(bo, "getBaselineKnobs", lambda _: [0] * 8)
    monkeypatch.setattr(bo, "evaluateKnobs", lambda knobs: 1.0)
    monkeypatch.setattr(bo, "gp_minimize", fake_minimize)
    monkeypatch.setattr(bo, "n_calls", 8)
    monkeypatch.setattr(bo, "AUTO_BUDGET", True)
    monkeypatch.setattr(bo, "_accepted_candidates", 0)

    bo.runBayesOpt()

    # 8 dimensions -> 5 initial + 1 baseline + max(10, 4*8) = 38 total.
    assert observed["n_calls"] == 37
    assert bo.LAST_OUTCOME["dimensions"] == 8
    assert bo.LAST_OUTCOME["call_budget"] == 38


def test_fixed_plan_without_dimensions_is_evaluated_once(monkeypatch):
    monkeypatch.setattr(bo, "createSearchSpace", lambda _: [])
    monkeypatch.setattr(bo, "getBaselineKnobs", lambda _: [])
    evaluated = []
    monkeypatch.setattr(
        bo, "evaluateKnobs", lambda knobs: evaluated.append(knobs) or 0.75
    )

    score, knobs = bo.runBayesOpt()

    assert evaluated == [[]]
    assert score == 0.75
    assert knobs == []



def test_identity_point_is_never_clamped_out_of_its_own_range(tmp_path):
    """The search space must contain the value the generated code declares.

    Clamping a declared default into a too-narrow range anchored BayesOpt on a
    value the code does not contain, so a candidate could pass the
    generation-time identity gate and then fail BayesOpt's identity check on
    the very same metric.
    """
    plan = tmp_path / "apx_all.json"
    plan.write_text(json.dumps([{
        "functionName": "f",
        "completeFunction": "void f(void){\n  float precision_scale = 1.0;\n}",
        "knobVariables": "['precision_scale']",
        "knobRanges": "[{'precision_scale': [0.9, 0.99]}]",
        "knobStepSize": "[{'precision_scale': 'Real'}]",
    }]))

    baseline = bo.getBaselineKnobs(str(plan))
    space = bo.createSearchSpace(str(plan))

    assert baseline == [1.0], "the declared default must survive verbatim"
    assert space[0].low <= 1.0 <= space[0].high, (
        "the dimension must be widened to contain the identity point"
    )
