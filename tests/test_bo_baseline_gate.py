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
    assert observed["x0"] == [[1]]
    assert observed["y0"] == [1.0]
    assert score == 1.0
    assert knobs == []
    assert bo.LAST_OUTCOME["selected"] == "original_program"


def test_auto_budget_scales_the_search_with_plan_dimensionality(monkeypatch):
    """A 3-knob plan and a 16-knob plan cannot share one adequate budget."""
    from skopt.space import Integer

    observed = {}

    def fake_minimize(func, space, n_calls, n_initial_points, random_state,
                      x0, y0):
        observed["n_calls"] = n_calls
        observed["x0"] = x0

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

    # 8 dimensions -> 38 total evaluations. Identity and the same fixed four
    # local probes used by every sufficiently large arm seed the surrogate;
    # dimensionality does not consume a larger share of the search budget.
    assert observed["n_calls"] == 33
    assert len(observed["x0"]) == 5
    assert bo.LAST_OUTCOME["dimensions"] == 8
    assert bo.LAST_OUTCOME["call_budget"] == 38
    assert bo.LAST_OUTCOME["local_identity_probe_count"] == 4
    assert bo.LAST_OUTCOME["global_search_calls"] == 33


def test_conservative_neighbors_change_only_one_knob_at_a_time():
    from skopt.space import Integer, Real

    points = bo.conservative_identity_neighbors(
        [Integer(0, 100), Integer(0, 1), Real(0.5, 1.0)],
        [100, 0, 1.0],
        fraction=0.10,
    )

    assert points == [
        [90, 0, 1.0],
        [100, 1, 1.0],
        [100, 0, 0.95],
    ]


def test_probe_cap_is_independent_of_plan_dimensionality():
    small = [[index] for index in range(5)]
    large = [[index] for index in range(19)]

    assert len(bo.evenly_spaced_probes(small, 4)) == 4
    assert len(bo.evenly_spaced_probes(large, 4)) == 4
    assert bo.evenly_spaced_probes(large, 4) == [[0], [6], [12], [18]]


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


def test_identity_point_is_exempt_from_the_dominance_gate(monkeypatch):
    """Scaffolding costs a cycle or two even with every knob at its no-op value.

    An approximated function carries knob declarations and the guards that
    bypass the approximation. At the identity point it is functionally the
    original but measured 20 power cycles against the pristine program's 19.
    Applying the dominance gate to the reference discarded the entire plan
    before the search began -- 8 of 9 replayed cells died exactly that way.
    """
    assert hasattr(bo, "_evaluating_identity_point")
    src = __import__("inspect").getsource(bo.evaluateKnobs)
    assert "_evaluating_identity_point" in src, (
        "the dominance gate must consult the identity-point flag"
    )
