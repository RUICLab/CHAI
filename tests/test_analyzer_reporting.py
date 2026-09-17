"""Tests for the two analyser quantities that were previously conflated."""

import importlib.util
import json
import os

_SPEC = importlib.util.spec_from_file_location(
    "analyze_ablation",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "analyze_ablation.py"),
)
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


def _cell(tmp_path, plans, violation_attempts=0, raw_false=None, removed=None,
          raw_checked=None, candidates_checked=None):
    logs = tmp_path / "logs"
    outputs = tmp_path / "outputs"
    logs.mkdir(parents=True)
    outputs.mkdir(parents=True)
    (tmp_path / "status.json").write_text(json.dumps({"exit_code": 0}))
    metrics = {
        "run_outcome": "valid_plan_generated",
        "operator_conformance_violation_count": violation_attempts,
        "functions_attempted": len(plans),
        "functions_succeeded": len(plans),
    }
    if raw_false is not None:
        metrics["raw_operator_claim_violation_count"] = raw_false
    if raw_checked is not None:
        metrics["raw_operator_claims_checked"] = raw_checked
    if candidates_checked is not None:
        metrics["raw_operator_candidates_checked"] = candidates_checked
    if removed is not None:
        metrics["removed_operator_claim_count"] = removed
    (logs / "experiment_metrics_app_MSP430_chai.json").write_text(
        json.dumps(metrics)
    )
    (outputs / "accepted_approximation_plans_app_MSP430_chai.json").write_text(
        json.dumps({"plans": plans})
    )
    return tmp_path


def test_attempt_counts_and_final_claims_are_different_quantities(tmp_path):
    """A rejected attempt is not a shipped claim.

    The analyser previously reported every unenforced attempt as "shipped",
    which counted retried and replaced candidates, and functions that never
    reached the accepted plan.
    """
    honest_plan = [{
        "functionName": "solve",
        "technique_numbers": [21],
        "completeFunction": GENUINE,
    }]
    cell_dir = _cell(tmp_path, honest_plan, violation_attempts=7)

    provenance = analyzer._final_provenance(str(cell_dir))
    assert provenance["functions_checked"] == 1
    assert provenance["false_claims_in_final_plan"] == 0, (
        "the accepted plan is clean even though 7 attempts were rejected"
    )


def test_a_false_claim_in_the_accepted_plan_is_counted(tmp_path):
    bad_plan = [{
        "functionName": "sqrt16",
        "technique_numbers": [21],
        "completeFunction": FALSE_T21,
    }]
    cell_dir = _cell(tmp_path, bad_plan)

    provenance = analyzer._final_provenance(str(cell_dir))
    assert provenance["functions_checked"] == 1
    assert provenance["false_claims_in_final_plan"] == 1


def test_missing_accepted_artifact_reports_none_not_zero(tmp_path):
    """Absence of evidence must not be reported as evidence of cleanliness."""
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (tmp_path / "status.json").write_text(json.dumps({"exit_code": 0}))
    assert analyzer._final_provenance(str(tmp_path)) is None


def test_analyzer_keeps_raw_removed_and_shipped_claims_distinct(tmp_path):
    honest_plan = [{
        "functionName": "solve",
        "technique_numbers": [21],
        "completeFunction": GENUINE,
    }]
    cell_dir = _cell(
        tmp_path, honest_plan, raw_false=6, removed=4,
        raw_checked=20, candidates_checked=8,
    )

    cell = analyzer.read_cell(str(cell_dir))

    assert cell["raw_false_claims"] == 6
    assert cell["raw_claims_checked"] == 20
    assert cell["raw_candidates_checked"] == 8
    assert cell["removed_false_claims"] == 4
    assert cell["final_provenance"]["false_claims_in_final_plan"] == 0


def test_old_campaign_does_not_render_missing_raw_measurement_as_zero(tmp_path):
    cell_dir = _cell(tmp_path, [], violation_attempts=7)

    cell = analyzer.read_cell(str(cell_dir))

    assert cell["raw_false_claims"] is None
    assert cell["removed_false_claims"] is None


def test_analyzer_reads_optimizer_budget_split(tmp_path):
    cell_dir = _cell(tmp_path, [])
    (cell_dir / "logs" / "best_knobs_app.csv").write_text(
        "knobs_list,error,power_cycles,original_power_cycles,"
        "power_cycle_ratio,optimization_metric,bo_outcome,identity_verified,"
        "bo_call_budget,bo_dimensions,local_identity_probes,global_search_calls\n"
        '"[]",0,19,19,1,1,baseline_selected,True,40,10,4,35\n'
    )

    cell = analyzer.read_cell(str(cell_dir))

    assert cell["bo_call_budget"] == 40
    assert cell["bo_dimensions"] == 10
    assert cell["local_identity_probes"] == 4
    assert cell["global_search_calls"] == 35


GENUINE = """void solve(int n) {
  /*Knob Variables Declaration Start*/
  int max_iterations = 64;
  /*Knob Variables Declaration End*/
  int iters = 0;
  for (int i = 0; i < n; i++) {
    work(i);
    iters++;
    if (iters >= max_iterations) { break; }
  }
}"""

FALSE_T21 = """static uint16_t sqrt16(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int max_iterations = 50;
  /*Knob Variables Declaration End*/
  uint16_t hi = 0xffff, lo = 0, mid = 0; uint32_t s = 0;
  int limit = (max_iterations * 100) / 100;
  while (s != x && hi - lo > 1) { mid = (hi + lo) >> 1; s = mid * mid;
    if (s < x) lo = mid; else hi = mid; }
  return mid;
}"""
