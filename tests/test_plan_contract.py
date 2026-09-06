import pytest

from lib.plan_contract import (
    check_implementation_matches_plan,
    parse_plan,
    validate_plan_against_catalogue,
)

WELL_FORMED = """
PLAN:
- OPERATOR: T1
  REGION: the inner convolution loop over j
  KNOB: iteration_pct | Integer | RANGE [50, 100] | IDENTITY 100
  BENEFIT: iterations
  PRECONDITION: image is at least 3x3
  INVARIANT: output stays a valid PGM of the same dimensions
- OPERATOR: T2
  REGION: the pixel_value accumulation
  KNOB: precision_mode | Integer | RANGE [0, 1] | IDENTITY 0
  BENEFIT: FP emulation
  PRECONDITION: pixel values fit in int32
  INVARIANT: output stays within 0..255
INTERACTIONS: none
RATIONALE: fewer iterations and integer arithmetic both reduce cycles.
"""


def test_well_formed_plan_parses_into_operator_records():
    plan, errors = parse_plan(WELL_FORMED)

    assert errors == []
    assert plan.operator_ids == {1, 2}
    assert plan.knob_names == {"iteration_pct", "precision_mode"}
    assert plan.records[0].region.startswith("the inner convolution loop")


def test_identity_outside_range_is_rejected():
    """An unreachable identity setting makes the safe baseline impossible."""
    plan, errors = parse_plan("""
PLAN:
- OPERATOR: T21
  REGION: the main loop
  KNOB: budget | Integer | RANGE [10, 50] | IDENTITY 100
  INVARIANT: still returns a value
INTERACTIONS: none
""")

    assert any("outside RANGE" in error for error in errors)


def test_missing_region_and_invariant_are_reported():
    plan, errors = parse_plan("""
PLAN:
- OPERATOR: T7
  KNOB: taps | Integer | RANGE [1, 8] | IDENTITY 8
INTERACTIONS: none
""")

    assert any("missing REGION" in error for error in errors)
    assert any("missing INVARIANT" in error for error in errors)


def test_plan_dimensionality_is_capped():
    records = "".join(
        f"""
- OPERATOR: T{tid}
  REGION: region {tid}
  KNOB: knob{tid} | Integer | RANGE [0, 1] | IDENTITY 0
  INVARIANT: fine
"""
        for tid in (1, 2, 3, 4, 7, 8)
    )
    _plan, errors = parse_plan("PLAN:" + records + "\nINTERACTIONS: none",
                               max_operators=4)

    assert any("at most 4" in error for error in errors)


def test_declining_is_a_valid_answer():
    plan, errors = parse_plan("PLAN: none\nRATIONALE: pure accessor.")

    assert plan.declined
    assert errors == []
    error, ids, actions = check_implementation_matches_plan(plan, {1})
    assert error == ""
    assert ids == {1}
    assert actions == []


def test_plan_outside_the_function_operator_contract_is_rejected():
    plan, _errors = parse_plan(WELL_FORMED)

    assert validate_plan_against_catalogue(plan, {1, 2, 21}) == []
    problems = validate_plan_against_catalogue(plan, {21})
    assert problems and "T1, T2" in problems[0]


def test_code_implementing_the_plan_exactly_is_accepted():
    """The link that did not previously exist between plan and code."""
    plan, _errors = parse_plan(WELL_FORMED)

    error, ids, actions = check_implementation_matches_plan(plan, {1, 2})
    assert error == ""
    assert ids == {1, 2}
    assert actions == []


def test_an_unplanned_operator_is_rejected():
    """Adding an operator after approval bypasses its REGION and INVARIANT."""
    plan, _errors = parse_plan(WELL_FORMED)

    error, _ids, _actions = check_implementation_matches_plan(plan, {1, 2, 7})
    assert "T7" in error
    assert "not in the approved plan" in error


def test_a_dropped_operator_narrows_provenance_instead_of_rejecting():
    """A subset of the plan is honest once the declared provenance is corrected.

    Rejecting these threw away otherwise valid candidates for being less
    ambitious than planned, which cost every CHAI cell in the fft campaign.
    """
    plan, _errors = parse_plan(WELL_FORMED)

    error, ids, actions = check_implementation_matches_plan(plan, {1})
    assert error == ""
    assert ids == {1}, "provenance must narrow to what the code contains"
    assert actions and "T2" in actions[0]


def test_implementing_none_of_the_plan_is_still_rejected():
    """Nothing planned was implemented, so there is no provenance to record."""
    plan, _errors = parse_plan(WELL_FORMED)

    error, _ids, _actions = check_implementation_matches_plan(plan, set())
    assert "none of the planned operators" in error
