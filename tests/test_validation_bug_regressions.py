"""Regressions for three validation defects that each produced a wrong verdict.

All three were found while putting two CNN benchmarks (pcam-iclib,
coughvid-iclib) through the pipeline. Each one made CHAI reject or mis-scope
something that was actually correct, so each test here asserts the *absence*
of a false verdict rather than the presence of a true one.
"""

import re

import pytest

from lib.plan_contract import parse_plan, validate_plan_against_hardware
from utils.validator import _declared_names, _shadowed_written_state


class _Profile:
    """Minimal stand-in for lib.hardware_profile.HardwareProfile."""

    def __init__(self, has_fpu, name="MSP430"):
        self.has_fpu = has_fpu
        self.name = name


# ---------------------------------------------------------------------------
# Bug 1: comma-separated declarators were invisible to the shadowing check
# ---------------------------------------------------------------------------

ORIGINAL_CONV = """static void conv_block(
    const float *input, int in_ch, int in_h, int in_w,
    const float *weight, const float *bias,
    float *output, int out_ch, int out_h, int out_w)
{
  int oc, oy, ox, ic, ky, kx;
  for (oc = 0; oc < out_ch; oc++) {
    for (oy = 0; oy < out_h; oy++) {
      for (ox = 0; ox < out_w; ox++) {
        float sum = bias[oc];
        output[(oc * out_h + oy) * out_w + ox] = sum;
      }
    }
  }
}"""


@pytest.mark.parametrize(
    "code, expected",
    [
        ("int oc, oy, ox, ic, ky, kx;", {"oc", "oy", "ox", "ic", "ky", "kx"}),
        ("int a = 1, b = 2;", {"a", "b"}),
        ("uint32_t in_bits, w_bits, mantissa_mask;",
         {"in_bits", "w_bits", "mantissa_mask"}),
        ("float sum = bias[oc];", {"sum"}),
        ("static float conv1_out[8][16][16];", {"conv1_out"}),
        ("int fill_row, accumulation_done;", {"fill_row", "accumulation_done"}),
    ],
)
def test_declared_names_handles_declarator_lists(code, expected):
    """A comma-separated list declares every name in it, not just the first."""
    assert expected <= _declared_names(code)


def test_declarator_list_is_not_reported_as_external_state():
    """The exact false rejection that blocked every generation attempt.

    The original declares its loop indices as `int oc, oy, ox, ic, ky, kx;`.
    A candidate that declares them inline instead -- valid C -- must not be
    accused of shadowing "externally visible state", because those indices are
    local to the original too. The old single-name regex could not match the
    comma-separated form, so the indices looked undeclared-but-written, i.e.
    external, and the candidate was rejected with feedback it could not act on.
    """
    candidate = ORIGINAL_CONV.replace("for (oc = 0;", "for (int oc = 0;")
    assert _shadowed_written_state(ORIGINAL_CONV, candidate) == set()


def test_genuine_shadowing_is_still_caught():
    """The fix must not blind the check to a real global being shadowed."""
    original = """void classify( )
{
  int c;
  float logit = 0.0f;
  for (c = 0; c < 4; c++) {
    logit += 1.0f;
  }
  result_logit = logit;
}"""
    candidate = original.replace(
        "  float logit = 0.0f;", "  float logit = 0.0f;\n  float result_logit;"
    )
    assert "result_logit" in _shadowed_written_state(original, candidate)


def test_keywords_are_never_reported_as_declared_names():
    assert not (_declared_names("int x;") & {"int"})


# ---------------------------------------------------------------------------
# Bug 2: a Real knob on a no-FPU target was only rejected after generation
# ---------------------------------------------------------------------------

def _plan(knob_type):
    text = f"""
PLAN:
- OPERATOR: T1
REGION: the oy loop in conv_block
PRECONDITION: adjacent rows are correlated
BENEFIT: removes multiply-accumulate work
INVARIANT: output keeps its shape at every setting
KNOB: perforation | {knob_type} | RANGE [0, 1] | IDENTITY 1

INTERACTIONS: none
"""
    plan, errors = parse_plan(text, function_code=ORIGINAL_CONV)
    assert not errors, errors
    return plan


def test_real_knob_rejected_at_plan_time_without_fpu():
    """Reject while the plan can still be revised, not after code generation.

    Left to the old path a Real knob was accepted here, rewritten to a C
    `double` by the candidate repair so the code matched the plan, and only
    then rejected for being floating point on a no-FPU target -- with feedback
    asking for the integer form the model had already written. Unescapable.
    """
    errors = validate_plan_against_hardware(_plan("Real"), _Profile(has_fpu=False))
    assert errors, "a Real knob must be rejected at plan time on a no-FPU target"
    assert "perforation" in errors[0]
    assert "Integer" in errors[0], "feedback must say what to do instead"


def test_integer_knob_accepted_without_fpu():
    assert validate_plan_against_hardware(
        _plan("Integer"), _Profile(has_fpu=False)
    ) == []


def test_real_knob_accepted_when_the_target_has_an_fpu():
    assert validate_plan_against_hardware(
        _plan("Real"), _Profile(has_fpu=True)
    ) == []


def test_hardware_plan_check_is_inert_without_a_profile():
    """Runs with --hardware omitted must keep their legacy behaviour."""
    assert validate_plan_against_hardware(_plan("Real"), None) == []


# ---------------------------------------------------------------------------
# Bug 3: operator applicability is a curated benchmark-name allowlist
# ---------------------------------------------------------------------------

def test_applicability_allowlists_are_documented_as_curated():
    """Guard the claim, not the mechanism.

    Applicability is decided by `app_name not in tech.applicable_apps`, a
    hand-maintained mapping. That is a legitimate design, but it means a new
    benchmark is auto-rejected from every operator whose list predates it, and
    some operators have empty lists and can never be selected at all. Any
    statement that a run "searched all 32 operators" is therefore unsupportable.

    This test records which operators are unreachable so the number cannot be
    quietly misreported, and fails if the registry grows a new operator with an
    empty allowlist without that being noticed.
    """
    from lib.technique_registry import TECHNIQUE_REGISTRY

    unreachable = sorted(
        tid
        for tid, tech in TECHNIQUE_REGISTRY.items()
        if tech.applicable_apps is not None and not tech.applicable_apps
    )
    # T15, T17, T18, T25, T27, T28 carry empty allowlists today.
    assert unreachable == [15, 17, 18, 25, 27, 28], (
        "the set of never-selectable operators changed; update the reported "
        f"operator count accordingly. Now unreachable: {unreachable}"
    )


def test_neural_network_operator_reason_is_not_stale():
    """T17's reason claimed no benchmark contains a neural network.

    Two CNN benchmarks now exist, so that justification is false. T17 is
    deliberately NOT enabled for them -- channel pruning changes tensor shapes
    and generally needs retraining, so it belongs as a model-level paired
    transformation with regenerated C artifacts rather than a function-local
    rewrite -- but the stated reason must not assert something untrue.
    """
    from lib.technique_registry import TECHNIQUE_REGISTRY

    # The pattern is validated against the exact wording it must catch, so
    # this test cannot silently become vacuous if the phrasing shifts. An
    # earlier version of it passed only because the regex missed "none of
    # these benchmarks contains" -- a test that cannot fail is worse than none.
    claims_no_network = re.compile(
        r"(?:\bno\b|\bnone\b)[^.]{0,40}\bbenchmarks?\b[^.]{0,20}"
        r"\b(?:contain|contains|have|has|include|includes)\b"
    )
    stale_wording = (
        "Trim neural connections. Not applicable: none of these "
        "benchmarks contains a neural network."
    )
    assert claims_no_network.search(stale_wording.lower()), (
        "the guard pattern no longer matches the wording it exists to catch"
    )

    description = (TECHNIQUE_REGISTRY[17].description or "").lower()
    assert not claims_no_network.search(description), (
        "T17's justification claims no benchmark contains a neural network, "
        "which pcam-iclib and coughvid-iclib contradict"
    )
    assert TECHNIQUE_REGISTRY[17].applicable_apps == [], (
        "T17 must stay unselected: pruning changes tensor shapes and needs "
        "retraining, so it is a model-level transformation rather than the "
        "function-local rewrite this pipeline performs"
    )
