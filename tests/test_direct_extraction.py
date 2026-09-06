from lib.plan_contract import parse_plan
from utils.direct_extraction import (
    build_plan_entry,
    extract_c_function,
    knob_metadata_from_plan,
)

PLAN_TEXT = """
PLAN:
- OPERATOR: T1
  REGION: the loop over i
  KNOB: iteration_pct | Integer | RANGE [50, 100] | IDENTITY 100
  BENEFIT: iterations
  PRECONDITION: n > 0
  INVARIANT: processes a prefix of the array
INTERACTIONS: none
"""

GENERATION = """APPLIED_TECHNIQUES: T1

Here is the approximated function:

```c
void work(int *data, int n)
{
  /*Knob Variables Declaration Start*/
  int iteration_pct = 100;
  /*Knob Variables Declaration End*/
  int limit = (n * iteration_pct) / 100;
  for (int i = 0; i < limit; i++) {
    if (data[i] > 0) { data[i] = data[i] >> 1; }
  }
}
```
"""


def test_extracts_a_balanced_definition_from_a_fenced_block():
    code = extract_c_function(GENERATION, "work")

    assert code.startswith("void work(int *data, int n)")
    assert code.count("{") == code.count("}")
    assert code.rstrip().endswith("}")


def test_extracts_without_a_code_fence():
    code = extract_c_function(
        "Some prose.\nint f(int a) { if (a) { return 1; } return 0; }\nMore prose.",
        "f",
    )

    assert code == "int f(int a) { if (a) { return 1; } return 0; }"


def test_knob_schema_comes_from_the_approved_plan():
    plan, errors = parse_plan(PLAN_TEXT)
    assert errors == []

    variables, ranges, steps = knob_metadata_from_plan(plan)

    assert variables == "['iteration_pct']"
    assert ranges == "[{'iteration_pct': [50, 100]}]"
    assert steps == "[{'iteration_pct': 'Integer'}]"


def test_builds_a_plan_entry_without_a_second_model_call():
    plan, _errors = parse_plan(PLAN_TEXT)

    entry = build_plan_entry("work", GENERATION, plan)

    assert entry is not None
    assert entry["functionName"] == "work"
    assert "iteration_pct" in entry["completeFunction"]
    assert entry["knobVariables"] == "['iteration_pct']"


def test_falls_back_when_a_planned_knob_is_absent_from_the_code():
    """Never build a search space over a variable the code does not declare."""
    plan, _errors = parse_plan(PLAN_TEXT)
    without_knob = GENERATION.replace("int iteration_pct = 100;", "")

    assert build_plan_entry("work", without_knob, plan) is None


def test_falls_back_when_the_function_cannot_be_located():
    plan, _errors = parse_plan(PLAN_TEXT)

    assert build_plan_entry("other_function", GENERATION, plan) is None


def test_knob_initializer_is_aligned_with_the_plan_identity():
    """Catch an identity mismatch statically instead of after a full sim run."""
    plan, _errors = parse_plan(PLAN_TEXT)
    misaligned = GENERATION.replace(
        "int iteration_pct = 100;", "int iteration_pct = 80;"
    )

    entry = build_plan_entry("work", misaligned, plan)

    assert entry is not None
    assert "int iteration_pct = 100;" in entry["completeFunction"]
    assert entry["planAlignmentActions"], "the repair must be reported"
    assert "80" in entry["planAlignmentActions"][0]


def test_no_alignment_action_when_the_code_already_matches():
    plan, _errors = parse_plan(PLAN_TEXT)

    entry = build_plan_entry("work", GENERATION, plan)

    assert entry["planAlignmentActions"] == []
