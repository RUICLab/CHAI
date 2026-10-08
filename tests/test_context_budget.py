"""Parent context must be bounded, and its size must be measured."""

from utils import token_budget
from utils.context import (
    enforce_callsite_operator_requirements,
    loop_invariant_callsite_evidence,
    summarize_direct_callsites,
    summarize_parent_contribution,
)

PARENT_CONVERSATION = [
    ("human", "What is the purpose of featurize?"),
    ("ai", "featurize computes mean and stddev features over an accelerometer window."),
    ("human", "Plan the approximation."),
    ("ai", "PLAN:\n- OPERATOR: T1\n  REGION: the feature loop\n" + "x" * 3000),
    ("human", "Write the code."),
    ("ai", """```c
void featurize(int *w, int n, float *out)
{
  /*Knob Variables Declaration Start*/
  int iteration_pct = 100;
  /*Knob Variables Declaration End*/
""" + "  /* body */\n" * 400 + "}\n```"),
    ("human", "Now as JSON."),
    ("ai", "{'approxmated_code': 'void featurize(int *w, int n, float *out)"
            " { " + "y" * 4000 + " }'}"),
]


def test_parent_summary_keeps_purpose_signature_and_knobs():
    summary = summarize_parent_contribution("featurize", PARENT_CONVERSATION)

    assert len(summary) == 2
    text = summary[1][1]
    assert "featurize computes mean and stddev" in text
    assert "featurize" in text


def test_parent_summary_is_far_smaller_than_the_raw_conversation():
    raw = sum(len(str(message)) for _role, message in PARENT_CONVERSATION)
    summary = summarize_parent_contribution("featurize", PARENT_CONVERSATION)
    compact = sum(len(str(message)) for _role, message in summary)

    assert compact < raw / 5, (
        f"parent context {compact} chars is not meaningfully smaller than "
        f"the raw {raw} chars"
    )


def test_empty_parent_conversation_contributes_nothing():
    assert summarize_parent_contribution("f", []) == []


def test_callsite_context_exposes_nested_repeated_calls_without_whole_file():
    entities = [
        {
            "functionName": "main",
            "completeFunction": """
            int main(void) {
              for (int outer = 0; outer < 100; ++outer) {
                for (int inner = 0; inner < 100; ++inner) {
                  init_search(patterns[outer]);
                  run_search(texts[inner]);
                }
              }
            }
            """,
        },
        {
            "functionName": "init_search",
            "completeFunction": "void init_search(char *p) { for (int i=0;i<32;i++) work(p,i); }",
        },
    ]

    context = summarize_direct_callsites("init_search", entities)

    assert len(context) == 1
    text = context[0][1]
    assert "outer < 100" in text and "inner < 100" in text
    assert "init_search(patterns[outer])" in text
    assert len(text) <= 3000

    evidence = loop_invariant_callsite_evidence("init_search", entities)
    assert evidence
    assert "innermost loop over inner" in evidence[0]


def test_callsite_reuse_detector_rejects_arguments_that_change_each_iteration():
    entities = [
        {
            "functionName": "main",
            "completeFunction": """
            int main(void) {
              for (int i = 0; i < 100; ++i) prepare(patterns[i]);
            }
            """,
        },
        {
            "functionName": "prepare",
            "completeFunction": "void prepare(char *p) { for (int j=0;j<32;j++) work(p,j); }",
        },
    ]

    assert loop_invariant_callsite_evidence("prepare", entities) == []


def test_outer_loop_invariance_does_not_hide_inner_loop_key_changes():
    entities = [
        {
            "functionName": "main",
            "completeFunction": """
            int main(void) {
              for (int y = 0; y < 100; ++y) {
                for (int i = 0; i < 100; ++i) search(texts[i]);
              }
            }
            """,
        },
        {
            "functionName": "search",
            "completeFunction": "int search(char *p) { while (*p) ++p; return 0; }",
        },
    ]

    assert loop_invariant_callsite_evidence("search", entities) == []


def test_t3_selection_requires_stable_key_callsite_evidence():
    entities = [
        {
            "functionName": "main",
            "completeFunction": """
            int main(void) {
              for (int y = 0; y < 8; ++y) {
                for (int i = 0; i < 8; ++i) prepare(patterns[y]);
                search(texts[y]);
              }
            }
            """,
        },
        {
            "functionName": "prepare",
            "completeFunction": "void prepare(char *p) { for (int j=0;j<32;j++) work(p,j); }",
        },
        {
            "functionName": "search",
            "completeFunction": "int search(char *p) { while (*p) ++p; return 0; }",
        },
    ]

    selected, removed, evidence = enforce_callsite_operator_requirements(
        {3, 21}, "prepare", entities
    )
    assert selected == {3, 21}
    assert removed == set()
    assert evidence

    selected, removed, evidence = enforce_callsite_operator_requirements(
        {3, 21}, "search", entities
    )
    assert selected == {21}
    assert removed == {3}
    assert evidence == []


def test_context_sizes_are_recorded_and_summarised():
    token_budget.reset()
    token_budget.record_context("small", [("system", "abc")], parent_count=0)
    token_budget.record_context("large", [("system", "x" * 70000)], parent_count=8)

    summary = token_budget.summary()

    assert summary["contexts_measured"] == 2
    assert summary["peak_function"] == "large"
    assert summary["peak_parents"] == 8
    assert summary["max_estimated_tokens"] > summary["min_estimated_tokens"]
    token_budget.reset()
