"""Parent context must be bounded, and its size must be measured."""

from utils import token_budget
from utils.context import summarize_parent_contribution

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
