"""Few-shot examples must be scoped to the operators a run can select."""

import config.globals as globals_module
from utils.initialization import (
    _split_examples_by_technique,
    loadNewTechniqueExamples,
)

EXAMPLES_PATH = "prompts/FewShotExamples/newTechniqueExamples.txt"


def _reset():
    globals_module.NEW_TECHNIQUE_EXAMPLES.clear()


def test_examples_split_by_technique_id():
    text = open(EXAMPLES_PATH).read()

    _preamble, sections = _split_examples_by_technique(text)

    assert set(sections) == {3, 21, 22, 23, 25, 26, 27, 29, 30}


def test_t3_example_is_outside_the_evaluation_suite_and_leak_detectable():
    text = open(EXAMPLES_PATH).read()
    _preamble, sections = _split_examples_by_technique(text)
    example = sections[3]

    assert "demo_t3_configure_filter" in example
    assert "init_search" not in example
    assert "strsearch" not in example


def test_withdrawn_t25_example_is_scoped_to_t25_not_its_neighbour():
    """The withdrawal notice must live in T25's own section.

    An earlier withdrawal used a heading the section parser does not match
    (``### Technique 25 ...`` rather than ``### Example N: ... (Technique 25)``),
    so T25 stopped parsing as a section entirely: the notice was absorbed into
    the preceding T30 example. Runs that could select T25 never saw the
    warning, while AR and Sobel runs -- which receive T30 -- were shown an
    irrelevant one.
    """
    text = open(EXAMPLES_PATH).read()
    _preamble, sections = _split_examples_by_technique(text)

    assert 25 in sections
    assert "WITHDRAWN" in sections[25]
    assert "WITHDRAWN" not in sections[30]


def test_withdrawal_notice_follows_t25_scoping():
    _reset()
    loadNewTechniqueExamples(include_auto_generated_cards=False,
                             feasible_ids={25})
    with_t25 = globals_module.NEW_TECHNIQUE_EXAMPLES[0][1]

    _reset()
    loadNewTechniqueExamples(include_auto_generated_cards=False,
                             feasible_ids={30})
    without_t25 = globals_module.NEW_TECHNIQUE_EXAMPLES[0][1]
    _reset()

    assert "WITHDRAWN" in with_t25
    assert "WITHDRAWN" not in without_t25


def test_auto_generated_card_examples_survive_edits_to_the_base_examples():
    """T31/T32 sit after the last base example and were once deleted whole by a
    regex that ran to end-of-file while replacing T25."""
    text = open(EXAMPLES_PATH).read()

    assert "AUTO-GENERATED FEW-SHOT START" in text
    assert "(Technique 31)" in text
    assert "(Technique 32)" in text


def test_scoping_drops_examples_for_infeasible_operators():
    """Sobel has no use for the nibble-lookup or link-estimator examples."""
    _reset()
    loadNewTechniqueExamples(include_auto_generated_cards=False,
                             feasible_ids={21, 22, 30})
    scoped = globals_module.NEW_TECHNIQUE_EXAMPLES[0][1]

    _reset()
    loadNewTechniqueExamples(include_auto_generated_cards=False)
    full = globals_module.NEW_TECHNIQUE_EXAMPLES[0][1]
    _reset()

    assert len(scoped) < len(full) / 1.5
    assert "Technique 22" in scoped
    assert "Technique 27" not in scoped
    assert "Technique 26" not in scoped


def test_scoping_falls_back_to_everything_when_nothing_matches():
    """Never hand the model an empty example block."""
    _reset()
    loadNewTechniqueExamples(include_auto_generated_cards=False, feasible_ids={99})
    text = globals_module.NEW_TECHNIQUE_EXAMPLES[0][1]
    _reset()

    assert "Technique 21" in text


def test_t21_example_preserves_the_identity_contract():
    """The example taught the model to change the original failure return."""
    text = open(EXAMPLES_PATH).read()
    _preamble, sections = _split_examples_by_technique(text)
    binary_search = sections[21]

    # The identity path must still return -1, as the original did.
    assert "return -1;" in binary_search
    approximated = binary_search.split("#### Approximated function")[1]
    assert approximated.rstrip().rstrip("`").rstrip().endswith("}")
    # The old example ended the function with `return mid;`, which is wrong at
    # early_exit_enabled = 0.
    assert not approximated.replace(" ", "").count("}\nreturnmid;")
