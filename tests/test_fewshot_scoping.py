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

    assert set(sections) == {21, 22, 23, 25, 26, 27, 29, 30}


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
