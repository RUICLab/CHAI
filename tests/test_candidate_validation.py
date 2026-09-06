import os

import pytest

from utils.error_analyzer import arError
from utils.validator import (
    _function_signature,
    _function_signature_text,
    _shadowed_written_state,
    _unsafe_loop_bound_knobs,
    _unsafe_loop_progress_knobs,
    normalizeSafeKnobDefaults,
    updateKnobValue,
)


def test_knob_update_changes_only_initialized_local_declaration():
    function = {
        "completeFunction": (
            "static int classify(int feature) {\n"
            "  int shift_bits = 1;\n"
            "  return feature >> shift_bits;\n"
            "}"
        )
    }

    updateKnobValue(function, 3, "shift_bits")

    assert "static int classify(int feature) {" in function["completeFunction"]
    assert "int shift_bits = 3;" in function["completeFunction"]


def test_knob_update_never_rewrites_a_function_parameter():
    function = {"completeFunction": "int classify(int shift_bits) { return shift_bits; }"}

    with pytest.raises(ValueError, match="local declaration"):
        updateKnobValue(function, 3, "shift_bits")

    assert function["completeFunction"] == "int classify(int shift_bits) { return shift_bits; }"


def test_function_signature_comparison_detects_added_knob_parameter():
    original = "static int classify(int feature, int model) { return feature + model; }"
    invalid = "static int classify(int feature, int model, int shift_bits) { return 0; }"

    assert _function_signature(original, "classify") != _function_signature(invalid, "classify")


def test_signature_does_not_absorb_preprocessor_or_comment_text():
    sobel = "#endif\n\nvoid sobel_filtering( )\n/* output image */\n{ return; }"
    lqi = (
        "// assuming a maximum size of the vector\n"
        "double calculateDeliveryRatio(PacketLog logs[], int size)\n"
        "{ return 0; }"
    )

    assert _function_signature_text(sobel, "sobel_filtering") == "void sobel_filtering( )"
    assert _function_signature_text(lqi, "calculateDeliveryRatio") == (
        "double calculateDeliveryRatio(PacketLog logs[], int size)"
    )


def test_ar_error_raises_on_build_failure_and_restores_cwd(tmp_path):
    (tmp_path / "Makefile").write_text("clean:\n\ttrue\nmain:\n\tfalse\n")
    cwd = os.getcwd()

    with pytest.raises(RuntimeError, match="build failed"):
        arError(str(tmp_path))

    assert os.getcwd() == cwd


def test_safe_knob_defaults_include_and_select_identity_point():
    plan = {
        "completeFunction": (
            "void work(void) {\n"
            "  double precision_scale = 0.9;\n"
            "  int quantization_factor = 10;\n"
            "}"
        ),
        "knobVariables": "['precision_scale', 'quantization_factor']",
        "knobRanges": (
            "[{'precision_scale': [0.5, 1.0]}, "
            "{'quantization_factor': [2, 20]}]"
        ),
        "knobStepSize": (
            "[{'precision_scale': 'Real'}, "
            "{'quantization_factor': 'Integer'}]"
        ),
    }

    actions = normalizeSafeKnobDefaults(plan)

    assert "double precision_scale = 1.0;" in plan["completeFunction"]
    assert "int quantization_factor = 1;" in plan["completeFunction"]
    assert "{'quantization_factor': [1, 20]}" in plan["knobRanges"]
    assert len(actions) == 3


def test_candidate_cannot_shadow_state_written_by_original():
    original = (
        "void init_search(const char *string) {\n"
        "  size_t i;\n"
        "  len = 3; table[i] = len; findme = (char *)string;\n"
        "}"
    )
    candidate = (
        "void init_search(const char *string) {\n"
        "  size_t i;\n"
        "  size_t len = 3;\n"
        "  int table[256];\n"
        "  char *findme;\n"
        "}"
    )

    assert _shadowed_written_state(original, candidate) == {"len", "table", "findme"}


def test_for_initializer_is_recognized_as_original_local_state():
    original = (
        "double ratio(int size) {\n"
        "  double total = 0;\n"
        "  for (int i = 0; i < size; i++) total += i;\n"
        "  return total;\n"
        "}"
    )
    candidate = (
        "double ratio(int size) {\n"
        "  double total = 0;\n"
        "  int i;\n"
        "  for (i = 0; i < size; i++) total += i;\n"
        "  return total;\n"
        "}"
    )

    assert _shadowed_written_state(original, candidate) == set()


def test_real_integer_loop_increment_is_rejected_before_execution():
    code = "for (int i = 0; i < size; i += task_skip_factor) { work(i); }"

    unsafe = _unsafe_loop_progress_knobs(
        code,
        ["task_skip_factor"],
        [{"task_skip_factor": [0.1, 1.0]}],
        [{"task_skip_factor": "Real"}],
    )

    assert unsafe == ["task_skip_factor"]


def test_positive_integer_loop_increment_is_allowed():
    code = "for (int i = 0; i < size; i += stride) { work(i); }"

    assert _unsafe_loop_progress_knobs(
        code,
        ["stride"],
        [{"stride": [1, 4]}],
        [{"stride": "Integer"}],
    ) == []


def test_loop_bound_multiplier_cannot_expand_input_domain():
    code = "for (int i = 0; i < size * perforation_factor; i++) { work(i); }"

    assert _unsafe_loop_bound_knobs(
        code,
        ["perforation_factor"],
        [{"perforation_factor": [1, 80]}],
    ) == ["perforation_factor"]
    assert _unsafe_loop_bound_knobs(
        code,
        ["perforation_factor"],
        [{"perforation_factor": [0.1, 1.0]}],
    ) == []

