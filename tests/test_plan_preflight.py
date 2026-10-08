import os
import pytest

from utils.knob_metadata import (
    KnobMetadataError,
    drop_fixed_knobs,
    normalize_knob_metadata,
)
from utils.utils import filterSourceTargets
from utils.source_functions import (
    find_source_function_line_range,
    recover_source_entities,
)


def _plan(ranges="[{'stride': [1, 4]}]", steps="[{'stride': 'Integer'}]"):
    return {
        "functionName": "work",
        "completeFunction": "int work(void) { int stride = 1; return stride; }",
        "knobVariables": "['stride']",
        "knobRanges": ranges,
        "knobStepSize": steps,
    }


def test_knob_metadata_normalizes_supported_container_forms():
    plan = _plan(ranges={"stride": ["1", "4"]}, steps={"stride": "int"})

    variables, ranges, steps = normalize_knob_metadata(plan, mutate=True)

    assert variables == ["stride"]
    assert ranges == [{"stride": [1, 4]}]
    assert steps == [{"stride": "Integer"}]
    assert plan["knobRanges"] == "[{'stride': [1, 4]}]"


def test_knob_metadata_rejects_symbolic_bounds_before_bo():
    with pytest.raises(KnobMetadataError, match="symbolic bounds"):
        normalize_knob_metadata(_plan("[{'stride': [1, MODEL_SIZE]}]"))


@pytest.mark.parametrize(
    "ranges,steps,message",
    [
        ("[{'other': [1, 4]}]", "[{'stride': 'Integer'}]", "same names"),
        ("[{'stride': [4, 4]}]", "[{'stride': 'Integer'}]", "smaller"),
        ("[{'stride': [1, 4]}]", "[{'stride': 'categorical'}]", "Integer or Real"),
    ],
)
def test_knob_metadata_rejects_non_tunable_or_misaligned_schemas(ranges, steps, message):
    with pytest.raises(KnobMetadataError, match=message):
        normalize_knob_metadata(_plan(ranges, steps))


def test_fixed_local_value_is_removed_from_tunable_metadata():
    plan = _plan(
        ranges="[{'stride': [2, 2]}]",
        steps="[{'stride': 'Integer'}]",
    )

    removed = drop_fixed_knobs(plan, mutate=True)
    variables, ranges, steps = normalize_knob_metadata(plan)

    assert removed == ["stride"]
    assert variables == []
    assert ranges == []
    assert steps == []


def test_source_target_filter_removes_pdg_only_nodes():
    targets = {"worker": "approximate", "cexp": "approximate", "helper": "do not approximate"}
    topology = ["cexp", "worker", "helper"]
    entities = [{"functionName": "worker", "completeFunction": "void worker(void) {}"}]

    selected, rejected = filterSourceTargets(targets, topology, entities)

    assert selected == ["worker"]
    assert rejected == [{
        "function": "cexp",
        "reason": "no extracted source definition (external/library/PDG-only node)",
    }]


def test_source_target_filter_keeps_local_computational_definitions():
    targets = {"fft_radix2": "approximate", "cos": "approximate"}
    topology = ["cos", "fft_radix2"]
    entities = [
        {"functionName": "cos", "completeFunction": "double cos(double x) { return x; }"},
        {"functionName": "fft_radix2", "completeFunction": "void fft_radix2(void) {}"},
    ]

    selected, rejected = filterSourceTargets(targets, topology, entities)

    assert selected == ["cos", "fft_radix2"]
    assert rejected == []


def test_source_recovery_handles_complex_return_type_and_nested_braces(tmp_path):
    source = (
        "double complex cexp(double complex z)\n"
        "{\n"
        "  if (z) { /* ignored } */ return z; }\n"
        "  return 0;\n"
        "}\n"
        "double cexp(double z);\n"
    )
    (tmp_path / "main.c").write_text(source)

    recovered = recover_source_entities(["cexp", "external"], ["main.c"], str(tmp_path))

    assert [entry["functionName"] for entry in recovered] == ["cexp"]
    assert recovered[0]["completeFunction"].endswith("}")
    assert find_source_function_line_range(source, "cexp") == (0, 4)


def test_source_recovery_ignores_calls_and_prototypes(tmp_path):
    source = "double cexp(double z);\nvoid run(void) { cexp(1.0); }\n"
    (tmp_path / "main.c").write_text(source)

    assert recover_source_entities(["cexp"], ["main.c"], str(tmp_path)) == []


def test_source_recovery_allows_comments_between_signature_and_body(tmp_path):
    source = (
        "void sobel_filtering( )\n"
        "  /* Input: image1[y][x] */\n"
        "  /* Output: image2[y][x] */\n"
        "{\n"
        "  return;\n"
        "}\n"
    )
    (tmp_path / "sobel.c").write_text(source)

    recovered = recover_source_entities(
        ["sobel_filtering"], ["sobel.c"], str(tmp_path)
    )

    assert [entry["functionName"] for entry in recovered] == ["sobel_filtering"]
    assert recovered[0]["completeFunction"].startswith("void sobel_filtering")



def test_a_declined_function_still_counts_as_attempted():
    """Denominators must be comparable across arms.

    CHAI's per-function selector legitimately declines a function with no
    applicable operator, but budget was still spent on it. Counting it only in
    the arms that reach planning reported CHAI "4/8" against the baseline's
    "8/9" for the same frozen target list -- not comparable quantities.
    """
    source = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "main.py"),
        encoding="utf-8",
    ).read()

    increment = "exp_functions_attempted += 1"
    assert source.count(increment) == 1, "exactly one place may count attempts"

    before_decline = source.index(increment)
    decline_branch = source.index("no applicable operator; ")
    assert before_decline < decline_branch, (
        "attempts must be counted before the per-function selector can skip a "
        "function, or arms report different denominators"
    )
