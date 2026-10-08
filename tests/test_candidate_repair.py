from utils.utils import findFileOfFunc
from utils import validator
from utils.validator import (
    buildFocusedRepairRequest,
    find_fewshot_identifier_leakage,
    normalizeKnobDeclarations,
    repairFunctionCandidate,
    validateFunctionContract,
)


def _plan(code, knobs=None, ranges=None, steps=None, name="target"):
    return {
        "functionName": name,
        "completeFunction": code,
        "knobVariables": knobs or "[]",
        "knobRanges": ranges or "[]",
        "knobStepSize": steps or "[]",
    }


def _source_workspace(tmp_path, monkeypatch, source):
    target = tmp_path / "target"
    target.mkdir()
    (target / "main.c").write_text(source, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(validator, "TARGET_FILES", ["main.c"])


def test_demo_only_identifiers_are_rejected_before_compilation():
    copied = """
    void init_search(const char *pattern) {
      demo_t3_configured_spec = pattern;
      demo_t3_compile_coefficient(pattern, 0);
    }
    """
    adapted = """
    void init_search(const char *pattern) {
      findme = pattern;
      table[0] = strlen(pattern);
    }
    """

    assert find_fewshot_identifier_leakage(copied) == [
        "demo_t3_compile_coefficient", "demo_t3_configured_spec"
    ]
    assert find_fewshot_identifier_leakage(adapted) == []


def test_uninitialized_shadows_of_original_state_are_removed(tmp_path, monkeypatch):
    _source_workspace(
        tmp_path,
        monkeypatch,
        """
        static size_t table[256];
        static size_t len;
        static char *findme;
        void init_search(const char *string) {
          size_t i;
          len = strlen(string);
          for (i=0; i<256; ++i) table[i] = len;
          findme = (char *)string;
        }
        """,
    )
    candidate = _plan(
        """
        void init_search(const char *string) {
          /*Knob Variables Declaration Start*/
          int exact_reuse_enabled = 0;
          /*Knob Variables Declaration End*/
          static size_t table[256];
          static size_t len;
          static char *findme;
          static const char *remembered;
          if (exact_reuse_enabled && remembered == string) return;
          len = strlen(string);
          for (size_t i=0; i<256; ++i) table[i] = len;
          findme = (char *)string;
          remembered = string;
        }
        """,
        knobs="['exact_reuse_enabled']",
        ranges="[{'exact_reuse_enabled': [0, 1]}]",
        steps="[{'exact_reuse_enabled': 'Integer'}]",
        name="init_search",
    )

    actions = repairFunctionCandidate(candidate)

    assert any("shadowing original file-scope state" in action for action in actions)
    assert "static size_t table[256];" not in candidate["completeFunction"]
    assert "static const char *remembered;" in candidate["completeFunction"]
    assert validateFunctionContract(candidate) == ""


def test_find_file_falls_back_when_lsp_omits_local_function(tmp_path, monkeypatch):
    _source_workspace(
        tmp_path,
        monkeypatch,
        "double complex cexp(double complex z) { return z; }\n",
    )

    assert findFileOfFunc("cexp", ["main.c"]) == "main.c"


def test_body_only_candidate_is_wrapped_with_original_signature(tmp_path, monkeypatch):
    _source_workspace(
        tmp_path,
        monkeypatch,
        "long btbl_bitcnt(long value) { return value; }\n",
    )
    candidate = _plan("{ return value >> 1; }", name="btbl_bitcnt")

    actions = repairFunctionCandidate(candidate)

    assert "wrapped body with original function signature" in actions
    assert candidate["completeFunction"].startswith("long btbl_bitcnt(long value)\n{")


def test_helper_definitions_receive_forward_prototypes(tmp_path, monkeypatch):
    _source_workspace(
        tmp_path,
        monkeypatch,
        "double target(double x) { return x; }\n",
    )
    candidate = _plan(
        "double target(double x) { return helper(x); }\n"
        "double helper(double x) { return x * 0.5; }"
    )

    actions = repairFunctionCandidate(candidate)

    assert "inserted 1 helper prototype(s)" in actions
    assert candidate["completeFunction"].startswith("double helper(double x);")


def test_missing_terminal_function_brace_is_appended(tmp_path, monkeypatch):
    _source_workspace(
        tmp_path,
        monkeypatch,
        "void sobel_filtering(void) { return; }\n",
    )
    candidate = _plan(
        "void sobel_filtering(void) {\n"
        "  int downsample_factor = 2;\n"
        "  {\n"
        "    if (downsample_factor) { return; }\n"
        "  }",
        knobs="['downsample_factor']",
        ranges="[{'downsample_factor': [1, 2]}]",
        steps="[{'downsample_factor': 'Integer'}]",
        name="sobel_filtering",
    )

    actions = repairFunctionCandidate(candidate)

    assert "appended 1 missing closing brace(s)" in actions
    assert candidate["completeFunction"].rstrip().endswith("}\n}")


def test_combined_knob_declaration_is_split_to_own_line():
    candidate = _plan(
        "double target(double x) {\n  double y = x, precision_scale = 0.8;\n  return y;\n}",
        knobs="['precision_scale']",
        ranges="[{'precision_scale': [0.1, 1.0]}]",
        steps="[{'precision_scale': 'Real'}]",
    )

    actions = normalizeKnobDeclarations(candidate)

    assert "split combined declaration for knob precision_scale" in actions
    assert "double y = x;" in candidate["completeFunction"]
    assert "double precision_scale = 0.8;" in candidate["completeFunction"]


def test_standalone_knob_assignment_becomes_typed_local():
    candidate = _plan(
        "double target(double x) {\n  precision_scale = 0.8;\n  return x;\n}",
        knobs="['precision_scale']",
        ranges="[{'precision_scale': [0.1, 1.0]}]",
        steps="[{'precision_scale': 'Real'}]",
    )

    actions = normalizeKnobDeclarations(candidate)

    assert "typed standalone assignment for knob precision_scale" in actions
    assert "double precision_scale = 0.8;" in candidate["completeFunction"]


def test_serialized_function_layout_and_pi_alias_are_repaired(tmp_path, monkeypatch):
    _source_workspace(
        tmp_path,
        monkeypatch,
        "#define PI 3.14159\ndouble target(double x) { return x; }\n",
    )
    candidate = _plan(
        r"double target(double x) {\n\tdouble y = 2 * M_PI * x;\n\treturn y;\n}"
    )

    actions = repairFunctionCandidate(candidate)

    assert "decoded literal newline/tab layout escapes" in actions
    assert "replaced unavailable M_PI with source-defined PI" in actions
    assert "\\n" not in candidate["completeFunction"]
    assert "2 * PI * x" in candidate["completeFunction"]


def test_duplicate_knob_declaration_is_removed(tmp_path, monkeypatch):
    _source_workspace(tmp_path, monkeypatch, "double target(double x) { return x; }\n")
    candidate = _plan(
        "double target(double x) {\n"
        "  int stride = 1;\n"
        "  int stride = 2;\n"
        "  return x;\n}",
        knobs="['stride']",
        ranges="[{'stride': [1, 4]}]",
        steps="[{'stride': 'Integer'}]",
    )

    actions = repairFunctionCandidate(candidate)

    assert "removed duplicate declaration for knob stride" in actions
    assert candidate["completeFunction"].count("int stride") == 1


def test_real_knob_declared_as_integer_is_changed_to_double():
    candidate = _plan(
        "double target(double x) {\n  int precision_scale = 1;\n  return x;\n}",
        knobs="['precision_scale']",
        ranges="[{'precision_scale': [0.5, 1.5]}]",
        steps="[{'precision_scale': 'Real'}]",
    )

    actions = normalizeKnobDeclarations(candidate)

    assert "changed Real knob precision_scale declaration to double" in actions
    assert "double precision_scale = 1;" in candidate["completeFunction"]


def test_focused_repair_contains_original_candidate_error_and_failed_operator(
    tmp_path, monkeypatch
):
    _source_workspace(
        tmp_path,
        monkeypatch,
        "double target(double x) { return x; }\n",
    )
    candidate = _plan("double target(double x) { return missing(x); }")

    prompt = buildFocusedRepairRequest(candidate, "undefined reference", {19})

    assert "double target(double x) { return x; }" in prompt
    assert "return missing(x)" in prompt
    assert "undefined reference" in prompt
    assert "T19" in prompt
