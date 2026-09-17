import ast
import subprocess
import os
import math
import re

from config.globals import TARGET_FILES
from utils.utils import findFileOfFunc, writeFunctionsToJson, Dprint
from lib.lsp import lsp_patcher
from utils.knob_metadata import KnobMetadataError, normalize_knob_metadata
from utils.source_functions import (
    find_source_function_signature,
    find_source_function_span,
)


# Deliberately distinctive identifiers in out-of-suite few-shot examples.
# Their presence in generated benchmark code is direct evidence that the model
# copied a demonstration instead of adapting the transformation to the source.
_FEWSHOT_SENTINEL_IDENTIFIERS = {
    "DemoT3FilterSpec",
    "demo_t3_coefficients",
    "demo_t3_configured_spec",
    "demo_t3_compile_coefficient",
    "demo_t3_configure_filter",
}


def find_fewshot_identifier_leakage(code):
    """Return sorted demonstration-only identifiers copied into *code*."""
    tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", code or ""))
    return sorted(tokens & _FEWSHOT_SENTINEL_IDENTIFIERS)


def _knob_declaration_pattern(knob):
    return re.compile(
        rf"^(?P<prefix>\s*(?:(?:const|volatile|static|register|unsigned|signed|long|short)\s+)*"
        rf"[A-Za-z_]\w*(?:\s*\*+)?\s+{re.escape(knob)}\s*=\s*)"
        rf"(?P<value>[^;]*)(?P<suffix>;.*)$"
    )


def updateKnobValue(functionDict, newValue, knob):
    """
    Update the value of a knob in the function code.

    Args:
        functionDict (dict): A dictionary containing the function code.
        newValue (any): The new value to assign to the knob.
        knob (str): The name of the knob to update.

    Returns:
        None
    """

    if not isinstance(knob, str) or not re.fullmatch(r"[A-Za-z_]\w*", knob):
        raise ValueError(f"Invalid knob name: {knob!r}")

    function_lines = functionDict["completeFunction"].split("\n")

    # A knob must be a local declaration.  The previous implementation matched
    # the first occurrence of its name, including a function parameter.  For
    # example, a parameter named ``shift_bits`` was transformed into ``{= 2;``.
    # Match only a typed declaration with an initializer and never an argument
    # list or a normal assignment statement.
    declaration = _knob_declaration_pattern(knob)

    for line_number, line in enumerate(function_lines):
        match = declaration.match(line)
        if match:
            function_lines[line_number] = f"{match.group('prefix').rstrip()} {newValue}{match.group('suffix')}"
            functionDict["completeFunction"] = "\n".join(function_lines)
            return

    raise ValueError(
        f"Knob '{knob}' has no initialized local declaration. Knobs must be "
        "declared inside the function body, not added to the function signature."
    )


def _function_signature_text(code, function_name):
    """Return the original C signature text, or ``None`` if absent."""
    return find_source_function_signature(code, function_name)


def _canonical_signature(signature):
    if not signature:
        return None
    signature = re.sub(r"/\*.*?\*/|//[^\n]*", "", signature, flags=re.DOTALL)
    return re.sub(r"\s+", "", signature)


def _function_signature(code, function_name):
    """Return a whitespace-normalized C signature, or ``None`` if absent."""
    return _canonical_signature(_function_signature_text(code, function_name))


def _read_original_function(function_name):
    source_file = findFileOfFunc(function_name, TARGET_FILES)
    if not source_file:
        return None, None, None
    path = os.path.join("target", source_file)
    try:
        with open(path, encoding="utf-8") as source:
            source_code = source.read()
    except OSError:
        return source_file, None, None
    span = find_source_function_span(source_code, function_name)
    original_function = source_code[span[0]:span[1]].strip() if span else None
    return source_file, source_code, original_function


def _strip_markdown_fence(code):
    match = re.fullmatch(r"\s*```(?:c|cpp)?\s*(.*?)\s*```\s*", code, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else code.strip()


def _c_brace_balance(code):
    """Count structural braces while ignoring comments and literals."""
    balance = 0
    state = "code"
    index = 0
    while index < len(code):
        char = code[index]
        nxt = code[index + 1] if index + 1 < len(code) else ""
        if state == "code":
            if char == '"':
                state = "string"
            elif char == "'":
                state = "char"
            elif char == "/" and nxt == "/":
                state = "line_comment"
                index += 1
            elif char == "/" and nxt == "*":
                state = "block_comment"
                index += 1
            elif char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
        elif state in {"string", "char"}:
            if char == "\\":
                index += 1
            elif (state == "string" and char == '"') or (
                state == "char" and char == "'"
            ):
                state = "code"
        elif state == "line_comment" and char == "\n":
            state = "code"
        elif state == "block_comment" and char == "*" and nxt == "/":
            state = "code"
            index += 1
        index += 1
    return balance


def _helper_prototypes(code, target_function):
    """Return prototypes for helper definitions emitted after the target."""
    definition = re.compile(
        r"(?m)^[ \t]*(?P<signature>(?:[A-Za-z_]\w*[ \t*]+)+"
        r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\))"
        r"(?:(?:[ \t\r\n]+)|(?:/\*.*?\*/)|(?://[^\n]*(?:\n|$)))*\{",
        re.DOTALL | re.MULTILINE,
    )
    target_match = next(
        (match for match in definition.finditer(code) if match.group("name") == target_function),
        None,
    )
    if not target_match:
        return []
    prototypes = []
    for match in definition.finditer(code):
        name = match.group("name")
        if name == target_function or match.start() < target_match.start():
            continue
        signature = match.group("signature").strip()
        prototype = f"{signature};"
        if prototype not in code[:target_match.start()] and prototype not in prototypes:
            prototypes.append(prototype)
    return prototypes


def normalizeKnobDeclarations(functionDict):
    """Normalize common LLM knob declarations into one local per line."""
    knob_variables, _, knob_steps = normalize_knob_metadata(functionDict, mutate=True)
    lines = functionDict.get("completeFunction", "").splitlines()
    actions = []
    declaration = re.compile(
        r"^(?P<indent>\s*)(?P<type>(?:(?:const|volatile|static|register)\s+)*"
        r"(?:(?:unsigned|signed)\s+)?(?:(?:long|short)\s+)?"
        r"(?:char|int|float|double|size_t|bool|_Bool|[A-Za-z_]\w*_t|[A-Z]\w*)"
        r"(?:\s*\*+)?)\s+(?P<decls>[^;]+);(?P<tail>.*)$"
    )
    integer_declaration = re.compile(
        r"^(?P<indent>\s*)(?P<qual>(?:(?:const|volatile|static|register)\s+)*)"
        r"(?P<type>(?:(?:unsigned|signed)\s+)?(?:(?:long|short)\s+)?"
        r"(?:char|int|size_t|bool|_Bool|[A-Za-z_]\w*_t))"
        r"(?P<rest>\s+)"
    )
    for index, knob in enumerate(knob_variables):
        existing_line = next(
            (number for number, line in enumerate(lines)
             if _knob_declaration_pattern(knob).match(line)),
            None,
        )
        if existing_line is not None:
            step_type = knob_steps[index][knob]
            if step_type == "Real":
                match = integer_declaration.match(lines[existing_line])
                if match:
                    lines[existing_line] = (
                        f"{match.group('indent')}{match.group('qual')}double"
                        f"{lines[existing_line][match.end('type'):]}"
                    )
                    actions.append(f"changed Real knob {knob} declaration to double")
            continue
        normalized = False
        for line_number, line in enumerate(lines):
            match = declaration.match(line)
            if not match or "," not in match.group("decls"):
                continue
            parts = [part.strip() for part in match.group("decls").split(",")]
            knob_parts = [
                part for part in parts
                if re.match(rf"^\**\s*{re.escape(knob)}\s*=", part)
            ]
            if not knob_parts:
                continue
            remaining = [part for part in parts if part not in knob_parts]
            replacement = []
            if remaining:
                replacement.append(
                    f"{match.group('indent')}{match.group('type')} "
                    f"{', '.join(remaining)};"
                )
            replacement.append(
                f"{match.group('indent')}{match.group('type')} {knob_parts[0]};"
                f"{match.group('tail')}"
            )
            lines[line_number:line_number + 1] = replacement
            actions.append(f"split combined declaration for knob {knob}")
            normalized = True
            break
        if normalized:
            continue
        step_type = knob_steps[index][knob]
        local_type = "int" if step_type == "Integer" else "double"
        assignment = re.compile(
            rf"^(?P<indent>\s*){re.escape(knob)}\s*=\s*(?P<value>[^;]+);(?P<tail>.*)$"
        )
        for line_number, line in enumerate(lines):
            match = assignment.match(line)
            if match:
                lines[line_number] = (
                    f"{match.group('indent')}{local_type} {knob} = "
                    f"{match.group('value')};{match.group('tail')}"
                )
                actions.append(f"typed standalone assignment for knob {knob}")
                break
    functionDict["completeFunction"] = "\n".join(lines)
    return actions


def _decode_literal_layout_escapes(code):
    """Decode a model response that serialized an entire C function on one line."""
    if "\\n" not in code or code.count("\n") > 2:
        return code, False
    return code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t"), True


def _dedupe_knob_declarations(function_dict):
    lines = function_dict.get("completeFunction", "").splitlines()
    try:
        knobs, _, _ = normalize_knob_metadata(function_dict)
    except KnobMetadataError:
        return []
    actions = []
    for knob in knobs:
        seen = False
        kept = []
        for line in lines:
            if _knob_declaration_pattern(knob).match(line):
                if seen:
                    actions.append(f"removed duplicate declaration for knob {knob}")
                    continue
                seen = True
            kept.append(line)
        lines = kept
    function_dict["completeFunction"] = "\n".join(lines)
    return actions


def _remove_uninitialized_shadow_declarations(code, names):
    """Remove safe local redeclarations of state the original function writes.

    Models often copy file-scope state into the replacement function as
    uninitialized ``static`` locals. That changes which object consumers read,
    so the ABI/state contract correctly rejects it. When the declaration has
    no initializer, removing it is a semantics-preserving repair: references
    resolve to the original file-scope object again. Initialized declarations
    are left untouched because deleting their initializer could hide work.
    """
    if not names:
        return code, []
    removed = []
    kept = []
    for line in code.splitlines():
        matched = None
        for name in names:
            declaration = re.match(
                rf"^\s*(?:(?:const|volatile|static|register|unsigned|signed|long|short)\s+)*"
                rf"[A-Za-z_]\w*(?:\s*\*+\s*|\s+)"
                rf"{re.escape(name)}\s*(?:\[[^]]*\])?\s*;\s*"
                rf"(?:/\*.*\*/\s*|//.*)?$",
                line,
            )
            if declaration:
                matched = name
                break
        if matched:
            removed.append(matched)
        else:
            kept.append(line)
    return "\n".join(kept), sorted(set(removed))


#: Knob-name tokens whose identity setting is "off" / zero.
_ZERO_IDENTITY_TOKENS = (
    "enabled", "enable_", "_enable", "disable", "shift_bits", "shift_amount",
    "quantize", "skip", "perforate", "decimat", "downsample", "prune",
    "precision_mode", "approx_mode", "early_exit", "lossy",
)

#: Knob-name tokens whose identity setting is the upper bound (full budget).
_UPPER_IDENTITY_TOKENS = (
    "budget", "iterations", "max_iter", "limit", "depth", "window",
    "resolution", "samples", "taps", "terms",
)

#: Knob-name tokens whose identity setting is 1 (unit factor / unit stride).
_UNIT_IDENTITY_TOKENS = (
    "factor", "scale", "stride", "sample_fraction", "fraction",
    "segment_len", "step_size", "fidelity", "ratio", "divisor", "every_nth",
)

#: Knob-name tokens expressed as a percentage, where 100 is the identity.
_PERCENT_IDENTITY_TOKENS = ("_pct", "percent", "percentage", "_rate")


def _neutral_knob_value(name, lower, upper):
    """Return a conservative no-approximation value for common knob names.

    This is a name-based heuristic and it stays one. It previously returned
    ``None`` for very common names -- ``threshold``, ``precision_mode``,
    ``iteration_pct``, ``nibble_stride`` -- which silently left whatever
    aggressive default the model invented as the "safe baseline". The tables
    below cover the naming conventions the prompts and operator cards actually
    teach, so the heuristic fires where the pipeline creates knobs. Anything it
    still cannot classify returns ``None`` and is caught downstream by the
    measured identity gate in ``runBayesOpt``, which is the real safety net.
    """
    lowered = name.lower()

    if any(token in lowered for token in _PERCENT_IDENTITY_TOKENS):
        # A percentage knob keeps everything at 100 when 100 is reachable.
        return 100 if lower <= 100 <= upper else upper

    if any(token in lowered for token in _ZERO_IDENTITY_TOKENS):
        return 0

    if lowered.startswith("max_") or any(
        token in lowered for token in _UPPER_IDENTITY_TOKENS
    ):
        return upper

    if any(token in lowered for token in _UNIT_IDENTITY_TOKENS):
        return 1

    if "threshold" in lowered or "tolerance" in lowered:
        # A convergence threshold approximates less as it gets tighter, so the
        # identity end of the range is the smallest value.
        return lower

    return None


def normalizeSafeKnobDefaults(function_dict):
    """Seed BO with a conservative identity point when the knob names allow it.

    ``runBayesOpt`` intentionally evaluates declared defaults first, but an LLM
    often emits an already-aggressive default (for example a quantization
    divisor of 10 with a range of 2..20). This repairs both the range and local
    initializer so the first point is a meaningful reference configuration.
    """
    variables, ranges, steps = normalize_knob_metadata(function_dict, mutate=True)
    actions = []
    for index, name in enumerate(variables):
        lower, upper = ranges[index][name]
        original_bounds = [lower, upper]
        neutral = _neutral_knob_value(name, lower, upper)
        if neutral is None:
            continue
        if neutral < lower:
            lower = neutral
        if neutral > upper:
            upper = neutral
        step_type = steps[index][name]
        neutral = int(neutral) if step_type == "Integer" else float(neutral)
        ranges[index][name] = [lower, upper]
        if ranges[index][name] != original_bounds:
            actions.append(
                f"expanded knob {name} range to include safe baseline {neutral}"
            )
        declaration = next(
            (
                _knob_declaration_pattern(name).match(line)
                for line in function_dict.get("completeFunction", "").splitlines()
                if _knob_declaration_pattern(name).match(line)
            ),
            None,
        )
        if declaration and str(declaration.group("value")).strip() != str(neutral):
            updateKnobValue(function_dict, neutral, name)
            actions.append(f"set safe baseline default for knob {name} to {neutral}")
    function_dict["knobRanges"] = str(ranges)
    function_dict["knobStepSize"] = str(steps)
    function_dict["knobVariables"] = str(variables)
    return actions


def _strip_c_comments(code):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", code, flags=re.DOTALL)


def _shadowed_written_state(original_function, approximate_function):
    """Find original non-local state that the candidate accidentally shadows."""
    declaration = re.compile(
        r"(?m)^\s*(?:(?:const|volatile|static|register|unsigned|signed|long|short)\s+)*"
        r"[A-Za-z_]\w*(?:(?:\s*\*+\s*)|\s+)([A-Za-z_]\w*)\s*"
        r"(?:\[[^;\n]*\]|=[^;\n]*|;)"
    )
    assignment = re.compile(
        r"\b([A-Za-z_]\w*)\s*(?:\[[^\]\n]*\]\s*)?"
        r"(?:\+\+|--|[+\-*/%]?=(?!=))"
    )
    for_declaration = re.compile(
        r"\bfor\s*\(\s*(?:(?:const|volatile|unsigned|signed|long|short)\s+)*"
        r"[A-Za-z_]\w*(?:\s*\*+\s*|\s+)([A-Za-z_]\w*)\s*="
    )
    original = _strip_c_comments(original_function)
    approximate = _strip_c_comments(approximate_function)
    original_locals = set(declaration.findall(original)) | set(
        for_declaration.findall(original)
    )
    approximate_locals = set(declaration.findall(approximate)) | set(
        for_declaration.findall(approximate)
    )
    written_identifiers = set(assignment.findall(original))
    original_external_state = written_identifiers - original_locals
    return original_external_state & approximate_locals


def _unsafe_loop_progress_knobs(code, variables, ranges, steps):
    """Return tunable increments that can truncate to zero and hang a loop."""
    unsafe = []
    for index, name in enumerate(variables):
        lower, _ = ranges[index][name]
        step_type = steps[index][name]
        increment_use = re.search(
            rf"\bfor\s*\([^;]*;[^;]*;[^)]*(?:\+=|-=)\s*{re.escape(name)}\b",
            code,
            re.DOTALL,
        )
        if increment_use and (step_type == "Real" or lower <= 0):
            unsafe.append(name)
    return unsafe


def _unsafe_loop_bound_knobs(code, variables, ranges):
    """Return multiplier knobs that can expand an original loop bound."""
    unsafe = []
    for index, name in enumerate(variables):
        _, upper = ranges[index][name]
        condition_uses_multiplier = re.search(
            rf"\bfor\s*\([^;]*;[^;]*(?:"
            rf"(?:[A-Za-z_]\w*|\d+)\s*\*\s*{re.escape(name)}|"
            rf"{re.escape(name)}\s*\*\s*(?:[A-Za-z_]\w*|\d+)"
            rf")[^;]*;",
            code,
            re.DOTALL,
        )
        if condition_uses_multiplier and upper > 1:
            unsafe.append(name)
    return unsafe


def repairFunctionCandidate(functionDict):
    """Apply conservative deterministic repairs before asking the LLM again."""
    function_name = functionDict.get("functionName", "")
    _, source_code, original_function = _read_original_function(function_name)
    if not source_code or not original_function:
        return []
    original_signature = _function_signature_text(source_code, function_name)
    code = _strip_markdown_fence(functionDict.get("completeFunction", ""))
    actions = []

    code, decoded_layout = _decode_literal_layout_escapes(code)
    if decoded_layout:
        actions.append("decoded literal newline/tab layout escapes")

    # The iclib FFT sources define PI, not the non-standard M_PI extension.
    # Apply this only when the source proves the alias is available and M_PI
    # is not, avoiding a platform-wide or semantics-changing substitution.
    has_pi = re.search(r"(?m)^\s*#\s*define\s+PI\b", source_code)
    has_m_pi = re.search(r"(?m)^\s*#\s*define\s+M_PI\b", source_code)
    if has_pi and not has_m_pi and re.search(r"\bM_PI\b", code):
        code = re.sub(r"\bM_PI\b", "PI", code)
        actions.append("replaced unavailable M_PI with source-defined PI")

    approximate_signature = _function_signature_text(code, function_name)
    if approximate_signature and _canonical_signature(approximate_signature) != _canonical_signature(original_signature):
        code = code.replace(approximate_signature, original_signature, 1)
        actions.append("restored exact original function signature")
    elif not approximate_signature:
        stripped = code.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            code = f"{original_signature}\n{stripped}"
            actions.append("wrapped body with original function signature")
        elif function_name not in stripped and ";" in stripped and not re.search(
            r"\b(?:here|code|function|approximation)\b", stripped, re.IGNORECASE
        ):
            code = f"{original_signature}\n{{\n{stripped}\n}}"
            actions.append("wrapped body-only candidate with original function")

    shadowed = _shadowed_written_state(original_function, code)
    code, removed_shadows = _remove_uninitialized_shadow_declarations(
        code, shadowed
    )
    if removed_shadows:
        actions.append(
            "removed local redeclaration(s) shadowing original file-scope state: "
            + ", ".join(removed_shadows)
        )

    prototypes = _helper_prototypes(code, function_name)
    if prototypes:
        code = "\n".join(prototypes) + "\n" + code
        actions.append(f"inserted {len(prototypes)} helper prototype(s)")
    brace_balance = _c_brace_balance(code)
    if 0 < brace_balance <= 2:
        code = code.rstrip() + "\n" + "\n".join("}" for _ in range(brace_balance))
        actions.append(f"appended {brace_balance} missing closing brace(s)")
    functionDict["completeFunction"] = code
    actions.extend(normalizeKnobDeclarations(functionDict))
    actions.extend(_dedupe_knob_declarations(functionDict))
    actions.extend(normalizeSafeKnobDefaults(functionDict))
    return actions


def buildFocusedRepairRequest(functionDict, error, failed_techniques=None):
    """Build a small, compiler-grounded repair instruction for the next attempt."""
    function_name = functionDict.get("functionName", "")
    _, source_code, original_function = _read_original_function(function_name)
    required_signature = _function_signature_text(source_code or "", function_name)
    failed_text = ""
    if failed_techniques:
        failed_text = (
            "\nThese operator IDs were present in the rejected candidate: "
            + ", ".join(f"T{identifier:02d}" for identifier in sorted(failed_techniques))
            + ". Repair the same operator when its implementation is appropriate; "
              "choose a different feasible operator only when it cannot be implemented safely."
        )
    return (
        "STRICT REPAIR MODE. Return one complete compilable replacement function.\n"
        f"Target: {function_name}\n"
        f"Required signature: {required_signature or 'unavailable'}\n"
        "Keep knobs as initialized local variables, one declaration per line. "
        "Do not use ellipses, prose, undeclared helpers, invented lookup tables, "
        "or body-only patches. If the compiler reports an undeclared identifier, "
        "remove that technique or define all required local data. Never apply >> "
        "or << directly to float/double values. Begin with APPLIED_TECHNIQUES and "
        "the exact catalog IDs used."
        f"{failed_text}\n\n"
        f"ORIGINAL FUNCTION:\n{original_function or 'unavailable'}\n\n"
        f"REJECTED CANDIDATE:\n{functionDict.get('completeFunction', '')}\n\n"
        f"VALIDATOR/COMPILER FAILURE:\n{str(error)[-6000:]}"
    )


def validateFunctionContract(functionDict):
    """Ensure an LLM approximation keeps the target function's public contract."""
    function_name = functionDict.get("functionName", "")
    approximate_code = functionDict.get("completeFunction", "")
    source_file = findFileOfFunc(function_name, TARGET_FILES)
    if not source_file:
        return f"Could not locate the original source for function '{function_name}'."

    try:
        with open(os.path.join("target", source_file), encoding="utf-8") as source:
            original_code = source.read()
    except OSError as error:
        return f"Could not read original source for '{function_name}': {error}"

    original_signature_text = _function_signature_text(original_code, function_name)
    approximate_signature_text = _function_signature_text(approximate_code, function_name)
    original_signature = _canonical_signature(original_signature_text)
    approximate_signature = _canonical_signature(approximate_signature_text)
    if not original_signature or not approximate_signature:
        return (
            f"Contract validation failed for '{function_name}': the complete C function "
            "definition and its opening brace must be present."
        )
    if original_signature != approximate_signature:
        return (
            f"Contract validation failed for '{function_name}': preserve the original "
            "function signature exactly (return type, qualifiers, parameter types, "
            "parameter order, and parameter count). Do not add knob parameters. "
            f"Required signature: {original_signature_text}"
        )
    if re.search(r"\.\.\.|rest\s+of\s+the\s+.*(?:remain|same)", approximate_code, re.IGNORECASE):
        return (
            f"Contract validation failed for '{function_name}': return the complete "
            "function body. Ellipses and 'rest remains the same' placeholders are not C code."
        )
    original_span = find_source_function_span(original_code, function_name)
    original_function = (
        original_code[original_span[0]:original_span[1]] if original_span else ""
    )
    shadowed_state = _shadowed_written_state(original_function, approximate_code)
    if shadowed_state:
        return (
            f"Contract validation failed for '{function_name}': the candidate declares "
            "local variables that shadow original externally visible state: "
            + ", ".join(sorted(shadowed_state))
            + ". Update the original state instead of redeclaring it."
        )
    try:
        knob_variables, knob_ranges, knob_steps = normalize_knob_metadata(
            functionDict, mutate=True
        )
    except KnobMetadataError as error:
        return f"Knob metadata validation failed for '{function_name}': {error}"
    for knob in knob_variables:
        if not any(
            _knob_declaration_pattern(knob).match(line)
            for line in approximate_code.splitlines()
        ):
            return (
                f"Knob declaration validation failed for '{function_name}': '{knob}' "
                "must be an initialized local variable on its own declaration line."
            )
    unsafe_progress = _unsafe_loop_progress_knobs(
        approximate_code, knob_variables, knob_ranges, knob_steps
    )
    if unsafe_progress:
        return (
            f"Loop-progress validation failed for '{function_name}': "
            + ", ".join(sorted(unsafe_progress))
            + " is used as an integer-loop increment but can truncate to zero. "
              "Use an Integer knob with a strictly positive lower bound, or use "
              "a Real factor only in the loop bound."
        )
    unsafe_bounds = _unsafe_loop_bound_knobs(
        approximate_code, knob_variables, knob_ranges
    )
    if unsafe_bounds:
        return (
            f"Loop-bound validation failed for '{function_name}': "
            + ", ".join(sorted(unsafe_bounds))
            + " multiplies an existing loop bound but its range exceeds 1, "
              "which can expand the iteration domain and access data out of bounds. "
              "For perforation/truncation factors, use a Real range within (0, 1]."
        )
    return ""


def compileFunction(path):
    """
    Compile the codebase.

    Args:
        path (str): The path to the codebase.

    Returns:
        int: 0 if the compilation is successful, 1 if it fails.
    """
    try:
        with open(os.path.join(path, "compiler_log.txt"), "w") as output_file:
            subprocess.run(
                "make main",
                shell=True,
                check=True,
                cwd=path,
                stdout=output_file,
                stderr=subprocess.STDOUT,
            )

        # Dprint("Success\n")
        return 0  # Successful compilation
    except Exception as e:
        Dprint(e)
        with open(os.path.join(path, "compiler_log.txt"), "r") as output_file:
            response = output_file.read()

        Dprint(response)
        return 1  # Compilation failed


def executeFunction(application, path):
    """
    Executes a command based on the given application name.

    Args:
        application (str): The name of the application.

    Returns:
        int: 0 if the command is executed successfully, 1 otherwise.
    """
    if application == "susan":
        command = f"./main 2092.pgm output.pgm -e"

        # Execute the command, and return 1 if it fails or there is a runtime error
        try:
            subprocess.run(
                command, shell=True, check=True, cwd=path, stdout=subprocess.DEVNULL
            )
            return 0
        except Exception as e:
            Dprint(e)
            return 1
    else:
        command = f"./main"

        # Execute the command, and return 1 if it fails or there is a runtime error
        try:
            subprocess.run(
                command, shell=True, check=True, cwd=path, stdout=subprocess.DEVNULL
            )
            return 0
        except Exception as e:
            Dprint(e)
            return 1


def validateFunction(functionDict):
    """
    Validates a function by iteratively updating knob values and checking if the function is safe.

    Args:
        functionDict (dict): A dictionary containing information about the function to be validated.

    Returns:
        None
    """

    # Extract knob variables, ranges, and step sizes from functionDict
    raw_vars = functionDict["knobVariables"]
    knob_variables = ast.literal_eval(raw_vars) if isinstance(raw_vars, str) else raw_vars
    raw_ranges = functionDict["knobRanges"]
    knob_ranges = ast.literal_eval(raw_ranges) if isinstance(raw_ranges, str) else raw_ranges
    raw_steps = functionDict["knobStepSize"]
    knob_step_sizes = ast.literal_eval(raw_steps) if isinstance(raw_steps, str) else raw_steps

    functionName = functionDict["functionName"]

    apx_codebase = f"compilation_testing/ApxFiles_{functionName}"
    apx_file = findFileOfFunc(functionName, TARGET_FILES)  # HARD CODED
    apx_file = os.path.join(apx_codebase, apx_file)

    json_file_path = f"apx_{functionName}.json"

    safeFlag = True

    for index, knob in enumerate(knob_variables):
        lower_bound, upper_bound = knob_ranges[index][knob]
        knob_step_size = knob_step_sizes[index][knob]

        # Debug
        # Dprint(f"Working on Knob: {knob}")
        # Dprint(f"Lower Bound: {lower_bound}")
        # Dprint(f"Upper Bound: {upper_bound}")
        # Dprint(f"Step Size: {knob_step_size}")

        # Update the knob value starting from upper bound to lower bound in a binary search fashion
        if knob_step_size == "Integer":
            current_value = math.ceil((upper_bound + lower_bound) / 2)
        elif knob_step_size == "Real":
            current_value = (upper_bound + lower_bound) / 2

        while current_value >= lower_bound:

            # Dprint(f"Updating the knob value to: {current_value}")

            # Update knob value and write to JSON
            updateKnobValue(functionDict, current_value, knob)
            wrapper_dict = {functionName: functionDict}
            writeFunctionsToJson(wrapper_dict, "approximated_functions/apx")

            # Patch the updated function code
            lsp_patcher(apx_codebase,json_file_path)

            # Compile the codebase
            compileFunction(apx_codebase)

            # Execute the file (add for other apps later)
            if "susan" in apx_file:
                successFlag = executeFunction("susan", apx_codebase)
            else:
                successFlag = executeFunction("other_App", apx_codebase)

            if successFlag == 0:
                if knob_step_size == "Integer":
                    current_value = math.ceil((current_value + upper_bound) / 2)
                elif knob_step_size == "Real":
                    current_value = (current_value + upper_bound) / 2
            else:
                upper_bound = current_value
                if knob_step_size == "Integer":
                    current_value = math.floor((current_value + lower_bound) / 2)
                elif knob_step_size == "Real":
                    current_value = (current_value + lower_bound) / 2

            # If the current value is equal to the upper bound, break the loop. The function is safe.
            if current_value >= upper_bound:
                break

            # If the current value is equal to the lower bound, break the loop. The function is unsafe.
            if current_value <= lower_bound:
                # Delete the json from the approximated functions
                safeFlag = False
                Dprint("rasied")
                
                quit()
                os.remove(json_file_path)
                break

        if not safeFlag:
            break

    Dprint(f"Successfully Validated {functionName}")


def validateEarlyExit(code_string, knob_names, knob_ranges):
    """
    Validates that early-exit approximation code follows safety constraints.
    
    Args:
        code_string (str): The approximated code with early-exit
        knob_names (list): List of knob variable names (e.g., ['convergence_threshold_knob'])
        knob_ranges (list): List of [min, max] ranges for each knob
    
    Returns:
        bool: True if code is safe for early-exit, False otherwise
    """
    import re
    
    # Rule 1: Check that convergence metric is computed BEFORE threshold check
    # Pattern: Look for convergence/metric computation (e.g., change = fabs(...))
    metric_patterns = [
        r'(change|error|distance|diff|metric)\s*=',  # Metric assignment
        r'fabs\s*\(',  # Absolute value computation
        r'sqrt\s*\(',  # Square root (often used in distance metrics)
    ]
    
    metric_found = any(re.search(pattern, code_string, re.IGNORECASE) for pattern in metric_patterns)
    if not metric_found:
        Dprint("Warning: No convergence metric computation found in early-exit code")
        return False
    
    # Rule 2: Check for safe exit patterns (break or return)
    safe_exit_patterns = [r'break\s*;', r'return\s*;', r'return\s+[\w\d_]+\s*;']
    safe_exit_found = any(re.search(pattern, code_string) for pattern in safe_exit_patterns)
    if not safe_exit_found:
        Dprint("Error: No safe exit statement (break or return) found in early-exit code")
        return False
    
    # Rule 3: Check for unsafe patterns (while True, infinite loops, hardware-specific code)
    unsafe_patterns = [
        r'while\s*\(\s*1\s*\)',  # while(1) infinite loop
        r'while\s*\(\s*true\s*\)',  # while(true) 
        r'for\s*\(\s*;\s*;\s*\)',  # for(;;) infinite loop
        r'__asm__|asm\s*\{',  # Inline assembly
        r'volatile\s+',  # Volatile (hardware-specific)
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, code_string, re.IGNORECASE):
            Dprint(f"Error: Unsafe pattern found in early-exit code: {pattern}")
            return False
    
    # Rule 4: Validate knob ranges (threshold should be in reasonable bounds)
    for knob_name, knob_range in zip(knob_names, knob_ranges):
        if len(knob_range) != 2:
            Dprint(f"Error: Invalid knob range for {knob_name}: {knob_range}")
            return False
        
        min_val, max_val = knob_range
        if min_val >= max_val:
            Dprint(f"Error: Knob range invalid for {knob_name}: min={min_val}, max={max_val}")
            return False
        
        # Check for pathological ranges (e.g., threshold causing division by zero)
        if 'threshold' in knob_name.lower() and min_val == max_val:
            Dprint(f"Warning: Zero-range threshold knob {knob_name} may be ineffective")
            return False
    
    Dprint(f"Early-exit validation passed for knobs: {knob_names}")
    return True
