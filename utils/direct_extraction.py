"""Build the executable plan entry directly from the generation response.

Why this exists
---------------
Generation used to be two model calls: write the C, then *reproduce that same C
inside a JSON object*. The second call wasted context and tokens, could
truncate or silently alter already-correct code, needed five retries plus
heuristic parsing, carried stringified Python lists inside JSON, and preserved
a misspelled ``approxmated_code`` key.

None of that is necessary once the plan is structured. The approved plan
already declares each knob's name, type, range and identity value, and the C
function is right there in the generation response. Extract the code, take the
knob schema from the contract the model already agreed to, and skip the round
trip. ``convertJson`` remains as the fallback for responses this cannot parse
and for the operator-only baseline, which has no structured plan.
"""

import re

_FENCE = re.compile(r"```(?:c|cpp|C)?\s*\n(.*?)```", re.DOTALL)


def extract_c_function(text, function_name):
    """Return the complete C definition of *function_name* from *text*.

    Returns ``""`` when no balanced definition can be found.
    """
    if not text:
        return ""

    candidates = [block for block in _FENCE.findall(text)]
    candidates.append(text)

    for block in candidates:
        definition = _extract_balanced_definition(block, function_name)
        if definition:
            return definition
    return ""


def _extract_balanced_definition(code, function_name):
    """Find ``function_name``'s definition and return it with balanced braces."""
    # Allow comments and K&R-style declarations between the parameter list and
    # the opening brace: `void f( ) /* doc */ {` is ordinary C and the 32B model
    # writes it. Requiring `)\s*{` dropped those functions entirely.
    gap = r"(?:\s|/\*.*?\*/|//[^\n]*\n|[A-Za-z_][\w \t*\[\]]*;)*"
    for match in re.finditer(
        rf"(^|\n)[^\n;]*?\b{re.escape(function_name)}\s*\([^;{{]*\){gap}\{{",
        code,
        re.DOTALL,
    ):
        start = match.start(0)
        if code[start] == "\n":
            start += 1
        brace_index = code.index("{", match.start(0))
        depth = 0
        for index in range(brace_index, len(code)):
            if code[index] == "{":
                depth += 1
            elif code[index] == "}":
                depth -= 1
                if depth == 0:
                    return code[start:index + 1].strip()
    return ""


def knob_metadata_from_plan(plan):
    """Render an approved plan's knob schema in the executable plan format.

    Returns ``(variables, ranges, step_sizes)`` as the stringified structures
    the rest of the pipeline expects, or ``None`` when the plan declares no
    knobs.
    """
    if plan is None or plan.declined:
        return None

    variables = []
    ranges = []
    steps = []
    seen = set()
    for record in plan.records:
        for knob in record.knobs:
            name = knob["name"]
            if name in seen:
                continue
            seen.add(name)
            variables.append(name)
            ranges.append({name: list(knob["range"])})
            steps.append({name: knob["type"]})

    if not variables:
        return None
    return str(variables), str(ranges), str(steps)


_DECLARATION = r"(^[^\S\n]*(?:[A-Za-z_]\w*[^\S\n]+)*{name}[^\S\n]*=[^\S\n]*)([^;]+)(;)"


def align_knob_defaults_with_plan(code, plan):
    """Set each knob's initializer to the identity value the plan declared.

    The plan states an IDENTITY value inside each knob's range, and the
    generation prompt requires the local initializer to be that value. A model
    frequently declares one and writes the other. Left alone, the identity gate
    in ``runBayesOpt`` correctly rejects the plan -- but it costs a full
    build-and-simulate cycle to discover a discrepancy that is visible
    statically. Align it here and report the change.

    Returns ``(code, actions)``.
    """
    actions = []
    for record in plan.records:
        for knob in record.knobs:
            name = knob["name"]
            identity = knob["identity"]
            pattern = re.compile(
                _DECLARATION.format(name=re.escape(name)), re.MULTILINE
            )
            match = pattern.search(code)
            if not match:
                continue
            current = match.group(2).strip()
            if current == str(identity):
                continue
            code = pattern.sub(
                lambda m: f"{m.group(1)}{identity}{m.group(3)}", code, count=1
            )
            actions.append(
                f"set knob {name} initializer to the plan's identity value "
                f"{identity} (code declared {current})"
            )
    return code, actions


def build_plan_entry(function_name, generation_text, plan):
    """Assemble an executable plan entry without a second model call.

    Returns ``None`` when the code cannot be extracted or the plan carries no
    knob schema, in which case the caller should fall back to ``convertJson``.
    """
    code = extract_c_function(generation_text, function_name)
    if not code:
        return None

    metadata = knob_metadata_from_plan(plan)
    if metadata is None:
        return None

    variables, ranges, steps = metadata

    # Every knob the contract declares must actually be declared in the code,
    # otherwise the search space would address variables that do not exist.
    declared = [knob["name"] for record in plan.records for knob in record.knobs]
    for name in declared:
        if not re.search(rf"\b{re.escape(name)}\s*=", code):
            return None

    code, alignment_actions = align_knob_defaults_with_plan(code, plan)

    return {
        "functionName": function_name,
        "completeFunction": code,
        "knobVariables": variables,
        "knobRanges": ranges,
        "knobStepSize": steps,
        "planAlignmentActions": alignment_actions,
    }
