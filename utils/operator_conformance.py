"""Static conformance checks for declared approximation operators.

The LLM must not obtain hardware approval merely by printing an operator ID.
These checks verify distinctive implementation structure before the candidate
can compile or enter Bayesian optimization.

Coverage
--------
Structural checks originally existed only for the ten added operators
(T21-T30), and hardware-semantic checks only for T2 and T6. Everything else --
T1, T3-T5, T7-T20 -- could be declared with no evidence that the transformation
was implemented at all, which is how a candidate whose only real change was a
loop bound came to be labelled "T2, T21". The ``BASE_OPERATOR_CONTRACTS`` table
below closes that gap for the base catalogue.

These are static checks. They prove that the distinctive *structure* of an
operator is present and knob-controlled; they do not prove the data-flow
semantics are correct. They are a floor, not a proof.
"""

import ast
import json
import re


def _has(code, pattern):
    """True when *pattern* occurs in real code, not merely in a comment.

    ``_strip_comments`` exists so that "a check cannot be satisfied by prose
    alone", but this predicate used to search the raw text, so a candidate that
    only mentioned an operator in a comment satisfied its keyword test.
    """
    return re.search(
        pattern, _strip_comments(code), re.IGNORECASE | re.MULTILINE | re.DOTALL
    ) is not None


def _strip_comments(code):
    """Remove comments so a check cannot be satisfied by prose alone."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", code, flags=re.DOTALL)


def _knob_names(code):
    """Return knob identifiers declared in the knob block."""
    match = re.search(
        r"/\*\s*Knob Variables Declaration Start\s*\*/(.*?)"
        r"/\*\s*Knob Variables Declaration End\s*\*/",
        code,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return set()
    return set(
        re.findall(r"\b([A-Za-z_]\w*)\s*=", match.group(1))
    )


def _knob_tainted_names(code, passes=3):
    """Return the knobs plus every variable derived from a knob.

    Requiring the knob identifier to appear on the same line as the construct
    it controls is too strict for ordinary C: a plan that computes
    ``int limit = (N * perforation_pct) / 100;`` and then loops to ``limit`` is
    a perfectly valid knob-controlled loop. Propagating the taint through
    assignments catches that without accepting an unrelated expression.
    """
    tainted = _knob_names(code)
    if not tainted:
        return tainted
    body = _strip_comments(code)
    # Only plain `x = expr;` assignments propagate taint. Compound forms
    # (`+=`, `*=`) accumulate unrelated data and previously dragged ordinary
    # accumulators into the tainted set, which made every loop look
    # knob-controlled.
    assignments = re.findall(
        r"^\s*(?:[A-Za-z_]\w*[\s*]+)*([A-Za-z_]\w*)\s*(?<![-+*/%<>!=])=(?!=)\s*([^;]+);",
        body,
        re.MULTILINE,
    )
    for _ in range(passes):
        grew = False
        for target, expression in assignments:
            if target in tainted:
                continue
            if any(
                re.search(rf"\b{re.escape(name)}\b", expression)
                for name in tainted
            ):
                tainted.add(target)
                grew = True
        if not grew:
            break
    return tainted


def _knob_controlled(code, pattern):
    """True when *pattern* appears in an expression involving a knob-derived value."""
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    for line in body.splitlines():
        if re.search(pattern, line, re.IGNORECASE) and any(
            re.search(rf"\b{re.escape(name)}\b", line) for name in tainted
        ):
            return True
    return False


#: Structural requirement for each base-catalogue operator.
#: ``check`` receives the comment-stripped code and the raw code.

def _braced_block(code, start):
    """Return (body, end_index) for the block whose '{' is at or after *start*."""
    opening = code.find("{", start)
    if opening == -1:
        return "", start
    depth = 0
    for position in range(opening, len(code)):
        if code[position] == "{":
            depth += 1
        elif code[position] == "}":
            depth -= 1
            if depth == 0:
                return code[opening + 1:position], position + 1
    return code[opening + 1:], len(code)


def _assigned_names(fragment):
    """Identifiers that receive a plain assignment inside *fragment*."""
    names = set()
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*=(?!=)", fragment):
        names.add(match.group(1))
    return names


_NIBBLE_INDEX = re.compile(r"&\s*0[xX]0?[fF]\b")


def _loop_header_and_body(code, match_start):
    """Return (header, body) for the for/while whose keyword starts at *match_start*."""
    opening = code.find("(", match_start)
    if opening == -1:
        return "", ""
    depth = 0
    close = opening
    for position in range(opening, len(code)):
        if code[position] == "(":
            depth += 1
        elif code[position] == ")":
            depth -= 1
            if depth == 0:
                close = position
                break
    header = code[opening + 1:close]
    rest = code[close + 1:]
    stripped = rest.lstrip()
    offset = len(rest) - len(stripped)
    if stripped.startswith("{"):
        body, _ = _braced_block(rest, offset)
    else:
        end = stripped.find(";")
        body = stripped if end == -1 else stripped[:end + 1]
    return header, body


def _knob_controls_nibble_traversal(code):
    """True when a knob governs how many nibbles a nibble-indexed loop processes.

    T27 is defined as processing *fewer* nibbles: at stride > 1 only every
    n-th nibble is counted, reducing both LUT accesses and loop iterations.
    Two weaker readings were tried and both are wrong:

    * a knob anywhere near a ``+=`` or ``[`` accepted a loop that walked all
      eight nibbles and merely added a knob as a correction term afterwards;
    * a knob inside the nibble-selection expression (``x >> (i * 4 * nsel)``)
      changes *which* nibble is read while still running eight iterations, so
      it adds error without removing work, and for nsel >= 2 it shifts by 32
      or more bits, which is undefined for a 32-bit value.

    So the knob must appear in the loop's continuation condition or its
    increment. A knob used only in the initialiser or the body does not bound
    traversal.
    """
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body_text = _strip_comments(code)
    for match in re.finditer(r"\b(for|while)\s*\(", body_text):
        keyword = match.group(1)
        header, body = _loop_header_and_body(body_text, match.start())
        if not _NIBBLE_INDEX.search(body):
            continue
        if keyword == "for":
            clauses = header.split(";")
            # Skip the initialiser: only the condition and increment bound how
            # many iterations run.
            governing = ";".join(clauses[1:]) if len(clauses) >= 2 else ""
        else:
            governing = header
        if any(
            re.search(r"\b" + re.escape(name) + r"\b", governing)
            for name in tainted
        ):
            return True
    return False


def _carries_value_past_guard(code):
    """True when a value produced under a knob-guarded branch is reused outside it.

    This is the structural signature of hold/reconstruction: decimate or skip
    inside a knob-controlled condition, then reuse the retained value for the
    positions that were skipped. The previous test looked for the words
    ``hold``/``reconstruct``/``interpolate``, which rejected a correct
    zero-order-hold that happened to name its carrier ``last_sample`` and
    accepted any candidate that merely mentioned the word in a comment.
    """
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    for match in re.finditer(r"\bif\s*\(", body):
        opening = body.find("(", match.start())
        depth = 0
        condition_end = opening
        for position in range(opening, len(body)):
            if body[position] == "(":
                depth += 1
            elif body[position] == ")":
                depth -= 1
                if depth == 0:
                    condition_end = position
                    break
        condition = body[opening + 1:condition_end]
        if not any(re.search(rf"\b{re.escape(n)}\b", condition) for n in tainted):
            continue
        guarded, after = _braced_block(body, condition_end)
        if not guarded:
            continue
        outside = body[:match.start()] + body[after:]
        for name in _assigned_names(guarded):
            if name in tainted:
                continue
            # The carrier must be read somewhere the guard does not cover.
            if re.search(rf"\b{re.escape(name)}\b", outside):
                return True
    return False


def _if_condition_and_statement(code, match_start):
    """Return an ``if`` condition, its immediate statement, and its end.

    This deliberately handles both braced and single-statement branches.  It
    is only a lightweight C scanner, but balancing parentheses avoids the
    truncated conditions produced by the old regular-expression checks.
    """
    opening = code.find("(", match_start)
    if opening == -1:
        return "", "", match_start
    depth = 0
    close = opening
    for position in range(opening, len(code)):
        if code[position] == "(":
            depth += 1
        elif code[position] == ")":
            depth -= 1
            if depth == 0:
                close = position
                break

    rest = code[close + 1:]
    stripped = rest.lstrip()
    offset = close + 1 + len(rest) - len(stripped)
    if stripped.startswith("{"):
        statement, end = _braced_block(code, offset)
        return code[opening + 1:close], statement, end
    end = code.find(";", offset)
    if end == -1:
        end = len(code) - 1
    return code[opening + 1:close], code[offset:end + 1], end + 1


def _has_recomputation(fragment):
    """Whether a cache-miss path performs observable work before key update."""
    return bool(
        re.search(r"\b(?:for|while|do)\b", fragment)
        or re.search(r"\b[A-Za-z_]\w*\s*\(", fragment)
        or re.search(r"\b[A-Za-z_]\w*\s*\[[^]]+\]\s*=", fragment)
        or re.search(r"\b[A-Za-z_]\w*\s*=(?!=)", fragment)
    )


def _starts_with_exit(fragment):
    """Whether a cache-hit branch immediately leaves the miss computation.

    A return somewhere later in a braced branch is insufficient: generated
    code duplicated the complete search in both sides of a purported T3 hit
    test, then returned from the search normally.  That changes control-flow
    shape but reuses no work.  T3's hit path must therefore start with the
    early return/goto that bypasses recomputation.
    """
    return bool(re.match(r"\s*(?:return\b[^;]*;|goto\s+[A-Za-z_]\w*\s*;)", fragment))


def _knob_guarded_keyed_reuse(code):
    """Recognise exact keyed reuse without relying on identifier spelling.

    T3 requires a hit guard which (1) is enabled by a declared knob, (2)
    compares a current key with a stored key, (3) exits before recomputation,
    and (4) updates the stored key after doing miss-path work.  Merely naming a
    variable ``cache`` or ``table`` proves none of these properties.
    """
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    for match in re.finditer(r"\bif\s*\(", body):
        condition, guarded, after = _if_condition_and_statement(body, match.start())
        if not any(re.search(rf"\b{re.escape(name)}\b", condition) for name in tainted):
            continue
        if not _starts_with_exit(guarded):
            continue

        pairs = re.findall(
            r"\b([A-Za-z_]\w*)\s*==\s*([A-Za-z_]\w*)\b", condition
        )
        for left, right in pairs:
            if left == right or left in tainted or right in tainted:
                continue
            tail = body[after:]
            for stored, current in ((left, right), (right, left)):
                update = re.search(
                    rf"\b{re.escape(stored)}\s*=(?!=)\s*"
                    rf"(?:\([^;)]*\)\s*)*\b{re.escape(current)}\b\s*;",
                    tail,
                )
                if update and _has_recomputation(tail[:update.start()]):
                    return True
    return False


def _knob_controlled_lossy_container(code):
    """Recognise a knob-sized hashed/bitset container without name heuristics."""
    body = _strip_comments(code)
    has_knob_index_reduction = _knob_controlled(code, r"[%&]")
    has_indexed_storage = bool(re.search(r"\b[A-Za-z_]\w*\s*\[[^]]+\]", body))
    has_compact_membership_logic = bool(
        re.search(r"(?:<<|>>|\||&|\^)\s*[^;]+", body)
    )
    return has_knob_index_reduction and has_indexed_storage and has_compact_membership_logic


def _calls_in(fragment):
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", fragment))
    return names - {"if", "for", "while", "switch", "sizeof", "return"}


def _knob_selected_call_surrogate(code):
    """Recognise a knob-selected arithmetic/table alternative to a call."""
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    for match in re.finditer(r"\bif\s*\(", body):
        condition, first, after = _if_condition_and_statement(body, match.start())
        if not any(re.search(rf"\b{re.escape(name)}\b", condition) for name in tainted):
            continue
        rest = body[after:].lstrip()
        if not re.match(r"else\b", rest):
            continue
        rest = re.sub(r"^else\b", "", rest, count=1).lstrip()
        if rest.startswith("{"):
            second, _ = _braced_block(rest, 0)
        else:
            end = rest.find(";")
            second = rest if end == -1 else rest[:end + 1]

        first_calls, second_calls = _calls_in(first), _calls_in(second)
        first_alt = bool(re.search(r"(?:[+*/-]|<<|>>|\[[^]]+\])", first))
        second_alt = bool(re.search(r"(?:[+*/-]|<<|>>|\[[^]]+\])", second))
        if (first_calls and not second_calls and second_alt) or (
            second_calls and not first_calls and first_alt
        ):
            return True
    return False


def _probabilistic_decision(code):
    """Recognise PRNG state generation feeding a knob-controlled decision."""
    body = _strip_comments(code)
    has_random_source = bool(
        re.search(r"\b(?:rand|random)\s*\(", body)
        or (re.search(r"\^", body) and re.search(r"(?:>>|<<)", body))
    )
    return has_random_source and bool(_knob_controlled_conditions(code))


def _knob_controlled_spatial_fill(code):
    """Recognise 2-D anchor sampling plus knob-bounded output filling.

    A plain two-dimensional perforation writes only anchor pixels. Spatial
    downsampling must additionally traverse the skipped block and write every
    output position. The canonical shape therefore has two knob-controlled
    anchor loops plus two knob-controlled fill loops and a 2-D array write.
    """
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    controlled_headers = [
        header for header in _loop_headers(body)
        if any(re.search(rf"\b{re.escape(name)}\b", header) for name in tainted)
    ]
    has_2d_output_write = bool(
        re.search(
            r"\b[A-Za-z_]\w*\s*\[[^]]+\]\s*\[[^]]+\]\s*=",
            body,
        )
    )
    return len(controlled_headers) >= 4 and has_2d_output_write


def _loop_headers(code):
    """Yield balanced for/while headers, including casts and nested calls."""
    for match in re.finditer(r"\b(?:for|while)\s*\(", code):
        opening = code.find("(", match.start())
        depth = 0
        for position in range(opening, len(code)):
            character = code[position]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    yield code[opening + 1:position]
                    break


def _knob_controls_a_loop(code):
    """True when a knob-derived value appears in a loop's own header.

    Checking "the function contains a loop AND a tainted name appears on some
    line with a comparison" was far too loose: an accumulator that happened to
    inherit taint made every unrelated loop count, so a knob computed into a
    local and then ignored still passed T1. The knob has to be in the header of
    the loop it supposedly controls.
    """
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    for head in _loop_headers(body):
        if any(re.search(rf"\b{re.escape(name)}\b", head) for name in tainted):
            return True
    return False


BASE_OPERATOR_CONTRACTS = {
    1: (
        "a loop whose bound or stride is genuinely controlled by a declared "
        "knob: the knob (or a value derived from it) must appear in the loop's "
        "own condition or increment, not merely somewhere in the function",
        lambda body, raw: _knob_controls_a_loop(raw),
    ),
    2: (
        "a knob-controlled change of numeric representation: an integer "
        "shift, mask, or narrowing cast. Multiplying a floating-point value "
        "by a scale knob is value scaling, not precision scaling",
        lambda body, raw: _knob_controlled(
            raw,
            r"(>>|<<|&\s*0[xX]|\(\s*(?:signed\s+|unsigned\s+)?"
            r"(?:char|short|int|long|int8_t|int16_t|int32_t|"
            r"uint8_t|uint16_t|uint32_t)\s*\))",
        ),
    ),
    3: (
        "exact keyed reuse: a knob-enabled key-equality hit guard exits "
        "before recomputation, while the miss path performs work and then "
        "updates the stored key",
        lambda body, raw: _knob_guarded_keyed_reuse(raw),
    ),
    4: (
        "a knob-controlled branch that skips work",
        lambda body, raw: _has(body, r"\bif\s*\(")
        and _knob_controlled(raw, r"\b(if|continue|goto)\b|[<>=!]="),
    ),
    5: (
        "a knob-sized hashed/bitset container: the knob controls index "
        "reduction into compact indexed storage with bitwise membership logic",
        lambda body, raw: _knob_controlled_lossy_container(raw),
    ),
    7: (
        "a knob-controlled early termination of an iterative computation",
        lambda body, raw: _has(body, r"\b(break|return)\b")
        and _knob_controlled(raw, r"\b(break|if)\b|[<>]=?"),
    ),
    8: (
        "a knob-selected simpler computation path",
        lambda body, raw: _has(body, r"\bif\s*\(")
        and _knob_controlled(raw, r"\bif\s*\("),
    ),
    9: (
        "a run-time precision switch driven by a knob",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and _has(body, r"(>>|<<|\(\s*(int|long|float|double)\s*\))"),
    ),
    10: (
        "a knob-controlled quantisation step (shift, mask, or divide-scale)",
        lambda body, raw: _knob_controlled(raw, r"(>>|<<|&|/|\*)"),
    ),
    11: (
        "a knob that trades output fidelity for reduced work",
        lambda body, raw: bool(_knob_names(raw))
        and _has(body, r"\b(if|for|while)\s*\("),
    ),
    12: (
        "a knob-controlled reduction of the data actually processed",
        lambda body, raw: _knob_controlled(raw, r"\[|\+=|[<>]=?"),
    ),
    13: (
        "a stored previous result reused under a knob-controlled condition",
        lambda body, raw: _carries_value_past_guard(raw),
    ),
    14: (
        "two computation paths of different fidelity selected by a knob",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and len(re.findall(r"\belse\b", body)) >= 1,
    ),
    16: (
        "a knob-controlled skip of zero or negligible elements",
        lambda body, raw: _has(body, r"\bif\s*\([^)]*(==\s*0|!=\s*0|<\s*\w+)")
        and bool(_knob_names(raw)),
    ),
    19: (
        "a cheaper surrogate replacing an expensive call, under a knob",
        lambda body, raw: _knob_selected_call_surrogate(raw),
    ),
    20: (
        "behaviour that adapts to the input under a knob-controlled test",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and _has(body, r"\bif\s*\("),
    ),
}


def nonconforming_base_operator_ids(code, technique_ids):
    """Return base-catalogue IDs whose required structure is absent."""
    body = _strip_comments(code)
    invalid = set()
    for technique_id in sorted(set(technique_ids) & set(BASE_OPERATOR_CONTRACTS)):
        _requirement, check = BASE_OPERATOR_CONTRACTS[technique_id]
        try:
            ok = bool(check(body, code))
        except Exception:
            ok = False
        if not ok:
            invalid.add(technique_id)
    return invalid



def _declared_knobs_used_in_body(code):
    """Return knobs that are declared and actually read in the function body.

    A knob that is declared and never used is decoration. The AR ``sqrt16``
    artifact declared ``max_iterations``, computed ``limit`` from it, and then
    ignored both -- yet was accepted as T21 because the function contained an
    ordinary ``return``.
    """
    knobs = _knob_names(code)
    if not knobs:
        return set()
    body = _strip_comments(code)
    marker = re.search(
        r"/\*\s*Knob Variables Declaration End\s*\*/", code, re.IGNORECASE
    )
    if marker:
        body = _strip_comments(code[marker.end():])
    return {
        knob for knob in knobs
        if len(re.findall(rf"\b{re.escape(knob)}\b", body)) > 0
    }


_CONDITION = re.compile(
    r"\b(?:if|while)\s*\(([^{;]*)\)|\bfor\s*\([^;]*;([^;]*);", re.DOTALL
)


def _knob_controlled_conditions(code):
    """Return the conditions whose expression depends on a knob."""
    tainted = _knob_tainted_names(code)
    if not tainted:
        return []
    body = _strip_comments(code)
    found = []
    for match in _CONDITION.finditer(body):
        expression = match.group(1) or match.group(2) or ""
        if any(
            re.search(rf"\b{re.escape(name)}\b", expression) for name in tainted
        ):
            found.append((match.start(), expression))
    return found


def _knob_guarded_exit(code):
    """True when a break/return is reachable only through a knob-controlled test.

    Requiring merely "the function contains a return" accepted every function
    ever written. The exit must be governed by the knob, which is the whole
    content of an early-exit operator.
    """
    conditions = _knob_controlled_conditions(code)
    if not conditions:
        return False
    body = _strip_comments(code)
    for position, _expression in conditions:
        # The guarded statement follows its condition; a short window keeps
        # this local without needing a full parser.
        window = body[position:position + 400]
        if re.search(r"\b(?:break|return|goto)\b", window):
            return True
    return False


def nonconforming_added_operator_ids(code, technique_ids):
    """Return added-operator IDs whose distinctive structure is absent.

    Keeping this separate from the rendered error lets the pipeline remove an
    unsupported *extra claim* while retaining a candidate that also contains a
    real operator.  A candidate whose only claim is unsupported is still
    rejected.
    """
    invalid = set()
    used_knobs = _declared_knobs_used_in_body(code)
    for technique_id in sorted(set(technique_ids) & set(range(21, 31))):
        # Common precondition for every added operator: at least one declared
        # knob must actually be read by the function. Without this a candidate
        # can declare a knob, never use it, and still claim the operator.
        if not used_knobs:
            invalid.add(technique_id)
            continue
        if technique_id == 21:
            ok = _knob_guarded_exit(code)
            requirement = (
                "a break/return that is reached only through a condition "
                "controlled by a declared knob. Declaring an iteration budget "
                "and leaving the loop condition unchanged is not an early exit"
            )
        elif technique_id == 22:
            ok = _knob_controlled_spatial_fill(code)
            requirement = (
                "two-dimensional knob-controlled anchor sampling plus bounded "
                "fill loops that write the skipped output positions"
            )
        elif technique_id == 23:
            ok = (
                (
                    _knob_controlled(code, r"\b(?:for|while)\s*\(|\+=|\[")
                    or _knob_controlled_conditions(code)
                )
                and _carries_value_past_guard(code)
            )
            requirement = (
                "a temporal decimation/stride knob plus reconstruction: a value "
                "retained under the knob-guarded branch and reused for the "
                "decimated positions"
            )
        elif technique_id == 24:
            ok = _has(code, r"\b(?:cabs|hypot|magnitude)\w*\s*\(")
            requirement = "magnitude extraction (for example cabs/hypot) from FFT values"
        elif technique_id == 25:
            ok = bool(_knob_controlled_conditions(code)) and _has(
                code, r"\b(?:lazy|cache|preprocess_budget)\w*\b"
            )
            requirement = "a lazy/cache/budget knob guarding deferred preprocessing"
        elif technique_id == 26:
            ok = _has(code, r">>\s*[A-Za-z_]\w*") and _has(
                code, r"\b(?:ewma|estimate|average|sample)\w*\b"
            )
            requirement = "an EWMA/estimate recurrence using an integer right-shift knob"
        elif technique_id == 27:
            has_lut = _has(
                code,
                r"(?:char|uint8_t|int)\s+[A-Za-z_]\w*\s*\[\s*16\s*\]\s*=\s*\{",
            )
            # The 4-bit shift may be written `i * 4)`, `i * 4 * sel)` or
            # `>> 4`; requiring the closing paren rejected the middle form.
            has_nibble_index = _has(code, r"&\s*0[xX]0?[fF]\b") and (
                _has(code, r"\*\s*4\b") or _has(code, r">>\s*4\b")
            )
            # The stride must be knob-controlled, but its identifier is the
            # model's to choose. Requiring the literal name ``nibble_stride``
            # tested vocabulary rather than behaviour and rejected every
            # candidate bc-iclib produced (15 declared, 0 accepted).
            has_stride = _knob_controls_nibble_traversal(code)
            ok = has_lut and has_nibble_index and has_stride
            requirement = (
                "a local initialized 16-entry LUT, 4-bit (& 0xF) indexing, "
                "and a knob controlling the nibble loop bound, increment, or selected nibble index"
            )
        elif technique_id == 28:
            ok = bool(_knob_controlled_conditions(code)) and _has(
                code, r"\b(?:fft|spectral|feature)\w*\b"
            )
            requirement = "tier/gate logic controlling expensive spectral features"
        elif technique_id == 29:
            ok = _has(code, r"\bradix_mode\b") and _has(
                code, r"(?:>>\s*[234]\b|&\s*0[xX][fF7])"
            )
            requirement = "a radix_mode knob with power-of-two shift/mask conversion"
        else:  # T30
            ok = (
                (
                    _knob_controlled(code, r"\[|\+=|\bfor\s*\(")
                    or _knob_controlled_conditions(code)
                )
                and _carries_value_past_guard(code)
            )
            requirement = (
                "segment/change knobs plus a representative value retained and "
                "reused for non-representative positions"
            )

        if not ok:
            invalid.add(technique_id)

    return invalid


def _knob_initialisers(code):
    """Return {knob: int_initialiser} parsed from the knob declaration block."""
    match = re.search(
        r"/\*\s*Knob Variables Declaration Start\s*\*/(.*?)"
        r"/\*\s*Knob Variables Declaration End\s*\*/",
        code,
        re.DOTALL,
    )
    if not match:
        return {}
    values = {}
    for decl in re.finditer(
        r"\b[A-Za-z_]\w*\s+([A-Za-z_]\w*)\s*=\s*(-?\d+)", match.group(1)
    ):
        values[decl.group(1)] = int(decl.group(2))
    return values


def _normalise_knob_ranges(knob_ranges):
    """Accept the several shapes knobRanges is stored in and return {knob: (lo, hi)}."""
    if isinstance(knob_ranges, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                knob_ranges = loader(knob_ranges)
                break
            except Exception:
                continue
    out = {}
    if isinstance(knob_ranges, dict):
        knob_ranges = [knob_ranges]
    for entry in knob_ranges or []:
        if not isinstance(entry, dict):
            continue
        for name, bounds in entry.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                try:
                    out[name] = (int(bounds[0]), int(bounds[1]))
                except (TypeError, ValueError):
                    continue
    return out


def _divisor_knobs(code):
    """Knob-derived names that appear as a '/' or '%' divisor in real code."""
    tainted = _knob_tainted_names(code)
    if not tainted:
        return set()
    body = _strip_comments(code)
    found = set()
    for match in re.finditer(r"[/%]\s*\(?\s*(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", body):
        if match.group(1) in tainted:
            found.add(match.group(1))
    return found


def validate_knob_safety(code, knob_ranges):
    """Reject knob declarations that are unsafe or inconsistent with their range.

    Two defects shipped in every arm of the ar-iclib campaign and neither
    operator-conformance nor the plan contract caught them:

    * ``int decimation_factor = 0;`` used as ``samplesInWindow % decimation_factor``
      divides by zero at the identity point itself, and the declared range
      ``[0, 8]`` lets BayesOpt evaluate it directly.
    * the same declaration carried the comment ``1 = identity`` while
      initialising to ``0``, so the code's own identity contradicted its plan.
    """
    errors = []
    ranges = _normalise_knob_ranges(knob_ranges)
    initialisers = _knob_initialisers(code)
    divisors = _divisor_knobs(code)

    for name in sorted(divisors):
        low, high = ranges.get(name, (None, None))
        if low is not None and low <= 0 <= high:
            errors.append(
                f"knob {name} is used as a divisor but its range [{low}, {high}] "
                f"includes 0; narrow the range so the divisor cannot be zero"
            )
        elif initialisers.get(name) == 0:
            errors.append(
                f"knob {name} is used as a divisor but is initialised to 0, "
                f"which divides by zero at the identity point"
            )

    for name, value in sorted(initialisers.items()):
        if name not in ranges:
            continue
        low, high = ranges[name]
        if not low <= value <= high:
            errors.append(
                f"knob {name} is initialised to {value}, which is outside its "
                f"declared range [{low}, {high}]"
            )

    return "; ".join(errors)


def validate_operator_conformance(code, technique_ids):
    """Return an actionable error when declared operators are not present."""
    invalid = nonconforming_added_operator_ids(code, technique_ids)
    errors = []

    # Base-catalogue operators used to be accepted on the strength of the label
    # alone. Require their distinctive structure too.
    for technique_id in sorted(nonconforming_base_operator_ids(code, technique_ids)):
        requirement, _check = BASE_OPERATOR_CONTRACTS[technique_id]
        errors.append(f"T{technique_id} requires {requirement}")

    requirements = {
        21: "an early break/return controlled by a threshold or iteration-budget knob",
        22: "two-dimensional knob-controlled anchor sampling plus bounded fill loops that write the skipped output positions",
        23: "a temporal decimation/stride knob plus a retained value reused for the decimated positions",
        24: "magnitude extraction (for example cabs/hypot) from FFT values",
        25: "a lazy/cache/budget knob guarding deferred preprocessing",
        26: "an EWMA/estimate recurrence using an integer right-shift knob",
        27: "a local initialized 16-entry LUT, 4-bit (& 0xF) indexing, and a knob controlling the nibble loop bound, increment, or selected nibble index",
        28: "tier/gate logic controlling expensive spectral features",
        29: "a radix_mode knob with power-of-two shift/mask conversion",
        30: "segment/change knobs plus a representative value reused for non-representative positions",
    }
    for technique_id in sorted(invalid):
        errors.append(f"T{technique_id} requires {requirements[technique_id]}")

    if not errors:
        return ""
    return (
        "Operator conformance validation failed: the declared operator ID does "
        "not match the generated C implementation. " + "; ".join(errors)
    )


def nonconforming_operator_ids(code, technique_ids):
    """Return every unsupported declared ID across the complete catalogue."""
    claimed = set(technique_ids)
    return (
        nonconforming_base_operator_ids(code, claimed)
        | nonconforming_added_operator_ids(code, claimed)
    )


def validate_hardware_operator_conformance(code, technique_ids, hw_profile):
    """Validate implementation semantics that depend on target capabilities.

    Generic operator conformance already requires T2 to contain a
    knob-controlled shift, mask, or narrowing cast. Do not repeat that check
    here using a small whitelist of knob names: valid code such as
    ``value >>= quant_shift`` was rejected merely because its knob was not
    called ``precision_mode``. This layer retains genuinely hardware-specific
    rules: no floating-point knobs on a no-FPU target, and real probabilistic
    control for T6.
    """
    if hw_profile is None:
        return ""

    claimed = set(technique_ids)
    errors = []

    if 6 in claimed:
        if not _probabilistic_decision(code):
            errors.append(
                "T6 requires an implemented random/pseudorandom decision controlled "
                "by a probability or sampling knob"
            )

    # A no-FPU target pays a software-emulation penalty for every float
    # operation. The approximation prompt itself used to demonstrate loop
    # perforation as `i < (1000 * 0.8)`, which *introduces* floating point on a
    # device that has none. Reject a knob that is declared as a float.
    if not hw_profile.has_fpu:
        knob_block = re.search(
            r"/\*\s*Knob Variables Declaration Start\s*\*/(.*?)"
            r"/\*\s*Knob Variables Declaration End\s*\*/",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if knob_block:
            float_knobs = re.findall(
                r"\b(?:float|double)\s+([A-Za-z_]\w*)", knob_block.group(1)
            )
            if float_knobs:
                errors.append(
                    "knob(s) "
                    + ", ".join(sorted(set(float_knobs)))
                    + " are declared as floating point on a target with no FPU; "
                    "every use costs software-emulated FP cycles. Express the "
                    "knob as an integer (for example a percentage in 0..100 or "
                    "a shift amount) and use integer arithmetic"
                )

    if not errors:
        return ""
    return "Hardware-specific operator conformance failed: " + "; ".join(errors)


def prune_unimplemented_added_operator_claims(function_dict, technique_ids):
    """Drop false added-operator labels when another claimed operator remains.

    This does not invent provenance. It only removes T21-T30 IDs that fail the
    static implementation contract. If removal would leave no claimed
    operator, no mutation occurs and normal validation rejects the candidate.
    """
    claimed = set(technique_ids)
    invalid = nonconforming_added_operator_ids(
        function_dict.get("completeFunction", ""), claimed
    )
    retained = claimed - invalid
    if not invalid or not retained:
        return claimed, []

    function_dict["technique_numbers"] = sorted(retained)
    action = (
        "discarded unimplemented operator claim(s): "
        + ", ".join(f"T{identifier}" for identifier in sorted(invalid))
    )
    return retained, [action]


def prune_unimplemented_operator_claims(function_dict, technique_ids):
    """Retain a valid implemented subset instead of rejecting the whole plan.

    Provenance correction used to apply only to T21-T30.  A candidate that
    genuinely implemented T1 but also mislabeled an inert reuse path as T13 was
    therefore discarded wholesale.  Dropping a statically disproven label does
    not weaken validation: the remaining IDs still have to pass their own
    contracts, and a candidate with no substantiated operator is left untouched
    so the normal validator rejects it.
    """
    claimed = set(technique_ids)
    code = function_dict.get("completeFunction", "")
    invalid = nonconforming_operator_ids(code, claimed)
    retained = claimed - invalid
    if not invalid or not retained:
        return claimed, []

    function_dict["technique_numbers"] = sorted(retained)
    action = (
        "discarded unimplemented operator claim(s): "
        + ", ".join(f"T{identifier}" for identifier in sorted(invalid))
    )
    return retained, [action]


def repair_operator_candidate(function_dict, technique_ids, hw_profile=None):
    """Install deterministic backends for operators with a canonical C form.

    T27 is a concrete 16-entry population-count transform, not an open-ended
    synthesis task. Keeping its implementation here prevents a small model
    from inventing an undeclared LUT or unsafe recursion while preserving the
    LLM's role in selecting the operator.
    """
    claimed = set(technique_ids)

    # On MSP430-class no-FPU targets, T2 must replace software-emulated
    # floating-point arithmetic with an integer/fixed-point realization. Sobel
    # has a canonical, algebraically equivalent integer form, so use it when a
    # model selects T2 but emits only floating-point value scaling.
    if (
        2 in claimed
        and function_dict.get("functionName") == "sobel_filtering"
        and hw_profile is not None
        and not hw_profile.has_fpu
        and validate_operator_conformance(
            function_dict.get("completeFunction", ""), {2}
        )
    ):
        function_dict["completeFunction"] = """void sobel_filtering(void)
{
  /*Knob Variables Declaration Start*/
  int precision_mode = 0;
  /*Knob Variables Declaration End*/

  int weight[3][3] = {{ -1,  0,  1 },
                      { -2,  0,  2 },
                      { -1,  0,  1 }};
  int x, y, i, j;

  if (precision_mode == 0) {
    double pixel_value;
    double min_value = DBL_MAX;
    double max_value = -DBL_MAX;

    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        pixel_value = 0.0;
        for (j = -1; j <= 1; j++) {
          for (i = -1; i <= 1; i++) {
            pixel_value += weight[j + 1][i + 1] * image1[y + j][x + i];
          }
        }
        if (pixel_value < min_value) min_value = pixel_value;
        if (pixel_value > max_value) max_value = pixel_value;
      }
    }
    if ((int)(max_value - min_value) == 0) {
      return;
    }

    x_size2 = x_size1;
    y_size2 = y_size1;
    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        pixel_value = 0.0;
        for (j = -1; j <= 1; j++) {
          for (i = -1; i <= 1; i++) {
            pixel_value += weight[j + 1][i + 1] * image1[y + j][x + i];
          }
        }
        pixel_value = MAX_BRIGHTNESS * (pixel_value - min_value) /
                      (max_value - min_value);
        image2[y][x] = (unsigned char)pixel_value;
      }
    }
  } else {
    int pixel_value_int;
    int min_int = 32767;
    int max_int = -32768;

    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        pixel_value_int =
            ((int)image1[y - 1][x + 1] - (int)image1[y - 1][x - 1]) +
            2 * ((int)image1[y][x + 1] - (int)image1[y][x - 1]) +
            ((int)image1[y + 1][x + 1] - (int)image1[y + 1][x - 1]);
        if (pixel_value_int < min_int) min_int = pixel_value_int;
        if (pixel_value_int > max_int) max_int = pixel_value_int;
      }
    }
    if (max_int == min_int) {
      return;
    }

    x_size2 = x_size1;
    y_size2 = y_size1;
    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        long scaled_value;
        pixel_value_int =
            ((int)image1[y - 1][x + 1] - (int)image1[y - 1][x - 1]) +
            2 * ((int)image1[y][x + 1] - (int)image1[y][x - 1]) +
            ((int)image1[y + 1][x + 1] - (int)image1[y + 1][x - 1]);
        scaled_value =
            (long)MAX_BRIGHTNESS * (long)(pixel_value_int - min_int);
        image2[y][x] =
            (unsigned char)(scaled_value / (long)(max_int - min_int));
      }
    }
  }
}"""
        function_dict["knobVariables"] = "['precision_mode']"
        function_dict["knobRanges"] = "[{'precision_mode': [0, 1]}]"
        function_dict["knobStepSize"] = "[{'precision_mode': 'Integer'}]"
        function_dict["technique_numbers"] = [2]
        return ["installed deterministic no-FPU T2 fixed-point Sobel backend"]

    if 27 not in claimed:
        return []
    if function_dict.get("functionName") != "btbl_bitcnt":
        return []
    if validate_operator_conformance(function_dict.get("completeFunction", ""), {27}) == "":
        return []

    function_dict["completeFunction"] = """int btbl_bitcnt(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int nibble_stride = 1;
  /*Knob Variables Declaration End*/

  static const char nibble_lut[16] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
  };
  int cnt = 0;
  int nibbles_counted = 0;
  for (int i = 0; i < 8; i += nibble_stride) {
    cnt += nibble_lut[(x >> (i * 4)) & 0xF];
    nibbles_counted++;
  }
  if (nibble_stride > 1 && nibbles_counted > 0) {
    cnt = (cnt * 8 + nibbles_counted / 2) / nibbles_counted;
  }
  return cnt;
}"""
    function_dict["knobVariables"] = "['nibble_stride']"
    function_dict["knobRanges"] = "[{'nibble_stride': [1, 4]}]"
    function_dict["knobStepSize"] = "[{'nibble_stride': 'Integer'}]"
    function_dict["technique_numbers"] = [27]
    return ["installed deterministic T27 nibble-lookup backend"]
