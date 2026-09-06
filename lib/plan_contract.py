"""Parse and enforce the structured approximation plan.

Why this exists
---------------
Planning used to ask only "what are the possible approximations?" and returned
free prose. Nothing downstream read it. Operator provenance was recovered later
by scraping ``APPLIED_TECHNIQUES:`` out of the *generated code* response, so the
coder could silently drop, swap, or add operators after planning, and a plan
that named eight operators routinely produced code implementing one.

This module turns the plan into a checkable artifact: a set of operator records
with a target region, a knob schema with an identity value, and a stated
invariant. Generation is then held to it.
"""

import re

#: One ``- OPERATOR: T<n>`` record and everything until the next record.
_RECORD_SPLIT = re.compile(r"^\s*-\s*OPERATOR\s*:", re.IGNORECASE | re.MULTILINE)

_FIELD_PATTERNS = {
    "region": re.compile(r"^\s*REGION\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    "benefit": re.compile(r"^\s*BENEFIT\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    "precondition": re.compile(
        r"^\s*PRECONDITION\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE
    ),
    "invariant": re.compile(r"^\s*INVARIANT\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
}

#: A KNOB line, captured loosely. A 7B model reliably produces the field
#: *content* but not the exact pipe-delimited punctuation, and rejecting it for
#: punctuation burns the whole retry budget without ever reaching generation.
#: Parse permissively, then validate strictly on semantics.
_KNOB_LINE = re.compile(r"^[ \t]*[-*]?[ \t]*KNOB\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_KNOB_NAME = re.compile(r"^\s*[|,]?\s*([A-Za-z_]\w*)")
_KNOB_TYPE = re.compile(r"\b(Integer|Real|int|float|double)\b", re.IGNORECASE)
_KNOB_RANGE = re.compile(
    r"(?:RANGE\s*)?[\[(]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*,\s*"
    r"(-?[\d.]+(?:[eE][+-]?\d+)?)\s*[\])]",
    re.IGNORECASE,
)
_KNOB_IDENTITY = re.compile(
    r"IDENTITY\s*[:=]?\s*(-?[\d.]+(?:[eE][+-]?\d+)?)", re.IGNORECASE
)

#: Shown verbatim when a KNOB line cannot be used. Small models copy examples
#: far more reliably than they follow an abstract grammar.
KNOB_FORMAT_EXAMPLE = (
    "KNOB: iteration_pct | Integer | RANGE [50, 100] | IDENTITY 100"
)


def _parse_knob_line(payload):
    """Return ``(knob_dict, error_or_None)`` for one KNOB line's content."""
    name_match = _KNOB_NAME.match(payload)
    if not name_match:
        return None, "the knob name is missing or is not a valid C identifier"
    name = name_match.group(1)
    remainder = payload[name_match.end():]

    type_match = _KNOB_TYPE.search(remainder)
    range_match = _KNOB_RANGE.search(remainder)
    identity_match = _KNOB_IDENTITY.search(remainder)

    if not range_match:
        return None, (
            f"knob {name} has no RANGE. Write it as: {KNOB_FORMAT_EXAMPLE}"
        )
    if not identity_match:
        return None, (
            f"knob {name} has a RANGE but no IDENTITY. IDENTITY is the value at "
            "which the function must reproduce the original behaviour exactly, "
            "and it is what the search is anchored to, so it cannot be omitted. "
            f"Write the whole line as: {KNOB_FORMAT_EXAMPLE}"
        )

    raw_type = (type_match.group(1).lower() if type_match else "integer")
    is_integer = raw_type in ("integer", "int")
    return {
        "name": name,
        "is_integer": is_integer,
        "low": range_match.group(1),
        "high": range_match.group(2),
        "identity": identity_match.group(1),
    }, None

_INTERACTIONS = re.compile(r"^\s*INTERACTIONS\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class PlanRecord:
    """One planned operator application."""

    def __init__(self, operator_id, region="", knobs=None, benefit="",
                 precondition="", invariant=""):
        self.operator_id = operator_id
        self.region = region
        self.knobs = knobs or []
        self.benefit = benefit
        self.precondition = precondition
        self.invariant = invariant

    def as_dict(self):
        return {
            "operator_id": self.operator_id,
            "region": self.region,
            "knobs": self.knobs,
            "benefit": self.benefit,
            "precondition": self.precondition,
            "invariant": self.invariant,
        }

    def __repr__(self):
        return f"PlanRecord(T{self.operator_id}, region={self.region!r})"


class ApprovedPlan:
    """The parsed, validated plan for one function."""

    def __init__(self, records=None, interactions="", declined=False, raw=""):
        self.records = records or []
        #: REGION descriptions that are plausible prose but quote no identifier.
        #: Recorded, never fatal.
        self.region_warnings = []
        self.interactions = interactions
        self.declined = declined
        self.raw = raw

    @property
    def operator_ids(self):
        return {record.operator_id for record in self.records}

    @property
    def knob_names(self):
        return {knob["name"] for record in self.records for knob in record.knobs}

    def as_dict(self):
        return {
            "declined": self.declined,
            "interactions": self.interactions,
            "operators": [record.as_dict() for record in self.records],
            "region_warnings": list(getattr(self, "region_warnings", [])),
        }

    def __repr__(self):
        if self.declined:
            return "ApprovedPlan(declined)"
        return f"ApprovedPlan({sorted(self.operator_ids)})"


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

#: Words a REGION may use that are not identifiers from the code.
_REGION_STOPWORDS = {
    "the", "a", "an", "in", "of", "on", "at", "to", "for", "over", "and", "or",
    "loop", "inner", "outer", "main", "call", "calls", "statement", "block",
    "body", "function", "value", "values", "array", "buffer", "variable",
    "computation", "accumulation", "expression", "branch", "condition", "this",
    "each", "every", "all", "same", "within", "inside", "that", "which", "its",
}


def _is_existing_symbol(name, function_code):
    """True when *name* is already a parameter or declared variable."""
    if not function_code:
        return False
    signature = re.match(r"[^{]*", function_code)
    if signature and re.search(rf"\b{re.escape(name)}\b", signature.group(0)):
        return True
    declaration = re.search(
        rf"\b(?:int|long|short|char|float|double|unsigned|signed|size_t|"
        rf"u?int\d+_t)[\w\s\*,]*\b{re.escape(name)}\b\s*[;,=\[]",
        function_code,
    )
    return bool(declaration)


def _fabricated_symbols(region, function_code):
    """Return code-shaped tokens in REGION that do not exist in the function.

    Only tokens that *look* like code are checked: snake_case, camelCase, or a
    name immediately followed by ``(`` or ``[``. Ordinary English words are not
    evidence of fabrication, and treating them as such threw away correct plans.
    """
    if not function_code:
        return set()
    present = set(_IDENTIFIER.findall(function_code))
    suspicious = set()
    for match in re.finditer(r"([A-Za-z_]\w*)\s*([([])?", region):
        token, adjacency = match.group(1), match.group(2)
        if token.lower() in _REGION_STOPWORDS:
            continue
        looks_like_code = (
            "_" in token
            or adjacency in ("(", "[")
            or (re.search(r"[a-z][A-Z]", token) is not None)
        )
        if looks_like_code and token not in present:
            suspicious.add(token)
    return suspicious


def _region_mentions_real_code(region, function_code):
    """True when REGION names at least one identifier present in the function.

    A plan whose REGION is generic prose ("the main loop") cannot be checked
    against the implementation later, and a plan naming a variable that does
    not exist is describing a different function.
    """
    if not function_code:
        return True
    code_identifiers = set(_IDENTIFIER.findall(function_code))
    region_words = [
        word for word in _IDENTIFIER.findall(region)
        if word.lower() not in _REGION_STOPWORDS
    ]
    if not region_words:
        return False
    return any(word in code_identifiers for word in region_words)


def parse_plan(text, max_operators=4, function_code=None):
    """Parse a planning response into an :class:`ApprovedPlan`.

    Returns ``(plan, errors)``. ``errors`` is a list of human-readable problems
    that should be sent back to the model; it is empty for a usable plan.

    When *function_code* is supplied, REGION is additionally checked against the
    function's actual identifiers, so a plan cannot name a loop or variable that
    does not exist in the function it claims to describe.
    """
    errors = []
    if not text:
        return ApprovedPlan(raw=""), ["The planning response was empty."]

    if re.search(r"^\s*PLAN\s*:\s*none\s*$", text, re.IGNORECASE | re.MULTILINE):
        return ApprovedPlan(declined=True, raw=text), []

    chunks = _RECORD_SPLIT.split(text)
    if len(chunks) < 2:
        return ApprovedPlan(raw=text), [
            "No PLAN block was found. Answer with `PLAN:` followed by one "
            "`- OPERATOR: T<number>` record per operator, or `PLAN: none`."
        ]

    records = []
    region_warnings = []
    for chunk in chunks[1:]:
        id_match = re.match(r"\s*T?(\d+)", chunk)
        if not id_match:
            errors.append(
                "An OPERATOR line did not name a catalogue ID in the form T<number>."
            )
            continue
        operator_id = int(id_match.group(1))

        fields = {}
        for name, pattern in _FIELD_PATTERNS.items():
            found = pattern.search(chunk)
            fields[name] = found.group(1).strip() if found else ""

        knobs = []
        saw_knob_line = False
        for knob_match in _KNOB_LINE.finditer(chunk):
            saw_knob_line = True
            parsed, knob_error = _parse_knob_line(knob_match.group(1))
            if knob_error:
                errors.append(f"T{operator_id}: {knob_error}")
                continue

            name = parsed["name"]
            is_integer = parsed["is_integer"]
            try:
                raw_values = [
                    float(parsed["low"]), float(parsed["high"]),
                    float(parsed["identity"]),
                ]
            except (TypeError, ValueError):
                errors.append(
                    f"T{operator_id}: knob {name} has a non-numeric RANGE or IDENTITY."
                )
                continue
            if is_integer and any(value != int(value) for value in raw_values):
                # int(2.5) -> 2 silently changed the declared search space.
                errors.append(
                    f"T{operator_id}: knob {name} is declared Integer but has "
                    f"non-integer bounds or identity. Use Real, or whole numbers."
                )
                continue
            cast = int if is_integer else float
            low_value, high_value, identity_value = (cast(v) for v in raw_values)

            if low_value > high_value:
                # Silently swapping hides a real modelling error: the model
                # stated a range it did not mean, and the "identity" it chose
                # may belong to the range it intended rather than this one.
                errors.append(
                    f"T{operator_id}: knob {name} has RANGE [{low_value}, "
                    f"{high_value}] with the low bound above the high bound. "
                    "State the range in ascending order."
                )
                continue
            if low_value == high_value:
                # A single-point range is not a knob: there is nothing to
                # search, and the "approximation" is pinned to one value. The
                # 7B model emits `RANGE [1, 1]` when it is describing a no-op
                # while claiming an operator was applied.
                errors.append(
                    f"T{operator_id}: knob {name} has the degenerate RANGE "
                    f"[{low_value}, {high_value}]. A single-point range gives "
                    "the optimizer nothing to search. Give a real range whose "
                    "IDENTITY end reproduces the original behaviour and whose "
                    "other end is more approximate."
                )
                continue
            if not (low_value <= identity_value <= high_value):
                # The model usually has the right instinct here (identity =
                # "no approximation") and simply drew the range around the
                # approximate values only. Say exactly how to repair it rather
                # than restating the rule.
                suggested_low = min(low_value, identity_value)
                suggested_high = max(high_value, identity_value)
                errors.append(
                    f"T{operator_id}: knob {name} has IDENTITY {identity_value} "
                    f"outside RANGE [{low_value}, {high_value}], so the "
                    "optimizer can never evaluate the unapproximated program. "
                    f"Widen the range to include it: "
                    f"RANGE [{suggested_low}, {suggested_high}] | "
                    f"IDENTITY {identity_value}."
                )
                continue
            if function_code and _is_existing_symbol(name, function_code):
                # The generation contract requires knobs to be new local
                # variables. Reusing a parameter or existing local silently
                # changes the function's interface or clobbers its state.
                errors.append(
                    f"T{operator_id}: knob {name} is already a parameter or "
                    "variable of this function. A knob must be a NEW local "
                    "variable introduced by the approximation, never an "
                    "existing symbol."
                )
                continue

            knobs.append({
                "name": name,
                "type": "Integer" if is_integer else "Real",
                "range": [low_value, high_value],
                "identity": identity_value,
            })

        if not fields["region"]:
            errors.append(
                f"T{operator_id}: missing REGION. Name the exact loop, call, or "
                "statement in this function that the operator changes."
            )
        elif function_code:
            bogus = _fabricated_symbols(fields["region"], function_code)
            if bogus:
                errors.append(
                    f"T{operator_id}: REGION names {', '.join(sorted(bogus))}, "
                    "which does not appear anywhere in this function. Name the "
                    "loop, variable, or callee that is actually present."
                )
            elif not _region_mentions_real_code(fields["region"], function_code):
                # Accurate prose that quotes no identifier is weak but not
                # wrong; record it rather than discarding an otherwise valid plan.
                region_warnings.append(
                    f"T{operator_id}: REGION {fields['region']!r} quotes no "
                    "identifier from the function"
                )
        if not fields["benefit"]:
            errors.append(
                f"T{operator_id}: missing BENEFIT. State which hardware cost "
                "this removes (iterations, FP emulation, memory traffic, calls)."
            )
        if not fields["precondition"]:
            errors.append(
                f"T{operator_id}: missing PRECONDITION. State what must be true "
                "of the inputs for this approximation to be safe."
            )
        if not knobs and not saw_knob_line:
            errors.append(
                f"T{operator_id}: missing a valid KNOB line. Use "
                f"`{KNOB_FORMAT_EXAMPLE}`."
            )
        if not fields["invariant"]:
            errors.append(
                f"T{operator_id}: missing INVARIANT. State what stays true of the "
                "output at every knob setting."
            )

        records.append(PlanRecord(
            operator_id=operator_id,
            region=fields["region"],
            knobs=knobs,
            benefit=fields["benefit"],
            precondition=fields["precondition"],
            invariant=fields["invariant"],
        ))

    if not records:
        errors.append("The PLAN block contained no parseable operator records.")

    seen_operators = set()
    for record in records:
        if record.operator_id in seen_operators:
            errors.append(
                f"T{record.operator_id} appears more than once. State one record "
                "per operator; if it applies in two places, name both in REGION."
            )
        seen_operators.add(record.operator_id)

    knob_declarations = {}
    for record in records:
        for knob in record.knobs:
            name = knob["name"]
            previous = knob_declarations.get(name)
            if previous is None:
                knob_declarations[name] = (record.operator_id, knob)
                continue
            previous_operator, previous_knob = previous
            if (
                previous_knob["range"] != knob["range"]
                or previous_knob["type"] != knob["type"]
                or previous_knob["identity"] != knob["identity"]
            ):
                errors.append(
                    f"Knob {name} is declared differently by T{previous_operator} "
                    f"and T{record.operator_id}. A knob has one type, one range, "
                    "and one identity value across the whole plan."
                )

    if len(records) > max_operators:
        errors.append(
            f"The plan lists {len(records)} operators but at most {max_operators} "
            "are allowed. Keep only the ones you can tie to a concrete REGION."
        )

    interactions_match = _INTERACTIONS.search(text)
    interactions = (
        interactions_match.group(1).strip() if interactions_match else ""
    )
    if not interactions:
        errors.append(
            "Missing `INTERACTIONS:` line. Write `INTERACTIONS: none` if the "
            "operators are independent."
        )

    plan = ApprovedPlan(records=records, interactions=interactions, raw=text)
    plan.region_warnings = region_warnings
    return plan, errors


def validate_plan_against_catalogue(plan, allowed_ids):
    """Return errors for planned operators outside the allowed set."""
    if plan.declined:
        return []
    invalid = sorted(plan.operator_ids - set(allowed_ids))
    if not invalid:
        return []
    allowed = ", ".join(f"T{tid}" for tid in sorted(allowed_ids))
    return [
        "The plan names operator(s) "
        + ", ".join(f"T{tid}" for tid in invalid)
        + f" which are not in this function's operator contract. Allowed: {allowed}."
    ]


def check_implementation_matches_plan(plan, implemented_ids):
    """Return ``(error, corrected_ids, actions)`` for a generated candidate.

    This is the link that did not previously exist: operator provenance was
    read out of the generation response alone, so the coder could add or drop
    operators after the plan was approved and nothing noticed.

    The two directions are not the same offence and must not be punished alike:

    * **Added** operators -- implemented but never planned -- are a real
      provenance violation. The plan was approved on the strength of a stated
      region, precondition and invariant that the added operator never had.
      These are rejected.
    * **Dropped** operators -- planned but not implemented -- do not
      misrepresent anything as long as the *declared* provenance is corrected
      to what the code actually contains. Rejecting these threw away otherwise
      valid candidates for being less ambitious than planned, which cost every
      CHAI cell in the fft smoke campaign. They are narrowed and recorded.

    A candidate implementing none of the planned operators is still rejected:
    there is nothing left to attribute.
    """
    if plan.declined or not plan.records:
        return "", set(implemented_ids), []

    planned = plan.operator_ids
    implemented = set(implemented_ids)

    added = sorted(implemented - planned)
    dropped = sorted(planned - implemented)
    kept = implemented & planned

    if added:
        expected = ", ".join(f"T{tid}" for tid in sorted(planned))
        return (
            "Generated code implements operator(s) "
            + ", ".join(f"T{tid}" for tid in added)
            + " that were not in the approved plan. Every operator must be "
            "planned first with its own REGION, PRECONDITION and INVARIANT. "
            f"Implement exactly the planned operators ({expected}), or return a "
            "new plan that includes what you intend to add.",
            implemented,
            [],
        )

    if not kept:
        expected = ", ".join(f"T{tid}" for tid in sorted(planned))
        return (
            f"Generated code implements none of the planned operators "
            f"({expected}). There is no operator provenance to record.",
            implemented,
            [],
        )

    actions = []
    if dropped:
        actions.append(
            "narrowed declared provenance to the operators actually "
            "implemented; planned but absent: "
            + ", ".join(f"T{tid}" for tid in dropped)
        )
    return "", kept, actions
