"""Strict normalization and validation for LLM-generated knob metadata."""

import ast
import json
import math
import re


class KnobMetadataError(ValueError):
    """Raised when a generated approximation cannot define a safe BO space."""


def _parse_container(value, field_name):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        raise KnobMetadataError(f"{field_name} must be a list or list-formatted string")

    stripped = value.strip()
    if not stripped:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
            if isinstance(parsed, (list, dict)):
                return parsed
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    raise KnobMetadataError(
        f"{field_name} is not a literal list/dict; symbolic bounds such as "
        "MODEL_SIZE are not allowed"
    )


def _as_entry_list(value, field_name):
    parsed = _parse_container(value, field_name)
    if isinstance(parsed, dict):
        return [{key: item} for key, item in parsed.items()]
    if not all(isinstance(item, dict) and len(item) == 1 for item in parsed):
        raise KnobMetadataError(f"{field_name} must contain one single-key object per knob")
    return parsed


def _coerce_number(value, knob_name, bound_name):
    if isinstance(value, bool):
        raise KnobMetadataError(f"{knob_name} {bound_name} bound cannot be boolean")
    if isinstance(value, (int, float)):
        number = value
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError as error:
            raise KnobMetadataError(
                f"{knob_name} {bound_name} bound must be numeric, got {value!r}"
            ) from error
    else:
        raise KnobMetadataError(
            f"{knob_name} {bound_name} bound must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(float(number)):
        raise KnobMetadataError(f"{knob_name} {bound_name} bound must be finite")
    return number


def normalize_knob_metadata(function_dict, *, mutate=False):
    """Return canonical variables/ranges/steps for one approximation plan.

    Accepted input may use JSON arrays, Python-literal strings, or dictionaries.
    Output is always ordered by ``knobVariables`` and contains finite numeric
    bounds only. C identifiers or expressions are intentionally rejected: BO
    cannot safely infer their values.
    """
    variables = _parse_container(function_dict.get("knobVariables", []), "knobVariables")
    if not isinstance(variables, list):
        raise KnobMetadataError("knobVariables must be a list")
    if not all(isinstance(name, str) and re.fullmatch(r"[A-Za-z_]\w*", name) for name in variables):
        raise KnobMetadataError("every knob variable must be a valid C identifier")
    if len(set(variables)) != len(variables):
        raise KnobMetadataError("knobVariables contains duplicate names")

    range_entries = _as_entry_list(function_dict.get("knobRanges", []), "knobRanges")
    step_entries = _as_entry_list(function_dict.get("knobStepSize", []), "knobStepSize")
    range_map = {next(iter(entry)): next(iter(entry.values())) for entry in range_entries}
    step_map = {next(iter(entry)): next(iter(entry.values())) for entry in step_entries}
    if len(range_map) != len(range_entries) or len(step_map) != len(step_entries):
        raise KnobMetadataError("knob metadata contains duplicate range or step entries")

    expected = set(variables)
    if set(range_map) != expected or set(step_map) != expected:
        raise KnobMetadataError(
            "knobVariables, knobRanges, and knobStepSize must contain exactly the same names"
        )

    normalized_ranges = []
    normalized_steps = []
    for name in variables:
        bounds = range_map[name]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise KnobMetadataError(f"{name} range must contain exactly [lower, upper]")

        raw_step = step_map[name]
        step_alias = str(raw_step).strip().lower()
        if step_alias in {"integer", "int"}:
            step = "Integer"
        elif step_alias in {"real", "float", "double"}:
            step = "Real"
        else:
            raise KnobMetadataError(f"{name} step type must be Integer or Real")

        lower = _coerce_number(bounds[0], name, "lower")
        upper = _coerce_number(bounds[1], name, "upper")
        if step == "Integer":
            if not float(lower).is_integer() or not float(upper).is_integer():
                raise KnobMetadataError(f"{name} Integer bounds must be whole numbers")
            lower, upper = int(lower), int(upper)
        else:
            lower, upper = float(lower), float(upper)
        if lower >= upper:
            raise KnobMetadataError(
                f"{name} lower bound must be smaller than its upper bound; "
                "fixed values are not tunable knobs"
            )

        normalized_ranges.append({name: [lower, upper]})
        normalized_steps.append({name: step})

    if mutate:
        function_dict["knobVariables"] = str(variables)
        function_dict["knobRanges"] = str(normalized_ranges)
        function_dict["knobStepSize"] = str(normalized_steps)

    return variables, normalized_ranges, normalized_steps


def drop_fixed_knobs(function_dict, *, mutate=False):
    """Remove metadata entries whose lower and upper bounds are identical.

    Local variables with fixed values are valid C constants, but they are not
    Bayesian-optimization dimensions. Small local models frequently emit them
    as knobs. Treating that recoverable metadata mistake as five JSON retries
    wastes calls and can discard an otherwise compilable approximation.

    Malformed or mismatched metadata is deliberately left to
    :func:`normalize_knob_metadata`, which will report the precise error.
    """
    variables = _parse_container(function_dict.get("knobVariables", []), "knobVariables")
    range_entries = _as_entry_list(function_dict.get("knobRanges", []), "knobRanges")
    step_entries = _as_entry_list(function_dict.get("knobStepSize", []), "knobStepSize")
    if not isinstance(variables, list):
        return []

    range_map = {next(iter(entry)): next(iter(entry.values())) for entry in range_entries}
    step_map = {next(iter(entry)): next(iter(entry.values())) for entry in step_entries}
    if set(range_map) != set(variables) or set(step_map) != set(variables):
        return []

    fixed = []
    for name in variables:
        bounds = range_map[name]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        try:
            lower = _coerce_number(bounds[0], name, "lower")
            upper = _coerce_number(bounds[1], name, "upper")
        except KnobMetadataError:
            continue
        if float(lower) == float(upper):
            fixed.append(name)

    if fixed and mutate:
        kept = [name for name in variables if name not in fixed]
        function_dict["knobVariables"] = str(kept)
        function_dict["knobRanges"] = str(
            [{name: range_map[name]} for name in kept]
        )
        function_dict["knobStepSize"] = str(
            [{name: step_map[name]} for name in kept]
        )
    return fixed


def validate_plan_entries(entries, *, mutate=False):
    """Validate every plan entry and return it for convenient pipeline use."""
    if not isinstance(entries, list):
        raise KnobMetadataError("approximation plan must be a JSON list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise KnobMetadataError(f"plan entry {index} must be an object")
        try:
            normalize_knob_metadata(entry, mutate=mutate)
        except KnobMetadataError as error:
            name = entry.get("functionName", f"entry {index}")
            raise KnobMetadataError(f"{name}: {error}") from error
    return entries

