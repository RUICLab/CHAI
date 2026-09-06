"""Behavioural verification that a declared operator actually does something.

Why this exists
---------------
Operator conformance is a pile of regular expressions asking "does this code
*look* like T21?". That question is a proxy, and the proxy kept failing: an
accepted artifact declared T21 while computing an iteration budget it never
used, and another declared T21 with a knob that appears nowhere in the body.
Each time the response was a better regular expression, which only moves the
gap somewhere else.

The property actually being claimed is behavioural, not syntactic:

    if changing the knob does not change what the program does,
    the operator is not implemented.

That check cannot be fooled by naming, cannot be satisfied by a comment, and
needs no per-operator pattern. It is also cheap, because the candidate is
already built and run for the identity gate immediately beforehand.

This is a *necessary* condition, not a sufficient one: a knob that changes the
output proves something is wired up, not that the transformation is the one
that was declared. Static conformance still runs alongside it.
"""

import os
import re
import shutil
import subprocess

from utils.error_analyzer import evaluate_error, ground_truth_files
from utils.knob_metadata import KnobMetadataError, normalize_knob_metadata
from utils.utils import Dprint


class SensitivityUnavailable(RuntimeError):
    """Raised when the check cannot run, as distinct from failing."""


def _set_knob_in_source(directory, knob, value):
    """Rewrite ``knob``'s initializer inside the knob-declaration block."""
    pattern = re.compile(
        rf"(/\*\s*Knob Variables Declaration Start\s*\*/.*?"
        rf"\b{re.escape(knob)}\s*=\s*)([-\w.]+)(.*?"
        rf"/\*\s*Knob Variables Declaration End\s*\*/)",
        re.DOTALL | re.IGNORECASE,
    )
    changed = False
    for name in os.listdir(directory):
        if not name.endswith(".c"):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8", errors="ignore") as handle:
            source = handle.read()
        patched, count = pattern.subn(rf"\g<1>{value}\g<3>", source)
        if count:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(patched)
            changed = True
    return changed


def _probe_value(knob_range, identity):
    """Return the range end furthest from the identity value."""
    low, high = knob_range[0], knob_range[1]
    if abs(high - identity) >= abs(identity - low):
        return high
    return low


def verify_knobs_have_effect(app_name, candidate_dir, plan_entry,
                             reference_dir="target", max_knobs=4):
    """Return ``(ok, message, per_knob)`` for the candidate's declared knobs.

    Each knob is driven to the end of its range furthest from its identity
    value; the candidate is rebuilt and re-run; and the resulting error is
    compared with the identity error. A knob that moves nothing is inert.
    """
    if not os.path.isdir(candidate_dir):
        raise SensitivityUnavailable(f"no candidate directory {candidate_dir}")

    try:
        variables, ranges, _steps = normalize_knob_metadata(plan_entry)
    except KnobMetadataError as error:
        raise SensitivityUnavailable(f"unusable knob metadata: {error}")
    if not variables:
        raise SensitivityUnavailable("candidate declares no knobs")

    artifacts = ground_truth_files(app_name)
    if not artifacts:
        raise SensitivityUnavailable(f"no reference artifact for {app_name}")
    for name in artifacts:
        source = os.path.join(reference_dir, name)
        if not os.path.exists(source):
            raise SensitivityUnavailable(f"reference {source} not generated")
        shutil.copy2(source, os.path.join(candidate_dir, name))

    probe_dir = candidate_dir.rstrip("/") + "__sensitivity"
    inert = []
    per_knob = {}

    for index, knob in enumerate(variables[:max_knobs]):
        bounds = ranges[index][knob]
        identity = _declared_value(candidate_dir, knob)
        if identity is None:
            per_knob[knob] = "no initializer found"
            continue
        probe = _probe_value(bounds, identity)
        if probe == identity:
            # A single-point range has nothing to vary; the plan contract
            # rejects these, but a baseline arm has no plan contract.
            inert.append(knob)
            per_knob[knob] = f"range [{bounds[0]}, {bounds[1]}] cannot vary"
            continue

        if os.path.exists(probe_dir):
            shutil.rmtree(probe_dir)
        shutil.copytree(candidate_dir, probe_dir)
        try:
            if not _set_knob_in_source(probe_dir, knob, probe):
                per_knob[knob] = "initializer not found in source"
                continue
            try:
                probed_error = evaluate_error(app_name, probe_dir)
            except Exception as failure:
                # Crashing at an extreme setting still proves the knob is wired
                # into behaviour, which is what this check asks.
                per_knob[knob] = f"changes behaviour (run failed: {failure})"
                continue

            identity_error = _identity_error(app_name, candidate_dir)
            if identity_error is None:
                raise SensitivityUnavailable("identity run did not produce an error")

            if abs(probed_error - identity_error) <= 1e-12:
                inert.append(knob)
                per_knob[knob] = (
                    f"inert: {knob}={identity} and {knob}={probe} both give "
                    f"error {identity_error}"
                )
            else:
                per_knob[knob] = (
                    f"effective: error {identity_error} -> {probed_error}"
                )
        finally:
            if os.path.exists(probe_dir):
                shutil.rmtree(probe_dir)

    Dprint(f"[knob sensitivity] {app_name}: {per_knob}")

    if inert and len(inert) == len([k for k in per_knob]):
        return False, (
            "None of the declared knobs changes the program's behaviour: "
            + "; ".join(f"{name} ({per_knob[name]})" for name in inert)
            + ". A knob that is declared, or computed into a local, but never "
            "reaches a loop bound, branch condition, or arithmetic path does "
            "not implement any approximation operator. Wire the knob into the "
            "computation it is supposed to control."
        ), per_knob

    if inert:
        return True, (
            "inert knob(s) present: " + ", ".join(inert)
        ), per_knob

    return True, "", per_knob


def _declared_value(directory, knob):
    """Read the knob's current initializer from the candidate source."""
    pattern = re.compile(
        rf"/\*\s*Knob Variables Declaration Start\s*\*/.*?"
        rf"\b{re.escape(knob)}\s*=\s*([-\w.]+)",
        re.DOTALL | re.IGNORECASE,
    )
    for name in os.listdir(directory):
        if not name.endswith(".c"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8",
                  errors="ignore") as handle:
            match = pattern.search(handle.read())
        if match:
            text = match.group(1)
            try:
                return float(text) if "." in text else int(text)
            except ValueError:
                return None
    return None


_identity_cache = {}


def _identity_error(app_name, candidate_dir):
    """Measure (once per candidate) the error at the declared identity point."""
    key = (app_name, os.path.abspath(candidate_dir))
    if key not in _identity_cache:
        try:
            _identity_cache[key] = evaluate_error(app_name, candidate_dir)
        except Exception:
            _identity_cache[key] = None
    return _identity_cache[key]


def reset_cache():
    _identity_cache.clear()
