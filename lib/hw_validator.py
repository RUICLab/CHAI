"""
Hardware Feasibility Validator (post-LLM)
==========================================

After the LLM produces its approximation output, this module extracts the
technique IDs the LLM chose and validates them against the hardware-feasible
set.  If any infeasible technique slipped through the prompt-level constraint,
the validator can:

  1. Build an error message asking the LLM to retry with only valid IDs.
  2. Hard-clamp the output by dropping any infeasible techniques.

This is the **hard guarantee** that the pipeline never applies an infeasible
technique, regardless of what the LLM returns.
"""

import json
import re
import os
from typing import Dict, List, Optional, Set, Tuple

from lib.technique_registry import TechniqueEntry
from utils.utils import Dprint


# ---------------------------------------------------------------------------
#  Extraction — pull technique IDs from various sources
# ---------------------------------------------------------------------------

def extract_technique_ids_from_text(text: str) -> Set[int]:
    """
    Scan free-form LLM output text for technique references.

    Looks for patterns like:
      - "Technique 27"  /  "Technique #27"  /  "technique_number: 27"
      - "T27"  /  "T 27"
      - "technique 27"
      - "#27"

    Supports technique IDs with 1-3 digits (T1 through T999).
    """
    ids: Set[int] = set()
    # "Technique #27", "technique 27", "Technique #27", etc.
    for m in re.finditer(r'(?i)technique[\s_#]*(\d{1,3})', text):
        ids.add(int(m.group(1)))
    # "technique_number: 42", "technique_number":42
    for m in re.finditer(r'(?i)technique_number["\s:]*(\d{1,3})', text):
        ids.add(int(m.group(1)))
    # Standalone "T27" or "T 27"
    for m in re.finditer(r'\bT\s?(\d{1,3})\b', text):
        ids.add(int(m.group(1)))
    return ids


def extract_technique_ids_from_json(json_path: str) -> Set[int]:
    """
    Extract ``technique_number`` fields from ``apx_all.json``.

    Returns an empty set if the file doesn't exist or has no such field.
    """
    ids: Set[int] = set()
    if not os.path.exists(json_path):
        return ids
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                raw_ids = entry.get("technique_numbers", entry.get("technique_number", []))
                if not isinstance(raw_ids, (list, tuple, set)):
                    raw_ids = [raw_ids] if raw_ids is not None else []
                ids.update(int(tn) for tn in raw_ids)
    except Exception:
        pass
    return ids


def extract_technique_ids_from_convo(convo: list) -> Set[int]:
    """
    Scan the conversation history list (list of (role, text) tuples)
    produced by the LLM flow for technique references.
    """
    ids: Set[int] = set()
    for role, text in convo:
        # Human messages contain the allowed catalogue and repair history.
        # Counting those IDs as model selections produces false provenance.
        if str(role).lower() == "ai":
            # Only the required provenance declaration counts. Explanations
            # often echo the full allowed catalogue and must not become a
            # false claim that every operator was applied.
            declaration = re.search(
                r"(?im)^\s*APPLIED_TECHNIQUES\s*:\s*([^\r\n]+)",
                str(text),
            )
            if declaration:
                ids |= {
                    int(match.group(1))
                    for match in re.finditer(
                        r"\bT\s*(\d{1,3})\b", declaration.group(1), re.IGNORECASE
                    )
                }
    return ids


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def validate_technique_ids(
    chosen_ids: Set[int],
    feasible: Dict[int, TechniqueEntry],
) -> Tuple[Set[int], Set[int]]:
    """
    Split *chosen_ids* into valid and invalid subsets.

    Returns
    -------
    (valid_ids, invalid_ids)
    """
    feasible_ids = set(feasible.keys())
    valid = chosen_ids & feasible_ids
    invalid = chosen_ids - feasible_ids
    return valid, invalid


def build_reprompt_error(
    invalid_ids: Set[int],
    feasible: Dict[int, TechniqueEntry],
) -> str:
    """
    Build an error string telling the LLM which technique IDs are invalid
    and restating the allowed list.  Suitable for injection via the
    ``{add_error}`` / ``prev_err`` channel.
    """
    lines = [
        "ERROR: Your response referenced infeasible technique(s) that are "
        "NOT supported on the target hardware:\n",
    ]
    for tid in sorted(invalid_ids):
        lines.append(f"  - Technique {tid} — NOT ALLOWED on this hardware")
    lines.append("")
    lines.append(
        "You may ONLY select from these technique IDs; selecting anything "
        "else is invalid:\n"
    )
    for tid in sorted(feasible):
        tech = feasible[tid]
        lines.append(f"  {tid}. {tech.name}")
    lines.append("")
    lines.append(
        "Please regenerate the approximated code using ONLY techniques "
        "from the allowed list above."
    )
    return "\n".join(lines)


def clamp_apx_json(
    json_path: str,
    feasible: Dict[int, TechniqueEntry],
) -> List[int]:
    """
    Remove entries from ``apx_all.json`` whose ``technique_number`` is
    not in the feasible set.

    Returns the list of technique IDs that were dropped.
    """
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    feasible_ids = set(feasible.keys())
    dropped: List[int] = []
    kept = []
    for entry in data:
        raw_ids = entry.get("technique_numbers", entry.get("technique_number", []))
        if not isinstance(raw_ids, (list, tuple, set)):
            raw_ids = [raw_ids] if raw_ids is not None else []
        entry_ids = {int(tn) for tn in raw_ids}
        invalid = sorted(entry_ids - feasible_ids)
        if invalid:
            dropped.extend(invalid)
            Dprint(f"[HW Validator] Clamped infeasible techniques {invalid} "
                   f"from apx_all.json")
        else:
            kept.append(entry)

    if dropped:
        with open(json_path, "w") as f:
            json.dump(kept, f, indent=2)

    return dropped
