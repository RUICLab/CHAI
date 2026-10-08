"""
Hardware-Aware Observability
=============================

Writes per-run output files that record:

  1. ``outputs/LLM_CONSTRAINTS_<app>_<hwname>.txt``
     — hardware summary, feasible list, ranked list, raw LLM technique
       selection text, and final validated technique IDs.

  2. ``outputs/hardware_rejections_<app>_<hwname>.json``
     — every rejected technique with structured reason.

These files are the "proof" that the menu changed and the LLM was forced
to pick from a hardware-specific menu.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from lib.hardware_profile import HardwareProfile
from lib.technique_registry import TechniqueEntry, TECHNIQUE_REGISTRY
from lib.constraint_gatekeeper import (
    filter_feasible_techniques,
    format_feasible_list,
)
from lib.hardware_cost_model import rank_techniques, format_ranked_list


_OUTPUTS_DIR = "outputs"


def _ensure_outputs_dir():
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Rejection JSON
# ---------------------------------------------------------------------------

def _build_rejection_reason(tech: TechniqueEntry, hw: HardwareProfile) -> Optional[str]:
    """Return a single-line reason string if *tech* is infeasible on *hw*."""
    reasons = []
    if tech.requires_fpu and not hw.has_fpu:
        reasons.append("requires FPU")
    if tech.requires_simd and not hw.has_simd:
        reasons.append("requires SIMD")
    if tech.memory_cost_kb > hw.ram_kb:
        reasons.append(f"needs {tech.memory_cost_kb} KB RAM (only {hw.ram_kb} KB)")
    return "; ".join(reasons) if reasons else None


def _safe_hardware_name(hw: HardwareProfile) -> str:
    """Return a filesystem-safe, stable label for a hardware profile."""
    return hw.name.replace(" ", "_").replace("/", "_")


def _decision_reason(
    technique_id: int,
    app_name: str,
    hw: HardwareProfile,
) -> str:
    """Explain why a technique ID is not admissible for this run."""
    tech = TECHNIQUE_REGISTRY.get(technique_id)
    if tech is None:
        return "unknown technique ID (not present in the CHAI registry)"

    reasons = []
    if tech.applicable_apps is not None and app_name not in tech.applicable_apps:
        reasons.append(f"not applicable to {app_name}")
    hw_reason = _build_rejection_reason(tech, hw)
    if hw_reason:
        reasons.append(hw_reason)
    return "; ".join(reasons) if reasons else "not in the feasible technique set"


def _technique_metadata(technique_id: int) -> Dict[str, Any]:
    """Return serialisable identifier metadata, including unknown IDs."""
    tech = TECHNIQUE_REGISTRY.get(technique_id)
    return {
        "technique_id": technique_id,
        "technique_name": tech.name if tech is not None else "Unknown technique",
    }


def collect_infeasible_plan_entries(
    json_path: str,
    app_name: str,
    hw: HardwareProfile,
    feasible: Dict[int, TechniqueEntry],
) -> List[Dict[str, Any]]:
    """Snapshot infeasible ``apx_all.json`` entries before they are clamped.

    Retaining the original entry lets an experiment report what plan was
    discarded, rather than merely reporting its technique ID.
    """
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    discarded = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        raw_ids = entry.get("technique_numbers", entry.get("technique_number", []))
        if not isinstance(raw_ids, (list, tuple, set)):
            raw_ids = [raw_ids] if raw_ids is not None else []
        for raw_id in raw_ids:
            try:
                technique_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if technique_id in feasible:
                continue
            record = _technique_metadata(technique_id)
            record.update({
                "plan_entry_index": index,
                "reason": _decision_reason(technique_id, app_name, hw),
                "plan_entry": entry,
            })
            discarded.append(record)
    return discarded


def _print_rejections_terminal(
    app_name: str,
    hw: HardwareProfile,
    entries: List[dict],
    feasible_count: int,
) -> None:
    """Print a readable rejected-technique table to the terminal."""
    print()
    print("=" * 78)
    print(f"  REJECTED / INFEASIBLE TECHNIQUES — app={app_name}  hardware={hw.name}")
    print("=" * 78)

    if not entries:
        print("  (none — every registered technique was feasible)")
        print("=" * 78)
        print()
        return

    id_w, name_w = 4, 30
    print(f"  {'ID':<{id_w}} {'Technique':<{name_w}} Reason")
    print("  " + "-" * 74)
    for e in entries:
        tid = str(e["technique_id"])
        name = str(e["technique_name"])[:name_w]
        reason = e["reason"]
        print(f"  {tid:<{id_w}} {name:<{name_w}} {reason}")

    print("  " + "-" * 74)
    print(f"  Feasible: {feasible_count}   Rejected: {len(entries)}   "
          f"Total registered: {feasible_count + len(entries)}")
    print("=" * 78)
    print()


def write_rejection_json(
    app_name: str,
    hw: HardwareProfile,
    feasible: Dict[int, TechniqueEntry],
    rejection_log: List[str],
    print_to_terminal: bool = True,
) -> str:
    """
    Write ``outputs/hardware_rejections_<app>_<hwname>.json``.

    If *print_to_terminal* is True (default), also prints a rejected-technique
    table with reasons to stdout before writing the file.

    Returns the path written.
    """
    _ensure_outputs_dir()

    from lib.technique_registry import get_applicable_techniques
    candidates = get_applicable_techniques(app_name)

    entries = []
    for tid, tech in sorted(TECHNIQUE_REGISTRY.items()):
        if tid in feasible:
            continue   # only record rejected
        reason_parts = []
        # Not applicable to this app?
        if tech.applicable_apps is not None and app_name not in tech.applicable_apps:
            reason_parts.append(f"not applicable to {app_name}")
        # HW constraint violation?
        hw_reason = _build_rejection_reason(tech, hw)
        if hw_reason:
            reason_parts.append(hw_reason)
        if not reason_parts:
            # Implicitly rejected because not in candidates (app filter)
            # but no HW issue either
            reason_parts.append(f"not applicable to {app_name}")
        entries.append({
            "technique_id": tid,
            "technique_name": tech.name,
            "reason": "; ".join(reason_parts),
        })

    if print_to_terminal:
        _print_rejections_terminal(app_name, hw, entries, feasible_count=len(feasible))

    safe_hw = hw.name.replace(" ", "_").replace("/", "_")
    path = os.path.join(_OUTPUTS_DIR,
                        f"hardware_rejections_{app_name}_{safe_hw}.json")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hw.name,
        "app": app_name,
        "total_techniques": len(TECHNIQUE_REGISTRY),
        "feasible_count": len(feasible),
        "rejected_count": len(entries),
        "rejections": entries,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    if print_to_terminal:
        print(f"[hw_observability] Rejection report saved to: {path}\n")

    return path


# ---------------------------------------------------------------------------
#  Complete approximation-plan decision audit
# ---------------------------------------------------------------------------

def _catalogue_exclusions(
    app_name: str,
    hw: HardwareProfile,
    feasible: Dict[int, TechniqueEntry],
) -> List[Dict[str, Any]]:
    """List applicable techniques removed by the pre-LLM gatekeeper."""
    from lib.technique_registry import get_applicable_techniques

    exclusions = []
    for technique_id, tech in sorted(get_applicable_techniques(app_name).items()):
        if technique_id in feasible:
            continue
        exclusion = _technique_metadata(technique_id)
        exclusion["reason"] = _decision_reason(technique_id, app_name, hw)
        exclusions.append(exclusion)
    return exclusions


def _invalid_reference_records(
    invalid_ids: Set[int],
    app_name: str,
    hw: HardwareProfile,
) -> List[Dict[str, Any]]:
    """Convert validator IDs into self-explanatory audit records."""
    records = []
    for technique_id in sorted(invalid_ids):
        record = _technique_metadata(technique_id)
        record["reason"] = _decision_reason(technique_id, app_name, hw)
        records.append(record)
    return records


def write_approximation_plan_report(
    app_name: str,
    hw: HardwareProfile,
    feasible: Dict[int, TechniqueEntry],
    invalid_ids: Optional[Set[int]] = None,
    clamped_entries: Optional[List[Dict[str, Any]]] = None,
    validated_ids: Optional[Set[int]] = None,
) -> Tuple[str, str]:
    """Write machine- and presentation-readable decision reports for a run.

    The report deliberately keeps three stages separate:

    * ``pre_llm_catalogue_exclusions`` are choices withheld from the model.
    * ``post_llm_rejected_references`` are model references caught by the
      validator (which cause a retry when encountered during generation).
    * ``final_clamped_plan_entries`` are concrete JSON plan entries removed
      before evaluation.

    Returning both paths makes it easy for the pipeline to surface the audit
    trail in its normal run output.
    """
    _ensure_outputs_dir()
    invalid_ids = invalid_ids or set()
    clamped_entries = clamped_entries or []
    validated_ids = validated_ids or set()
    exclusions = _catalogue_exclusions(app_name, hw, feasible)
    invalid_records = _invalid_reference_records(invalid_ids, app_name, hw)

    safe_hw = _safe_hardware_name(hw)
    base_name = f"approximation_plan_decisions_{app_name}_{safe_hw}"
    json_path = os.path.join(_OUTPUTS_DIR, base_name + ".json")
    markdown_path = os.path.join(_OUTPUTS_DIR, base_name + ".md")

    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().isoformat(),
        "application": app_name,
        "hardware": {
            "name": hw.name,
            "ram_kb": hw.ram_kb,
            "has_fpu": hw.has_fpu,
            "has_simd": hw.has_simd,
            "energy_budget_mj": hw.energy_budget_mj,
            "clock_mhz": hw.clock_mhz,
        },
        "summary": {
            "feasible_technique_count": len(feasible),
            "pre_llm_catalogue_exclusion_count": len(exclusions),
            "post_llm_rejected_reference_count": len(invalid_records),
            "final_clamped_plan_entry_count": len(clamped_entries),
            "validated_technique_ids": sorted(validated_ids),
        },
        "pre_llm_catalogue_exclusions": exclusions,
        "post_llm_rejected_references": invalid_records,
        "final_clamped_plan_entries": clamped_entries,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(markdown_path, "w") as f:
        f.write("# CHAI approximation-plan decision audit\n\n")
        f.write(f"- Application: `{app_name}`\n")
        f.write(f"- Hardware: `{hw.name}`\n")
        f.write(f"- Generated: {payload['timestamp']}\n\n")
        f.write("## Summary\n\n")
        f.write("| Decision stage | Count | Meaning |\n")
        f.write("| --- | ---: | --- |\n")
        f.write(f"| Feasible techniques | {len(feasible)} | Available to the LLM |\n")
        f.write(f"| Pre-LLM exclusions | {len(exclusions)} | Removed before prompting |\n")
        f.write(f"| Post-LLM rejected references | {len(invalid_records)} | Caught by validator / retry path |\n")
        f.write(f"| Final clamped plan entries | {len(clamped_entries)} | Removed from `apx_all.json` before evaluation |\n\n")

        def write_records(title: str, records: List[Dict[str, Any]], entry: bool = False):
            f.write(f"## {title}\n\n")
            if not records:
                f.write("None.\n\n")
                return
            f.write("| Technique | Name | Reason |\n")
            f.write("| ---: | --- | --- |\n")
            for record in records:
                name = str(record["technique_name"]).replace("|", "\\|")
                reason = str(record["reason"]).replace("|", "\\|")
                f.write(f"| {record['technique_id']} | {name} | {reason} |\n")
            if entry:
                f.write("\nThe JSON companion file retains each original discarded plan entry.\n")
            f.write("\n")

        write_records("Pre-LLM catalogue exclusions", exclusions)
        write_records("Post-LLM rejected references", invalid_records)
        write_records("Final clamped plan entries", clamped_entries, entry=True)

    return json_path, markdown_path


# ---------------------------------------------------------------------------
#  Constraint / Proof TXT
# ---------------------------------------------------------------------------

def write_constraints_txt(
    app_name: str,
    hw: HardwareProfile,
    feasible: Dict[int, TechniqueEntry],
    ranked: List,
    llm_raw_response: str = "",
    validated_ids: Optional[Set[int]] = None,
    invalid_ids: Optional[Set[int]] = None,
    clamped_ids: Optional[List[int]] = None,
) -> str:
    """
    Write ``outputs/LLM_CONSTRAINTS_<app>_<hwname>.txt``.

    Returns the path written.
    """
    _ensure_outputs_dir()

    safe_hw = hw.name.replace(" ", "_").replace("/", "_")
    path = os.path.join(_OUTPUTS_DIR,
                        f"LLM_CONSTRAINTS_{app_name}_{safe_hw}.txt")

    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  HARDWARE-AWARE APPROXIMATION — CONSTRAINT PROOF\n")
        f.write("=" * 70 + "\n\n")

        # Section 1: Hardware summary
        f.write("HARDWARE PROFILE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {hw.startup_line()}\n")
        f.write(f"  {hw.summary()}\n\n")

        # Section 2: Feasible technique list
        f.write("FEASIBLE TECHNIQUES\n")
        f.write("-" * 40 + "\n")
        f.write(format_feasible_list(feasible) + "\n\n")

        # Section 3: Ranked list
        if ranked:
            f.write("RANKED BY COST (cheapest first)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ranked_list(ranked) + "\n\n")

        # Section 4: Raw LLM technique-selection response
        f.write("RAW LLM TECHNIQUE-SELECTION RESPONSE\n")
        f.write("-" * 40 + "\n")
        if llm_raw_response:
            f.write(llm_raw_response[:5000] + "\n")
        else:
            f.write("  (not captured — --no_llm mode or LLM not run)\n")
        f.write("\n")

        # Section 5: Validated IDs
        f.write("FINAL VALIDATED TECHNIQUE IDs\n")
        f.write("-" * 40 + "\n")
        if validated_ids is not None:
            f.write(f"  Valid  : {sorted(validated_ids)}\n")
        else:
            f.write("  (validation not performed)\n")
        if invalid_ids:
            f.write(f"  Invalid: {sorted(invalid_ids)} — "
                    "these were rejected by the post-LLM validator\n")
        if clamped_ids:
            f.write(f"  Clamped: {clamped_ids} — "
                    "these were force-removed from apx_all.json\n")
        f.write("\n")

        f.write(f"Generated: {datetime.now().isoformat()}\n")

    return path
