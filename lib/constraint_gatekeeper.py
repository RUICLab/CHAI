"""
Constraint Gatekeeper
======================

Filters approximation techniques against the loaded HardwareProfile,
removing any that would exceed memory, require missing hardware units
(FPU / SIMD), or are not applicable to the current benchmark.

The surviving set of technique IDs is passed downstream so that the LLM
is instructed to choose *only* from feasible techniques.
"""

from typing import Dict, List, Tuple

from lib.hardware_profile import HardwareProfile
from lib.technique_registry import (
    TechniqueEntry,
    get_applicable_techniques,
    get_all_techniques,
)
from lib.hardware_cost_model import energy_budget_headroom
from utils.utils import Dprint


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def filter_feasible_techniques(
    app_name: str,
    hw: HardwareProfile,
) -> Tuple[Dict[int, TechniqueEntry], List[str]]:
    """
    Return the subset of techniques **feasible** on *hw* for *app_name*.

    Parameters
    ----------
    app_name : str
        The benchmark / application name (e.g. ``"sobel-iclib"``).
    hw : HardwareProfile
        Target hardware description.

    Returns
    -------
    feasible : dict[int, TechniqueEntry]
        Technique ID → entry for every technique that passes all checks.
    rejection_log : list[str]
        Human-readable strings explaining each rejection (for diagnostics).
    """
    candidates = get_applicable_techniques(app_name)
    feasible: Dict[int, TechniqueEntry] = {}
    rejection_log: List[str] = []
    advisory_log: List[str] = []

    for tid, tech in candidates.items():
        rejected = False

        # --- FPU check ---
        if tech.requires_fpu and not hw.has_fpu:
            rejection_log.append(
                f"  T{tid:02d} {tech.name}: REJECTED — requires FPU "
                f"(target {hw.name} has no FPU)"
            )
            rejected = True

        # --- SIMD check ---
        if tech.requires_simd and not hw.has_simd:
            rejection_log.append(
                f"  T{tid:02d} {tech.name}: REJECTED — requires SIMD "
                f"(target {hw.name} has no SIMD)"
            )
            rejected = True

        # --- Memory check ---
        if tech.memory_cost_kb > hw.ram_kb:
            rejection_log.append(
                f"  T{tid:02d} {tech.name}: REJECTED — needs "
                f"{tech.memory_cost_kb} KB but target has {hw.ram_kb} KB RAM"
            )
            rejected = True

        # --- Energy budget: ADVISORY ONLY ---
        #
        # ``energy_budget_mj`` is in millijoules; ``estimate_cost`` returns
        # hand-assigned score units. Converting between them needs a calibration
        # constant that has never been measured on the target, so this must not
        # remove an operator: an uncalibrated constant silently deciding which
        # operators an arm may use would be a confound, not a control. It is
        # recorded as advisory metadata and used for ranking only. Promote it to
        # a hard gate only after measuring per-operator overhead on the device.
        if not rejected:
            headroom = energy_budget_headroom(tech, hw)
            if headroom is not None and headroom[0] > headroom[1]:
                advisory_log.append(
                    f"  T{tid:02d} {tech.name}: ADVISORY — overhead score "
                    f"{headroom[0]} exceeds the uncalibrated energy screen "
                    f"{headroom[1]:.4f} (from {hw.energy_budget_mj} mJ). "
                    "Not rejected; ranking metadata only."
                )

        if not rejected:
            feasible[tid] = tech

    # Summary
    n_total = len(get_all_techniques())
    n_app = len(candidates)
    n_pass = len(feasible)
    n_rej = n_app - n_pass

    Dprint(f"[Gatekeeper] {n_total} total techniques → "
           f"{n_app} applicable to '{app_name}' → "
           f"{n_pass} feasible on {hw.name} ({n_rej} rejected)")

    if advisory_log:
        Dprint("[Gatekeeper] Advisory (not rejections):")
        for line in advisory_log:
            Dprint(line)

    return feasible, rejection_log


# ---------------------------------------------------------------------------
#  Formatting helpers (used by main.py to build the LLM constraint text)
# ---------------------------------------------------------------------------

def format_feasible_list(feasible: Dict[int, TechniqueEntry]) -> str:
    """
    Produce a numbered list suitable for injection into the LLM prompt
    telling the model which techniques are allowed.
    """
    if not feasible:
        return ("No hardware-feasible techniques found. "
                "Apply only generic code-quality improvements.")

    lines = [
        "HARDWARE CONSTRAINT — You may ONLY choose from the following "
        "approximation techniques (all others are infeasible on the "
        "target hardware):\n",
    ]
    for tid in sorted(feasible):
        tech = feasible[tid]
        lines.append(f"  {tid}. {tech.name} — {tech.description}")

    lines.append("")   # trailing newline
    return "\n".join(lines)


def format_rejection_report(rejection_log: List[str]) -> str:
    """Pretty-print the rejection log for the terminal."""
    if not rejection_log:
        return "  (no techniques rejected)"
    return "\n".join(rejection_log)
