"""
Hardware Cost Model Plugin
===========================

Estimates the *execution cost* of each surviving technique on the target
hardware and ranks them from cheapest to most expensive.

The cost score for a technique is::

    cost_score = compute_cost_relative * (100 / clock_mhz)

Lower scores are better (less overhead on the target platform).

The ranked list can be fed to the LLM so it prefers cheaper techniques
first, or it can simply inform the terminal output.
"""

from typing import Dict, List, Tuple

import os

from lib.hardware_profile import HardwareProfile
from lib.technique_registry import TechniqueEntry
from utils.utils import Dprint


# ---------------------------------------------------------------------------
#  Cost estimation
# ---------------------------------------------------------------------------

#: Relative energy cost of the RAM an operator occupies, per KB, expressed in
#: the same arbitrary units as ``compute_cost_relative``. On an intermittently
#: powered device, retained state is the dominant cost of a power cycle, so
#: memory pressure is weighted rather than treated as a rounding term.
MEMORY_ENERGY_WEIGHT = 2.0

#: Extra cost charged to an operator that needs floating point on a target with
#: no FPU. The gatekeeper removes operators that *require* an FPU, but several
#: operators are implementable either way and should be ranked accordingly.
SOFT_FP_PENALTY = 3.0


def estimate_cost(tech: TechniqueEntry, hw: HardwareProfile) -> float:
    """
    Produce a scalar cost score for applying *tech* on *hw* (lower is better).

    This is an *overhead* score, not a savings estimate: it ranks how much the
    operator's own machinery costs on this target. It deliberately does not
    claim to predict the saving an operator delivers for a particular function,
    which depends on that function's structure and is not knowable from the
    registry. Treat the ranking as a prior over which operators to try first.

    Terms:

    * clock-scaled compute overhead, so a slower part penalises expensive
      operators more;
    * memory pressure as a fraction of available RAM, weighted by
      ``MEMORY_ENERGY_WEIGHT`` because retained state dominates the cost of a
      power cycle on an intermittently powered device;
    * a soft floating-point penalty when the target has no FPU, since every
      float operation is software-emulated.
    """
    compute_factor = tech.compute_cost_relative * (100.0 / hw.clock_mhz)

    mem_pressure = (tech.memory_cost_kb / hw.ram_kb) if hw.ram_kb > 0 else 0.0

    fp_penalty = 0.0
    if not hw.has_fpu and _description_implies_float(tech):
        fp_penalty = SOFT_FP_PENALTY * tech.compute_cost_relative

    cost = compute_factor + mem_pressure * MEMORY_ENERGY_WEIGHT + fp_penalty

    return round(cost, 4)


_FLOAT_HINTS = (
    "float", "floating", "precision", "spectral", "fft", "magnitude",
    "interpolat", "probabilis", "model",
)


def _description_implies_float(tech: TechniqueEntry) -> bool:
    """Heuristic: does this operator usually involve floating-point work?"""
    text = f"{tech.name} {tech.description}".lower()
    return any(hint in text for hint in _FLOAT_HINTS)


#: Cost-score units treated as equivalent to one millijoule of budget.
#:
#: This is a **screening constant, not a calibrated energy model.** The cost
#: score above is in arbitrary units derived from hand-assigned
#: ``compute_cost_relative`` values; ``energy_budget_mj`` is in millijoules.
#: Comparing them requires a scale factor, and this one is a stated assumption
#: rather than a measurement. It exists so ``energy_budget_mj`` -- which was
#: parsed from every profile, printed in the startup banner, and then consulted
#: by nothing -- becomes load-bearing and falsifiable instead of decorative.
#:
#: Do not present energy-budget rejections as measured energy results. To
#: calibrate this properly, measure per-operator overhead on the target and
#: replace both this constant and ``compute_cost_relative``.
COST_SCORE_UNITS_PER_MJ = float(
    os.getenv("CHAI_COST_SCORE_UNITS_PER_MJ", "1.0")
)


def energy_budget_headroom(tech: TechniqueEntry, hw: HardwareProfile):
    """Return ``(score, allowance)`` for the energy screen, or ``None``.

    ``None`` means the profile declares no usable budget.
    """
    budget = getattr(hw, "energy_budget_mj", None)
    if not budget or budget <= 0:
        return None
    return estimate_cost(tech, hw), float(budget) * COST_SCORE_UNITS_PER_MJ


def exceeds_energy_budget(tech: TechniqueEntry, hw: HardwareProfile) -> bool:
    """True when an operator's overhead score exceeds the screened budget.

    See :data:`COST_SCORE_UNITS_PER_MJ` for why this is a screen rather than an
    energy measurement.
    """
    headroom = energy_budget_headroom(tech, hw)
    if headroom is None:
        return False
    score, allowance = headroom
    return score > allowance


# ---------------------------------------------------------------------------
#  Ranking
# ---------------------------------------------------------------------------

def rank_techniques(
    feasible: Dict[int, TechniqueEntry],
    hw: HardwareProfile,
) -> List[Tuple[int, str, float]]:
    """
    Rank the *feasible* techniques by ascending cost on *hw*.

    Returns
    -------
    list of (technique_id, name, cost_score)
        Sorted cheapest-first.
    """
    scored: List[Tuple[int, str, float]] = []
    for tid, tech in feasible.items():
        score = estimate_cost(tech, hw)
        scored.append((tid, tech.name, score))

    scored.sort(key=lambda x: x[2])

    Dprint("[Cost Model] Ranked techniques (cheapest first):")
    for tid, name, score in scored:
        Dprint(f"  T{tid:02d} {name}: cost = {score}")

    return scored


def select_by_opt_level(ranked_techniques, opt_level):
    """
    ranked_techniques: list sorted cheapest/lowest-cost first
    opt_level: 0-3, higher = more techniques allowed through
    """
    n = len(ranked_techniques)
    if n == 0:
        return []

    # proportion of the ranked list to expose at each level
    fractions = {0: 0.25, 1: 0.50, 2: 0.75, 3: 1.0}
    cutoff = max(1, int(n * fractions[opt_level]))

    return ranked_techniques[:cutoff]


def format_ranked_list(
    ranked: List[Tuple[int, str, float]],
) -> str:
    """
    Produce a human-readable ranked table for the terminal log.
    """
    lines = ["  Rank  ID  Technique                          Cost"]
    lines.append("  " + "-" * 52)
    for rank, (tid, name, score) in enumerate(ranked, start=1):
        lines.append(f"  {rank:>4}  {tid:>2}  {name:<35s} {score:.4f}")
    return "\n".join(lines)
    
    
