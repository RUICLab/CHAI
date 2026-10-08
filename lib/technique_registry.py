"""
Technique Registry
===================

Central catalogue of every approximation technique known to CheckMate.

Each entry records:
  - human-readable name
  - hardware requirements  (FPU, SIMD)
  - memory footprint       (KB overhead)
  - relative compute cost   (unitless; higher = more overhead from the
    technique itself, *not* the savings it delivers)
  - applicable benchmarks/apps (None → all apps)

The gatekeeper and cost model query this registry to filter and rank
techniques before they reach the LLM.

Data-driven extension
---------------------
Technique Cards stored as JSON files under ``techniques/cards/*.json``
are loaded at import time and merged into the registry.  Card-based
entries always override built-in entries with the same ``technique_id``.
"""

import glob
import json
import os
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
#  Data structure
# ---------------------------------------------------------------------------

class TechniqueEntry:
    """One row in the registry."""

    def __init__(
        self,
        technique_id: int,
        name: str,
        *,
        requires_fpu: bool = False,
        requires_simd: bool = False,
        memory_cost_kb: float = 0.0,
        compute_cost_relative: float = 1.0,
        applicable_apps: Optional[List[str]] = None,
        description: str = "",
    ):
        self.technique_id = technique_id
        self.name = name
        self.requires_fpu = requires_fpu
        self.requires_simd = requires_simd
        self.memory_cost_kb = memory_cost_kb
        self.compute_cost_relative = compute_cost_relative
        self.applicable_apps = applicable_apps     # None → any app
        self.description = description

    def __repr__(self):
        return (f"TechniqueEntry({self.technique_id}, {self.name!r}, "
                f"fpu={self.requires_fpu}, simd={self.requires_simd}, "
                f"mem={self.memory_cost_kb} KB)")


# ---------------------------------------------------------------------------
#  Full registry — techniques 1-30
# ---------------------------------------------------------------------------

TECHNIQUE_REGISTRY: Dict[int, TechniqueEntry] = {

    # --- Original 20 techniques from the approximation_techniques list ---

    1: TechniqueEntry(
        1, "Loop Perforation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        description="Truncate loop iterations to trade accuracy for speed.",
    ),
    2: TechniqueEntry(
        2, "Precision Scaling",
        memory_cost_kb=0.0,
        compute_cost_relative=0.2,
        description="Lower numerical precision.",
    ),
    3: TechniqueEntry(
        3, "Function Memoization",
        memory_cost_kb=1.0,
        compute_cost_relative=0.5,
        description=(
            "Exact keyed reuse: return a previously computed result when the "
            "current key equals the stored key; otherwise recompute and update "
            "the key. Pointer identity is safe only while the pointed-to value "
            "cannot mutate between calls."
        ),
    ),
    4: TechniqueEntry(
        4, "Task Skipping",
        memory_cost_kb=0.0,
        compute_cost_relative=0.1,
        description="Omit non-critical tasks.",
    ),
    5: TechniqueEntry(
        5, "Approximate Data Structures",
        memory_cost_kb=0.5,
        compute_cost_relative=0.6,
        applicable_apps=["stringsearch-iclib"],
        description=(
            "Bloom filters, skip lists and similar probabilistic containers. "
            "Requires an existing lookup structure to replace."
        ),
    ),
    6: TechniqueEntry(
        6, "Probabilistic Algorithms",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        description="Randomised algorithms for faster solutions.",
    ),
    7: TechniqueEntry(
        7, "Truncated Computations",
        memory_cost_kb=0.0,
        compute_cost_relative=0.2,
        description="Stop algorithm before full convergence.",
    ),
    8: TechniqueEntry(
        8, "Simplified Models",
        memory_cost_kb=0.0,
        compute_cost_relative=0.4,
        description="Use less complex algorithms.",
    ),
    9: TechniqueEntry(
        9, "Dynamic Precision Adjustment",
        requires_fpu=True,
        memory_cost_kb=0.0,
        compute_cost_relative=0.5,
        description="Adjust precision at run-time.",
    ),
    10: TechniqueEntry(
        10, "Quantization",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        description="Reduce value range.",
    ),
    11: TechniqueEntry(
        11, "Energy-Aware Computation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.4,
        description="Modify fidelity for energy efficiency.",
    ),
    12: TechniqueEntry(
        12, "Data Approximation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        description="Compressed or simplified data.",
    ),
    13: TechniqueEntry(
        13, "Selective Re-computation",
        memory_cost_kb=0.5,
        compute_cost_relative=0.4,
        description="Reuse existing results when possible.",
    ),
    14: TechniqueEntry(
        14, "Multi-Fidelity Modeling",
        requires_fpu=True,
        memory_cost_kb=1.0,
        compute_cost_relative=0.8,
        applicable_apps=["ar-iclib", "accept-activityrec"],
        description=(
            "Combine models of varying accuracy. Requires the application to "
            "contain more than one model of the same quantity."
        ),
    ),
    15: TechniqueEntry(
        15, "Relaxed Consistency",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        applicable_apps=[],
        description=(
            "Ease consistency in distributed systems. Not applicable to "
            "single-core bare-metal benchmarks: there is no replica or "
            "coherence protocol to relax."
        ),
    ),
    16: TechniqueEntry(
        16, "Sparse Computation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        applicable_apps=["fft-iclib", "ar-iclib", "accept-activityrec"],
        description=(
            "Skip zero elements in sparse data. Requires a genuinely sparse "
            "array or spectrum, not merely a loop over dense data."
        ),
    ),
    17: TechniqueEntry(
        17, "Neural Network Pruning",
        requires_fpu=True,
        memory_cost_kb=2.0,
        compute_cost_relative=0.9,
        applicable_apps=[],
        description=(
            "Trim neural connections. pcam-iclib and coughvid-iclib are both "
            "CNNs, so this operator is relevant to them in principle. It is "
            "still left unselected for a structural reason: pruning changes "
            "tensor shapes and weight values and generally requires "
            "retraining or fine-tuning, which makes it a model-level paired "
            "transformation that must regenerate the exported C artifact. "
            "This pipeline applies function-local rewrites to a fixed "
            "artifact, so enabling pruning as a local operator would be "
            "incorrect rather than merely inapplicable."
        ),
    ),
    18: TechniqueEntry(
        18, "Approximate Joins",
        memory_cost_kb=0.5,
        compute_cost_relative=0.5,
        applicable_apps=[],
        description=(
            "Non-exact database joins. Not applicable: none of these "
            "benchmarks performs a relational join."
        ),
    ),
    19: TechniqueEntry(
        19, "Surrogate Functions",
        memory_cost_kb=0.5,
        compute_cost_relative=0.6,
        description="Replace complex functions with simpler approximants.",
    ),
    20: TechniqueEntry(
        20, "Adaptive Algorithms",
        memory_cost_kb=0.0,
        compute_cost_relative=0.4,
        description="Adjust behaviour based on input.",
    ),

    # --- Techniques 21-30: detailed CheckMate techniques ---

    21: TechniqueEntry(
        21, "Early-Exit Approximation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.2,
        description=(
            "Terminate iterative computation early when a convergence metric "
            "falls below a threshold knob."
        ),
    ),
    22: TechniqueEntry(
        22, "Spatial Downsampling",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        applicable_apps=["sobel-iclib", "accept-sobel", "susan", "segment-bm"],
        description=(
            "Process every n-th pixel in x/y, fill skipped pixels."
        ),
    ),
    23: TechniqueEntry(
        23, "Temporal Decimation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.3,
        applicable_apps=["ar-iclib", "accept-activityrec", "fft-iclib", "lqi-iclib",
                         "link-estimator"],
        description=(
            "Decimate time-series data and reconstruct via ZOH or "
            "linear interpolation."
        ),
    ),
    24: TechniqueEntry(
        24, "Magnitude-Only FFT",
        requires_fpu=True,
        memory_cost_kb=1.0,
        compute_cost_relative=0.7,
        applicable_apps=["ar-iclib", "accept-activityrec", "fft-iclib"],
        description=(
            "Discard FFT phase and keep magnitude only."
        ),
    ),
    25: TechniqueEntry(
        25, "Lazy Preprocessing",
        memory_cost_kb=0.5,
        compute_cost_relative=0.4,
        # Withdrawn for the current function-at-a-time generator.  A correct
        # lazy skip table requires validity metadata in init_search *and* a
        # checked lookup in strsearch; the old single-function example kept
        # the dominant 256-entry initialisation and saved no work.
        applicable_apps=[],
        description=(
            "Withdrawn: requires a paired producer/consumer transformation."
        ),
    ),
    26: TechniqueEntry(
        26, "Bit-Shift EWMA",
        memory_cost_kb=0.0,
        compute_cost_relative=0.1,
        # lqi-iclib was mapped here in error: its source accumulates a
        # delivery ratio with double arithmetic and contains no EWMA
        # recurrence for this operator to replace. Offering T26 there gave the
        # planner a transformation with no matching structure.
        applicable_apps=["link-estimator"],
        description=(
            "Replace an existing FP EWMA recurrence with integer bit-shift "
            "arithmetic. Requires the source to already compute an "
            "exponentially weighted moving average."
        ),
    ),
    27: TechniqueEntry(
        27, "Nibble Lookup",
        memory_cost_kb=0.016,     # 16-byte nibble LUT
        compute_cost_relative=0.2,
        # Retired from the six-benchmark campaign after a canonical bc-iclib
        # MSP430/RF_2/68uF microbenchmark.  Original byte-LUT recursion cost 5
        # cycles; stride 1/2/3/4 cost 15/11/9/8 respectively.  Every T27 point
        # was Pareto-dominated.  Its smaller table is a footprint benefit, but
        # footprint is not rewarded by the experiment's objective.
        applicable_apps=[],
        description=(
            "16-entry nibble LUT for popcount; stride knob skips nibbles."
        ),
    ),
    28: TechniqueEntry(
        28, "Hierarchical Feature Extraction",
        requires_fpu=True,
        memory_cost_kb=0.5,
        compute_cost_relative=0.6,
        # T28 assumes a cheap time-domain tier and an expensive spectral tier
        # with a classifier consuming both. ar-iclib and accept-activityrec
        # compute only mean magnitude and stddev magnitude; there is no
        # spectral tier for the gate to skip, so the operator had no matching
        # structure in either benchmark (0 declared, 0 accepted).
        applicable_apps=[],
        description=(
            "Tier-1/tier-2 feature gating to skip expensive spectral "
            "computations. Requires the source to compute a cheap feature "
            "tier and a separate expensive spectral tier."
        ),
    ),
    29: TechniqueEntry(
        29, "Radix Variation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.1,
        # radix-bm is not one of the six MSP430 benchmarks, so T29 can never be
        # selected in that experiment while still costing prompt budget and
        # offering another label to mis-apply. Its knob is also categorical:
        # only radix 2, 8, 10 and 16 have defined semantics, so an Integer
        # RANGE [2, 16] lets the optimizer choose 3..15, which are undefined.
        applicable_apps=["radix-bm"],
        description=(
            "Replace base-10 division with power-of-two bit-shift ops. "
            "KNOB IS CATEGORICAL: only radix 2, 8, 10, 16 are defined; do not "
            "declare it as an Integer RANGE."
        ),
    ),
    30: TechniqueEntry(
        30, "Pattern Segmentation",
        memory_cost_kb=0.0,
        compute_cost_relative=0.2,
        applicable_apps=["sobel-iclib", "accept-sobel", "susan", "ar-iclib",
                         "accept-activityrec", "segment-bm"],
        description=(
            "Compute at representative points and fill non-representative "
            "positions via hold-last or interpolation."
        ),
    ),
}


# ---------------------------------------------------------------------------
#  Data-driven extension — load Technique Cards from techniques/cards/*.json
# ---------------------------------------------------------------------------

_CARDS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "techniques", "cards"
)


def _load_cards_into_registry(
    registry: Dict[int, TechniqueEntry],
    cards_dir: str = _CARDS_DIR,
) -> List[str]:
    """
    Scan *cards_dir* for ``*.json`` files, validate each against the
    Technique Card schema, convert to ``TechniqueEntry`` and merge
    into *registry*.

    Returns a list of loaded card filenames (for diagnostics).
    """
    loaded: List[str] = []
    if not os.path.isdir(cards_dir):
        return loaded

    for path in sorted(glob.glob(os.path.join(cards_dir, "*.json"))):
        try:
            with open(path, "r") as f:
                card = json.load(f)
        except Exception:
            continue  # skip unparseable files silently

        # Inline validation (avoid circular import of full schema module
        # at import time — only need technique_id + feasibility fields)
        tid = card.get("technique_id")
        if not isinstance(tid, int) or tid < 1:
            continue

        feas = card.get("feasibility", {})
        entry = TechniqueEntry(
            technique_id=tid,
            name=card.get("name", f"Technique {tid}"),
            requires_fpu=feas.get("requires_fpu", False),
            requires_simd=feas.get("requires_simd", False),
            memory_cost_kb=feas.get("memory_cost_kb", 0.0),
            compute_cost_relative=card.get("cost_model", {}).get(
                "compute_cost_relative", 1.0
            ),
            applicable_apps=feas.get("applicable_apps"),
            description=card.get("description_llm", card.get("name", "")),
        )
        registry[tid] = entry
        loaded.append(os.path.basename(path))

    return loaded


# Auto-load cards at import time
_loaded_cards = _load_cards_into_registry(TECHNIQUE_REGISTRY)


def get_loaded_card_files() -> List[str]:
    """Return the list of card filenames loaded at import time."""
    return list(_loaded_cards)


def load_all_cards(cards_dir: str = _CARDS_DIR) -> list:
    """
    Load and return raw card dicts from *cards_dir* (for use by
    prompt_updater and other tools that need the full card data).
    """
    cards = []
    if not os.path.isdir(cards_dir):
        return cards
    for path in sorted(glob.glob(os.path.join(cards_dir, "*.json"))):
        try:
            with open(path, "r") as f:
                cards.append(json.load(f))
        except Exception:
            continue
    return cards


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def get_all_techniques() -> Dict[int, TechniqueEntry]:
    """Return the full registry dict."""
    return TECHNIQUE_REGISTRY


def get_applicable_techniques(app_name: str) -> Dict[int, TechniqueEntry]:
    """
    Return only the techniques whose ``applicable_apps`` list includes
    *app_name*, or those with ``applicable_apps=None`` (universally
    applicable).
    
    """
    # ``applicable_apps is None`` means "universally applicable".
    # ``applicable_apps == []`` means "applicable to nothing", which is how
    # categorically inapplicable operators (distributed-systems consistency,
    # neural-network pruning, relational joins) are excluded from every
    # benchmark in this suite. The two must not be conflated.
    return {
        tid: t for tid, t in TECHNIQUE_REGISTRY.items()
        if (t.applicable_apps is None or app_name in t.applicable_apps)
    }


def technique_names(ids: List[int]) -> List[str]:
    """Map a list of technique IDs to their human-readable names."""
    return [TECHNIQUE_REGISTRY[i].name for i in ids if i in TECHNIQUE_REGISTRY]
