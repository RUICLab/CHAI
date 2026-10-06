"""
Prompt Updater — anchor-bounded file updates
==============================================

Provides functions to:

1. Inject rendered text between ``# === AUTO-GENERATED START ===`` /
   ``# === AUTO-GENERATED END ===`` anchors in any text file.
2. Render technique-card data into blocks suitable for each prompt asset.

Safety guarantees:
- Only text between anchors is modified; everything above/below is kept.
- If anchors are missing they are appended at the end of the file.
- No arbitrary file writes — every update goes through
  ``inject_between_anchors``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

START_ANCHOR = "# === AUTO-GENERATED START ==="
END_ANCHOR   = "# === AUTO-GENERATED END ==="

# Few-shot variant uses different anchors to avoid collisions
FEWSHOT_START_ANCHOR = "### === AUTO-GENERATED FEW-SHOT START ==="
FEWSHOT_END_ANCHOR   = "### === AUTO-GENERATED FEW-SHOT END ==="


# ── Core anchor injection ────────────────────────────────────────────────

def inject_between_anchors(
    filepath: str,
    new_block: str,
    start_anchor: str = START_ANCHOR,
    end_anchor: str = END_ANCHOR,
) -> bool:
    """
    Replace text between *start_anchor* and *end_anchor* in *filepath*
    with *new_block*.  Returns True if the file was modified.

    If anchors are missing, they are appended at the end of the file.
    """
    if not os.path.exists(filepath):
        # Create the file with anchors + content
        with open(filepath, "w") as f:
            f.write(f"{start_anchor}\n{new_block}\n{end_anchor}\n")
        return True

    with open(filepath, "r") as f:
        content = f.read()

    if start_anchor not in content:
        # Append anchors at end
        content = content.rstrip("\n") + "\n"
        content += f"{start_anchor}\n{new_block}\n{end_anchor}\n"
        with open(filepath, "w") as f:
            f.write(content)
        return True

    # Split on anchors
    before_start = content.split(start_anchor)[0]
    after_end_parts = content.split(end_anchor)
    after_end = after_end_parts[-1] if len(after_end_parts) > 1 else ""

    updated = (
        before_start
        + start_anchor + "\n"
        + new_block + "\n"
        + end_anchor
        + after_end
    )

    with open(filepath, "w") as f:
        f.write(updated)
    return True


# ── Render functions ─────────────────────────────────────────────────────

def render_techniques_block(cards: List[Dict[str, Any]]) -> str:
    """
    Render the block for ``prompts/approximation_techniques.txt``.

    Each card produces one numbered entry identical in style to the
    static T1–T30 entries.
    """
    lines: List[str] = []
    for card in sorted(cards, key=lambda c: c["technique_id"]):
        tid = card["technique_id"]
        desc = card["description_llm"].strip()
        lines.append(f"{tid}. {card['name']}: {desc}")
    return "\n".join(lines)


def render_planning_block(cards: List[Dict[str, Any]]) -> str:
    """
    Render additional planning guidance for
    ``prompts/planning_step.txt``.
    """
    lines: List[str] = []
    for card in sorted(cards, key=lambda c: c["technique_id"]):
        pg = card.get("planning_guidance", {})
        lines.append(f"### Technique {card['technique_id']} — {card['name']}")
        lines.append(f"**When to use:** {pg.get('when_to_use', 'N/A')}")
        lines.append(f"**When NOT to use:** {pg.get('when_not_to_use', 'N/A')}")
        lines.append("")
    return "\n".join(lines)


def render_rules_block(cards: List[Dict[str, Any]]) -> str:
    """
    Render implementation rules for ``prompts/approximate_vPDG1.txt``.
    """
    lines: List[str] = []
    for card in sorted(cards, key=lambda c: c["technique_id"]):
        pg = card.get("planning_guidance", {})
        rules = pg.get("implementation_rules", [])
        if not rules:
            continue
        lines.append(
            f"=== Technique {card['technique_id']} ({card['name']}) "
            f"implementation rules ==="
        )
        for rule in rules:
            lines.append(f"  - {rule}")
        lines.append("")
    return "\n".join(lines)


def render_fewshot_block(cards: List[Dict[str, Any]]) -> str:
    """
    Render few-shot examples for
    ``prompts/FewShotExamples/newTechniqueExamples.txt``.
    """
    lines: List[str] = []
    example_num = 1
    for card in sorted(cards, key=lambda c: c["technique_id"]):
        for ex in card.get("few_shot_examples", []):
            lines.append(
                f"### Card-Generated Example {example_num}: "
                f"{card['name']} (Technique {card['technique_id']})"
            )
            lines.append("")
            lines.append(f"**User request:** {ex['user_request']}")
            lines.append("")
            resp = ex.get("assistant_response_example", "")
            lines.append(f"**Approximated code:**")
            lines.append("")
            lines.append(resp)
            lines.append("")
            example_num += 1
    return "\n".join(lines)


# ── High-level "update all prompt assets" ────────────────────────────────

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "prompts")


def update_all_prompt_assets(
    cards: List[Dict[str, Any]],
    *,
    prompt_dir: str | None = None,
    dry_run: bool = False,
) -> List[str]:
    """
    Inject auto-generated blocks into every prompt asset.

    Returns a list of files that were (or would be) modified.
    """
    base = prompt_dir or os.path.abspath(_PROMPT_DIR)
    changed: List[str] = []

    assets = [
        (
            os.path.join(base, "approximation_techniques.txt"),
            render_techniques_block(cards),
            START_ANCHOR,
            END_ANCHOR,
        ),
        (
            os.path.join(base, "planning_step.txt"),
            render_planning_block(cards),
            START_ANCHOR,
            END_ANCHOR,
        ),
        (
            os.path.join(base, "approximate_vPDG1.txt"),
            render_rules_block(cards),
            START_ANCHOR,
            END_ANCHOR,
        ),
        (
            os.path.join(base, "FewShotExamples", "newTechniqueExamples.txt"),
            render_fewshot_block(cards),
            FEWSHOT_START_ANCHOR,
            FEWSHOT_END_ANCHOR,
        ),
    ]

    for filepath, block, sa, ea in assets:
        if dry_run:
            changed.append(filepath)
            continue
        if inject_between_anchors(filepath, block, sa, ea):
            changed.append(filepath)

    return changed
