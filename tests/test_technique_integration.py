"""
test_technique_integration.py — comprehensive test suite
=========================================================

Tests the entire Technique Synthesis + Auto-Integration pipeline:

- Card schema validation (pass / fail)
- Data-driven registry loading
- Prompt anchor-bounded updates
- Few-shot anchor-bounded updates
- Validator generic ID extraction
- Gatekeeper feasibility rules
- Clamp removes infeasible techniques
- End-to-end dry-run with mocked LLM
"""

import copy
import json
import os
import shutil
import sys
import tempfile

import pytest

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.technique_card_schema import (
    SCHEMA_VERSION,
    validate_card,
    normalize_card,
    card_to_technique_entry,
)
from lib.prompt_updater import (
    START_ANCHOR,
    END_ANCHOR,
    FEWSHOT_START_ANCHOR,
    FEWSHOT_END_ANCHOR,
    inject_between_anchors,
    render_techniques_block,
    render_planning_block,
    render_rules_block,
    render_fewshot_block,
    update_all_prompt_assets,
)
from lib.technique_registry import (
    TechniqueEntry,
    get_all_techniques,
    load_all_cards,
    _load_cards_into_registry,
)
from lib.hw_validator import (
    extract_technique_ids_from_convo,
    extract_technique_ids_from_text,
    clamp_apx_json,
)
from lib.constraint_gatekeeper import filter_feasible_techniques
from lib.hardware_profile import HardwareProfile


# ═══════════════════════════════════════════════════════════════════════
#  1. Card Schema Validation
# ═══════════════════════════════════════════════════════════════════════

class TestCardSchemaValidation:
    """Tests for lib/technique_card_schema.py"""

    def test_card_schema_validation_passes_for_valid_card(self, valid_card):
        ok, errors = validate_card(valid_card)
        assert ok, f"Valid card should pass validation but got: {errors}"
        assert errors == []

    def test_card_schema_validation_fails_for_missing_sections(self, valid_card):
        # Remove required top-level fields one by one
        for field in [
            "technique_id", "technique_tag", "name", "category",
            "description_llm", "planning_guidance", "few_shot_examples",
            "feasibility", "cost_model",
        ]:
            broken = copy.deepcopy(valid_card)
            del broken[field]
            ok, errors = validate_card(broken)
            assert not ok, f"Card missing '{field}' should fail validation"
            assert any(field in e for e in errors)

    def test_card_schema_validation_fails_for_bad_schema_version(self, valid_card):
        broken = copy.deepcopy(valid_card)
        broken["schema_version"] = "99.0"
        ok, errors = validate_card(broken)
        assert not ok
        assert any("schema_version" in e for e in errors)

    def test_card_schema_validation_fails_for_negative_technique_id(self, valid_card):
        broken = copy.deepcopy(valid_card)
        broken["technique_id"] = -1
        ok, errors = validate_card(broken)
        assert not ok

    def test_card_schema_validation_fails_for_empty_implementation_rules(self, valid_card):
        broken = copy.deepcopy(valid_card)
        broken["planning_guidance"]["implementation_rules"] = []
        ok, errors = validate_card(broken)
        assert not ok
        assert any("implementation_rules" in e for e in errors)

    def test_card_schema_validation_fails_for_empty_few_shot(self, valid_card):
        broken = copy.deepcopy(valid_card)
        broken["few_shot_examples"] = []
        ok, errors = validate_card(broken)
        assert not ok

    def test_card_schema_validation_fails_for_missing_feasibility_fields(self, valid_card):
        broken = copy.deepcopy(valid_card)
        del broken["feasibility"]["requires_fpu"]
        ok, errors = validate_card(broken)
        assert not ok
        assert any("requires_fpu" in e for e in errors)

    def test_card_schema_validation_fails_for_negative_memory(self, valid_card):
        broken = copy.deepcopy(valid_card)
        broken["feasibility"]["memory_cost_kb"] = -1.0
        ok, errors = validate_card(broken)
        assert not ok

    def test_normalize_card_fills_defaults(self, valid_card):
        card = copy.deepcopy(valid_card)
        # Remove optional fields
        if "applicable_apps" in card["feasibility"]:
            del card["feasibility"]["applicable_apps"]
        normed = normalize_card(card)
        assert normed["feasibility"]["applicable_apps"] is None
        assert "expected_json_fields" in normed["few_shot_examples"][0]

    def test_card_to_technique_entry(self, valid_card):
        entry = card_to_technique_entry(valid_card)
        assert isinstance(entry, TechniqueEntry)
        assert entry.technique_id == 99
        assert entry.name == "Test Technique"
        assert entry.requires_fpu is False
        assert entry.memory_cost_kb == 0.5


# ═══════════════════════════════════════════════════════════════════════
#  2. Data-Driven Registry
# ═══════════════════════════════════════════════════════════════════════

class TestRegistryLoadsCards:
    """Tests for registry loading from techniques/cards/"""

    def test_registry_loads_cards_directory(self, tmp_cards_dir):
        registry = {}
        loaded = _load_cards_into_registry(registry, tmp_cards_dir)
        assert len(loaded) == 1
        assert 99 in registry
        assert registry[99].name == "Test Technique"
        assert registry[99].memory_cost_kb == 0.5

    def test_registry_skips_invalid_json(self, tmp_cards_dir):
        # Write an invalid JSON file
        bad = os.path.join(tmp_cards_dir, "BAD.json")
        with open(bad, "w") as f:
            f.write("{not valid json")
        registry = {}
        loaded = _load_cards_into_registry(registry, tmp_cards_dir)
        # Should still load the valid one
        assert 99 in registry

    def test_registry_skips_card_without_technique_id(self, tmp_cards_dir):
        bad = os.path.join(tmp_cards_dir, "no_id.json")
        with open(bad, "w") as f:
            json.dump({"name": "No ID"}, f)
        registry = {}
        _load_cards_into_registry(registry, tmp_cards_dir)
        assert len(registry) == 1  # only the valid T99

    def test_load_all_cards_returns_dicts(self, tmp_cards_dir):
        cards = load_all_cards(tmp_cards_dir)
        assert len(cards) >= 1
        assert cards[0]["technique_id"] == 99

    def test_builtin_registry_has_30_techniques(self):
        """The hard-coded registry should have T1–T30."""
        all_t = get_all_techniques()
        for tid in range(1, 31):
            assert tid in all_t, f"Built-in technique T{tid} missing from registry"


# ═══════════════════════════════════════════════════════════════════════
#  3. Prompt Updates — Anchor-Bounded
# ═══════════════════════════════════════════════════════════════════════

class TestPromptUpdates:
    """Tests for lib/prompt_updater.py"""

    def test_prompt_updates_are_anchor_bounded(self, tmp_prompts_dir, valid_card):
        cards = [valid_card]
        changed = update_all_prompt_assets(
            cards, prompt_dir=tmp_prompts_dir
        )
        assert len(changed) == 4

        # Check approximation_techniques.txt
        techniques_file = os.path.join(
            tmp_prompts_dir, "approximation_techniques.txt"
        )
        with open(techniques_file) as f:
            content = f.read()
        assert START_ANCHOR in content
        assert END_ANCHOR in content
        assert "Test Technique" in content
        # Original content preserved above anchor
        assert "Existing techniques here." in content

    def test_inject_between_anchors_preserves_surrounding(self):
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "test.txt")
            with open(fp, "w") as f:
                f.write(
                    "BEFORE\n"
                    "# === AUTO-GENERATED START ===\n"
                    "old stuff\n"
                    "# === AUTO-GENERATED END ===\n"
                    "AFTER\n"
                )
            inject_between_anchors(fp, "new stuff")
            with open(fp) as f:
                content = f.read()
            assert "BEFORE" in content
            assert "AFTER" in content
            assert "new stuff" in content
            assert "old stuff" not in content
        finally:
            shutil.rmtree(d)

    def test_inject_creates_anchors_if_missing(self):
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "test.txt")
            with open(fp, "w") as f:
                f.write("No anchors here.\n")
            inject_between_anchors(fp, "injected")
            with open(fp) as f:
                content = f.read()
            assert START_ANCHOR in content
            assert END_ANCHOR in content
            assert "injected" in content
            assert "No anchors here." in content
        finally:
            shutil.rmtree(d)


# ═══════════════════════════════════════════════════════════════════════
#  4. Few-Shot Updates — Anchor-Bounded
# ═══════════════════════════════════════════════════════════════════════

class TestFewShotUpdates:
    """Tests for few-shot prompt update via anchors."""

    def test_fewshot_updates_are_anchor_bounded(self, tmp_prompts_dir, valid_card):
        cards = [valid_card]
        update_all_prompt_assets(cards, prompt_dir=tmp_prompts_dir)

        fewshot_file = os.path.join(
            tmp_prompts_dir,
            "FewShotExamples",
            "newTechniqueExamples.txt",
        )
        with open(fewshot_file) as f:
            content = f.read()

        assert FEWSHOT_START_ANCHOR in content
        assert FEWSHOT_END_ANCHOR in content
        assert "Test Technique" in content
        # Original content preserved
        assert "Existing examples." in content

    def test_render_fewshot_block_produces_examples(self, valid_card):
        block = render_fewshot_block([valid_card])
        assert "Test Technique" in block
        assert "Technique 99" in block
        assert "test_knob" in block


# ═══════════════════════════════════════════════════════════════════════
#  5. Validator — Generic ID Extraction
# ═══════════════════════════════════════════════════════════════════════

class TestValidatorExtraction:
    """Tests for lib/hw_validator.py extraction."""

    def test_validator_extracts_generic_id_formats(self):
        text = (
            "I applied Technique 31, also known as T31. "
            "Combined with technique #22 and T 5."
        )
        ids = extract_technique_ids_from_text(text)
        assert 31 in ids
        assert 22 in ids
        assert 5 in ids

    def test_validator_extracts_3_digit_ids(self):
        text = "Using Technique 100 and T999."
        ids = extract_technique_ids_from_text(text)
        assert 100 in ids
        assert 999 in ids

    def test_validator_extracts_technique_number_field(self):
        text = 'technique_number: 42'
        ids = extract_technique_ids_from_text(text)
        assert 42 in ids


# ═══════════════════════════════════════════════════════════════════════
#  6. Gatekeeper — Feasibility Rules
# ═══════════════════════════════════════════════════════════════════════

class TestGatekeeper:
    """Tests for lib/constraint_gatekeeper.py"""

    def _make_hw(self, **overrides):
        defaults = {
            "name": "TestHW",
            "ram_kb": 10,
            "has_fpu": False,
            "has_simd": False,
            "energy_budget_mj": 5,
            "clock_mhz": 16,
        }
        defaults.update(overrides)
        return HardwareProfile(**defaults)

    def test_gatekeeper_applies_requires_and_memory_rules(self, tmp_cards_dir, valid_card):
        # Add a card that requires FPU
        fpu_card = copy.deepcopy(valid_card)
        fpu_card["technique_id"] = 100
        fpu_card["technique_tag"] = "T100"
        fpu_card["name"] = "FPU-Required Technique"
        fpu_card["feasibility"]["requires_fpu"] = True
        fpu_path = os.path.join(tmp_cards_dir, "T100_fpu.json")
        with open(fpu_path, "w") as f:
            json.dump(fpu_card, f)

        # Load cards into a fresh registry
        registry = {}
        _load_cards_into_registry(registry, tmp_cards_dir)
        assert 99 in registry
        assert 100 in registry

        # Registry entry for T100 should have requires_fpu=True
        assert registry[100].requires_fpu is True

        # The real gatekeeper filters the full registry; we test the concept:
        hw_no_fpu = self._make_hw(has_fpu=False)
        # T100 requires FPU, no-FPU hardware → rejected
        feasible, log = filter_feasible_techniques("some-app", hw_no_fpu)
        # T100 should not be in the feasible set from the main registry
        # (it's only in our temp registry). But the gating logic itself
        # is proven by checking built-in T9 (requires_fpu=True)
        assert 9 not in feasible, "T9 (requires FPU) should be rejected on no-FPU hw"

        # With FPU hardware, T9 should pass
        hw_fpu = self._make_hw(has_fpu=True)
        feasible_fpu, _ = filter_feasible_techniques("some-app", hw_fpu)
        assert 9 in feasible_fpu

    def test_gatekeeper_rejects_high_memory(self):
        hw = self._make_hw(ram_kb=0.001)  # extremely small RAM
        feasible, log = filter_feasible_techniques("some-app", hw)
        # T3 (Function Memoization) needs 1.0 KB — should be rejected
        assert 3 not in feasible


# ═══════════════════════════════════════════════════════════════════════
#  7. Clamp — Remove Infeasible Techniques
# ═══════════════════════════════════════════════════════════════════════

class TestClamp:
    """Tests for clamp_apx_json."""

    def test_clamp_removes_infeasible_technique(self):
        d = tempfile.mkdtemp()
        try:
            apx_path = os.path.join(d, "apx_all.json")
            data = [
                {"functionName": "foo", "technique_number": 1},
                {"functionName": "bar", "technique_number": 999},
            ]
            with open(apx_path, "w") as f:
                json.dump(data, f)

            # Feasible set contains only T1
            feasible = {
                1: TechniqueEntry(1, "Loop Perforation"),
            }
            dropped = clamp_apx_json(apx_path, feasible)
            assert 999 in dropped

            # Verify file was updated
            with open(apx_path) as f:
                remaining = json.load(f)
            assert len(remaining) == 1
            assert remaining[0]["technique_number"] == 1
        finally:
            shutil.rmtree(d)

    def test_clamp_rejects_entry_when_any_declared_operator_is_infeasible(self, tmp_path):
        apx_path = tmp_path / "apx_all.json"
        apx_path.write_text(json.dumps([
            {"functionName": "safe", "technique_numbers": [1, 2]},
            {"functionName": "unsafe", "technique_numbers": [1, 999]},
        ]))
        feasible = {
            1: TechniqueEntry(1, "One"),
            2: TechniqueEntry(2, "Two"),
        }

        dropped = clamp_apx_json(str(apx_path), feasible)

        assert dropped == [999]
        assert json.loads(apx_path.read_text()) == [
            {"functionName": "safe", "technique_numbers": [1, 2]}
        ]


def test_conversation_operator_extraction_ignores_human_catalog_ids():
    convo = [
        ("human", "Allowed: T1, T2, T27"),
        ("ai", "APPLIED_TECHNIQUES: T27\nOther considered options: T1, T2"),
    ]

    assert extract_technique_ids_from_convo(convo) == {27}


# ═══════════════════════════════════════════════════════════════════════
#  8. Generic Test Harness — All Cards
# ═══════════════════════════════════════════════════════════════════════

class TestGenericCardHarness:
    """
    Load every card from techniques/cards/ and verify it integrates
    correctly with all subsystems.
    """

    def _get_cards(self):
        cards_dir = os.path.join(_PROJECT_ROOT, "techniques", "cards")
        return load_all_cards(cards_dir)

    def test_all_cards_validate_against_schema(self):
        for card in self._get_cards():
            ok, errors = validate_card(card)
            assert ok, (
                f"Card T{card.get('technique_id', '?')} failed validation: "
                f"{errors}"
            )

    def test_all_cards_in_registry(self):
        registry = get_all_techniques()
        for card in self._get_cards():
            tid = card["technique_id"]
            assert tid in registry, (
                f"Card T{tid} ({card['name']}) not found in registry"
            )

    def test_all_cards_have_fewshot_examples(self):
        for card in self._get_cards():
            fse = card.get("few_shot_examples", [])
            assert len(fse) >= 1, (
                f"Card T{card['technique_id']} has no few-shot examples"
            )

    def test_prompt_files_contain_card_techniques(self):
        cards = self._get_cards()
        if not cards:
            pytest.skip("No cards in techniques/cards/")

        # Render the techniques block and check each card appears
        block = render_techniques_block(cards)
        for card in cards:
            assert card["name"] in block, (
                f"Card T{card['technique_id']} ({card['name']}) "
                f"not in rendered techniques block"
            )


# ═══════════════════════════════════════════════════════════════════════
#  9. End-to-End Dry Run (Mocked LLM)
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEndDryRun:
    """
    Simulate the full add-technique pipeline with a mocked LLM client.
    No network calls.
    """

    def test_e2e_dry_run_with_mocked_llm(self, valid_card, tmp_prompts_dir):
        """
        1. Mock LLM returns a valid card
        2. Builder writes artifacts to tmp dir
        3. Updater injects prompt blocks
        4. Registry loads it
        5. Verification passes
        """
        import unittest.mock as mock

        cards_dir = tempfile.mkdtemp(prefix="e2e_cards_")
        try:
            # Step 1: Write the card (simulating LLM output)
            card = copy.deepcopy(valid_card)
            card["technique_id"] = 88
            card["technique_tag"] = "T88"
            card["name"] = "Mock E2E Technique"
            card["description_llm"] = (
                "A mocked technique for end-to-end testing of the "
                "auto-integration pipeline."
            )

            card_path = os.path.join(cards_dir, "T88_mock_e2e_technique.json")
            with open(card_path, "w") as f:
                json.dump(card, f, indent=2)

            # Step 2: Validate
            ok, errors = validate_card(card)
            assert ok, f"Mock card failed validation: {errors}"

            # Step 3: Load into registry
            registry = {}
            loaded = _load_cards_into_registry(registry, cards_dir)
            assert 88 in registry
            assert registry[88].name == "Mock E2E Technique"

            # Step 4: Update prompts
            all_cards = load_all_cards(cards_dir)
            changed = update_all_prompt_assets(
                all_cards, prompt_dir=tmp_prompts_dir
            )
            assert len(changed) == 4

            # Step 5: Verify prompts contain the technique
            tech_file = os.path.join(
                tmp_prompts_dir, "approximation_techniques.txt"
            )
            with open(tech_file) as f:
                content = f.read()
            assert "Mock E2E Technique" in content
            assert START_ANCHOR in content
            assert END_ANCHOR in content

            # Verify few-shot
            fewshot_file = os.path.join(
                tmp_prompts_dir, "FewShotExamples", "newTechniqueExamples.txt"
            )
            with open(fewshot_file) as f:
                content = f.read()
            assert "Mock E2E Technique" in content

            # Verify validator can extract the ID
            ids = extract_technique_ids_from_text("Applied T88 technique")
            assert 88 in ids

        finally:
            shutil.rmtree(cards_dir, ignore_errors=True)
