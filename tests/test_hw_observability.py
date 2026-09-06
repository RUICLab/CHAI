"""Tests for the hardware-aware discarded-plan audit report."""

import json

from lib import hw_observability
from lib.hardware_profile import HardwareProfile
from lib.technique_registry import TechniqueEntry


def test_plan_audit_preserves_clamped_entry_and_reason(tmp_path, monkeypatch):
    """A final clamp must leave a reportable copy of the discarded plan."""
    monkeypatch.setattr(hw_observability, "_OUTPUTS_DIR", str(tmp_path))
    plan_path = tmp_path / "apx_all.json"
    original_invalid_entry = {
        "function_name": "filter",
        "technique_number": 999,
        "knob_variables": {"stride": 4},
    }
    plan_path.write_text(json.dumps([
        {"function_name": "safe", "technique_number": 1},
        original_invalid_entry,
    ]))

    hardware = HardwareProfile(
        name="Test MCU",
        ram_kb=8,
        has_fpu=False,
        has_simd=False,
        energy_budget_mj=1,
        clock_mhz=16,
    )
    feasible = {1: TechniqueEntry(1, "Safe technique")}

    discarded = hw_observability.collect_infeasible_plan_entries(
        str(plan_path), "sobel-iclib", hardware, feasible
    )
    assert len(discarded) == 1
    assert discarded[0]["technique_id"] == 999
    assert discarded[0]["plan_entry"] == original_invalid_entry
    assert "unknown technique ID" in discarded[0]["reason"]

    json_report, markdown_report = hw_observability.write_approximation_plan_report(
        "sobel-iclib",
        hardware,
        feasible,
        invalid_ids={999},
        clamped_entries=discarded,
        validated_ids={1},
    )
    payload = json.loads(open(json_report).read())
    assert payload["summary"]["post_llm_rejected_reference_count"] == 1
    assert payload["summary"]["final_clamped_plan_entry_count"] == 1
    assert payload["final_clamped_plan_entries"][0]["plan_entry"] == original_invalid_entry
    assert "Final clamped plan entries" in open(markdown_report).read()


def test_plan_audit_supports_multiple_operator_ids(tmp_path):
    plan_path = tmp_path / "apx_all.json"
    plan_path.write_text(json.dumps([
        {"function_name": "filter", "technique_numbers": [1, 999]},
    ]))
    hardware = HardwareProfile(
        name="Test MCU", ram_kb=8, has_fpu=False, has_simd=False,
        energy_budget_mj=1, clock_mhz=16,
    )

    discarded = hw_observability.collect_infeasible_plan_entries(
        str(plan_path), "sobel-iclib", hardware,
        {1: TechniqueEntry(1, "Safe technique")},
    )

    assert [entry["technique_id"] for entry in discarded] == [999]

