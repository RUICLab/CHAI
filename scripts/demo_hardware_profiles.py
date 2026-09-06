#!/usr/bin/env python3
"""
scripts/demo_hardware_profiles.py
==================================

Demonstrates that different hardware profiles produce different feasible
technique menus for the same benchmark.

Loads three profiles and shows which techniques survive gatekeeper
filtering for ``ar-iclib``, proving that the "menu changes" when the
hardware changes.

Run from CheckMate root:
    python scripts/demo_hardware_profiles.py
"""

import sys, os

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Stub Dprint so library modules work stand-alone
import utils.utils as _u
_u.Dprint = lambda *a, **kw: None

from lib.hardware_profile import load_hardware_profile
from lib.constraint_gatekeeper import (
    filter_feasible_techniques,
    format_feasible_list,
    format_rejection_report,
)
from lib.hardware_cost_model import rank_techniques, format_ranked_list
from lib.hw_observability import (
    write_approximation_plan_report,
    write_rejection_json,
    write_constraints_txt,
)

# -----------------------------------------------------------------------

PROFILES = [
    ("config/hardware_msp430.json",  "MSP430"),
    ("config/hardware_arm_a53.json", "ARM A53"),
    ("config/hardware_custom.json",  "Custom"),
]

APP = "ar-iclib"

SEP  = "=" * 65
SEP2 = "-" * 65
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = {}   # profile_name → set of feasible technique IDs
assertions_ok = 0
assertions_fail = 0


def check(label, condition):
    global assertions_ok, assertions_fail
    if condition:
        print(f"  [{PASS}] {label}")
        assertions_ok += 1
    else:
        print(f"  [{FAIL}] {label}")
        assertions_fail += 1


# ===================== Per-profile reports ==============================

for json_path, label in PROFILES:
    hw = load_hardware_profile(json_path)
    feasible, rejection_log = filter_feasible_techniques(APP, hw)
    ranked = rank_techniques(feasible, hw)

    fids = set(feasible.keys())
    results[label] = fids

    print(f"\n{SEP}")
    print(f"  {label}  →  {APP}")
    print(f"  {hw.startup_line()}")
    print(SEP)
    print(format_ranked_list(ranked))
    print()
    if rejection_log:
        print("  Rejected:")
        print(format_rejection_report(rejection_log))
    print()

    # Write observability outputs
    rej_path = write_rejection_json(APP, hw, feasible, rejection_log)
    con_path = write_constraints_txt(APP, hw, feasible, ranked)
    audit_json, audit_md = write_approximation_plan_report(APP, hw, feasible)
    print(f"  Audit JSON: {audit_json}")
    print(f"  Audit Markdown: {audit_md}")
    print(f"  → {rej_path}")
    print(f"  → {con_path}")

# ===================== Assertions =======================================

print(f"\n{SEP}")
print("  ASSERTIONS")
print(SEP)

msp_ids = results["MSP430"]
arm_ids = results["ARM A53"]
cust_ids = results["Custom"]

# MSP430 must NOT have FPU-dependent techniques 24 and 28
check("MSP430: T24 (Magnitude-Only FFT) is ABSENT",   24 not in msp_ids)
check("MSP430: T28 (HFE) is ABSENT",                  28 not in msp_ids)

# ARM A53 MUST have them (has FPU + SIMD + plenty of RAM)
check("ARM A53: T24 (Magnitude-Only FFT) is PRESENT", 24 in arm_ids)
check("ARM A53: T28 (HFE) is PRESENT",                28 in arm_ids)

# Custom RISC-V profile: has_fpu=false → 24/28 must be absent
cust_hw = load_hardware_profile("config/hardware_custom.json")
if not cust_hw.has_fpu:
    check("Custom (no FPU): T24 is ABSENT",  24 not in cust_ids)
    check("Custom (no FPU): T28 is ABSENT",  28 not in cust_ids)
else:
    check("Custom (FPU): T24 may be present", 24 in cust_ids)
    check("Custom (FPU): T28 may be present", 28 in cust_ids)

# The menus must actually differ between MSP430 and ARM A53
check("Menu DIFFERS between MSP430 and ARM A53", msp_ids != arm_ids)

# ===================== Diff-like summary ================================

all_ids = msp_ids | arm_ids | cust_ids

print(f"\n{SEP}")
print("  TECHNIQUE DIFF ACROSS PROFILES")
print(SEP)
print(f"  {'ID':>3}  {'MSP430':^8}  {'ARM A53':^8}  {'Custom':^8}  Technique")
print(f"  {'---':>3}  {'--------':^8}  {'--------':^8}  {'--------':^8}  ---------")

for tid in sorted(all_ids):
    from lib.technique_registry import TECHNIQUE_REGISTRY as TR
    name = TR[tid].name if tid in TR else "?"
    m = "  ✓" if tid in msp_ids else "  ✗"
    a = "  ✓" if tid in arm_ids else "  ✗"
    c = "  ✓" if tid in cust_ids else "  ✗"
    # Highlight rows where at least one differs
    marker = ""
    if not (tid in msp_ids) == (tid in arm_ids) == (tid in cust_ids):
        marker = "  ← DIFFERS"
    print(f"  {tid:>3}  {m:^8}  {a:^8}  {c:^8}  {name}{marker}")

# ===================== Summary ==========================================

print(f"\n{SEP}")
print(f"  RESULTS: {assertions_ok} passed, {assertions_fail} failed")
print(SEP)
print()
print("  Files to diff for 'menu changed' proof:")
print(f"    diff outputs/LLM_CONSTRAINTS_{APP}_MSP430.txt "
      f"outputs/LLM_CONSTRAINTS_{APP}_ARM_Cortex-A53.txt")
print(f"    diff outputs/hardware_rejections_{APP}_MSP430.json "
      f"outputs/hardware_rejections_{APP}_ARM_Cortex-A53.json")
print()

sys.exit(1 if assertions_fail else 0)
