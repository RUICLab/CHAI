# CheckMate: Automatic Technique Integration System

## What It Does

This system lets you add a brand-new approximation technique to the entire CheckMate pipeline with **a single command**. You describe what you want in plain English, and the framework:

1. Calls an LLM (Claude) to generate a structured **Technique Card** JSON
2. Validates it against a strict schema
3. Writes the card file to `techniques/cards/`
4. Updates **all 4 prompt files** that the pipeline reads at runtime
5. Auto-registers the technique in the registry (no code edits needed)
6. Runs the test suite to verify everything integrated correctly
7. Prints a report

```
┌─────────────────────────────────────────────────────┐
│  YOU TYPE:                                          │
│                                                     │
│  python -m tools.add_technique \                    │
│    --request "Add 4-bit quantization for sensors"   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  LLM generates a       │
          │  Technique Card JSON   │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Schema validation     │
          │  (20+ checks)          │
          └────────────┬───────────┘
                       │
          ┌────────────┼───────────────────┐
          │            │                   │
          ▼            ▼                   ▼
  ┌──────────────┐ ┌───────────┐ ┌──────────────────┐
  │ Card file    │ │ 4 prompt  │ │ Registry         │
  │ saved to     │ │ files     │ │ auto-loads card   │
  │ techniques/  │ │ updated   │ │ at import time    │
  │ cards/       │ │ (anchors) │ │ (no code edit)    │
  └──────────────┘ └───────────┘ └──────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  31 pytest tests run   │
          │  (all must pass)       │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  REPORT printed        │
          │  ✓ Done                │
          └────────────────────────┘
```

---

## How to Use It

### Option 1: LLM-generated (one command)

```bash
python -m tools.add_technique \
  --request "Add 4-bit quantization for integer sensor data"
```

The LLM generates the full Technique Card automatically.

### Option 2: Pre-built card (skip LLM)

```bash
python -m tools.add_technique \
  --card techniques/cards/T31_4bit_quantization.json
```

Use this when you already have a card JSON file ready.

### Option 3: Dry-run (preview only)

```bash
python -m tools.add_technique \
  --request "Add bit-plane slicing" --dry-run
```

Shows what would happen without writing any files.

### All CLI flags

| Flag | Description |
|------|-------------|
| `--request TEXT` | Natural-language description of the technique |
| `--card FILE` | Path to existing Technique Card JSON (skips LLM) |
| `--dry-run` | Preview changes without writing files |
| `--out DIR` | Override output directory (default: `techniques/cards/`) |
| `--skip-tests` | Skip the pytest verification gate |
| `--skip-smoke` | Skip optional smoke run |

---

## What Happens Step by Step

### Step A — Card Generation

When you pass `--request`, the CLI calls Claude (Anthropic API) with a structured system prompt that includes:
- The full Technique Card JSON schema
- The next available technique ID
- Rules for filling each field correctly

The LLM returns a single JSON object. If Anthropic fails, it falls back to OpenAI.

API key is loaded from `.env` in the project root (same file the main pipeline uses).

### Step B — Schema Validation

The card is validated by `lib/technique_card_schema.py` with 20+ checks:

- All 10 required top-level fields present (`schema_version`, `technique_id`, `technique_tag`, `name`, `category`, `description_llm`, `planning_guidance`, `few_shot_examples`, `feasibility`, `cost_model`)
- `schema_version` matches `"1.0"`
- `technique_id` is a positive integer
- `technique_tag` starts with `"T"`
- `description_llm` is at least 10 characters
- `planning_guidance` has `when_to_use`, `when_not_to_use`, `implementation_rules` (≥1 rule)
- `few_shot_examples` has at least 1 entry, each with `user_request` and `assistant_response_example`
- `feasibility` has `requires_fpu` (bool), `requires_simd` (bool), `memory_cost_kb` (≥0)
- `cost_model` has `compute_cost_relative` (≥0)

If validation fails, the error messages are printed and the pipeline stops.

### Step C — Artifact Generation

**C.1 — Card file written** to `techniques/cards/T<id>_<name_slug>.json`

**C.2 — Registry auto-loads** the card. `lib/technique_registry.py` scans `techniques/cards/*.json` at import time and converts each card to a `TechniqueEntry`. No code changes needed — drop a JSON file in the directory and it appears in the registry.

**C.3 — Four prompt files updated** (anchor-bounded):

| File | What gets injected |
|------|--------------------|
| `prompts/approximation_techniques.txt` | Technique description for the LLM (one numbered line) |
| `prompts/planning_step.txt` | When to use / when not to use guidance |
| `prompts/approximate_vPDG1.txt` | Implementation rules (do/don't bullets) |
| `prompts/FewShotExamples/newTechniqueExamples.txt` | Before/after C code examples |

### Step D — Test Gate

Runs `pytest tests/` (31 tests) covering:
- Card schema validation (pass + 7 failure modes)
- Registry loads cards from directory
- Prompt updates are anchor-bounded
- Few-shot updates are anchor-bounded
- Validator extracts technique IDs in all formats
- Gatekeeper applies FPU/SIMD/memory rules
- Clamp removes infeasible techniques
- Generic harness validates every card on disk
- End-to-end dry-run with mocked LLM

### Step E — Report

Prints a summary of every file created/modified.

---

## The Safety System: Anchor-Bounded Updates

The framework **never** overwrites user-maintained content. Every prompt file has anchor markers:

```
...user-maintained content stays untouched...
# === AUTO-GENERATED START ===
...only this region is replaced...
# === AUTO-GENERATED END ===
```

The few-shot examples file uses its own anchors:
```
### === AUTO-GENERATED FEW-SHOT START ===
### === AUTO-GENERATED FEW-SHOT END ===
```

`lib/prompt_updater.py` implements `inject_between_anchors()` which:
1. Reads the file
2. Splits on the start anchor
3. Splits on the end anchor
4. Replaces only the middle section
5. Writes back

If anchors are missing from a file, they are appended at the end (safe default).

---

## Technique Card Schema (v1.0)

The canonical source of truth for any technique is a single JSON file:

```json
{
  "schema_version": "1.0",
  "technique_id": 31,
  "technique_tag": "T31",
  "name": "4-Bit Quantization",
  "category": "Precision Reduction",

  "description_llm": "2-5 sentences injected into the LLM prompt...",

  "planning_guidance": {
    "when_to_use": "...",
    "when_not_to_use": "...",
    "implementation_rules": [
      "Use ONLY integer right-shift (>>) for quantization",
      "When quantize_enabled==0, output MUST be identical to original"
    ]
  },

  "few_shot_examples": [
    {
      "user_request": "Apply 4-bit quantization to the pixel processing function.",
      "expected_json_fields": {
        "knobVariables": "['quantize_enabled', 'shift_bits']",
        "knobRanges": "[{'quantize_enabled': [0, 1]}, {'shift_bits': [0, 4]}]"
      },
      "assistant_response_example": "```c\nvoid process_pixels(...) {\n    ...\n}\n```"
    }
  ],

  "feasibility": {
    "requires_fpu": false,
    "requires_simd": false,
    "memory_cost_kb": 0.0,
    "applicable_apps": null
  },

  "cost_model": {
    "compute_cost_relative": 0.15,
    "memory_cost_kb": 0.0
  }
}
```

### How each field drives the pipeline

| Card Field | Where it's used |
|------------|-----------------|
| `technique_id` / `technique_tag` | Registry key, validator extraction, prompt numbering |
| `name` | Registry entry name, prompt descriptions, report output |
| `description_llm` | Injected into `approximation_techniques.txt` — the LLM reads this when deciding what approximations to apply |
| `planning_guidance.when_to_use` | Injected into `planning_step.txt` — guides the LLM's planning step |
| `planning_guidance.when_not_to_use` | Same — tells LLM when to avoid this technique |
| `planning_guidance.implementation_rules` | Injected into `approximate_vPDG1.txt` — strict do/don't rules for code generation |
| `few_shot_examples` | Injected into `newTechniqueExamples.txt` — before/after code examples |
| `feasibility.requires_fpu` | Gatekeeper rejects technique if target hardware has no FPU |
| `feasibility.requires_simd` | Gatekeeper rejects technique if target hardware has no SIMD |
| `feasibility.memory_cost_kb` | Gatekeeper rejects if technique needs more RAM than target has |
| `feasibility.applicable_apps` | `null` = all apps; list = only those benchmarks |
| `cost_model.compute_cost_relative` | Cost model ranks techniques (cheaper first) for the LLM |

---

## How It Connects to the Existing Pipeline

```
                          ┌─────────────┐
                          │  Technique  │
                          │  Card JSON  │
                          └──────┬──────┘
                                 │
                    ┌────────────┼────────────────┐
                    │            │                │
                    ▼            ▼                ▼
          ┌──────────────┐ ┌──────────┐ ┌──────────────┐
          │ technique_   │ │ prompts/ │ │ few-shot     │
          │ registry.py  │ │ (3 files)│ │ examples     │
          │ (auto-load)  │ │ (anchors)│ │ (anchors)    │
          └──────┬───────┘ └────┬─────┘ └──────┬───────┘
                 │              │               │
                 ▼              ▼               │
          ┌──────────────┐ ┌──────────┐         │
          │ constraint_  │ │ LLM gets │◄────────┘
          │ gatekeeper   │ │ technique│
          │ (filters by  │ │ desc +   │
          │ FPU/SIMD/RAM)│ │ rules +  │
          └──────┬───────┘ │ examples │
                 │         └────┬─────┘
                 ▼              │
          ┌──────────────┐     │
          │ hardware_    │     │
          │ cost_model   │     │
          │ (ranks by    │     │
          │ cost score)  │     │
          └──────┬───────┘     │
                 │             │
                 ▼             ▼
          ┌─────────────────────────┐
          │  LLM generates          │
          │  approximated C code    │
          │  with knob variables    │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  hw_validator.py        │
          │  extracts technique IDs │
          │  (T31, Technique 31,    │
          │   #31, technique_number)│
          │  clamps infeasible ones │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  Bayesian Optimization  │
          │  tunes knob values      │
          │  (100 iterations)       │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  Fused Simulation       │
          │  (MSP430 cross-compile  │
          │   → checkpoint count)   │
          └─────────────────────────┘
```

**Key point:** The gatekeeper, cost model, and validator all read from the registry. The registry auto-loads cards. So dropping a card JSON file into `techniques/cards/` is all that's needed — no manual code edits anywhere.

---

## File Index

| File | Role |
|------|------|
| `tools/add_technique.py` | CLI entry point — orchestrates the full pipeline |
| `lib/technique_card_schema.py` | Schema definition, validation, normalization, LLM prompt builder |
| `lib/prompt_updater.py` | Anchor-bounded text injection into prompt files |
| `lib/technique_registry.py` | Data-driven registry — loads `techniques/cards/*.json` at import time |
| `lib/constraint_gatekeeper.py` | Filters techniques by hardware feasibility (reads registry) |
| `lib/hardware_cost_model.py` | Ranks techniques by cost score (reads registry) |
| `lib/hw_validator.py` | Extracts technique IDs from LLM output, clamps infeasible ones |
| `techniques/cards/*.json` | Technique Card files — the single source of truth |
| `prompts/approximation_techniques.txt` | Technique descriptions (has auto-generated section) |
| `prompts/planning_step.txt` | Planning guidance (has auto-generated section) |
| `prompts/approximate_vPDG1.txt` | Implementation rules (has auto-generated section) |
| `prompts/FewShotExamples/newTechniqueExamples.txt` | Few-shot code examples (has auto-generated section) |
| `tests/test_technique_integration.py` | 31 tests covering the full integration |
| `tests/conftest.py` | Shared test fixtures |

---

## Running Tests

```bash
cd /home/nsola5/CheckMate
source .venv/bin/activate
python -m pytest tests/ -v
```

All 31 tests should pass:

```
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_passes_for_valid_card PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_missing_sections PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_bad_schema_version PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_negative_technique_id PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_empty_implementation_rules PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_empty_few_shot PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_missing_feasibility_fields PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_schema_validation_fails_for_negative_memory PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_normalize_card_fills_defaults PASSED
tests/test_technique_integration.py::TestCardSchemaValidation::test_card_to_technique_entry PASSED
tests/test_technique_integration.py::TestRegistryLoadsCards::test_registry_loads_cards_directory PASSED
tests/test_technique_integration.py::TestRegistryLoadsCards::test_registry_skips_invalid_json PASSED
tests/test_technique_integration.py::TestRegistryLoadsCards::test_registry_skips_card_without_technique_id PASSED
tests/test_technique_integration.py::TestRegistryLoadsCards::test_load_all_cards_returns_dicts PASSED
tests/test_technique_integration.py::TestRegistryLoadsCards::test_builtin_registry_has_30_techniques PASSED
tests/test_technique_integration.py::TestPromptUpdates::test_prompt_updates_are_anchor_bounded PASSED
tests/test_technique_integration.py::TestPromptUpdates::test_inject_between_anchors_preserves_surrounding PASSED
tests/test_technique_integration.py::TestPromptUpdates::test_inject_creates_anchors_if_missing PASSED
tests/test_technique_integration.py::TestFewShotUpdates::test_fewshot_updates_are_anchor_bounded PASSED
tests/test_technique_integration.py::TestFewShotUpdates::test_render_fewshot_block_produces_examples PASSED
tests/test_technique_integration.py::TestValidatorExtraction::test_validator_extracts_generic_id_formats PASSED
tests/test_technique_integration.py::TestValidatorExtraction::test_validator_extracts_3_digit_ids PASSED
tests/test_technique_integration.py::TestValidatorExtraction::test_validator_extracts_technique_number_field PASSED
tests/test_technique_integration.py::TestGatekeeper::test_gatekeeper_applies_requires_and_memory_rules PASSED
tests/test_technique_integration.py::TestGatekeeper::test_gatekeeper_rejects_high_memory PASSED
tests/test_technique_integration.py::TestClamp::test_clamp_removes_infeasible_technique PASSED
tests/test_technique_integration.py::TestGenericCardHarness::test_all_cards_validate_against_schema PASSED
tests/test_technique_integration.py::TestGenericCardHarness::test_all_cards_in_registry PASSED
tests/test_technique_integration.py::TestGenericCardHarness::test_all_cards_have_fewshot_examples PASSED
tests/test_technique_integration.py::TestGenericCardHarness::test_prompt_files_contain_card_techniques PASSED
tests/test_technique_integration.py::TestEndToEndDryRun::test_e2e_dry_run_with_mocked_llm PASSED
```

---

## Example: What Happened When We Ran It

```bash
$ python -m tools.add_technique --request "Add 4-bit quantization for sensor data"

[add-technique] Calling LLM for technique card (next_id=32) …
[add-technique] Card validated: T32 — 4-bit Quantization for Sensor Data
  Written: techniques/cards/T32_4_bit_quantization_for_sensor_data.json
  Updated: prompts/approximation_techniques.txt
  Updated: prompts/planning_step.txt
  Updated: prompts/approximate_vPDG1.txt
  Updated: prompts/FewShotExamples/newTechniqueExamples.txt

[add-technique] Running pytest gate …
31 passed in 0.24s
  Tests: PASSED

============================================================
TECHNIQUE AUTO-INTEGRATION REPORT
============================================================
  Technique Card: T32 — 4-bit Quantization for Sensor Data
  Card file: techniques/cards/T32_4_bit_quantization_for_sensor_data.json
  Prompt: prompts/approximation_techniques.txt
  Prompt: prompts/planning_step.txt
  Prompt: prompts/approximate_vPDG1.txt
  Prompt: prompts/FewShotExamples/newTechniqueExamples.txt
  Registry: auto-loaded from card file at import time
  Tests: PASSED
============================================================
```

The LLM generated a Technique Card with:
- Sensor-specific description with linear/non-linear quantization schemes
- Planning guidance (when to use 4-bit quant, when not to)
- 6 implementation rules (packed nibble storage, bounds checking, etc.)
- A full before/after C code example with 12-bit ADC → 4-bit packed storage
- Feasibility: no FPU, no SIMD, 0.1 KB memory
- Cost: 0.2 relative compute cost

The next time `main.py` runs, the pipeline will see T32 in the registry, the gatekeeper will include/exclude it based on hardware, the LLM will read its description and rules in the prompt, and the validator will recognize `T32` / `Technique 32` in the output.
