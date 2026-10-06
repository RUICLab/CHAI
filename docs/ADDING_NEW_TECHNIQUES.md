# Adding New Approximation Techniques to CheckMate

## Quick Start — Automated Integration (Recommended)

CheckMate includes a **Technique Synthesis + Auto-Integration** CLI that adds any new approximation technique from a single natural-language request. It calls the LLM to produce a Technique Card, validates it, and updates **all** pipeline artifacts automatically.

```bash
# LLM-generated (one command does everything)
python -m tools.add_technique \
  --request "Add 4-bit quantization for integer sensor data"

# Pre-built card (skip LLM call)
python -m tools.add_technique \
  --card techniques/cards/T31_4bit_quantization.json

# Dry-run — preview changes without writing anything
python -m tools.add_technique \
  --request "Add bit-plane slicing" --dry-run
```

**What happens behind the scenes:**

| Step | What | Files touched |
|------|------|---------------|
| A | LLM generates a **Technique Card** JSON | `techniques/cards/T<N>_<name>.json` |
| B | Card validated against canonical schema | — |
| C.1 | Card file written | `techniques/cards/` |
| C.2 | Technique description injected into prompts | `prompts/approximation_techniques.txt` |
| C.3 | Planning guidance injected | `prompts/planning_step.txt` |
| C.4 | Implementation rules injected | `prompts/approximate_vPDG1.txt` |
| C.5 | Few-shot examples injected | `prompts/FewShotExamples/newTechniqueExamples.txt` |
| C.6 | Registry auto-loads card at import time | `lib/technique_registry.py` (no edit needed) |
| D | pytest gate runs | `tests/test_technique_integration.py` |
| E | Report printed | stdout |

**Safety:** All prompt updates are **anchor-bounded** — the framework only writes between `# === AUTO-GENERATED START ===` / `# === AUTO-GENERATED END ===` markers. User-maintained content is never touched.

---

## Architecture: Technique Card → Full Pipeline

```
                    ┌──────────────────────────┐
                    │  USER REQUEST (text)      │
                    │  "Add 4-bit quantization" │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  LLM generates           │
                    │  Technique Card JSON      │
                    │  (strict schema v1.0)     │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  Schema Validation       │
                    │  technique_card_schema.py │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
   ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
   │ Card File        │ │ Prompt       │ │ Registry      │
   │ techniques/      │ │ Updater      │ │ auto-loads    │
   │ cards/*.json     │ │ (anchors)    │ │ at import     │
   └─────────────────┘ └──────────────┘ └───────┬───────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │ Existing Pipeline      │
                                   │ • Gatekeeper (auto)    │
                                   │ • Cost Model (auto)    │
                                   │ • Validator (auto)     │
                                   │ • LLM prompt injection │
                                   │ • BayesOpt tuning      │
                                   └────────────────────────┘
```

---

## Technique Card Schema (v1.0)

Every technique is defined by a single JSON file in `techniques/cards/`. The schema is enforced by `lib/technique_card_schema.py`.

```json
{
  "schema_version": "1.0",
  "technique_id": 31,
  "technique_tag": "T31",
  "name": "4-Bit Quantization",
  "category": "Precision Reduction",
  "description_llm": "...(injected into LLM prompt)...",
  "planning_guidance": {
    "when_to_use": "...",
    "when_not_to_use": "...",
    "implementation_rules": ["Rule 1", "Rule 2"]
  },
  "few_shot_examples": [{
    "user_request": "Apply 4-bit quantization to the pixel processing function.",
    "expected_json_fields": {"knobVariables": "...", "knobRanges": "..."},
    "assistant_response_example": "```c\nvoid foo() {...}\n```"
  }],
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

**Required fields** — validation fails if any are missing:
- `schema_version`, `technique_id`, `technique_tag`, `name`, `category`
- `description_llm` (≥10 chars)
- `planning_guidance` with `when_to_use`, `when_not_to_use`, `implementation_rules` (≥1 rule)
- `few_shot_examples` (≥1 entry, each with `user_request` + `assistant_response_example`)
- `feasibility` with `requires_fpu`, `requires_simd`, `memory_cost_kb`
- `cost_model` with `compute_cost_relative`

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `lib/technique_card_schema.py` | Schema definition, `validate_card()`, `normalize_card()`, LLM prompt builder |
| `lib/prompt_updater.py` | Anchor-bounded injection into prompt/few-shot files |
| `lib/technique_registry.py` | Data-driven: loads `techniques/cards/*.json` at import time |
| `tools/add_technique.py` | CLI entry point: orchestrates card→validate→write→update→test |
| `tests/test_technique_integration.py` | 31 tests covering schema, registry, prompts, validator, gatekeeper, clamp, E2E |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Manual Process (Legacy)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. approximation_techniques.txt   ← natural-language description  │
│     (read by LLM at prompt time)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  2. technique_registry.py          ← TechniqueEntry(...)           │
│     hardware requirements, memory, cost, applicable apps           │
├─────────────────────────────────────────────────────────────────────┤
│  3. constraint_gatekeeper.py       ← auto-filters using registry   │
│     removes techniques that need FPU/SIMD/too much RAM             │
├─────────────────────────────────────────────────────────────────────┤
│  4. hardware_cost_model.py         ← auto-ranks by cost            │
│     cheapest techniques shown first to LLM                         │
├─────────────────────────────────────────────────────────────────────┤
│  5. LLM (Claude)                   ← reads technique descriptions  │
│     + feasible list → generates approximated C code with knobs     │
├─────────────────────────────────────────────────────────────────────┤
│  6. bo.py (BayesOpt)               ← tunes knobs automatically     │
│     100 iterations, evaluates error + checkpoints per config       │
├─────────────────────────────────────────────────────────────────────┤
│  7. Fused simulator                ← measures cycles/checkpoints   │
│     cross-compiles → simulates on MSP430 → returns cycle count     │
└─────────────────────────────────────────────────────────────────────┘
```

Once a technique is defined in steps 1–2, **everything else is automatic**. The gatekeeper filters it, the cost model ranks it, the LLM generates code for it, and BayesOpt tunes it.

---

## Step-by-Step: Adding a New Technique

### Step 1 — Write the Natural-Language Description

**File:** `prompts/approximation_techniques.txt`

Append one line at the end (before the `# === AUTO-GENERATED` markers). The format is:

```
<ID>. <Name> (<category>): <1-3 sentence description of what it does, 
what it trades, and where it applies>. Knobs: <knob_name> [min, max] Type.
```

**Example (Technique 22 — Spatial Downsampling):**

```
22. Spatial downsampling (input fidelity approximation): Reduces computational 
intensity by processing an image or spatial data at a coarser resolution (every 
n-th pixel in x and y dimensions) while maintaining the same output dimensions. 
Uses safe fill strategies (nearest-neighbor, linear interpolation) for skipped 
pixels. Trades spatial detail for energy efficiency; applicable to image 
processing, edge detection, and filtering operations. 
Knob: downsample_factor [1, 2, 4].
```

**Key rules:**
- Include the technique **category** in parentheses (e.g., "arithmetic simplification", "data-level approximation")
- State what is **traded** (accuracy, precision, fidelity) for what gain (speed, energy)
- List **applicable domains** so the LLM knows when to use it
- Specify **knob names, ranges, and types** (Integer / Real) — this is what the LLM will use to generate `/*Knob Variables Declaration Start*/` blocks

> **That's the only "input" you provide.** The LLM reads this description and generates the actual C code with knobs automatically.

### Step 2 — Register in the Technique Registry

**File:** `lib/technique_registry.py`

Add an entry to the `TECHNIQUE_REGISTRY` dict:

```python
31: TechniqueEntry(
    31, "My New Technique",
    requires_fpu=False,       # Does it need floating-point hardware?
    requires_simd=False,      # Does it need SIMD instructions?
    memory_cost_kb=0.0,       # Extra RAM overhead in KB
    compute_cost_relative=0.3, # Relative overhead (0.0–1.0, lower = cheaper)
    applicable_apps=["sobel-iclib", "ar-iclib"],  # Which benchmarks? None = all
    description="One-line description for logs.",
),
```

**Field guide:**

| Field | Purpose | Example |
|-------|---------|---------|
| `technique_id` | Unique integer, sequential | `31` |
| `name` | Human-readable name | `"Spatial Downsampling"` |
| `requires_fpu` | Set `True` if technique uses float math that can't be emulated | `False` for MSP430-safe |
| `requires_simd` | Set `True` if technique needs vector instructions | `False` for MSP430 |
| `memory_cost_kb` | Extra RAM the technique adds (LUT tables, buffers) | `0.016` for a 16-byte LUT |
| `compute_cost_relative` | How expensive the technique overhead is (0–1 scale) | `0.1` = very cheap |
| `applicable_apps` | List of benchmark names this applies to, or `None` for all | `["sobel-iclib"]` |
| `description` | Short text for logs/reports | `"Process every n-th pixel"` |

**This is what enables automatic hardware filtering.** The constraint gatekeeper checks:
- `requires_fpu` vs `hw.has_fpu`
- `requires_simd` vs `hw.has_simd`  
- `memory_cost_kb` vs `hw.ram_kb`
- `applicable_apps` vs current benchmark name

Techniques that fail any check are **automatically rejected** and never shown to the LLM.

### Step 3 (Optional) — Add an Error Analyzer

**File:** `utils/error_analyzer.py`

If your technique targets a **new benchmark** (not one of the existing 6), add an error function:

```python
def myNewBenchmarkError(pathToCodebase):
    """Compare approximated output to ground truth for my-new-benchmark."""
    # ... read outputs, compute error metric (0.0 = perfect, 1.0 = max error)
    return error
```

Then register it in `lib/bo.py` inside `evaluateKnobs()`:

```python
elif app_name == "my-new-benchmark":
    error = myNewBenchmarkError("knob_tuning")
```

For existing benchmarks (sobel, ar, fft, lqi, stringsearch, bc), this is **not needed** — the error analyzers already exist.

### That's It — 3 Files, Done

No other code changes needed. The pipeline handles everything else:

1. **Hardware gatekeeper** reads the registry → filters by FPU/SIMD/RAM/app
2. **Cost model** ranks surviving techniques cheapest-first  
3. **LLM prompt** includes the natural-language description + feasible list
4. **LLM** generates approximated C code with `/*Knob Variables Declaration Start/End*/` blocks
5. **JSON extractor** parses knob names, ranges, step sizes from the LLM output
6. **BayesOpt** creates a search space from knob ranges and runs 100 iterations
7. **Each iteration**: writes knobs → cross-compiles → Fused simulates → measures cycles/checkpoints → computes error
8. **Best configuration** (lowest checkpoints within error bound) is saved

---

## The 10 Techniques We Added (T21–T30)

| ID | Name | Category | Applicable Apps | Key Knobs |
|----|------|----------|-----------------|-----------|
| 21 | Early-Exit Approximation | Control-flow | All | `exit_threshold` |
| 22 | Spatial Downsampling | Input fidelity | sobel, susan | `downsample_factor [1,4]` |
| 23 | Temporal Decimation | Data-level | ar, fft, lqi | `decimation_factor [1,10]` |
| 24 | Magnitude-Only FFT | Frequency-domain | ar, fft | `n_fft, keep, phase_policy` |
| 25 | Lazy Preprocessing | Preprocessing | stringsearch | `lazy_budget [16,256]` |
| 26 | Bit-Shift EWMA | Arithmetic | lqi | `shift_factor [0,4]` |
| 27 | Nibble Lookup | Computation | bc (bitcount) | `nibble_stride [1,4]` |
| 28 | Hierarchical Feature Extraction | Feature-level | ar | `hfe_gate_tau, hfe_full_rate` |
| 29 | Radix Variation | Arithmetic | radix-bm | `radix_mode [2,16]` |
| 30 | Pattern Segmentation | Data-level | sobel, ar, susan | `segment_len [1,8], policy` |

---

## Results: What the Techniques Achieved

All 6 benchmarks evaluated on **MSP430** (no FPU, 10 KB RAM, 16 MHz) with **RF_1 trace**, **68 µF capacitor**, **100 BayesOpt iterations**, **30% error bound**.

| Benchmark | Techniques Selected | Original CPs | Best CPs | Reduction | Error |
|-----------|-------------------|--------------|----------|-----------|-------|
| **sobel-iclib** | T21 + T22 + T30 | 48 | 3 | **93.75%** | 15.5% |
| **ar-iclib** | T21 + T23 + T30 | 14 | 3 | **78.57%** | 1.6% |
| **fft-iclib** | T21 + T23 | 9 | 5 | **44.44%** | 0.5% |
| **lqi-iclib** | T21 + T23 + T26 | 3 | 2 | **33.33%** | 21.5% |
| **stringsearch-iclib** | T21 + T25 | 21 | 14 | **33.33%** | 0.0% |
| **bc-iclib** | T21 + T27 | 3 | 3 | **0.00%** | 27.7% |

### Sobel Breakdown (93.75% Reduction)

The LLM combined three techniques on `sobel_filtering()`:

**Original:** Every pixel, full 3×3 convolution, two complete passes.

**Approximated (best knobs `[4, 1, 1, 146]`):**
- **T22** `downsample_factor=4` → process every 4th pixel (16× fewer convolutions), fill with nearest-neighbor
- **T30** `segment_pattern=1` → cross-pattern convolution (5 multiplies instead of 9)
- **T21** `early_exit_enabled=1, threshold=146` → exit pass-1 early once min/max range exceeds 146 after 50 samples

Result: 48 → 3 checkpoints, 15.5% output error (within 30% bound).

---

## Hardware-Aware Filtering in Action

When you target MSP430 (`has_fpu=false`, `has_simd=false`, `ram_kb=10`):

| Technique | FPU? | SIMD? | Memory | Verdict |
|-----------|------|-------|--------|---------|
| T21 Early-Exit | No | No | 0 KB | **PASS** |
| T22 Spatial Downsampling | No | No | 0 KB | **PASS** (if app matches) |
| T23 Temporal Decimation | No | No | 0 KB | **PASS** (if app matches) |
| T24 Magnitude-Only FFT | **Yes** | No | 1 KB | **REJECTED** — needs FPU |
| T25 Lazy Preprocessing | No | No | 0.5 KB | **PASS** (stringsearch only) |
| T26 Bit-Shift EWMA | No | No | 0 KB | **PASS** (lqi only) |
| T27 Nibble Lookup | No | No | 0.016 KB | **PASS** (bc only) |
| T28 Hierarchical Features | **Yes** | No | 0.5 KB | **REJECTED** — needs FPU |
| T29 Radix Variation | No | No | 0 KB | **PASS** (radix-bm only) |
| T30 Pattern Segmentation | No | No | 0 KB | **PASS** (if app matches) |

The gatekeeper + `applicable_apps` filter ensures each benchmark only sees relevant, feasible techniques. For example, sobel-iclib on MSP430 sees only **T21, T22, T30**.

---

## Running the Pipeline

### With LLM (generates fresh approximated code)
```bash
cd CheckMate
source .venv/bin/activate
python main.py --bm_name sobel-iclib \
    --hardware config/hardware_msp430.json \
    --trace "../traces/RF_1.csv"
```

### Without LLM (uses pre-generated `apx_all.json`)
```bash
python main.py --bm_name sobel-iclib --no_llm \
    --hardware config/hardware_msp430.json \
    --trace "../traces/RF_1.csv"
```

### Key Output Files

| File | Location | Content |
|------|----------|---------|
| Best knobs | `logs/best_knobs_{app}.csv` | Winning knob config, error, checkpoints |
| BayesOpt trace | `logs/{app}_{cap}_{trace}.csv` | All evaluated configurations |
| Original checkpoints | `logs/original_checkpoints_{app}-{cap}_{trace}.txt` | Baseline checkpoint count |
| Experiment metrics | `logs/experiment_metrics_{app}_{hw}.json` | API calls, compile failures, etc. |
| Hardware rejections | `outputs/hardware_rejections_{app}_{hw}.json` | Which techniques were filtered and why |
| LLM constraints proof | `outputs/LLM_CONSTRAINTS_{app}_{hw}.txt` | Full audit trail of HW filtering + LLM response |

---

## File Index

| File | Purpose |
|------|---------|
| `prompts/approximation_techniques.txt` | Natural-language technique descriptions (LLM reads this) |
| `lib/technique_registry.py` | Hardware requirements, costs, applicable apps per technique |
| `lib/constraint_gatekeeper.py` | Filters techniques against hardware profile |
| `lib/hardware_cost_model.py` | Ranks techniques by execution cost on target hardware |
| `lib/hardware_profile.py` | Loads hardware JSON (MSP430, ARM Cortex, etc.) |
| `lib/hw_validator.py` | Post-LLM validation — ensures LLM only used feasible techniques |
| `lib/hw_observability.py` | Writes rejection JSON + constraints proof text |
| `lib/bo.py` | Bayesian optimisation loop (100 iterations, knob tuning) |
| `lib/llm.py` | LLM API calls (technique selection, code generation) |
| `utils/checkpoints.py` | Build → Fused simulate → cycle count pipeline |
| `utils/error_analyzer.py` | Per-benchmark error computation (image diff, signal error, etc.) |
| `config/hardware_msp430.json` | MSP430 hardware profile |
| `config/config.py` | Error bound (0.30), API selection, debug flags |

---

## Quick Reference: Adding Technique 31

1. **`prompts/approximation_techniques.txt`** — append:
   ```
   31. My Technique (my-category): Description of what it does and trades. 
   Knobs: my_knob [1, 10] Integer.
   ```

2. **`lib/technique_registry.py`** — add to `TECHNIQUE_REGISTRY`:
   ```python
   31: TechniqueEntry(
       31, "My Technique",
       memory_cost_kb=0.0,
       compute_cost_relative=0.3,
       applicable_apps=["sobel-iclib"],
       description="Short description.",
   ),
   ```

3. **Run:**
   ```bash
   python main.py --bm_name sobel-iclib \
       --hardware config/hardware_msp430.json \
       --trace "../traces/RF_1.csv"
   ```

The LLM will read your description, generate approximated C code with knobs, and BayesOpt will automatically find the best knob configuration.
