# CHAI: Constraint-Guided Hardware-Aware Approximation for Intermittent Computing

CHAI is an LLM-driven framework for automatically generating, validating, and tuning **software-level approximations** of C/embedded applications targeted at intermittent and energy-constrained platforms. CHAI takes a benchmark application, a hardware profile (e.g. MSP430, ARM Cortex-A53, RISC-V), and an energy trace, then produces approximated code variants that are (a) feasible on the target hardware, (b) compile and run correctly, and (c) tuned by Bayesian optimization to reduce checkpointing overhead while keeping output error within a configurable bound.

> CHAI is the rebranded codebase of the published CheckMate project. Internal directories, tools, and Docker images may still use the `CheckMate` / `checkmate-image` names — this is intentional and works as-is. Throughout this README the framework is referred to as **CHAI**.

---

## 1. Project Overview

CHAI links four ideas into a single pipeline:

1. **Hardware-aware constraint gatekeeping** — a structured hardware profile filters the catalogue of approximation techniques *before* the LLM ever sees them, so the model can only choose techniques that physically fit the target.
2. **LLM planning + transformation** — a multi-step prompt pipeline (purpose identification → planning → approximation → JSON conversion) rewrites each function with knob variables.
3. **Validation + post-LLM clamping** — generated code is compiled and re-validated against the hardware constraint set; any infeasible techniques the LLM tried to slip in are clamped out.
4. **Hardware-in-the-loop fine tuning** — knobs are tuned with scikit-optimize Bayesian Optimization against a modified [Fused](https://github.com/UoS-EEC/fused) intermittent-computing simulator driven by real-world voltage traces.

---

## 2. Purpose / Motivation

Intermittent computing devices (e.g. batteryless RFID-scale sensors) execute in short power-cycle bursts and must checkpoint state across power failures. Approximate computing reduces the work done per cycle, but choosing *which* approximation to apply is hardware-dependent: e.g. floating-point precision scaling is meaningless on an MSP430 with no FPU.

CHAI exists to:

- **Automate** the search for approximations that respect hardware capabilities (FPU, SIMD, RAM).
- **Generate, validate, and evaluate** approximate code transformations end-to-end.
- **Tune** the resulting approximations against a real energy trace so that the chosen knob values measurably reduce the number of power cycles to complete the workload.
- **Enforce** that the LLM's choices are physically realizable on the target — through a constraint gatekeeper, post-LLM validator, and a final clamping step.

CHAI is aimed at researchers and engineers working on intermittent / energy-harvesting embedded and IoT systems who want a reproducible pipeline for hardware-aware approximate computing.

---

## 3. Repository Structure

```
CheckMate/                              # repository root (project name: CHAI)
├── main.py                             # master pipeline script (entry point)
├── requirements.txt                    # Python dependencies
├── setup.sh                            # one-shot system setup script
├── install_fused_script.sh             # builds the Fused intermittent-computing simulator
├── Dockerfile                          # full reproducible environment
├── .env                                # API keys + model selection (you create this)
│
├── config/                             # global configuration + hardware profiles
│   ├── config.py                       # pipeline switches (ERROR_BOUND, debug flags, etc.)
│   ├── globals.py                      # shared run-time globals (e.g. PLATFORM_ARCHITECTURE)
│   ├── hardware_msp430.json            # built-in hardware profile (10 KB RAM, no FPU/SIMD)
│   ├── hardware_arm_a53.json           # built-in hardware profile (FPU + SIMD)
│   └── hardware_custom.json            # built-in hardware profile (RISC-V RV32IMC)
│
├── lib/                                # core CHAI modules
│   ├── llm.py                          # LLM interaction (purpose / plan / approximate / convert)
│   ├── lsp.py                          # clangd LSP wrapper for function extraction
│   ├── pdg.py                          # function call graph (FCG) construction via egypt
│   ├── bo.py                           # Bayesian Optimization knob tuning loop
│   ├── hardware_profile.py             # HardwareProfile loader
│   ├── constraint_gatekeeper.py        # filters infeasible techniques
│   ├── hardware_cost_model.py          # ranks surviving techniques by cost
│   ├── hw_validator.py                 # post-LLM technique-ID validation + clamping
│   ├── hw_observability.py             # writes rejection logs + constraint proofs
│   ├── prompt_updater.py               # prompt anchor injection for new techniques
│   ├── technique_registry.py           # catalogue of all known techniques
│   └── technique_card_schema.py        # JSON schema for Technique Cards
│
├── techniques/cards/                   # JSON technique-card definitions (auto-loaded)
├── tools/add_technique.py              # CLI to synthesize and integrate a new technique
├── prompts/                            # LLM prompt templates + few-shot examples
├── benchmark_applications/             # original (untouched) C benchmark sources
│                                       # NOTE: tracked as deletes in git status — see Notes below.
├── eval-apps/                          # working copies of benchmarks compiled for the target
│   ├── CMakeLists.txt                  # cross-compile entry (TARGET_ARCH=msp430|cm0)
│   ├── cmake/                          # cm0/msp430 toolchain files
│   ├── cmsis/, iclib/, support/        # platform support libs
│   └── <benchmark>/                    # one folder per benchmark (sobel-iclib, ar-iclib, …)
│
├── llm-prerun/                         # cached LLM outputs per benchmark
│                                       # used with --no_llm to skip the API entirely
├── traces/                             # sample voltage traces (RF_*.csv, Solar_*.csv)
├── fusedBin/                           # location for the Fused simulator binary + its config
│   ├── fused                           # (you place the built binary here)
│   ├── fusedConfig/                    # config templates (config.yaml.in)
│   └── rt_plotter.py, find_optimals.py # auxiliary scripts
│
├── utils/                              # utility helpers
│   ├── initialization.py               # loads target files, prompts, examples
│   ├── compiler.py                     # compile-test wrapper
│   ├── validator.py                    # runtime validation + knob updates
│   ├── checkpoints.py                  # drives the Fused checkpointing simulation
│   ├── error_analyzer.py               # benchmark-specific error / ground-truth functions
│   ├── error_metrics.py                # generic error metrics (RMSE, etc.)
│   ├── trace_utils.py                  # deterministic seeds + approximation-trace logging
│   └── utils.py, json_handling.py, …
│
├── scripts/demo_hardware_profiles.py   # demonstrates feasibility filtering on each profile
├── tests/test_technique_integration.py # pytest gate for technique integration
├── docs/                               # detailed framework docs
│   ├── WORKFLOW_DETAILED.md            # full architectural walk-through
│   ├── ADDING_NEW_TECHNIQUES.md        # how to extend the technique registry
│   └── TECHNIQUE_AUTO_INTEGRATION.md   # auto-integration internals
└── logs/                               # per-run logs, metrics JSON, BO CSV results
```

---

## 4. Installation / Setup

CHAI runs on Linux (tested on Ubuntu 22.04). The two recommended installation paths are **Docker** (easiest) and **manual** (development).

### Required tools

| Tool                       | Purpose                                              |
| -------------------------- | ---------------------------------------------------- |
| Python 3.10+               | Runs the pipeline                                    |
| `pip`, `venv`              | Python package management                            |
| `clangd`                   | LSP server used to extract function definitions      |
| `egypt` (Perl)             | Builds the C function call graph from `.expanded`    |
| `graphviz` (`dot`)         | Required by `egypt`                                  |
| MSP430 / ARM GCC toolchain | Cross-compiles benchmarks for the target            |
| CMake + Ninja              | Builds Fused and benchmark binaries                  |
| `fused` (built separately) | Intermittent-computing simulator                     |
| LLM API key / local service| Anthropic (Claude), OpenAI, HuggingFace, or local Ollama |

### Option A — Docker

```bash
docker build -t checkmate-image .
docker run -it checkmate-image
```

The Dockerfile already installs every system dependency, clones and builds Fused, copies the binary into `fusedBin/`, and installs Python requirements.

> The image tag is still `checkmate-image` — that is the image name baked into the Dockerfile.

### Option B — Manual setup

```bash
# 1. System dependencies (egypt, clangd, graphviz, perl)
bash setup.sh

# 2. Build the Fused simulator
git clone https://github.com/rafayy769/fused-checkmate.git
bash install_fused_script.sh
cp fused-checkmate/build/fused fusedBin/

# 3. Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
clangd --version
egypt --version          # man egypt also works
dot -V
ls fusedBin/fused        # must exist before fine-tuning
```

If `egypt` is installed in a non-standard location, export `EGYPT_DIR=/path/to/dir-containing-egypt` before running `main.py`. The script also auto-detects `~/.local/bin`, `~/bin`, and `/usr/local/bin`.

---

## 5. Configuration

### 5.1 `.env` — LLM API selection

Create `.env` in the repository root:

```dotenv
# Pick exactly one provider.
# For Anthropic Claude:
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-20250514

# For OpenAI:
# OPENAI_API_KEY=sk-...
# LLM_MODEL=gpt-4o-2024-11-20

# For local HuggingFace / Llama:
# HF_TOKEN=hf_...
# LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

The active provider is selected by `CHAI_API_NAME` in `.env` (`Anthropic`, `OpenAI`, `HuggingFace`, or `Ollama`). If unset, it defaults to OpenAI.

For local Ollama, pull a model and configure the OpenAI-compatible endpoint:

```dotenv
CHAI_API_NAME=Ollama
LLM_MODEL=qwen2.5-coder:7b
# Use this URL when CHAI runs in Docker on Docker Desktop:
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_API_KEY=ollama
```

### 5.2 `config/config.py` — pipeline switches

| Variable                       | Meaning                                                                 |
| ------------------------------ | ----------------------------------------------------------------------- |
| `ERROR_BOUND`                  | Maximum tolerated output error during knob tuning (default `0.30`)      |
| `APPROXIMATION_CONTEXT`        | Where to inject the technique catalogue in the prompt (`0` or `1`)      |
| `GIVE_FORMAT_EXAMPLES`         | Inject JSON-format few-shot examples (`0`/`1`)                          |
| `GIVE_LOOP_PERF_EXMAPLES`      | Inject loop-perforation examples (`0`/`1`)                              |
| `GIVE_NEW_TECHNIQUE_EXAMPLES`  | Inject newly-added technique examples (`0`/`1`)                         |
| `TEXT_PLANING`                 | Use plan-then-approximate flow (`1`) vs single-shot annotation (`0`)    |
| `CHAI_API_NAME`                | `Anthropic` / `OpenAI` / `HuggingFace` / `Ollama`                       |
| `DEBUG`                        | Verbose `Dprint` logging (`0`/`1`)                                      |

### 5.3 Hardware profiles — `config/hardware_*.json`

A hardware profile tells the constraint gatekeeper what the target supports:

```json
{
  "name": "MSP430",
  "ram_kb": 10,
  "has_fpu": false,
  "has_simd": false,
  "energy_budget_mj": 5,
  "clock_mhz": 16
}
```

Built-in profiles:

| File                          | Target                |
| ----------------------------- | --------------------- |
| `config/hardware_msp430.json` | TI MSP430 (no FPU/SIMD, 10 KB RAM) |
| `config/hardware_arm_a53.json`| ARM Cortex-A53 (FPU + SIMD)        |
| `config/hardware_custom.json` | RISC-V RV32IMC (no FPU/SIMD, 64 KB RAM) |

Pass one with `--hardware <path>` (or `--hardware_profile <path>`) on the command line.

### 5.4 Technique registry — `techniques/cards/*.json`

The catalogue of approximation techniques is auto-loaded from JSON cards in `techniques/cards/`. Each card declares the technique's name, planning guidance, few-shot examples, hardware feasibility, and cost-model parameters. Built-in techniques (1–30) are also embedded in `lib/technique_registry.py`. See `docs/ADDING_NEW_TECHNIQUES.md` for the schema.

### 5.5 Energy traces — `traces/*.csv`

Voltage traces drive the Fused simulator. Each row is one voltage sample in volts. CHAI normalizes the trace (including any UTF-8 BOM) before passing it to Fused. Override the trace list with `--trace <path>`.

### 5.6 Benchmarks — `eval-apps/<name>/`

Each benchmark folder must contain at least:
- `*.c` / `*.h` — source
- `Makefile` — build instructions
- `application.txt` — declares the entry / target function name
- (for cross-compiled flows) `CMakeLists.txt`

---

## 6. How to Run

The single entry point is `main.py`. The full pipeline (plan → approximate → validate → BayesOpt) is invoked by passing a benchmark name. `--bm_name` is **required**; argparse will reject the command otherwise.

### 6.1 End-to-end run

```bash
python3 main.py --bm_name sobel-iclib
```

This executes, in order:

1. Fresh-run cleanup of transient directories.
2. (Optional) Hardware-aware filtering when `--hardware` is supplied.
3. FCG construction (`lib/pdg.py`) and function extraction (`lib/lsp.py`).
4. LLM target-function selection, planning, approximation, JSON conversion.
5. Compile and validate each approximated function.
6. Post-LLM clamping of any infeasible techniques.
7. Ground-truth generation for the original program.
8. Bayesian Optimization knob tuning per (trace, capacitor) pair via the Fused simulator.
9. Results CSV + per-run JSON metrics in `logs/`.

### 6.2 Hardware-aware run

```bash
python3 main.py --bm_name sobel-iclib --hardware config/hardware_msp430.json
```

Adds: gatekeeper filtering of infeasible techniques, ranked cost output, post-LLM validation and clamping, and observability files in `outputs/`: rejection JSON, a constraints proof TXT, and a complete discarded-plan audit (`approximation_plan_decisions_*.json` + `.md`).

### 6.3 Without an API key (offline / repro)

```bash
python3 main.py --bm_name sobel-iclib --no_llm
```

Reuses cached LLM responses from `llm-prerun/<benchmark>/` and proceeds directly to BayesOpt fine tuning. Recommended for first-time installs to verify the toolchain.

### 6.4 Override the energy trace

```bash
python3 main.py --bm_name sobel-iclib --trace ../traces/RF_2.csv
```

### 6.5 Approximation tracing

```bash
python3 main.py --bm_name sobel-iclib --trace_approx 1
```

Sets deterministic seeds (`set_deterministic_seeds(42)`) and writes a structured before/after metrics report at the end.

### 6.6 Adding a new technique

```bash
python3 tools/add_technique.py --request "Add 4-bit quantization for sensor data"
# or supply a local technique card:
python3 tools/add_technique.py --card techniques/cards/T31_4bit_quantization.json
```

This synthesizes a Technique Card via the LLM, validates it against the schema, writes it to `techniques/cards/`, updates prompts/few-shots at anchored locations, and runs the pytest gate.

### 6.7 Demo hardware-profile filtering only

```bash
python3 scripts/demo_hardware_profiles.py
```

### 6.8 Available benchmarks

| Application                | `--bm_name` value      |
| -------------------------- | ---------------------- |
| Activity Recognition       | `ar-iclib`             |
| Sobel Filter               | `sobel-iclib`          |
| String Search              | `stringsearch-iclib`   |
| Fast Fourier Transform     | `fft-iclib`            |
| Link Quality Indicator     | `lqi-iclib`            |
| Bitcount                   | `bc-iclib`             |

---

## 6b. Running the comparison campaign

**Read [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) before
running or citing any comparison.** It is the frozen protocol; the settings it
describes are enforced in code.

"CHAI is better" is ambiguous between a larger operator catalogue and a
hardware-aware control plane. The campaign is a 2x2 that separates them:

| Arm | Catalogue | Control plane | Isolates |
| --- | --- | --- | --- |
| A `checkmate_baseline` | T1-T20 | off | reference |
| B `operator_only_plus10` | T1-T30 | off | B - A = operator effect |
| C `chai_plus10` | T1-T30 | on | C - B = control-plane effect |
| D `chai_baseline_ops` | T1-T20 | on | D - A = control plane alone |

```bash
# Full campaign: 6 benchmarks x 4 arms x 3 seeds, frozen target functions
bash scripts/run_ablation_campaign.sh

# Narrow it while iterating
APPS="sobel-iclib fft-iclib" SEEDS=2026 BO_CALLS=20 \
  bash scripts/run_ablation_campaign.sh

# Decompose the result
python3 scripts/analyze_ablation.py experiment_results/<run_id>
```

The analysis prints the operator effect and the control-plane effect
separately. The claim "it is not just the ten operators" is supported only if
the control-plane column shows a larger improvement than the operator column.

Results under `experiment_results/` that carry a `SUPERSEDED.md` predate this
protocol and must not be cited.

## 7. Input and Output

### Inputs

| What                       | Where                                      |
| -------------------------- | ------------------------------------------ |
| Benchmark source code      | `benchmark_applications/<name>/` → copied into `eval-apps/<name>/` at run start |
| Application entry function | `<benchmark>/application.txt`              |
| Hardware profile           | `config/hardware_*.json` (selected by `--hardware`) |
| LLM API key + model        | `.env`                                     |
| Energy trace               | `traces/*.csv` (selected by `--trace` or default list in `main.py`) |
| Capacitor values           | Hard-coded in `main.py` (`capacitors = [...]`)  |
| Pre-run LLM responses      | `llm-prerun/<bm_name>/` (used with `--no_llm`) |
| Approximation techniques   | `techniques/cards/*.json` + `lib/technique_registry.py` |
| Prompt templates           | `prompts/*.txt`                            |

### Outputs

| What                                | Where                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------- |
| Extracted function JSON             | `functions/`                  *(temporary)*                                |
| Approximated function JSONs         | `approximated_functions/apx_*.json`, `apx_all.json`                        |
| Compile-test artifacts              | `compilation_testing/`        *(temporary)*                                |
| Knob-tuning workspace               | `knob_tuning/`                                                             |
| Fused build outputs                 | `fusedBin/app.hex`, `fusedBin/cycles.dump`                                 |
| BayesOpt per-trace CSV              | `logs/<app>_<capacitor>_<trace>.csv`                                       |
| Best-knob summary CSV               | `logs/best_knobs_<app>.csv`                                                |
| Original (baseline) checkpoint count| `logs/original_checkpoints*.txt`                                           |
| Hardware rejection report           | `logs/<app>_<hw>_rejections.json`                                          |
| Hardware constraints proof          | `logs/<app>_<hw>_constraints.txt`                                          |
| Experiment metrics                  | `logs/experiment_metrics_<app>_<hw>.json`                                  |
| Conversation history                | `logs/conv1.txt`                                                           |

---

## 8. Main Components / Architecture

```
┌─ main.py ────────────────────────────────────────────────────────────────┐
│                                                                          │
│  1. Fresh-run cleanup                                                    │
│  2. Hardware-aware filtering          ──► lib/constraint_gatekeeper.py   │
│       (lib/hardware_profile.py)            lib/hardware_cost_model.py    │
│  3. FCG + function extraction         ──► lib/pdg.py, lib/lsp.py         │
│  4. LLM stages (per function)         ──► lib/llm.py                     │
│       a. purposeIdentificationFunction      prompts/purpose_id_*.txt     │
│       b. planStepFunction (HW-aware)        prompts/planning_step.txt    │
│       c. approximateFunction                prompts/approximate_*.txt    │
│       d. convertJson                        prompts/convert_json.txt     │
│  5. Compile-test                      ──► utils/compiler.py              │
│  6. Post-LLM validation + clamping    ──► lib/hw_validator.py            │
│  7. Observability                     ──► lib/hw_observability.py        │
│  8. Ground-truth generation           ──► utils/error_analyzer.py        │
│  9. BayesOpt fine tuning              ──► lib/bo.py                      │
│       per (trace, capacitor):              utils/checkpoints.py          │
│                                            fusedBin/fused (simulator)    │
│ 10. Results aggregation               ──► logs/best_knobs_<app>.csv      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key components

- **LLM pipeline (`lib/llm.py`)** — orchestrates the four prompt stages. All raw prompts live in `prompts/`. Hardware constraint text is injected into the platform-architecture description before planning, so the LLM sees the hardware filter inline.
- **Technique registry (`lib/technique_registry.py` + `techniques/cards/*.json`)** — single source of truth for the catalogue of approximation techniques. Each entry declares hardware requirements (FPU/SIMD/memory), compute cost, and applicable apps.
- **Constraint gatekeeper (`lib/constraint_gatekeeper.py`)** — pre-LLM filter. Discards techniques that need an FPU the target lacks, that exceed RAM, or that aren't applicable to the current benchmark.
- **Hardware cost model (`lib/hardware_cost_model.py`)** — ranks surviving techniques by an estimated execution cost on the target (`compute_cost_relative * 100 / clock_mhz`, plus memory pressure).
- **Hardware validator (`lib/hw_validator.py`)** — post-LLM check. Re-extracts technique IDs from the LLM's output, reprompts once on infeasible choices, and clamps any infeasible entries from `apx_all.json`.
- **Validation pipeline (`utils/compiler.py`, `utils/validator.py`)** — compile-tests each approximated function (with up to 3 retries), then validates runtime behaviour by setting knobs to neutral defaults and comparing against the original.
- **Checkpoint orchestration (`utils/checkpoints.py`)** — cross-compiles each candidate to MSP430 hex and runs Fused with the configured trace/capacitor. Reports power-cycle count.
- **Bayesian optimization (`lib/bo.py`)** — uses `skopt.gp_minimize` over the search space defined by knob ranges in `apx_all.json`. The objective combines output error and checkpoint count; configurations exceeding `ERROR_BOUND` are rejected.
- **Auto technique integration (`tools/add_technique.py`, `lib/technique_card_schema.py`, `lib/prompt_updater.py`)** — synthesizes a Technique Card from a natural-language request, validates it, and edits anchored regions in the prompt files automatically.

---

## 9. Example Usage

End-to-end run on the Sobel benchmark, targeting MSP430, against trace `RF_2.csv`, without making any LLM API calls (offline reproducibility):

```bash
# 1. Clone and set up
git clone <this-repo> CheckMate
cd CheckMate
bash setup.sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build Fused once
git clone https://github.com/rafayy769/fused-checkmate.git
bash install_fused_script.sh
cp fused-checkmate/build/fused fusedBin/

# 3. Run CHAI offline (uses cached LLM responses from llm-prerun/sobel-iclib/)
python3 main.py --bm_name sobel-iclib --no_llm \
                --hardware config/hardware_msp430.json \
                --trace traces/RF_2.csv

# 4. Inspect results
ls logs/
cat logs/best_knobs_sobel-iclib.csv
cat logs/experiment_metrics_sobel-iclib_MSP430.json
```

Expected console highlights:

- `[HW] name=MSP430, clock=16MHz, ram=10KB, fpu=0, simd=0`
- A "HARDWARE-AWARE APPROXIMATION PLANNING" banner listing feasible / rejected techniques.
- A `EXPERIMENT METRICS SUMMARY` block with `Functions succeeded`, `HW reprompts triggered`, `Techniques clamped`, etc.
- A final BayesOpt result table showing `checkpoint_reduction` < 1.0 (lower = better).

---

## 10. Troubleshooting

| Symptom                                                            | Likely cause / fix                                                                                                                            |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError` on first run                                 | Activate the virtualenv: `source .venv/bin/activate && pip install -r requirements.txt`.                                                       |
| `egypt: command not found`                                         | Run `bash setup.sh` again, install manually from https://www.gson.org/egypt/, or export `EGYPT_DIR=<dir-containing-egypt>` before launching `main.py`. |
| `clangd --version` fails                                           | `sudo apt install clangd`.                                                                                                                     |
| `Hardware profile not found: …`                                    | The `--hardware` path is wrong. Use `config/hardware_msp430.json` (relative to repo root).                                                    |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` not set                     | Populate `.env` and ensure `API_NAME` in `config/config.py` matches the provider whose key you set.                                            |
| `MSP430_GCC_ROOT not set` or `msp430-gcc: command not found`       | Re-run `install_fused_script.sh -t` (sets toolchain env vars), or export `MSP430_GCC_ROOT=$HOME/.local/msp430-gcc` and add it to `PATH`.       |
| Fused fails / `fusedBin/fused` missing                             | Build Fused (`install_fused_script.sh`) and copy the binary to `fusedBin/`.                                                                    |
| `[ERROR] Failed to get original checkpoints`                       | The benchmark's original code didn't compile or run on Fused. Check `eval-apps/<bm>/Makefile` and the `application.txt` entry-function name.   |
| `[Max retries] <fn> failed compilation after 3 attempts`           | The LLM produced invalid C three times in a row. Inspect `compilation_testing/<fn>/` for the raw build error.                                  |
| `createRTLFile: PDG generation failed after 5 attempts`            | The LLM could not produce a buildable Makefile in 5 tries. Inspect `logs/rtl_chat_log.txt` and `dependency_graphs/target/compiler_log.txt`. Raise the cap with `CHECKMATE_PDG_MAX_RETRIES=10`. |
| `[HW Validator] LLM chose infeasible technique(s) … reprompting`   | Informational — CHAI is reprompting once with the constraint set. If it persists, the technique catalogue may be too restrictive for the app.  |
| `All BayesOpt configs exceeded ERROR_BOUND`                        | Loosen `ERROR_BOUND` in `config/config.py`, or revise the technique knob ranges in `apx_all.json`.                                             |
| `benchmark_applications/` missing in working tree                  | See "Notes for Developers" §11 — recover via `git checkout HEAD -- benchmark_applications` or pull the original sources.                       |

---

## 11. Notes for Developers

### Adding a new approximation technique

Two paths:

- **Automated** — `python3 tools/add_technique.py --request "..."` synthesizes a Technique Card via the LLM, validates it, writes the JSON to `techniques/cards/`, and patches the prompt anchors. Run the pytest gate (`pytest tests/`) afterwards.
- **Manual** — write a JSON card under `techniques/cards/T<id>_<name>.json` matching the schema in `lib/technique_card_schema.py`. The registry auto-loads it at next import. See `docs/ADDING_NEW_TECHNIQUES.md` for the field-by-field guide.

### Adding a new hardware profile

Drop a JSON file in `config/` matching the schema in `lib/hardware_profile.py` (`name`, `ram_kb`, `has_fpu`, `has_simd`, `energy_budget_mj`, `clock_mhz`). Pass it with `--hardware <path>`.

### Adding a new benchmark

1. Create `benchmark_applications/<name>/` (or, given the current repo state, `eval-apps/<name>/`) with the source files, a `Makefile`, an `application.txt` declaring the entry function, and a `CMakeLists.txt` if you want cross-compiled MSP430/ARM builds.
2. Add a benchmark-specific error function in `utils/error_analyzer.py` and a ground-truth generator entry there.
3. (Optional) Add a `llm-prerun/<name>/` folder if you want `--no_llm` reproducibility.

### Adding a new validation check

Plug into `utils/validator.py::validateFunction` or add a stage between `compileTest` and the `convertJson` step in `main.py`.

### Adding a new evaluation metric

Add the metric function in `utils/error_metrics.py`, then wire it into the relevant benchmark's error function in `utils/error_analyzer.py` and (if it should drive BayesOpt) into the objective function in `lib/bo.py`.

### Repository state caveat

`git status` currently shows the entire `benchmark_applications/` tree as deleted. Restore the folder before running benchmarks that read from it:

```bash
git checkout HEAD -- benchmark_applications
```

`main.py` copies `benchmark_applications/<bm>/` into `eval-apps/<bm>/` at the start of every run, so the source must be present.

### Tests

```bash
pytest tests/
```

### Detailed docs

- `docs/WORKFLOW_DETAILED.md` — full architectural walk-through.
- `docs/ADDING_NEW_TECHNIQUES.md` — technique card author guide.
- `docs/TECHNIQUE_AUTO_INTEGRATION.md` — internals of `tools/add_technique.py`.
- `docs/CHAI_VS_OPERATOR_ONLY.md` — why a larger operator catalogue is not CHAI's hardware-aware planning and enforcement layer.

---
