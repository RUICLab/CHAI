from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import JsonOutputParser
import shutil
import os
import subprocess
import ast
import json
import pandas as pd

# Ensure egypt (Perl call-graph tool) is on PATH for PDG generation.
# Honor $EGYPT_DIR if set, otherwise try a few conventional install locations.
import shutil as _shutil
if not _shutil.which("egypt"):
    _candidate_egypt_dirs = [
        os.environ.get("EGYPT_DIR"),
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/bin"),
        "/usr/local/bin",
    ]
    for _egypt_dir in _candidate_egypt_dirs:
        if _egypt_dir and os.path.isfile(os.path.join(_egypt_dir, "egypt")):
            if _egypt_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _egypt_dir + ":" + os.environ.get("PATH", "")
            break

# Ensure MSP430 toolchain env vars are set for cross-compilation
if not os.environ.get("MSP430_GCC_ROOT"):
    _msp_gcc = os.path.expanduser("~/.local/msp430-gcc")
    if os.path.isdir(_msp_gcc):
        os.environ["MSP430_GCC_ROOT"] = _msp_gcc
if not os.environ.get("MSP430_INC"):
    _msp_inc = os.path.expanduser("~/.local/msp430-inc")
    if os.path.isdir(_msp_inc):
        os.environ["MSP430_INC"] = _msp_inc
import argparse
import glob as _glob_mod
from tqdm import tqdm
from colorama import Back, Fore, Style
import warnings
from tabulate import tabulate
# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
#  Centralized Fresh-Run Preparation
# ---------------------------------------------------------------------------
def _prepare_fresh_run(app_name_for_cleanup=None):
    """Remove ALL transient artifacts so every run starts clean.

    This single function replaces the scattered ad-hoc cleanups
    (compilation_testing, knob_tuning, dependency_graphs, etc.) that
    previously caused stale-file bugs.

    Called once, right after argument parsing, before any pipeline work.
    """
    os.makedirs("logs", exist_ok=True)

    transient_dirs = [
        "compilation_testing",
        "knob_tuning",
        "functions",
        "approximated_functions",
        "dependency_graphs",
    ]
    for d in transient_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"[cleanup] Removed stale {d}/")

    # Clean eval-apps/build/ (CMake cache) — stale object files cause the
    # cross-compiler to silently reuse old hex, which was root-cause of
    # the "zero cycle reduction" bug.
    build_dir = "eval-apps/build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print("[cleanup] Removed stale eval-apps/build/")

    # Restore eval-apps/{app}/ to pristine benchmark state so that leftover
    # approximated .c/.h files from a previous run don't leak in.
    if app_name_for_cleanup:
        bm_src = f"benchmark_applications/{app_name_for_cleanup}/"
        ea_dst = f"eval-apps/{app_name_for_cleanup}/"
        if os.path.isdir(bm_src):
            if os.path.exists(ea_dst):
                shutil.rmtree(ea_dst)
            shutil.copytree(bm_src, ea_dst)
            print(f"[cleanup] Restored eval-apps/{app_name_for_cleanup}/ from benchmark_applications/")

    # Clean target/ directory (will be repopulated immediately after)
    if os.path.exists("target"):
        for old in _glob_mod.glob(os.path.join("target", '*')):
            if os.path.isfile(old):
                os.remove(old)
            elif os.path.isdir(old):
                shutil.rmtree(old)
        print("[cleanup] Cleaned target/")

    # Remove stale Fused artifacts
    for stale in ["fusedBin/app.hex", "fusedBin/cycles.dump", "fusedBin/trace_input.csv"]:
        if os.path.exists(stale):
            os.remove(stale)
            print(f"[cleanup] Removed stale {stale}")

    # Clean stale log files (but not CSV results — those are valuable)
    for log_f in _glob_mod.glob("logs/*.txt"):
        os.remove(log_f)
    for log_f in _glob_mod.glob("logs/*.json"):
        os.remove(log_f)
    for log_f in _glob_mod.glob("logs/*.jsonl"):
        os.remove(log_f)

    print("[cleanup] Fresh-run preparation complete.\n")

from config.config import (
    TEXT_PLANING,
    ERROR_BOUND,
)

from utils.initialization import (
    loadTargetFiles,
    loadGlobalContext,
    parseFunctions,
    getPromptTemplates,
    loadFormatExamples,
    loadLoopPerfExamples,
    loadNewTechniqueExamples,
    loadCodeBaseSummary,
)

from utils.context import (
    manufacturerContext
)

from utils.json_handling import (
    loadEntities,
    joinJsonFiles
)

from utils.utils import (
    formatMessageForHistory,
    copyFiles,
    getAppName,
    getFunctionData,
    writeFunctionsToJson,
    filterSourceTargets,
    Dprint
)

from utils.compiler import(
    compileTest
)

from utils.validator import(
    validateFunction,
    validateFunctionContract,
    repairFunctionCandidate,
    buildFocusedRepairRequest,
)
from lib.hw_validator import extract_technique_ids_from_convo, validate_technique_ids

from utils.trace_utils import (
    set_deterministic_seeds,
)
from utils.fused_trace import prepare_fused_trace
from utils.source_functions import recover_source_entities
from utils import llm_accounting
from utils import token_budget
from utils.target_build import validate_target_build
from utils.identity_gate import IdentityCheckUnavailable, verify_identity
from utils.direct_extraction import build_plan_entry
from lib.plan_contract import (
    check_implementation_matches_plan,
    parse_plan,
    validate_plan_against_catalogue,
)
from utils.optimization_metrics import (
    objective_description,
    power_cycle_metrics,
)
from utils.retry_context import (
    generation_attempt_context,
    json_conversion_context,
)
from utils.operator_conformance import (
    prune_unimplemented_added_operator_claims,
    repair_operator_candidate,
    validate_hardware_operator_conformance,
    validate_operator_conformance,
)
from lib.technique_registry import get_applicable_techniques

parser = argparse.ArgumentParser()

parser.add_argument("--bm_name", type=str, help="Benchmark app/ Evaluation app name", required=True)
parser.add_argument("--no_llm", action="store_true", help="include if you do not want to run LLM", required=False)
parser.add_argument(
    "--prerun_dir",
    type=str,
    default=None,
    help="Directory containing no-LLM artifacts (for example an approximated_functions/ plan). "
         "Defaults to llm-prerun/<bm_name>.",
)
parser.add_argument(
    "--artifact_label",
    type=str,
    default=None,
    help="Optional suffix for accepted-plan output artifacts, keeping controlled conditions separate.",
)
parser.add_argument("--trace_approx", type=int, choices=[0, 1], default=0, help="Enable approximation tracing (1=enabled, 0=disabled)")
parser.add_argument("--hardware", type=str, default=None,
                    help="Path to a hardware profile JSON (e.g. config/hardware_msp430.json). "
                         "Enables hardware-aware technique filtering.")
parser.add_argument("--hardware_profile", type=str, default=None,
                    help="Alias for --hardware. Path to a hardware profile JSON.")
parser.add_argument(
    "--hardware_control",
    choices=["chai", "prompt_only"],
    default="chai",
    help="How to use --hardware. 'chai' enables filtering, cost ranking, "
         "post-LLM validation, and clamping. 'prompt_only' gives the same raw "
         "profile to the LLM but deliberately disables CHAI's control plane "
         "for an operator-only comparison baseline.",
)
parser.add_argument("--trace", type=str, default=None,
                    help="Override traces list with a single trace path (e.g. ../traces/RF_1.csv)")
parser.add_argument(
    "--traces",
    type=str,
    default=None,
    help="Comma-separated list of trace paths to sweep. A result at one trace "
         "is a single operating point, not a characterisation.",
)
parser.add_argument(
    "--capacitors",
    type=str,
    default=None,
    help="Comma-separated list of capacitor values to sweep (e.g. "
         "'22e-6,68e-6,220e-6'). Defaults to 68e-6.",
)
parser.add_argument(
    "--preflight_only",
    action="store_true",
    help="Run real initialization, PDG, entity recovery, and catalog setup, then exit before LLM calls.",
)
parser.add_argument(
    "--operator_catalog",
    choices=["all", "checkmate_plus_10", "checkmate_20"],
    default="all",
    help="Fixed operator menu. checkmate_20 exposes only the original T1-T20 "
         "catalogue; checkmate_plus_10 exposes T1-T30 (the original twenty "
         "plus the ten added operators); all additionally exposes "
         "auto-generated technique cards. Holding this fixed across a pair of "
         "runs is what separates the operator-catalogue effect from the "
         "control-plane effect.",
)
parser.add_argument(
    "--canonical_backends",
    choices=["off", "on"],
    default="off",
    help="Whether deterministic hand-written operator backends may replace a "
         "model-generated candidate (the fixed-point Sobel T2 backend and the "
         "T27 nibble-lookup backend). OFF for any comparison: a win produced by "
         "a hand-written benchmark-specific implementation measures that "
         "implementation, not the planner. Turn ON only to report the "
         "system-backend condition as a separate, separately-labelled arm.",
)
parser.add_argument(
    "--identity_gate",
    choices=["strict", "advisory", "off"],
    default="strict",
    help="Verify during generation that the declared knob defaults reproduce "
         "the original function. 'strict' (default) rejects the candidate and "
         "asks the model to repair it. Checking this only inside BayesOpt meant "
         "a broken identity point discarded the whole function's plan with no "
         "chance of repair.",
)
parser.add_argument(
    "--target_build_gate",
    choices=["strict", "advisory", "off"],
    default="strict",
    help="How to treat the MSP430 cross-build and memory check. 'strict' "
         "(default) rejects a candidate unless the target build is verified, "
         "so a missing toolchain can never be mistaken for a validated run. "
         "'advisory' logs the outcome without blocking. 'off' disables it.",
)
parser.add_argument(
    "--target_functions",
    type=str,
    default=None,
    help="Path to a JSON file freezing the list of functions to approximate. "
         "Target selection is otherwise an LLM call made independently in each "
         "run, so paired conditions silently approximated different function "
         "sets (AR: 7 functions in one arm, 9 in the other) and the comparison "
         "was uncontrolled. Pass the same file to every arm of a comparison.",
)
parser.add_argument(
    "--write_target_functions",
    type=str,
    default=None,
    help="Write the target functions this run selected to a JSON file, so a "
         "later paired run can be frozen to the same set.",
)
parser.add_argument(
    "--per_function_operators",
    dest="per_function_operators",
    action="store_true",
    default=True,
    help="CHAI control plane: ask the model which feasible operators actually "
         "apply to each individual function before generation. Without this "
         "every function sees the identical app-wide set.",
)
parser.add_argument(
    "--no_per_function_operators",
    dest="per_function_operators",
    action="store_false",
    help="Disable per-function operator narrowing (app-wide set for every function).",
)
parser.add_argument(
    "--max_operators_per_function",
    type=int,
    default=4,
    help="Cap on operators selected for one function. Bounds plan "
         "dimensionality so the BO budget is not spread over 16 knobs.",
)
parser.add_argument("--opt_level", type=int, choices=[0, 1, 2, 3], default=3,
                    help="How much of the hardware cost-ranked feasible pool to keep: "
                         "level 0 keeps the cheapest 25%%, level 1 the cheapest 50%%, "
                         "level 2 the cheapest 75%%, level 3 keeps all. Applied once "
                         "app-wide; per-function narrowing happens afterwards.")

args = parser.parse_args()

# Resolve --hardware_profile alias
if args.hardware_profile and not args.hardware:
    args.hardware = args.hardware_profile

Dprint(f"Runing Benchmark App: {args.bm_name}")
app_name = args.bm_name
_CATALOG_RANGES = {
    "checkmate_20": set(range(1, 21)),
    "checkmate_plus_10": set(range(1, 31)),
}
operator_catalog_ids = _CATALOG_RANGES.get(args.operator_catalog)
include_auto_generated_cards = operator_catalog_ids is None
catalog_feasible = get_applicable_techniques(app_name)
if operator_catalog_ids is not None:
    catalog_feasible = {
        tid: technique for tid, technique in catalog_feasible.items()
        if tid in operator_catalog_ids
    }
catalog_constraint_text = (
    f"APPLICATION-SPECIFIC OPERATOR CONTRACT for {app_name}:\n"
    "Select and declare operators ONLY from this applicable set:\n"
    + "\n".join(
        f"  T{tid}: {technique.name}"
        for tid, technique in sorted(catalog_feasible.items())
    )
    + "\nDo not mention or apply any other operator ID."
)
if operator_catalog_ids is not None:
    print(
        f"[catalog] Fixed operator menu '{args.operator_catalog}': "
        f"{len(operator_catalog_ids)} IDs "
        f"(T{min(operator_catalog_ids)}-T{max(operator_catalog_ids)}); "
        "auto-generated cards excluded."
    )

# --- Prepare a clean slate BEFORE anything else ---
_prepare_fresh_run(app_name_for_cleanup=app_name)

alreadyRun = False
if args.no_llm:
    Dprint(f"Using LLM prerun outputs...")
    alreadyRun = True

    def copy_prerun_outputs(bm_name):
        source_dir = args.prerun_dir or os.path.join("llm-prerun", bm_name)
        if not os.path.exists(source_dir):
            Dprint(f"Source directory does not exist: {source_dir}")
            return

        for folder in os.listdir(source_dir):
            src_path = os.path.join(source_dir, folder)
            dest_path = os.path.join(".", folder)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                Dprint(f"Copied {folder} to current directory.")
            else:
                Dprint(f"Skipping non-folder item: {folder}")

    copy_prerun_outputs(args.bm_name)
    

# ---------------------------------------------------------------------------
#  Hardware-Aware Approximation Planning
# ---------------------------------------------------------------------------
hw_profile = None          # None → legacy behaviour (no filtering)
hw_constraint_text = ""    # injected into LLM prompts when non-empty
selection_constraint_text = catalog_constraint_text
hw_feasible = None         # dict[int, TechniqueEntry]  |  None  (app-wide, post opt_level slice)
hw_ranked = None           # sorted list of (id, name, cost)  |  None  (app-wide, post opt_level slice)
hw_rejection_log = []      # human-readable rejection lines (app-level, for startup printout)
hw_llm_raw_responses = []  # collected raw LLM planning/approx outputs
hw_validated_ids = set()   # final validated technique IDs
hw_invalid_ids = set()     # IDs rejected by post-LLM validator
hw_reprompted_ids = set()  # LLM references rejected before JSON conversion
hw_clamped_ids = []        # IDs force-removed from apx_all.json
hw_clamped_entries = []    # original JSON plan entries removed by the clamp

# ---------------------------------------------------------------------------
#  Experiment Instrumentation Counters
# ---------------------------------------------------------------------------
exp_api_calls = 0          # total LLM API invocations
exp_compile_failures = 0   # total compilation failures
exp_compile_retries_per_func = {}  # function → retry count
exp_hw_reprompts = 0       # HW validator reprompts
exp_technique_clamped = 0  # techniques force-removed post-LLM
exp_functions_attempted = 0
exp_functions_succeeded = 0
exp_target_build_status = {}   # function -> MSP430 build/memory gate result
exp_function_operator_sets = {}  # function -> operators selected for it
exp_functions_no_operator = 0    # functions the selector correctly declined
exp_approved_plans = {}          # function -> structured, approved plan
MAX_PLAN_CONTRACT_RETRIES = 3    # plan reprompts before giving up on a function
exp_plan_mismatches = 0          # generations that departed from the approved plan
exp_plan_contract_failures = 0   # functions with no parseable, in-contract plan
exp_candidate_rejections = {}    # rejection kind -> count (semantic vs build)
exp_canonical_backend_used = {}  # function -> canonical backend actions applied
exp_post_repair_mismatches = 0   # candidates that diverged from the plan after repair
exp_conformance_violations = {}  # function -> operator-label violations (both arms)
exp_identity_rejections = 0      # candidates whose declared identity was not the identity
exp_direct_extractions = 0       # plan entries built without a JSON round trip
exp_json_conversions = 0         # fallback JSON conversion calls
exp_target_build_skipped = False  # True when the cross toolchain was absent


def record_generation_rejection(
    function_name, attempt, reason, candidate=None, technique_ids=None
):
    """Persist full generation/repair failures for post-run diagnosis."""
    from datetime import datetime

    os.makedirs("logs", exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "app": app_name,
        "function": function_name,
        "attempt": attempt,
        "reason": str(reason),
        "technique_ids": sorted(technique_ids or []),
        "candidate": candidate,
    }
    with open(
        f"logs/generation_rejections_{app_name}.jsonl", "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(payload) + "\n")


def load_frozen_targets(path, app_name):
    """Load a frozen (target list, code summary) pair for *app_name*.

    Accepts the current schema ``{"app", "target_functions", "code_summary"}``
    and the earlier ``{app_name: {...}}`` / bare-list shapes.
    """
    with open(path, "r", encoding="utf-8") as file:
        frozen = json.load(file)

    summary = ""
    if isinstance(frozen, dict) and "target_functions" in frozen:
        stored_app = frozen.get("app")
        if stored_app and stored_app != app_name:
            raise SystemExit(
                f"[target selection] {path} was frozen for {stored_app!r} but "
                f"this run is {app_name!r}. Use one frozen file per benchmark."
            )
        targets = frozen["target_functions"]
        summary = frozen.get("code_summary", "") or ""
    elif isinstance(frozen, dict) and app_name in frozen:
        targets = frozen[app_name]
    else:
        targets = frozen

    if isinstance(targets, list):
        targets = {name: "approximate" for name in targets}
    if not isinstance(targets, dict) or not targets:
        raise SystemExit(
            f"[target selection] {path} contains no frozen target functions "
            f"for {app_name}"
        )
    if not any(str(choice).lower() == "approximate" for choice in targets.values()):
        raise SystemExit(
            f"[target selection] {path} marks no function as 'approximate' "
            f"for {app_name}"
        )
    if not summary:
        print(
            Fore.YELLOW
            + f"[target selection] WARNING: {path} carries no frozen code "
              "summary; arms will differ in that part of their context."
            + Style.RESET_ALL
        )
    return targets, summary


_ground_truth_ready = False


def ensure_ground_truth(app_name):
    """Generate the reference output once, on first use."""
    global _ground_truth_ready
    if _ground_truth_ready:
        return True
    try:
        generateGroundTruth(app_name)
        _ground_truth_ready = True
        return True
    except Exception as error:
        print(
            Fore.YELLOW
            + f"[ground truth] Could not generate reference output for "
              f"{app_name}: {error}"
            + Style.RESET_ALL
        )
        return False


def build_function_constraint_text(function_feasible, function_name):
    """Render the operator contract for one function.

    ``function_feasible`` is ``None`` in prompt-only (operator-only baseline)
    mode, where no control plane exists and the app-wide catalogue text is used
    unchanged. Under CHAI it is the narrowed per-function set.
    """
    if function_feasible is None:
        return selection_constraint_text

    lines = [
        f"OPERATOR CONTRACT for {function_name} on this target:",
        "These are the only operators that are BOTH feasible on this hardware "
        "and applicable to this function. Select only from this list:",
    ]
    for tid, technique in sorted(function_feasible.items()):
        lines.append(f"  T{tid}: {technique.name} — {technique.description}")
    lines.append(
        "Do not mention or apply any other operator ID. If none of these fit "
        "this function, say so rather than inventing an operator."
    )
    text = "\n".join(lines)
    if hw_profile is not None and not hw_profile.has_fpu:
        text += (
            "\n\nNO-FPU IMPLEMENTATION CONTRACT:\n"
            "- This target has no floating-point unit. Every float operation is "
            "software-emulated and costs cycles.\n"
            "- If selecting T2 Precision Scaling, implement a knob-controlled "
            "integer or fixed-point arithmetic path. Multiplying a "
            "floating-point output by a scale knob is invalid: it retains the "
            "software-emulated FP cost and saves nothing.\n"
            "- If selecting T6 Probabilistic Algorithms, implement an actual "
            "random or pseudorandom decision controlled by a probability or "
            "sampling knob.\n"
            "- Never introduce a new floating-point literal or float variable "
            "to express a loop bound or a stride. Use integer arithmetic.\n"
        )
    return text


def record_generation_repair(function_name, attempt, actions):
    """Persist deterministic repairs separately from LLM retry evidence."""
    from datetime import datetime

    os.makedirs("logs", exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "app": app_name,
        "function": function_name,
        "attempt": attempt,
        "actions": list(actions),
    }
    with open(
        f"logs/generation_repairs_{app_name}.jsonl", "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(payload) + "\n")


if args.hardware and args.hardware_control == "chai":
    from lib.hardware_profile import load_hardware_profile
    from lib.constraint_gatekeeper import (
        filter_feasible_techniques,
        format_feasible_list,
        format_rejection_report,
    )
    from lib.hardware_cost_model import (
        rank_techniques,
        format_ranked_list,
        select_by_opt_level,
    )
    from lib.hw_validator import (
        extract_technique_ids_from_text,
        extract_technique_ids_from_convo,
        extract_technique_ids_from_json,
        validate_technique_ids,
        build_reprompt_error,
        clamp_apx_json,
    )
    from lib.hw_observability import (
        collect_infeasible_plan_entries,
        write_approximation_plan_report,
        write_rejection_json,
        write_constraints_txt,
    )

    hw_profile = load_hardware_profile(args.hardware)

    # Compact startup line (Task A.4)
    print(hw_profile.startup_line())

    # App-wide feasible set (hardware + applicable_apps filtering).
    hw_feasible, hw_rejection_log = filter_feasible_techniques(app_name, hw_profile)
    if operator_catalog_ids is not None:
        hw_feasible = {
            tid: technique for tid, technique in hw_feasible.items()
            if tid in operator_catalog_ids
        }

    # Keep the FULL app-level feasible set around for observability / final
    # clamp step, since hw_feasible itself gets overwritten below by the
    # opt_level-sliced version that every function will actually use.
    hw_feasible_full = hw_feasible

    # Full ranking, shown once for visibility/debugging.
    hw_ranked_full = rank_techniques(hw_feasible_full, hw_profile)

    # ---- Apply opt_level slicing ONCE, app-wide ----
    # opt_level selects how aggressively the cost model prunes the pool: lower
    # levels keep only the cheapest fraction of the hardware-ranked list.
    #
    # The previous implementation did the opposite. It sorted cheapest-first
    # and then sliced with ``hw_ranked_full[3 - opt_level:]``, which *removed*
    # the cheapest operators and exposed the expensive tail — the reverse of
    # what a hardware cost model is for. ``select_by_opt_level`` implements the
    # intended behaviour and was dead code until now.
    hw_ranked = select_by_opt_level(hw_ranked_full, args.opt_level)
    hw_feasible = {tid: hw_feasible_full[tid] for tid, _, _ in hw_ranked}
    dropped = len(hw_ranked_full) - len(hw_ranked)
    hw_constraint_text = format_feasible_list(hw_feasible)
    selection_constraint_text = (
        hw_constraint_text
        + "\nHARDWARE COST RANKING (prefer lower-cost feasible techniques "
          "when they fit the function):\n"
        + format_ranked_list(hw_ranked)
    )
    if not hw_profile.has_fpu:
        selection_constraint_text += (
            "\n\nNO-FPU IMPLEMENTATION CONTRACT:\n"
            "- If selecting T2 Precision Scaling, implement a knob-controlled "
            "integer or fixed-point arithmetic path. Multiplying a floating-point "
            "output by a scale knob is invalid and retains software-emulated FP cost.\n"
            "- If selecting T6 Probabilistic Algorithms, implement an actual "
            "random or pseudorandom decision controlled by a probability/sampling knob."
        )

    print(f"  [OPT] Hardware feasible set size (app-wide): {len(hw_feasible_full)} "
          f"→ opt_level={args.opt_level} → kept the {len(hw_feasible)} "
          f"cheapest-ranked (dropped {dropped} most expensive) → "
          f"{len(hw_feasible)}/{len(hw_feasible_full)} operators available "
          f"for {app_name}")

    # ---- Terminal output ----
    print(Back.CYAN + Fore.BLACK)
    print("=" * 60)
    print("  HARDWARE-AWARE APPROXIMATION PLANNING")
    print("=" * 60)
    print(Style.RESET_ALL)
    print(f"  Target : {hw_profile.summary()}")
    print(f"  App    : {app_name}")
    print()
    print(Fore.GREEN + "  Full app-wide feasible ranking (debug):" + Style.RESET_ALL)
    print(format_ranked_list(hw_ranked_full))
    print()
    print(Fore.GREEN + f"  Opt-level {args.opt_level} slice — kept the {len(hw_ranked)} cheapest-ranked operators:" + Style.RESET_ALL)
    print(format_ranked_list(hw_ranked))
    print()
    if hw_rejection_log:
        print(Fore.RED + "  Rejected techniques (app-wide):" + Style.RESET_ALL)
        print(format_rejection_report(hw_rejection_log))
        print()
    print("-" * 60)
    print()

    # ---- Write observability: rejection JSON (Task D.2) ----
    rej_path = write_rejection_json(app_name, hw_profile, hw_feasible_full, hw_rejection_log)
    print(f"  → Rejection log: {rej_path}")
elif args.hardware:
    from lib.hardware_profile import load_hardware_profile

    hw_profile = load_hardware_profile(args.hardware)
    raw_profile_text = (
        "TARGET HARDWARE INPUT (raw profile; no automatic filtering, ranking, "
        "validation, or clamping):\n"
        f"  name={hw_profile.name}\n"
        f"  ram_kb={hw_profile.ram_kb}\n"
        f"  has_fpu={str(hw_profile.has_fpu).lower()}\n"
        f"  has_simd={str(hw_profile.has_simd).lower()}\n"
        f"  energy_budget_mj={hw_profile.energy_budget_mj}\n"
        f"  clock_mhz={hw_profile.clock_mhz}\n"
        "Use these facts when choosing from the complete operator catalogue."
    )
    selection_constraint_text = raw_profile_text + "\n\n" + catalog_constraint_text
    hw_feasible_full = None
    hw_ranked_full = None
    print(hw_profile.startup_line())
    print("[comparison] Hardware profile supplied in prompt-only mode; "
          "CHAI gate/rank/validate/clamp disabled.")
else:
    hw_feasible_full = None
    hw_ranked_full = None

copyFiles(f"benchmark_applications/{app_name}/",f"eval-apps/{app_name}/")

# copy only the Makefile, .c and .h files in eval_apps/{app_name}/ to target/
def copyEvalAppSource(app_name, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    source_dir = os.path.join("eval-apps", app_name)
    for pattern in ("*.c", "*.h"):
        for source_path in _glob_mod.glob(os.path.join(source_dir, pattern)):
            shutil.copy2(source_path, target_dir)
    for filename in ("Makefile", "application.txt"):
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_dir)

copyEvalAppSource(app_name, "target/")

from lib.llm import (
    purposeIdentificationFunction,
    annotateFunction,
    planStepFunction,
    approximateFunction,
    convertJson,
    findTargetFunctions,
    findApplicableTechniquesForFunction,
)

import lib.bo as bo_module
from lib.bo import runBayesOpt

from lib.pdg import (
    initPDGGen
)
from utils.models import AnnotateData, ApproximatedData
from config.globals import CHAT_HISTORY, PLATFORM_ARCHITECTURE, ENTITIES, TARGET_FILES
from config.config import GIVE_FORMAT_EXAMPLES, GIVE_LOOP_PERF_EXMAPLES
from utils.error_analyzer import generateGroundTruth
from utils.checkpoints import checkpointOrchestration
import csv

# ----------- README FOLLOW ALONG START HERE -----------

# Enable deterministic tracing if requested
if args.trace_approx:
    Dprint("Approximation tracing ENABLED - set deterministic seeds")
    set_deterministic_seeds(42)

PDG, topological_order = initPDGGen()

# LLM API initialization already been done in lib/llm.py.


# Load filenames of target files

loadTargetFiles("target/")
platform_archi = PLATFORM_ARCHITECTURE

# Load global context (System prompt, approximation summry, few shot examples)
loadGlobalContext()
loadFormatExamples()
loadLoopPerfExamples()
# Load only the few-shot examples for operators this run can actually select.
# Loading all ten for every application cost ~10.6K characters of context on
# every model call, most of it about operators the gatekeeper had already ruled
# out for this target.
_example_scope = (
    set(hw_feasible_full) if hw_feasible_full is not None
    else set(catalog_feasible)
)
loadNewTechniqueExamples(
    include_auto_generated_cards=include_auto_generated_cards,
    feasible_ids=_example_scope,
)

# Parse target files to extract entities (functions, structs, global variables) and load them.
parseFunctions()
loadEntities()

# Recover any locally defined functions clangd omitted. The dependency graph
# may still contain external/library nodes; those remain unresolved and are
# rejected by filterSourceTargets below.
known_entity_names = {
    entity.get("functionName") for entity in ENTITIES if isinstance(entity, dict)
}
missing_pdg_entities = [
    function for function in topological_order if function not in known_entity_names
]
recovered_entities = recover_source_entities(
    missing_pdg_entities, TARGET_FILES, base_dir="target"
)
if recovered_entities:
    ENTITIES.extend(recovered_entities)
    print(
        "[entity recovery] Recovered local source definitions: "
        + ", ".join(sorted(entity["functionName"] for entity in recovered_entities))
    )

if args.preflight_only:
    entity_names = {
        entity.get("functionName") for entity in ENTITIES if isinstance(entity, dict)
    }
    unresolved_nodes = sorted(
        set(topological_order) - entity_names - {"main"}
    )
    print(
        f"[preflight] app={app_name} entities={len(entity_names)} "
        f"recovered={len(recovered_entities)} unresolved={unresolved_nodes}"
    )

    # Freezing the target list is a preflight activity: a comparison campaign
    # runs target selection ONCE and hands the same file to every arm, instead
    # of letting each arm make its own independent selection call.
    if args.write_target_functions:
        preflight_prompts = getPromptTemplates(
            include_auto_generated_cards=include_auto_generated_cards
        )
        preflight_targets, _preflight_summary = findTargetFunctions(
            preflight_prompts["targetFunctionsPrompt"], fallback_app_name=app_name
        )
        os.makedirs(os.path.dirname(args.write_target_functions) or ".", exist_ok=True)
        with open(args.write_target_functions, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "app": app_name,
                    "target_functions": preflight_targets,
                    "code_summary": _preflight_summary,
                },
                file,
                indent=2,
            )
        selected = sorted(
            name for name, choice in (preflight_targets or {}).items()
            if str(choice).lower() == "approximate"
        )
        print(
            f"[preflight] Froze {len(selected)} target function(s) for {app_name} "
            f"in {args.write_target_functions}: {selected}"
        )
    raise SystemExit(0)

if not alreadyRun:

    # Load and create prompt templates.
    prompts = getPromptTemplates(
        include_auto_generated_cards=include_auto_generated_cards
    )

    # Load output scheama.
    annotatedDataParser = JsonOutputParser(pydantic_object=AnnotateData)
    approximatedDataParser = JsonOutputParser(pydantic_object=ApproximatedData)

    # Load output format instructions for LLM.
    output_format_instructions_anno = annotatedDataParser.get_format_instructions()
    output_format_instructions_apx = approximatedDataParser.get_format_instructions()


    # Intialize approximated functions dirctrionary
    approximated_functions_dict = {}
    
    # Get list of target functions and code base summary.
    #
    # A frozen file supplies BOTH the target list and the code-base summary and
    # skips this call entirely. Calling findTargetFunctions and then overwriting
    # only the target list would still give each arm a differently-worded
    # summary in its context -- an uncontrolled difference between conditions --
    # and would spend two model calls per arm to produce a discarded answer.
    if args.target_functions:
        target_functions, code_summary = load_frozen_targets(
            args.target_functions, app_name
        )
        selected = sorted(
            name for name, choice in target_functions.items()
            if str(choice).lower() == "approximate"
        )
        print(
            f"[target selection] FROZEN from {args.target_functions}: "
            f"{len(selected)} target(s) {selected}; code-base summary reused; "
            "no model call made."
        )
    else:
        target_functions, code_summary = findTargetFunctions(
            prompts["targetFunctionsPrompt"], fallback_app_name=app_name
        )

    if args.write_target_functions:
        os.makedirs(os.path.dirname(args.write_target_functions) or ".", exist_ok=True)
        with open(args.write_target_functions, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "app": app_name,
                    "target_functions": target_functions,
                    "code_summary": code_summary,
                },
                file,
                indent=2,
            )
        print(f"[target selection] Wrote frozen target list: {args.write_target_functions}")

    # Load code_base summary
    loadCodeBaseSummary(code_summary)

    # Exclude PDG-only/library nodes before any LLM call. For FFT this blocks
    # libc/libm functions such as cos/exp/sin/cexp that have no source body.
    filtered_topological_order, rejected_source_targets = filterSourceTargets(
        target_functions, topological_order, ENTITIES
    )
    if rejected_source_targets:
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/target_rejections_{app_name}.json", "w") as file:
            json.dump(rejected_source_targets, file, indent=2)
        for rejected in rejected_source_targets:
            print(
                f"[target validation] Skipping {rejected['function']}: "
                f"{rejected['reason']}"
            )
    # Iterate over all functions
    for this_function in filtered_topological_order:
        print(Back.WHITE)
        print(Fore.BLACK)
        print("APPROXIMATING FUNCTION: " + this_function + "\n")
        print(Style.RESET_ALL)
        """
            Step 0: Parent Entities (Functions) Context
        """

        this_context = manufacturerContext(PDG, this_function)
        Dprint("\n\n\n\n --- start \n\n")
        Dprint(this_context)

        """
            Step 0.5: Hardware-aware operator constraint text

            hw_feasible is the app-wide feasible set, computed once before this
            loop (hardware capability + RAM gate, applicable_apps, then the
            opt_level cost slice). It is the *starting* set for this function;
            Step 1.5 narrows it further to the operators whose structure
            actually exists in this function.
        """
        function_feasible = hw_feasible
        if hw_feasible_full is not None:
            print(f"  [OPT] {this_function}: {len(hw_feasible)}/{len(hw_feasible_full)} "
                  f"techniques feasible app-wide (opt_level={args.opt_level})")

        """
            Step 1: Identify this_function's purpose
        """

        # chain = purposePrompt | llmLangChain

        exp_api_calls += 1  # purpose identification call
        this_purpose_convo = purposeIdentificationFunction(
            this_function,
            this_context,
            prompts["purposePrompt"]
        )

        # Add purpose identification convo to history(context) object
        this_context = this_context + this_purpose_convo
        Dprint("\n\n\n\n --- Perpose \n\n")
        Dprint(this_context)

        """
            Step 1.5: Per-function operator narrowing (CHAI control plane only)

            Every function previously received the identical app-wide feasible
            set. The registry only tags operators by applicable_apps, so a
            spectral operator was offered to a bounds-check helper and a
            loop operator to a function with no loop. That is why FFT's plan
            carried unrelated operator suggestions. Narrowing per function is
            a control-plane decision and therefore belongs to CHAI, not to the
            operator-only baseline.
        """
        this_tech_select_convo = []
        # Counted here, before any arm-specific branch, so every arm reports the
        # same denominator for the same frozen target list.
        exp_functions_attempted += 1
        if hw_feasible is not None and args.per_function_operators:
            exp_api_calls += 1
            this_tech_select_convo, function_operator_ids = (
                findApplicableTechniquesForFunction(
                    this_function=this_function,
                    this_context=this_context,
                    feasible_techniques=hw_feasible,
                    techniqueSelectionPrompt=prompts["operatorSelectionPrompt"],
                    max_operators=args.max_operators_per_function,
                    ranked_fallback=hw_ranked,
                )
            )
            exp_function_operator_sets[this_function] = sorted(function_operator_ids)

            if not function_operator_ids:
                print(
                    Fore.CYAN
                    + f"  [OPT] {this_function}: no applicable operator; "
                      "skipping without spending generation budget."
                    + Style.RESET_ALL
                )
                exp_functions_no_operator += 1
                CHAT_HISTORY[this_function] = (
                    this_purpose_convo + this_tech_select_convo
                )
                continue

            function_feasible = {
                tid: hw_feasible[tid]
                for tid in function_operator_ids
                if tid in hw_feasible
            }
            print(
                f"  [OPT] {this_function}: narrowed to "
                f"{len(function_feasible)}/{len(hw_feasible)} operators "
                f"{sorted(function_feasible)}"
            )
            this_context = this_context + this_tech_select_convo

        """
            Step 2: Annotate the function
        """

        this_plan_anno_convo = None
        function_code_annotated = ""
        approved_plan = None

        # The constraint text handed to planning and generation must describe
        # the operators actually allowed for THIS function. Previously it was
        # the app-wide list, so the model was invited to use operators the
        # control plane would later reject.
        function_constraint_text = build_function_constraint_text(
            function_feasible, this_function
        )

        if TEXT_PLANING:
            # Inject hardware-aware planning text into the platform architecture
            # description so the LLM sees the constraint during planning.
            hw_aware_archi = platform_archi + "\n\n" + function_constraint_text

            exp_api_calls += 1  # planning step call
            this_plan_anno_convo = planStepFunction(
                this_function=this_function,
                this_context=this_context,
                planningPrompt=prompts["planningPrompt"],
                platform_architecure=hw_aware_archi,
                max_operators=args.max_operators_per_function,
            )

            # Parse the plan into a checkable contract. Under CHAI's control
            # plane an unusable or out-of-contract plan is sent back before any
            # code is generated; in prompt-only mode the plan stays advisory so
            # the baseline keeps its original, unenforced behaviour.
            plan_text = "\n".join(
                str(message) for role, message in this_plan_anno_convo if role == "ai"
            )
            this_function_code = (
                getFunctionData(ENTITIES, this_function) or {}
            ).get("completeFunction", "")
            approved_plan, plan_errors = parse_plan(
                plan_text,
                max_operators=args.max_operators_per_function,
                function_code=this_function_code,
            )
            if function_feasible is not None:
                plan_errors = plan_errors + validate_plan_against_catalogue(
                    approved_plan, set(function_feasible)
                )
                plan_attempts = 0
                while plan_errors and plan_attempts < MAX_PLAN_CONTRACT_RETRIES:
                    plan_attempts += 1
                    print(
                        Fore.YELLOW
                        + f"  [plan contract] {this_function}: rejecting plan "
                          f"(attempt {plan_attempts}) — {plan_errors[0]}"
                        + Style.RESET_ALL
                    )
                    record_generation_rejection(
                        this_function,
                        plan_attempts,
                        "plan contract violation: " + "; ".join(plan_errors),
                        candidate={"plan_text": plan_text},
                    )
                    exp_api_calls += 1
                    this_plan_anno_convo = planStepFunction(
                        this_function=this_function,
                        this_context=this_context,
                        planningPrompt=prompts["planningPrompt"],
                        platform_architecure=hw_aware_archi,
                        max_operators=args.max_operators_per_function,
                        prev_err=(
                            "Your previous plan was rejected. Fix these problems "
                            "and answer again in the required format:\n- "
                            + "\n- ".join(plan_errors)
                        ),
                    )
                    plan_text = "\n".join(
                        str(message)
                        for role, message in this_plan_anno_convo
                        if role == "ai"
                    )
                    approved_plan, plan_errors = parse_plan(
                        plan_text,
                        max_operators=args.max_operators_per_function,
                        function_code=this_function_code,
                    )
                    plan_errors = plan_errors + validate_plan_against_catalogue(
                        approved_plan, set(function_feasible)
                    )

                exp_approved_plans[this_function] = approved_plan.as_dict()

                if approved_plan.declined:
                    print(
                        Fore.CYAN
                        + f"  [plan contract] {this_function}: planner found no "
                          "applicable operator; skipping generation."
                        + Style.RESET_ALL
                    )
                    exp_functions_no_operator += 1
                    CHAT_HISTORY[this_function] = (
                        this_purpose_convo
                        + this_tech_select_convo
                        + this_plan_anno_convo
                    )
                    continue

                if plan_errors:
                    print(
                        Fore.RED
                        + f"  [plan contract] {this_function}: no valid plan after "
                          f"{MAX_PLAN_CONTRACT_RETRIES} attempts — skipping."
                        + Style.RESET_ALL
                    )
                    exp_plan_contract_failures += 1
                    CHAT_HISTORY[this_function] = (
                        this_purpose_convo
                        + this_tech_select_convo
                        + this_plan_anno_convo
                    )
                    continue
        else:
            function_code_annotated, this_plan_anno_convo = annotateFunction(
                this_function, 
                this_context, 
                prompts["annotationPrompt"], 
                output_format_instructions_anno,
                annotatedDataParser,
            )


        # Add annotation convo to history(context) object
        this_context = this_context + this_plan_anno_convo # Add the annotation conversation context for approximation prompt
        Dprint("\n\n\n\n --- Planning \n\n")
        Dprint(this_context)

        err_approximation = ""
        # Prepend hardware constraints to the approximation error channel
        # so the LLM sees them in {add_error} on the first iteration.
        if function_constraint_text:
            err_approximation = function_constraint_text + "\n"
        compile_retries = 0
        MAX_COMPILE_RETRIES = 3
        selection_retries = 0
        MAX_SELECTION_RETRIES = 3
        failed_technique_ids = set()
        base_generation_context = list(this_context)
        generation_attempt = 0
        this_approx_convo = []
        this_approx_json_convo = []
        while True:
            """
                Step 3: Approximate the function and test Compilation and Validation
            """
            
            generation_attempt += 1
            # After a failed candidate, use only the focused compiler-grounded
            # request. Keeping every rejected response in the long context made
            # Ollama/JSON conversion repeat the same broken code verbatim.
            attempt_context = generation_attempt_context(
                base_generation_context, generation_attempt
            )
            exp_api_calls += 1  # approximation call
            this_approx_convo = approximateFunction(
                this_function=this_function,
                this_context=attempt_context,
                approximation_prompt=prompts["approximationPrompt"],
                prev_err=err_approximation
            )

            current_technique_ids = extract_technique_ids_from_convo(
                this_approx_convo
            )

            # Operator provenance must come from the model response itself;
            # never infer it from the prompt or from the generated C shape.
            if not current_technique_ids:
                selection_retries += 1
                reason = (
                    "Operator provenance missing. Begin the response with "
                    "APPLIED_TECHNIQUES: T<number>, ... using at least one "
                    "exact ID from the supplied catalogue."
                )
                record_generation_rejection(
                    this_function,
                    generation_attempt,
                    reason,
                    candidate={"conversation": this_approx_convo},
                )
                if selection_retries >= MAX_SELECTION_RETRIES:
                    print(
                        Fore.RED
                        + f"  [Operator provenance] {this_function} omitted "
                          f"catalog IDs after {MAX_SELECTION_RETRIES} attempts - skipping."
                        + Style.RESET_ALL
                    )
                    exp_compile_retries_per_func[this_function] = compile_retries
                    approximated_functions_dict.pop(this_function, None)
                    failed_plan_path = f"approximated_functions/apx_{this_function}.json"
                    if os.path.exists(failed_plan_path):
                        os.remove(failed_plan_path)
                    break
                err_approximation = reason
                continue

            # --- Plan conformance: generated code must match the approved plan ---
            # Operator provenance was previously read only from the generation
            # response, so the coder could add or drop operators after planning
            # and nothing compared the two. Hold it to the contract.
            if (
                function_feasible is not None
                and approved_plan is not None
                and approved_plan.records
            ):
                plan_mismatch, corrected_ids, plan_actions = (
                    check_implementation_matches_plan(
                        approved_plan, current_technique_ids
                    )
                )
                if not plan_mismatch and plan_actions:
                    # The candidate implemented a subset of the plan. Record
                    # what it actually contains rather than discarding it.
                    current_technique_ids = corrected_ids
                    record_generation_repair(
                        this_function, generation_attempt, plan_actions
                    )
                if plan_mismatch:
                    selection_retries += 1
                    record_generation_rejection(
                        this_function,
                        generation_attempt,
                        plan_mismatch,
                        candidate={"conversation": this_approx_convo},
                        technique_ids=current_technique_ids,
                    )
                    print(
                        Fore.RED
                        + f"  [plan conformance] {this_function}: {plan_mismatch}"
                        + Style.RESET_ALL
                    )
                    exp_plan_mismatches += 1
                    if selection_retries >= MAX_SELECTION_RETRIES:
                        print(
                            Fore.RED
                            + f"  [plan conformance] {this_function} never matched "
                              f"its approved plan after {MAX_SELECTION_RETRIES} "
                              "attempts - skipping."
                            + Style.RESET_ALL
                        )
                        exp_compile_retries_per_func[this_function] = compile_retries
                        approximated_functions_dict.pop(this_function, None)
                        failed_plan_path = (
                            f"approximated_functions/apx_{this_function}.json"
                        )
                        if os.path.exists(failed_plan_path):
                            os.remove(failed_plan_path)
                        break
                    err_approximation = plan_mismatch
                    continue

            # --- Post-LLM catalogue/app/hardware feasibility check ---
            selection_feasible = (
                function_feasible if function_feasible is not None else catalog_feasible
            )
            if selection_feasible is not None:
                chosen = current_technique_ids
                valid, invalid = validate_technique_ids(chosen, selection_feasible)
                if invalid:
                    selection_retries += 1
                    Dprint(f"[HW Validator] Infeasible IDs in LLM output: {invalid}")
                    print(Fore.RED + f"  [HW Validator] LLM chose infeasible technique(s): "
                          f"{sorted(invalid)} — reprompting once..." + Style.RESET_ALL)
                    if function_feasible is not None:
                        err_approximation = build_reprompt_error(invalid, function_feasible)
                        hw_invalid_ids |= invalid
                        hw_reprompted_ids |= invalid
                        exp_hw_reprompts += 1
                    else:
                        err_approximation = (
                            f"ERROR: operator IDs {sorted(invalid)} are not applicable "
                            f"to {app_name} in the fixed T1-T30 catalogue. Choose only: "
                            + ", ".join(f"T{tid}" for tid in sorted(selection_feasible))
                        )
                    # Collect raw response for observability
                    for role, txt in this_approx_convo:
                        if role == "ai":
                            hw_llm_raw_responses.append(str(txt))
                    record_generation_rejection(
                        this_function,
                        generation_attempt,
                        err_approximation,
                        candidate={"conversation": this_approx_convo},
                        technique_ids=current_technique_ids,
                    )
                    if selection_retries >= MAX_SELECTION_RETRIES:
                        print(
                            Fore.RED
                            + f"  [HW Validator] {this_function} still selected "
                              f"infeasible IDs after {MAX_SELECTION_RETRIES} attempts - skipping."
                            + Style.RESET_ALL
                        )
                        exp_compile_retries_per_func[this_function] = compile_retries
                        approximated_functions_dict.pop(this_function, None)
                        failed_plan_path = f"approximated_functions/apx_{this_function}.json"
                        if os.path.exists(failed_plan_path):
                            os.remove(failed_plan_path)
                        break
                    continue

            # Collect raw LLM response for observability
            if hw_feasible is not None:
                for role, txt in this_approx_convo:
                    if role == "ai":
                        hw_llm_raw_responses.append(str(txt))

            """
                Step 4: Convert approximation to JSON format and Compile and Validate
            """

            # Prefer direct extraction: the C is already in the generation
            # response and the approved plan already declares the knob schema,
            # so asking the model to retype its own code inside JSON is a
            # wasted call that can truncate or alter correct code.
            approximate_function = None
            this_approx_json_convo = []
            generation_text = "\n".join(
                str(message) for role, message in this_approx_convo if role == "ai"
            )
            if function_feasible is not None and approved_plan is not None:
                approximate_function = build_plan_entry(
                    this_function, generation_text, approved_plan
                )
                if approximate_function is not None:
                    exp_direct_extractions += 1
                    Dprint(
                        f"[direct extraction] {this_function}: built plan entry "
                        "from the generation response and the approved plan; "
                        "skipped the JSON conversion call."
                    )
                    alignment_actions = approximate_function.pop(
                        "planAlignmentActions", []
                    )
                    if alignment_actions:
                        record_generation_repair(
                            this_function, generation_attempt, alignment_actions
                        )

            if approximate_function is None:
                exp_api_calls += 1  # JSON conversion call
                exp_json_conversions += 1
                conversion_context = json_conversion_context(
                    base_generation_context,
                    this_approx_convo,
                    generation_attempt,
                )
                this_approx_json_convo, approximate_function = convertJson(
                    this_context=conversion_context,
                    convert_json_prompt=prompts["convertJsonPrompt"],
                    this_function=this_function,
                    output_format_parser=approximatedDataParser,
                )

            # Persist exact model-declared provenance with the executable plan.
            approximate_function["technique_numbers"] = sorted(current_technique_ids)

            # Canonical backends replace the model's candidate wholesale with a
            # hand-written, benchmark-specific implementation. That is a
            # legitimate system-level result but it is NOT evidence about the
            # planner, so it is off unless explicitly requested.
            if args.canonical_backends == "on":
                deterministic_repairs = repair_operator_candidate(
                    approximate_function,
                    current_technique_ids,
                    hw_profile if hw_feasible is not None else None,
                )
                if deterministic_repairs:
                    exp_canonical_backend_used[this_function] = list(
                        deterministic_repairs
                    )
            else:
                deterministic_repairs = []
            if "installed deterministic T27 nibble-lookup backend" in deterministic_repairs:
                discarded_claims = set(current_technique_ids) - {27}
                if discarded_claims:
                    deterministic_repairs.append(
                        "discarded unimplemented operator claim(s): "
                        + ", ".join(f"T{tid}" for tid in sorted(discarded_claims))
                    )
                # The canonical backend replaces the entire candidate, so T27
                # is the only operator present in the executable result.
                current_technique_ids = {27}
                approximate_function["technique_numbers"] = [27]
            if "installed deterministic no-FPU T2 fixed-point Sobel backend" in deterministic_repairs:
                discarded_claims = set(current_technique_ids) - {2}
                if discarded_claims:
                    deterministic_repairs.append(
                        "discarded operator claim(s) replaced by hardware backend: "
                        + ", ".join(f"T{tid}" for tid in sorted(discarded_claims))
                    )
                current_technique_ids = {2}
                approximate_function["technique_numbers"] = [2]
            deterministic_repairs.extend(
                repairFunctionCandidate(approximate_function)
            )

            # Small local models sometimes describe a valid T1/T2 transform
            # but append an added-operator ID that is not present in the C.
            # Preserve the executable, declared transform while removing only
            # the statically disproven extra claim. If no real claim remains,
            # the conformance validator below rejects the candidate normally.
            # Pruning false operator claims is a provenance correction, i.e.
            # control plane. Leave the baseline's declared labels untouched so
            # its mislabelling rate stays measurable.
            if hw_feasible is not None:
                current_technique_ids, provenance_repairs = (
                    prune_unimplemented_added_operator_claims(
                        approximate_function, current_technique_ids
                    )
                )
                deterministic_repairs.extend(provenance_repairs)
            if deterministic_repairs:
                record_generation_repair(
                    this_function, generation_attempt, deterministic_repairs
                )

            # Plan conformance was checked on the model's declared IDs before
            # any repair ran. Deterministic repairs, operator pruning, and
            # backend substitution all mutate technique_numbers afterwards, so
            # the executable candidate could still diverge from the approved
            # plan. Re-check the final artifact.
            if (
                function_feasible is not None
                and approved_plan is not None
                and approved_plan.records
            ):
                final_ids = set(
                    approximate_function.get("technique_numbers", [])
                ) or set(current_technique_ids)
                post_repair_mismatch, corrected_final, post_actions = (
                    check_implementation_matches_plan(approved_plan, final_ids)
                )
                if not post_repair_mismatch and post_actions:
                    current_technique_ids = corrected_final
                    approximate_function["technique_numbers"] = sorted(corrected_final)
                    record_generation_repair(
                        this_function, generation_attempt, post_actions
                    )
                if post_repair_mismatch:
                    selection_retries += 1
                    exp_plan_mismatches += 1
                    exp_post_repair_mismatches += 1
                    reason = (
                        "after deterministic repair the executable candidate no "
                        "longer matches the approved plan: " + post_repair_mismatch
                    )
                    record_generation_rejection(
                        this_function,
                        generation_attempt,
                        reason,
                        candidate=approximate_function,
                        technique_ids=sorted(final_ids),
                    )
                    print(
                        Fore.RED
                        + f"  [plan conformance/post-repair] {this_function}: {reason}"
                        + Style.RESET_ALL
                    )
                    approximated_functions_dict.pop(this_function, None)
                    if selection_retries >= MAX_SELECTION_RETRIES:
                        exp_compile_retries_per_func[this_function] = compile_retries
                        failed_plan_path = (
                            f"approximated_functions/apx_{this_function}.json"
                        )
                        if os.path.exists(failed_plan_path):
                            os.remove(failed_plan_path)
                        break
                    err_approximation = reason
                    continue

            approximated_functions_dict[this_function] = approximate_function
            writeFunctionsToJson(approximated_functions_dict, 'approximated_functions/apx')

            # Verify that model-declared added operators are actually present;
            # printing "T27" is not proof of a nibble lookup implementation.
            # Track *why* a candidate was rejected. Every rejection below used
            # to increment exp_compile_failures, so semantic and ABI rejections
            # were reported as compilation failures and the metrics could not
            # distinguish "the model wrote invalid C" from "the model claimed an
            # operator it did not implement".
            rejection_kind = None

            # Does the generated C actually contain the operators it declares?
            #
            # This is measured in BOTH arms but ENFORCED only under CHAI. The
            # protocol defines the operator-only baseline as an expanded menu
            # with no enforcement; rejecting its candidates for mislabelling
            # would hand the baseline a piece of CHAI's control plane and then
            # credit the result to the operator catalogue. Recording the
            # violation unenforced also yields the measurement that matters:
            # how often an unsupervised model's operator labels are fiction.
            conformance_error = validate_operator_conformance(
                approximate_function.get("completeFunction", ""),
                current_technique_ids,
            )
            if conformance_error:
                exp_conformance_violations.setdefault(this_function, []).append(
                    {
                        "attempt": generation_attempt,
                        "declared": sorted(current_technique_ids),
                        "error": conformance_error,
                        "enforced": hw_feasible is not None,
                    }
                )

            err_comp = conformance_error if hw_feasible is not None else ""
            if err_comp:
                rejection_kind = "operator_conformance"
            elif conformance_error:
                print(
                    Fore.YELLOW
                    + f"  [operator provenance] {this_function}: declared "
                      f"{sorted(current_technique_ids)} but the implementation "
                      "does not match. Recorded, NOT enforced (operator-only "
                      "baseline has no control plane)."
                    + Style.RESET_ALL
                )
            if not err_comp and hw_feasible is not None:
                err_comp = validate_hardware_operator_conformance(
                    approximate_function.get("completeFunction", ""),
                    current_technique_ids,
                    hw_profile,
                )
                if err_comp:
                    rejection_kind = "hardware_conformance"
            # Reject an ABI change before the compiler's more verbose errors.
            if not err_comp:
                err_comp = validateFunctionContract(approximate_function)
                if err_comp:
                    rejection_kind = "function_contract"
            if not err_comp:
                err_comp = compileTest(this_function)
                if err_comp:
                    rejection_kind = "host_compilation"

            # Host gcc is not the target. The acceptance gate above builds with
            # the fallback Makefile's `CC=gcc ... -lm`, so a candidate full of
            # software-emulated floating point, or one that overflows FRAM,
            # used to pass generation and fail only once BayesOpt was already
            # spending simulation budget on it. Under CHAI's control plane the
            # candidate must cross-compile for MSP430 and fit the device.
            # Verify the declared identity point NOW, while the model can
            # still be asked to fix it. Checking it inside BayesOpt meant the
            # only possible response was to discard the whole function's plan.
            if not err_comp and args.identity_gate != "off":
                if ensure_ground_truth(app_name):
                    try:
                        identity_ok, identity_message = verify_identity(
                            app_name, f"compilation_testing/ApxFiles_{this_function}"
                        )
                    except IdentityCheckUnavailable as unavailable:
                        identity_ok, identity_message = True, ""
                        print(
                            Fore.YELLOW
                            + f"  [identity gate] skipped for {this_function}: "
                              f"{unavailable}"
                            + Style.RESET_ALL
                        )
                    if not identity_ok:
                        exp_identity_rejections += 1
                        if args.identity_gate == "strict":
                            err_comp = identity_message
                            rejection_kind = "identity"
                            print(
                                Fore.RED
                                + f"  [identity gate] {this_function}: declared "
                                  "defaults are not the identity point"
                                + Style.RESET_ALL
                            )
                        else:
                            print(
                                Fore.YELLOW
                                + f"  [identity gate] advisory: {this_function} "
                                  "defaults are not the identity point"
                                + Style.RESET_ALL
                            )

            if (
                not err_comp
                and hw_feasible is not None
                and args.target_build_gate != "off"
            ):
                target_result = validate_target_build(
                    app_name, f"compilation_testing/ApxFiles_{this_function}"
                )
                exp_target_build_status[this_function] = target_result.as_dict()
                if target_result.status == "failed":
                    err_comp = target_result.message
                    rejection_kind = "target_build"
                elif target_result.status == "skipped":
                    exp_target_build_skipped = True
                    if args.target_build_gate == "strict":
                        # Fail closed. A gate that cannot run is not a gate that
                        # passed; treating "skipped" as acceptable is how an
                        # unvalidated run comes to look like a validated one.
                        err_comp = (
                            "MSP430 target build could not be verified: "
                            f"{target_result.message}. Under "
                            "--target_build_gate strict an unverifiable target "
                            "build invalidates the candidate. Install the cross "
                            "toolchain, or pass --target_build_gate advisory and "
                            "label the run as unvalidated."
                        )
                        rejection_kind = "target_build_unverified"
                        print(
                            Fore.RED
                            + f"  [target build] STRICT: rejecting {this_function} "
                              f"— {target_result.message}"
                            + Style.RESET_ALL
                        )
                    else:
                        print(
                            Fore.YELLOW
                            + f"  [target build] advisory: gate skipped for "
                              f"{this_function}: {target_result.message}"
                            + Style.RESET_ALL
                        )

            # err_val = validateFunction(approximated_functions_dict[this_function])

            # if err_comp or err_val:
            if err_comp:
                compile_retries += 1
                exp_candidate_rejections[rejection_kind or "unknown"] = (
                    exp_candidate_rejections.get(rejection_kind or "unknown", 0) + 1
                )
                if rejection_kind in ("host_compilation", "target_build"):
                    exp_compile_failures += 1
                failed_technique_ids |= current_technique_ids
                record_generation_rejection(
                    this_function,
                    compile_retries,
                    err_comp,
                    candidate=approximate_function,
                    technique_ids=current_technique_ids,
                )
                if compile_retries >= MAX_COMPILE_RETRIES:
                    print(Fore.RED + f"  [Max retries] {this_function} failed compilation after {MAX_COMPILE_RETRIES} attempts — skipping." + Style.RESET_ALL)
                    exp_compile_retries_per_func[this_function] = compile_retries
                    # Failed repair attempts are diagnostics, not executable
                    # approximation plans. Exclude them from apx_all.json so
                    # BayesOpt never evaluates their build failure.
                    approximated_functions_dict.pop(this_function, None)
                    failed_plan_path = f"approximated_functions/apx_{this_function}.json"
                    if os.path.exists(failed_plan_path):
                        os.remove(failed_plan_path)
                    break
                err_approximation = buildFocusedRepairRequest(
                    approximate_function,
                    err_comp,
                    failed_techniques=failed_technique_ids,
                )
                continue

            exp_functions_succeeded += 1
            if hw_feasible is not None:
                hw_validated_ids |= (
                    current_technique_ids & set(hw_feasible.keys())
                )
            exp_compile_retries_per_func[this_function] = compile_retries
            this_context = (
                base_generation_context
                + this_approx_convo
                + this_approx_json_convo
            )
            break

        """
            Step 5: Save conversation history
        """
        CHAT_HISTORY[this_function] = (
            this_purpose_convo
            + this_tech_select_convo
            + this_plan_anno_convo
            + this_approx_convo
            + this_approx_json_convo
        )

        with open("logs/conv1.txt","a") as file:
            file.write(f"Function {this_function}" + "\n")
            file.write(str(approximated_functions_dict) + "\n")
            file.write(str(CHAT_HISTORY) + "\n")

if not alreadyRun:
    if exp_functions_succeeded == 0:
        print(Back.RED + Fore.WHITE)
        print(
            f"NO FUNCTION WAS SUCCESSFULLY APPROXIMATED "
            f"(0/{exp_functions_attempted} attempted)\n"
        )
        print(Style.RESET_ALL)
    elif exp_functions_succeeded < exp_functions_attempted:
        print(Back.YELLOW + Fore.BLACK)
        print(
            f"PARTIAL: {exp_functions_succeeded}/{exp_functions_attempted} "
            f"functions approximated\n"
        )
        print(Style.RESET_ALL)
    else:
        print(Back.GREEN + Fore.BLACK)
        print(
            f"ALL FUNCTIONS APPROXIMATED "
            f"({exp_functions_succeeded}/{exp_functions_attempted})\n"
        )
        print(Style.RESET_ALL)

# Assemble validated plans before hardware clamping and before the early
# no-plan exit. Previously the clamp ran before apx_all.json existed.
if not alreadyRun:
    if os.path.isdir("approximated_functions"):
        joinJsonFiles("approximated_functions/", "apx", "apx_all.json")
    else:
        os.makedirs("approximated_functions", exist_ok=True)
        with open("approximated_functions/apx_all.json", "w") as file:
            json.dump([], file)

# ---------------------------------------------------------------------------
#  Post-LLM: final clamp + observability (Tasks C.2 / D.1)
# ---------------------------------------------------------------------------
if hw_feasible_full is not None:
    # Clamp apx_all.json — remove any infeasible entries that survived.
    # Use the FULL app-level feasible set here: a technique chosen under
    # the opt_level slice is still globally feasible hardware-wise, so
    # clamping must check against hw_feasible_full, not the sliced
    # hw_feasible.
    apx_path = "approximated_functions/apx_all.json"
    if os.path.exists(apx_path):
        # Also pick up technique_number from JSON
        json_ids = extract_technique_ids_from_json(apx_path)
        _, json_invalid = validate_technique_ids(json_ids, hw_feasible_full)
        hw_invalid_ids |= json_invalid
        # Preserve exact infeasible JSON plan entries before the hard clamp
        # removes them, so the run can report discarded plans precisely.
        hw_clamped_entries = collect_infeasible_plan_entries(
            apx_path, app_name, hw_profile, hw_feasible_full
        )
        hw_clamped_ids = clamp_apx_json(apx_path, hw_feasible_full)
        if hw_clamped_ids:
            print(Fore.RED + f"  [HW Validator] Clamped infeasible technique(s) "
                  f"from apx_all.json: {hw_clamped_ids}" + Style.RESET_ALL)

    # Write constraints proof TXT (Task D.1)
    raw_resp = "\n---\n".join(hw_llm_raw_responses) if hw_llm_raw_responses else ""
    con_path = write_constraints_txt(
        app_name, hw_profile, hw_feasible_full, hw_ranked_full,
        llm_raw_response=raw_resp,
        validated_ids=hw_validated_ids or None,
        invalid_ids=hw_invalid_ids or None,
        clamped_ids=hw_clamped_ids or None,
    )
    print(f"  → Constraints proof: {con_path}")

    decision_json, decision_md = write_approximation_plan_report(
        app_name,
        hw_profile,
        hw_feasible,
        invalid_ids=hw_reprompted_ids,
        clamped_entries=hw_clamped_entries,
        validated_ids=hw_validated_ids,
    )
    print(f"  → Approximation-plan audit: {decision_json}")
    print(f"  → Approximation-plan summary: {decision_md}")

# Preserve accepted code, knob spaces, and exact model-declared operator IDs.
# The transient approximation directory is intentionally deleted after a run.
os.makedirs("outputs", exist_ok=True)
accepted_plan_path = (
    f"outputs/accepted_approximation_plans_{app_name}_"
    f"{(hw_profile.name + '_' + args.hardware_control) if hw_profile else 'baseline'}"
    f"{('_' + args.artifact_label) if args.artifact_label else ''}.json"
)
accepted_plans = []
apx_path = "approximated_functions/apx_all.json"
if os.path.exists(apx_path):
    with open(apx_path, encoding="utf-8") as file:
        accepted_plans = json.load(file)
with open(accepted_plan_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "app": app_name,
            "hardware_profile": hw_profile.name if hw_profile else None,
            "hardware_control": args.hardware_control if hw_profile else "none",
            "operator_catalog": args.operator_catalog,
        "target_functions_frozen": bool(args.target_functions),
        "target_functions_source": args.target_functions or "llm_selection",
            "plans": accepted_plans,
        },
        file,
        indent=2,
    )
print(f"  -> Accepted-plan archive: {accepted_plan_path}")

# ---------------------------------------------------------------------------
#  Experiment Metrics Summary
# ---------------------------------------------------------------------------
if not alreadyRun:
    exp_technique_clamped = len(hw_clamped_ids) if isinstance(hw_clamped_ids, list) else 0
    print("\n" + "=" * 60)
    print("  EXPERIMENT METRICS SUMMARY")
    print("=" * 60)
    print(f"  Application           : {app_name}")
    print(f"  Hardware profile      : {hw_profile.name if hw_profile else 'None (baseline)'}")
    print(f"  Hardware control      : {args.hardware_control if hw_profile else 'none'}")
    print(f"  Hardware-aware        : {'YES' if hw_feasible_full is not None else 'NO'}")
    print(f"  ---")
    print(f"  Functions attempted   : {exp_functions_attempted}")
    print(f"  Functions succeeded   : {exp_functions_succeeded}")
    print(f"  Total LLM API calls   : {llm_accounting.total()} "
          f"(loop-site count was {exp_api_calls})")
    print("  LLM calls by stage    :")
    print(llm_accounting.format_report())
    print("  Context size (estimated from characters):")
    print(token_budget.format_summary())
    print("  Prompt tokens (measured):")
    print(token_budget.format_measured_summary())
    print(f"  Build failures        : {exp_compile_failures}")
    print(f"  Rejections by kind    : {exp_candidate_rejections}")
    print(f"  Compile retries/func  : {exp_compile_retries_per_func}")
    print(f"  HW reprompts triggered: {exp_hw_reprompts}")
    print(f"  Techniques clamped    : {exp_technique_clamped}")
    if hw_feasible_full is not None:
        print(f"  Feasible techniques   : {len(hw_feasible_full)}")
        print(f"  Rejected techniques   : {len(hw_rejection_log)}")
        print(f"  Invalid IDs caught    : {sorted(hw_invalid_ids) if hw_invalid_ids else 'None'}")
        print(f"  Validated IDs         : {sorted(hw_validated_ids) if hw_validated_ids else 'None'}")
    print("=" * 60)
    
    # Write metrics to JSON for comparison
    import time
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app": app_name,
        "opt_level": args.opt_level,
        "hardware_profile": hw_profile.name if hw_profile else "None",
        "hardware_control": args.hardware_control if hw_profile else "none",
        "hardware_aware": hw_feasible_full is not None,
        "operator_catalog": args.operator_catalog,
        "target_functions_frozen": bool(args.target_functions),
        "target_functions_source": args.target_functions or "llm_selection",
        "run_outcome": (
            "valid_plan_generated" if exp_functions_succeeded > 0 else "no_valid_plan"
        ),
        "functions_attempted": exp_functions_attempted,
        "functions_succeeded": exp_functions_succeeded,
        "api_calls": llm_accounting.total(),
        "api_calls_by_stage": llm_accounting.snapshot(),
        "api_calls_generation_loop_only": exp_api_calls,
        "context_budget_estimated": token_budget.summary(),
        "context_budget_measured": token_budget.measured_summary(),
        "compile_failures": exp_compile_failures,
        "candidate_rejections_by_kind": exp_candidate_rejections,
        "compile_retries_per_func": exp_compile_retries_per_func,
        "hw_reprompts": exp_hw_reprompts,
        "techniques_clamped": exp_technique_clamped,
        "target_build_gate": (
            "skipped" if exp_target_build_skipped
            else ("enforced" if hw_feasible_full is not None else "not_applicable")
        ),
        "identity_gate_mode": args.identity_gate,
        "identity_rejections": exp_identity_rejections,
        "target_build_gate_mode": args.target_build_gate,
        "target_build_status": exp_target_build_status,
        "canonical_backends": args.canonical_backends,
        "canonical_backend_used": exp_canonical_backend_used,
        "planner_generated_only": (
            args.canonical_backends == "off" and not exp_canonical_backend_used
        ),
        "per_function_operator_selection": bool(
            args.per_function_operators and hw_feasible_full is not None
        ),
        "max_operators_per_function": args.max_operators_per_function,
        "function_operator_sets": exp_function_operator_sets,
        "functions_without_applicable_operator": exp_functions_no_operator,
        "plan_conformance_rejections": exp_plan_mismatches,
        "plan_conformance_post_repair_rejections": exp_post_repair_mismatches,
        "operator_conformance_enforced": hw_feasible_full is not None,
        "operator_conformance_violations": exp_conformance_violations,
        "operator_conformance_violation_count": sum(
            len(v) for v in exp_conformance_violations.values()
        ),
        "plan_contract_failures": exp_plan_contract_failures,
        "direct_plan_extractions": exp_direct_extractions,
        "json_conversion_calls": exp_json_conversions,
        "approved_plans": exp_approved_plans,
        "feasible_techniques": len(hw_feasible_full) if hw_feasible_full else None,
        "rejected_techniques": len(hw_rejection_log) if hw_rejection_log else 0,
        "invalid_ids": sorted(hw_invalid_ids) if hw_invalid_ids else [],
        "validated_ids": sorted(hw_validated_ids) if hw_validated_ids else [],
    }
    hw_label = (
        f"{hw_profile.name}_{args.hardware_control}"
        if hw_profile else "baseline"
    )
    metrics_path = f"logs/experiment_metrics_{app_name}_{hw_label}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → Metrics saved: {metrics_path}")

    context_report = token_budget.write_report(
        f"logs/context_budget_{app_name}_{hw_label}.json"
    )
    if context_report:
        print(f"  → Context measurements: {context_report}")

    if exp_functions_succeeded == 0:
        print(
            Fore.RED
            + "[preflight] No valid approximation plan was generated; "
              "stopping before ground truth, BO, and Fused."
            + Style.RESET_ALL
        )
        # Exit 0 deliberately. An arm that cannot produce a valid plan has
        # *lost* under the protocol, and that must be counted against it in the
        # comparison. Exiting non-zero made the analyser mark the whole cell
        # INVALID and drop it, which hides the failure instead of scoring it.
        # A genuine crash still surfaces as a non-zero exit from an exception;
        # `run_outcome: no_valid_plan` in the metrics distinguishes the two.
        raise SystemExit(0)

# Read application.txt file from target/ and read the function name
# app_names = ["lqi-iclib", "stringsearch-iclib"]
# app_name = getAppName()
# app_name = app_names[0]

# Generate the ground truth
generateGroundTruth(app_name)
# quit()

copyFiles("target", "knob_tuning")
# quit()
if not alreadyRun:
    # Search for a Makefile in compilation_testing/ and copy it in knob_tuning/. The Makefile is in any of the subdirectories of compilation_testing/
    os.system(r"find compilation_testing/ -name Makefile -exec cp {} knob_tuning/ \;")
    # NOTE: Do NOT copy compilation_testing Makefiles to target/ — target/ keeps
    # the original benchmark Makefile for ground-truth generation.

# Validate copied Makefiles: try to compile, and if it fails, generate a simple deterministic one
def _ensure_working_makefile(directory):
    """Ensure the Makefile in 'directory' can compile. If not, write a fallback."""
    import glob, subprocess as _sp
    makefile_path = os.path.join(directory, 'Makefile')
    # Try compiling with existing Makefile
    try:
        _sp.run("make clean", shell=True, cwd=directory, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        result = _sp.run("make main", shell=True, cwd=directory, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return  # Existing Makefile works
    except Exception:
        pass

    # Generate a fallback Makefile from the .c files present
    c_files = sorted(glob.glob(os.path.join(directory, '*.c')))
    if not c_files:
        return
    c_names = [os.path.basename(f) for f in c_files]
    o_names = [f.replace('.c', '.o') for f in c_names]
    lines = ['CC=gcc', 'CFLAGS=-c -O2 -DLOCAL_RUN', '']
    for c, o in zip(c_names, o_names):
        lines += [f'{o}: {c}', f'\t$(CC) $(CFLAGS) {c} -o {o}', '']
    lines += [f"main: {' '.join(o_names)}", f"\t$(CC) {' '.join(o_names)} -o main -lm", '']
    lines += ['clean:', f"\trm -f {' '.join(o_names)} main", '']
    with open(makefile_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[Makefile] Wrote fallback Makefile for {directory}")

for _mf_dir in ['knob_tuning', 'target']:
    _ensure_working_makefile(_mf_dir)

file_path = f"approximated_functions/apx_all.json"
destination_file_path = "knob_tuning"
shutil.copy2(file_path, destination_file_path)

# --- Ensure the knob_tuning Makefile compiles the LLM-patched source file ---
# apx_all.json's filePath tells us which .c file contains the approximated code.
# The original eval-app Makefile may reference a different .c file (e.g. a
# pre-existing hand-crafted approximation), so we regenerate the Makefile to
# compile the correct source.
def _fix_knob_tuning_makefile():
    apx_json_path = os.path.join("knob_tuning", "apx_all.json")
    if not os.path.exists(apx_json_path):
        return
    with open(apx_json_path, "r") as f:
        apx_data = json.load(f)
    if not apx_data:
        return
    # Collect all unique source files referenced by the approximated functions
    src_files = set()
    for entry in apx_data:
        base = os.path.basename(entry.get("filePath", ""))
        if base:
            src_files.add(base)
    if not src_files:
        return
    # Read current Makefile and check if it already compiles the right file(s)
    mk_path = os.path.join("knob_tuning", "Makefile")
    if os.path.exists(mk_path):
        with open(mk_path, "r") as f:
            mk_content = f.read()
        # If the Makefile already references all the patched source files, leave it
        if all(sf in mk_content for sf in src_files):
            return
    # Generate a Makefile that compiles the patched source file(s)
    src_list = sorted(src_files)
    src_str = " ".join(src_list)
    makefile = (
        f"CC = gcc\n"
        f"CFLAGS = -DLOCAL_RUN\n"
        f"\n"
        f"main: {src_str}\n"
        f"\t$(CC) $(CFLAGS) -o main {src_str} -lm\n"
        f"\n"
        f"clean:\n"
        f"\trm -f main\n"
    )
    with open(mk_path, "w") as f:
        f.write(makefile)
    print(f"[Makefile] Updated knob_tuning/Makefile to compile: {src_str}")

_fix_knob_tuning_makefile()



# Traces and capacitors were hard-coded to a single value each, with the
# alternatives commented out. A result measured at one energy-harvesting trace
# and one capacitor is a single operating point, not a characterisation, so
# make the sweep configurable without editing source.
if args.capacitors:
    capacitors = [value.strip() for value in args.capacitors.split(",") if value.strip()]
else:
    capacitors = ["68e-6"]

if args.trace:
    traces = [args.trace]
elif args.traces:
    traces = [value.strip() for value in args.traces.split(",") if value.strip()]
else:
    traces = ["../traces/RF_2.csv"]

print(
    f"[sweep] {len(traces)} trace(s) x {len(capacitors)} capacitor(s) = "
    f"{len(traces) * len(capacitors)} operating point(s): "
    f"traces={traces} capacitors={capacitors}"
)

# Create a DataFrame to store the best knobs and the error, checkpoints
columns = [
    "knobs_list",
    "error",
    "power_cycles",
    "original_power_cycles",
    "power_cycle_ratio",
    "power_cycle_reduction",
    "optimization_metric",
    "trace",
    "capacitor",
    "bo_outcome",
    "identity_verified",
    "candidates_accepted",
]
df = pd.DataFrame(columns=columns)

def generateConfigFile(trace, capacitor):

    # copy the generic config.yaml.in file from fusedBin/fusedConfig/ to fusedBin/config.yaml
    shutil.copy2("fusedBin/fusedConfig/config.yaml.in", "fusedBin/config.yaml")

    # Fused only accepts a BOM-free, single-number-per-line trace relative to
    # its own working directory. Normalize a local copy before configuring it.
    normalized_trace = prepare_fused_trace(trace)

    # append the trace and capacitor size at the end
    with open("fusedBin/config.yaml", "a") as f:
        f.write(f"VoltageTraceFile: \"./{normalized_trace.name}\"\n")
        f.write(f"CapacitorValue: {capacitor}\n")

for trace in tqdm(traces, desc="Optimizing for trace: "):
    for capacitor in tqdm(capacitors, desc="Optimizing for capacitor: "):

        # generate the corresponding config file for fused
        generateConfigFile(trace, capacitor)

        # Get checkpoint of the original unapproximated code
        original_checkpoints = checkpointOrchestration('target/',app_name)

        # Guard: if the original build/simulation failed, skip this config
        if original_checkpoints is None:
            print(Fore.RED + f"  [ERROR] Failed to get original checkpoints for {trace}/{capacitor} — skipping." + Style.RESET_ALL)
            continue

        # Write the original checkpoints to a file logs/original_checkpoints.txt
        with open("logs/original_checkpoints.txt", "w") as f:
            f.write(str(original_checkpoints))

        trace_name = trace.split("/")[-1].split(".")[0]
        capacitor_number = capacitor.split("e")[0]

        # Create a csv file logs/{appName}_{capacitor}_{trace}.csv
        with open(f"logs/{app_name}_{capacitor_number}_{trace_name}.csv", "w") as f:
            f.write("knobs_list,error,checkpoints\n")

        # Save the {trace,capacitor} pair in a file. Path = "logs/trace_capacitor.txt"
        with open("logs/trace_capacitor.txt", "w") as f:
            f.write(f"{trace},{capacitor}")

        best_score, best_knobs = runBayesOpt()
        bo_outcome = dict(getattr(bo_module, "LAST_OUTCOME", {}) or {})

        if best_score == float('inf') and best_knobs == []:
            print(Fore.YELLOW + f"  [WARNING] No valid knob dimensions for {trace}/{capacitor} — skipping this configuration." + Style.RESET_ALL)
            continue

        Dprint(f"Best knobs: {best_knobs}")
        Dprint(f"Best score (E+C): {best_score}")

        # Find the error and checkpoints using the best knobs from logs/{appName}_{capacitor}_{trace}.csv
        best_error = None
        best_checkpoints = None
        with open(f"logs/{app_name}_{capacitor_number}_{trace_name}.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row

            # Match the best knobs with the knobs_list in the csv file
            for row in reader:
                checkpoints = row[-1]
                error = row[-2]
                knobs_list_str = row[:-2]  # Extract the knobs list as a string

                # Convert knobs_list_str (which is something like "[1','3','4]") to actual list
                # Remove unwanted characters and convert it to a proper list
                knobs_list_str = (','.join(knobs_list_str)).replace("'", "")  # Merge if knobs_list_str is split across multiple columns
                # knobs_list = ast.literal_eval(knobs_list_str.replace("'", ""))  # Convert the string to a list of integers

                Dprint(str(knobs_list_str) , str(best_knobs), str(knobs_list_str) == str(best_knobs))
                if str(knobs_list_str) == str(best_knobs):
                    best_error = error
                    best_checkpoints = checkpoints
                    break

        # Guard: if no config passed the error bound the CSV has no
        # matching row, so best_checkpoints / best_error are still None.
        if best_checkpoints is None or best_error is None:
            print(Fore.YELLOW + f"  [WARNING] All BayesOpt configs exceeded ERROR_BOUND ({ERROR_BOUND}) for {trace}/{capacitor} — skipping." + Style.RESET_ALL)
            continue

        # Check for empty or all-NA columns
        print(best_knobs)
        power_cycle_ratio, power_cycle_reduction = power_cycle_metrics(
            best_checkpoints, original_checkpoints
        )
        new_row = pd.DataFrame({
            columns[0]: [best_knobs],
            columns[1]: best_error,
            columns[2]: best_checkpoints,
            columns[3]: original_checkpoints,
            columns[4]: power_cycle_ratio,
            columns[5]: power_cycle_reduction,
            columns[6]: best_score,
            columns[7]: trace,
            columns[8]: capacitor,
            columns[9]: bo_outcome.get("state", "unknown"),
            columns[10]: bo_outcome.get("identity_verified", False),
            columns[11]: bo_outcome.get("candidates_accepted", 0),
        })

        # Append the filtered row
        df = pd.concat([df,new_row])
        with open(f"logs/original_checkpoints_{app_name}-{capacitor}_{trace_name}.txt", "w") as f:
            f.write(str(original_checkpoints))

# Save the DataFrame to a csv file
df.to_csv(f"logs/best_knobs_{app_name}.csv", index=False)

print("CheckMate has ended...")
print(f"Results {app_name}:")
print(df)

# Synthetic trace data was removed because it was not benchmark output.
if False:
    trace_path = None
    print(f"\n✓ Trace saved to: {trace_path}\n")

if args.trace_approx:
    trace_status_path = f"outputs/approximation_trace_status_{app_name}.json"
    trace_status = {
        "app": app_name,
        "status": "not_generated",
        "reason": (
            "No benchmark-specific runtime trace adapter is implemented. "
            "Synthetic demo metrics are intentionally not reported as experiment evidence."
        ),
    }
    with open(trace_status_path, "w", encoding="utf-8") as file:
        json.dump(trace_status, file, indent=2)
    print(f"[trace] Real approximation trace unavailable; status: {trace_status_path}")

# ---------------------------------------------------------------------------
#  End-of-Run Cleanup: prepare workspace for the next fresh run
# ---------------------------------------------------------------------------
def _post_run_cleanup():
    """Clean transient artifacts so the next run starts completely fresh.
    
    Results (logs/*.csv, best_knobs_*.csv) are preserved.
    Everything else that could cause stale-state problems is removed.
    """
    cleanup_dirs = [
        "compilation_testing",
        "knob_tuning",
        "functions",
        "approximated_functions",
        "dependency_graphs",
        "eval-apps/build",
    ]
    for d in cleanup_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Restore eval-apps/{app}/ from benchmark so it's pristine for next run
    bm_src = f"benchmark_applications/{app_name}/"
    ea_dst = f"eval-apps/{app_name}/"
    if os.path.isdir(bm_src):
        if os.path.exists(ea_dst):
            shutil.rmtree(ea_dst)
        shutil.copytree(bm_src, ea_dst)

    # Clean target/ (will be repopulated next run)
    if os.path.exists("target"):
        shutil.rmtree("target")

    # Clean stale Fused artifacts
    for stale in ["fusedBin/app.hex", "fusedBin/cycles.dump", "fusedBin/config.yaml", "fusedBin/trace_input.csv"]:
        if os.path.exists(stale):
            os.remove(stale)

    print("[cleanup] Post-run cleanup complete — workspace ready for next run.")

# ---------------------------------------------------------------------------
#  Archive tuned result before cleanup wipes it
# ---------------------------------------------------------------------------
import datetime as _dt
if os.path.isdir("knob_tuning"):
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _archive_dir = f"results_archive/{app_name}_{_ts}"
    os.makedirs("results_archive", exist_ok=True)
    shutil.copytree("knob_tuning", _archive_dir)
    print(f"[archive] Saved tuned result to {_archive_dir}/")

_post_run_cleanup()
