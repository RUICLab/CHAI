import json
import os
import random
import subprocess
import copy
from colorama import Fore, Style
from skopt import gp_minimize
from skopt.space import Integer, Real
from tqdm import tqdm


from utils.utils import getAppName, Dprint, itrPrint
from utils.validator import updateKnobValue
from config.globals import TARGET_FILES
from config.config import ERROR_BOUND
from utils.error_analyzer import (
    susanError, lqiError, stringSearchError, fftError, sobelError, arError,
    bitcountError, radixbmError, segmentbmError, acceptSobelError,
    acceptActivityrecError, describe_error_bound, error_bound_for,
    evaluate_error,
)
from utils.checkpoints import checkpointOrchestration
from lib.lsp import lsp_patcher
from utils.knob_metadata import (
    KnobMetadataError,
    normalize_knob_metadata,
    validate_plan_entries,
)
from utils.bo_config import recommended_call_budget, validate_bo_call_budget
from utils.optimization_metrics import (
    REQUIRE_IMPROVEMENT,
    constrained_optimization_metric,
    dominates_original,
    is_identity_error,
    objective_description,
    power_cycle_metrics,
)

#: Measurement published by the most recent successful ``evaluateKnobs`` call.
#: ``runBayesOpt`` reads this to verify the declared identity point, which the
#: scalar objective alone cannot express.
LAST_EVALUATION = None

#: Structured description of how the most recent campaign terminated. Written
#: by ``runBayesOpt`` and consumed by ``main.py`` so a run in which every
#: candidate failed is never reported as "the baseline won".
LAST_OUTCOME = {}

#: Number of candidates in the current campaign that passed every hard gate.
_accepted_candidates = 0

#: True while the declared identity point is being measured. The identity point
#: establishes the reference the search is anchored to, so it is exempt from the
#: dominance gate -- the scaffolding an approximated function carries even when
#: every knob is at its no-op setting legitimately costs a cycle or two.
_evaluating_identity_point = False

# Keep the published/default search budget, while allowing a reproducible
# reduced budget for end-to-end smoke/comparison campaigns.
N_INITIAL_POINTS = 5
n_calls = validate_bo_call_budget(
    int(os.getenv("CHAI_BO_CALLS", "100")),
    initial_points=N_INITIAL_POINTS,
)
#: When set, the per-run budget is raised to match the plan's dimensionality.
#: A fixed 40 calls adequately searches 3 knobs and barely touches 16.
AUTO_BUDGET = os.getenv("CHAI_BO_AUTO_BUDGET", "1").strip().lower() not in (
    "0", "false", "no", "off"
)

cur_call = 0
# progress_bar = tqdm(total=n_calls, desc="Bayesian Optimization Iteration")

def _declared_default(code, knob_name, step_type):
    """Return the knob's initial value as written in the generated C."""
    import re as _re

    match = _re.search(rf'{_re.escape(knob_name)}\s*=\s*([0-9.eE\-+]+)', code)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return int(value) if step_type == "Integer" else value


def _range_including_default(range_values, default, step_type):
    """Widen *range_values* so the declared identity point is reachable.

    A declared default outside its own range means the optimizer can never
    evaluate the unapproximated program. Clamping the default into the range
    (the previous behaviour) silently anchored the search on a value the code
    does not contain, so the same candidate could pass the generation-time
    identity gate and then fail BayesOpt's identity check on the same metric.
    Widening keeps the two consistent and records that the plan was malformed.
    """
    low, high = range_values[0], range_values[1]
    if default is None:
        return [low, high], None
    if low <= default <= high:
        return [low, high], None
    new_low, new_high = min(low, default), max(high, default)
    if step_type == "Integer":
        new_low, new_high = int(new_low), int(new_high)
    return [new_low, new_high], (
        f"widened range [{low}, {high}] to [{new_low}, {new_high}] so the "
        f"declared default {default} is reachable"
    )


def createSearchSpace(pathToJson):
    with open(pathToJson, "r") as file:
        approximation_data = json.load(file)
    validate_plan_entries(approximation_data)

    space = []
    for approximation_dict in approximation_data:
        _, knob_ranges, knob_step_size = normalize_knob_metadata(approximation_dict)
        code = approximation_dict.get("completeFunction", "")

        for i, knob_range in enumerate(knob_ranges):
            for knob_name, range_values in knob_range.items():
                step_type = knob_step_size[i][knob_name]
                default = _declared_default(code, knob_name, step_type)
                bounds, widened = _range_including_default(
                    range_values, default, step_type
                )
                if widened:
                    Dprint(f"[BayesOpt] knob {knob_name}: {widened}")
                if step_type == "Integer":
                    space.append(Integer(bounds[0], bounds[1], name=knob_name))
                if step_type == "Real":
                    space.append(Real(bounds[0], bounds[1], name=knob_name))

    return space


def getBaselineKnobs(pathToJson):
    """Extract baseline (default) knob values from JSON, in the same order as createSearchSpace."""
    with open(pathToJson, "r") as file:
        approximation_data = json.load(file)
    validate_plan_entries(approximation_data)

    baseline = []
    for approximation_dict in approximation_data:
        _, knob_ranges, knob_step_size = normalize_knob_metadata(approximation_dict)
        code = approximation_dict.get("completeFunction", "")

        for i, knob_range in enumerate(knob_ranges):
            for knob_name, range_values in knob_range.items():
                step_type = knob_step_size[i][knob_name]
                val = _declared_default(code, knob_name, step_type)
                if val is None:
                    # No initializer found: fall back to the conservative end.
                    val = (
                        float(range_values[0]) if step_type == "Real"
                        else int(range_values[0])
                    )
                # Do NOT clamp. createSearchSpace widens the dimension to
                # include this value, so the identity point stays exactly what
                # the generated code contains.
                baseline.append(val)

    return baseline


def _app_name_or_unknown():
    """Resolve the app name without making a missing target tree fatal here."""
    try:
        return getAppName()
    except (OSError, IndexError):
        return "unknown-app"


def runBayesOpt():
    tuning_folder = "knob_tuning/apx_all.json"
    seed_override = os.getenv("CHAI_BO_SEED")
    random_state = int(seed_override) if seed_override is not None else random.randint(0, 100)
    global cur_call, LAST_EVALUATION, LAST_OUTCOME, _accepted_candidates
    cur_call = 0
    _accepted_candidates = 0
    LAST_EVALUATION = None
    LAST_OUTCOME = {
        "state": "unknown",
        "objective": objective_description(
            error_bound=error_bound_for(_app_name_or_unknown())
        ),
        "identity_verified": False,
        "candidates_evaluated": 0,
        "candidates_accepted": 0,
        "selected": None,
    }

    campaign_app = _app_name_or_unknown()
    campaign_bound, bound_reason = describe_error_bound(campaign_app)
    print(
        f"[BayesOpt] Frozen objective: "
        f"{objective_description(error_bound=campaign_bound)}"
    )
    print(f"[BayesOpt] {campaign_app} error bound {campaign_bound}: {bound_reason}")

    try:
        search_space = createSearchSpace(tuning_folder)
        baseline_x0 = getBaselineKnobs(tuning_folder)
    except (OSError, json.JSONDecodeError, KnobMetadataError) as error:
        app_name = getAppName()
        _record_evaluation_rejection(app_name, [], f"invalid optimization plan: {error}")
        print(f"[BayesOpt] ERROR: Invalid approximation plan — {error}")
        LAST_OUTCOME["state"] = "invalid_plan"
        LAST_OUTCOME["reason"] = str(error)
        return float('inf'), []
    if len(search_space) == 0:
        print("[BayesOpt] No tunable dimensions; evaluating the fixed plan once.")
        fixed_score = evaluateKnobs([])
        if fixed_score == 100.0:
            _record_evaluation_rejection(
                getAppName(),
                [],
                "fixed approximation plan failed build/runtime or accuracy bound",
            )
            LAST_OUTCOME["state"] = "fixed_plan_rejected"
            return float('inf'), []
        LAST_OUTCOME["state"] = "fixed_plan_selected"
        LAST_OUTCOME["candidates_evaluated"] = 1
        LAST_OUTCOME["candidates_accepted"] = 1
        LAST_OUTCOME["selected"] = "fixed_plan"
        return fixed_score, []

    # Validate the declared identity point before spending the search budget.
    # A model can emit compilable code whose "safe" defaults no longer compute
    # the original function at all. Exploring around that broken reference is
    # not a meaningful approximation experiment.
    global _evaluating_identity_point
    _evaluating_identity_point = True
    try:
        baseline_score = evaluateKnobs(baseline_x0)
    finally:
        _evaluating_identity_point = False
    app_name = _app_name_or_unknown()
    if baseline_score == 100.0:
        _record_evaluation_rejection(
            app_name,
            baseline_x0,
            "declared safe baseline failed build/runtime or accuracy bound; "
            "discarding the entire approximation plan",
        )
        print(
            "[BayesOpt] ERROR: Declared safe baseline is invalid; "
            "discarding plan before optimization."
        )
        LAST_OUTCOME["state"] = "baseline_invalid"
        return float('inf'), []

    # The generation prompt requires an identity setting inside every knob
    # range ("at this setting the function must preserve the original
    # behavior").  Requiring only error <= ERROR_BOUND accepted plans whose
    # declared reference point was already 18% wrong, which makes every
    # downstream comparison meaningless.  Measure it instead of trusting it.
    baseline_measurement = LAST_EVALUATION
    if baseline_measurement is None:
        print(
            "[BayesOpt] WARNING: no baseline measurement was published; "
            "identity verification skipped."
        )
    elif not is_identity_error(baseline_measurement["error"]):
        reason = (
            f"declared safe baseline is not an identity point: measured error "
            f"{baseline_measurement['error']} exceeds the identity tolerance. "
            f"The plan's knob defaults do not reproduce the original function."
        )
        _record_evaluation_rejection(app_name, baseline_x0, reason)
        print(Fore.RED + f"[BayesOpt] ERROR: {reason}" + Style.RESET_ALL)
        LAST_OUTCOME["state"] = "baseline_not_identity"
        LAST_OUTCOME["baseline_error"] = baseline_measurement["error"]
        return float('inf'), []
    else:
        LAST_OUTCOME["identity_verified"] = True
        LAST_OUTCOME["baseline_error"] = baseline_measurement["error"]
        LAST_OUTCOME["baseline_power_cycles"] = baseline_measurement["power_cycles"]

    # Item 11: size the budget from the search space rather than a fixed
    # constant. A 3-knob plan and a 16-knob plan cannot share one budget and
    # both be adequately searched, and an arm that gets a relatively larger
    # budget than another is a confound.
    effective_calls = n_calls
    if AUTO_BUDGET:
        recommended = recommended_call_budget(
            len(search_space), initial_points=N_INITIAL_POINTS
        )
        if recommended > effective_calls:
            print(
                f"[BayesOpt] Search space has {len(search_space)} dimension(s); "
                f"raising budget from {effective_calls} to the recommended "
                f"{recommended} calls."
            )
            effective_calls = recommended
    LAST_OUTCOME["dimensions"] = len(search_space)
    LAST_OUTCOME["call_budget"] = effective_calls

    remaining_calls = effective_calls - 1
    res = gp_minimize(
        evaluateKnobs,
        search_space,
        n_calls=remaining_calls,
        n_initial_points=min(N_INITIAL_POINTS, remaining_calls),
        random_state=random_state
    )

    LAST_OUTCOME["candidates_evaluated"] = cur_call
    LAST_OUTCOME["candidates_accepted"] = _accepted_candidates

    # ``gp_minimize`` returns the 100.0 penalty when every candidate was
    # rejected.  Comparing the baseline against that penalty silently reports
    # "the baseline won" for a campaign in which nothing worked at all.
    if _accepted_candidates <= 1:
        # Only the identity point itself was ever accepted.
        print(
            Fore.YELLOW
            + "[BayesOpt] No approximate configuration beat the unapproximated "
              "program under the frozen objective."
            + Style.RESET_ALL
        )
        LAST_OUTCOME["state"] = "no_improvement_found"
        LAST_OUTCOME["selected"] = "identity_baseline"
        return baseline_score, baseline_x0

    if baseline_score <= res.fun:
        LAST_OUTCOME["state"] = "baseline_selected"
        LAST_OUTCOME["selected"] = "identity_baseline"
        return baseline_score, baseline_x0

    LAST_OUTCOME["state"] = "approximation_selected"
    LAST_OUTCOME["selected"] = "approximate_configuration"
    return res.fun, res.x


def _record_evaluation_rejection(app_name, knobs, reason):
    """Persist an invalid candidate instead of letting a tuning run abort."""
    from datetime import datetime

    os.makedirs("logs", exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "app": app_name,
        "knobs": list(knobs),
        "reason": str(reason),
    }
    path = f"logs/optimization_rejections_{app_name}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_optimizer_value) + "\n")


def _json_optimizer_value(value):
    """Convert NumPy/scikit-optimize values without hiding unknown types."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _safe_error_evaluation(app_name, knobs, evaluator):
    """Return ``None`` if generated code cannot be evaluated.

    ``None`` deliberately remains distinct from a numerical error so reports
    never describe a build/runtime failure as an accuracy-bound violation.
    """
    original_cwd = os.getcwd()
    try:
        return evaluator()
    except Exception as error:
        # Evaluators historically chdir into knob_tuning. Restore the campaign
        # root before writing the rejection record or the JSONL file is created
        # inside a transient directory that cleanup later removes.
        os.chdir(original_cwd)
        _record_evaluation_rejection(app_name, knobs, f"error evaluation failed: {error}")
        itrPrint(f"> Error evaluation rejected: {error}")
        return None
    finally:
        # Several benchmark evaluators chdir into knob_tuning/.  If their
        # build or output parsing fails, restore the process CWD before the
        # next BayesOpt iteration addresses its plan file.
        os.chdir(original_cwd)


def _build_candidate(path):
    """Compile a patched candidate before attempting benchmark-specific error.

    Every evaluator used to own its own build logic, and several ignored exit
    codes.  This common gate prevents a failed build from becoming a fake
    numerical error measurement.
    """
    clean = subprocess.run("make clean", shell=True, cwd=path, capture_output=True, text=True)
    build = subprocess.run("make main", shell=True, cwd=path, capture_output=True, text=True)
    if clean.returncode != 0:
        return f"candidate clean failed: {clean.stdout}{clean.stderr}".strip()
    if build.returncode != 0:
        return f"candidate build failed: {build.stdout}{build.stderr}".strip()
    return ""


def evaluateKnobs(*params):

    global TARGET_FILES
    global cur_call, LAST_EVALUATION, _accepted_candidates

    cur_call += 1

    candidate_knobs = list(params[0])
    knobs_list = copy.copy(candidate_knobs)
    # knobs_list = [1, 1.0]

    # Dprint(knobs_list)

    with open("knob_tuning/apx_all.json", "r") as file:
        json_content = file.read()

    # Parse and revalidate the plan before every mutation. This is cheap and
    # prevents stale/corrupt metadata from reaching the source patcher.
    try:
        approximation_data = json.loads(json_content)
        validate_plan_entries(approximation_data)
    except (json.JSONDecodeError, KnobMetadataError) as error:
        app_name = getAppName()
        _record_evaluation_rejection(app_name, candidate_knobs, f"invalid optimization plan: {error}")
        itrPrint(f"> Candidate rejected — invalid optimization plan: {error}")
        return 100.0

    app_name = getAppName()
    try:
        for approximation_dict in approximation_data:
            knob_variables, _, _ = normalize_knob_metadata(approximation_dict)
            # Dprint(knob_variables)

            for var in knob_variables:
                new_value = knobs_list.pop(0)
                # Dprint(var)
                # Dprint(new_value)
                updateKnobValue(approximation_dict, new_value, var)
    except (ValueError, IndexError, SyntaxError) as error:
        _record_evaluation_rejection(app_name, candidate_knobs, f"invalid knob declaration: {error}")
        itrPrint(f"> Candidate rejected — invalid knob declaration: {error}")
        return 100.0

    # json.dump(approximation_data, fp="knob_tuning/apx_all.json")

    # Save approximation_data in knob_tuning/apx_all.json, write your own logic here
    # open("knob_tuning/apx_all.json", "w").write(json.dumps(approximation_data))
    open("knob_tuning/apx_all.json", "w").write(json.dumps(approximation_data, indent=2))

    lsp_patcher("knob_tuning/", "apx_all.json")

    build_error = _build_candidate("knob_tuning")
    if build_error:
        _record_evaluation_rejection(app_name, candidate_knobs, build_error)
        itrPrint(
            f"> Candidate rejected — build failure: {build_error[:300]}"
        )
        return 100.0

    # Get the Error
    # One dispatch table, shared with the generation-time identity gate, so a
    # candidate cannot pass one check and fail the other on a different metric.
    error = _safe_error_evaluation(
        app_name, candidate_knobs, lambda: evaluate_error(app_name, "knob_tuning")
    )

    if error is None:
        # The evaluator already logged the exact build/run/output failure.
        # Do not mislabel this as exceeding the numerical error bound.
        return 100.0

    Dprint(f"Error Returned from Eval Func: {error}")

    # Benchmark-specific accuracy bound. A single suite-wide 0.30 treated an
    # image-similarity loss and an exact integer-answer loss as equivalent
    # amounts of damage; see ERROR_METRIC_SPEC in utils/error_analyzer.py.
    bound = error_bound_for(app_name)
    if error > bound:
        _record_evaluation_rejection(
            app_name,
            candidate_knobs,
            f"accuracy error {error} exceeds the {app_name} bound {bound}",
        )
        Dprint(f"Error {error:.4f} exceeds bound {bound} — returning penalty")
        itrPrint(f"> Error: {error} [REJECTED — exceeds {bound}]")
        return 100.0  # large penalty value

    Dprint(f"Error After Bound Check: {error}")

    # logs/trace_capacitor.txt contains the trace and capacitor size
    with open("logs/trace_capacitor.txt", "r") as f:
        trace, capacitor = f.read().split(",")

    # Read the original power-cycle count from logs/original_checkpoints.txt
    with open("logs/original_checkpoints.txt", "r") as f:
        original_power_cycles = int(f.read())

    # Get the simulated power-on reset count for this candidate
    power_cycles = checkpointOrchestration("knob_tuning/", app_name)

    # Guard: if build/simulation failed, return penalty so BayesOpt skips this config
    if power_cycles is None:
        Dprint("[evaluateKnobs] checkpointOrchestration returned None — penalizing")
        _record_evaluation_rejection(
            app_name, candidate_knobs, "MSP430 build or Fused simulation failed"
        )
        return 100.0  # same penalty as error-bound violation

    power_cycle_ratio, power_cycle_reduction = power_cycle_metrics(
        power_cycles, original_power_cycles
    )

    Dprint("Power-cycle Remaining Ratio: ", power_cycle_ratio)
    Dprint("Power-cycle Reduction: ", power_cycle_reduction)

    # Publish the raw measurement so ``runBayesOpt`` can verify the declared
    # identity point.  The scalar objective cannot express "error was zero".
    LAST_EVALUATION = {
        "knobs": list(candidate_knobs),
        "error": float(error),
        "power_cycles": int(power_cycles),
        "original_power_cycles": int(original_power_cycles),
        "power_cycle_ratio": power_cycle_ratio,
    }

    # Hard dominance gate: a configuration that needs MORE power cycles than
    # the unapproximated program is not an approximation result.  Without this
    # the objective happily selected plans at 1.89x and 2.21x the original.
    if (
        REQUIRE_IMPROVEMENT
        and not _evaluating_identity_point
        and not dominates_original(power_cycles, original_power_cycles)
    ):
        _record_evaluation_rejection(
            app_name,
            candidate_knobs,
            f"power cycles {power_cycles} exceed the unapproximated program's "
            f"{original_power_cycles} (ratio {power_cycle_ratio:.4f}); "
            "candidate is dominated by doing nothing",
        )
        itrPrint(
            f"> Power cycles: {power_cycles}/{original_power_cycles} "
            f"[REJECTED — worse than the original program]"
        )
        return 100.0

    trace_name = trace.split("/")[-1].split(".")[0]
    capacitor_number = capacitor.split("e")[0]
    # Appends the knobs, error and power cycles to logs/{appName}_{capacitor}_{trace}.csv
    with open(f"logs/{app_name}_{capacitor_number}_{trace_name}.csv", "a") as f:
        f.write(f"'{params[0]}',{error},{power_cycles}\n")

    optimization_metric = constrained_optimization_metric(
        error, power_cycles, original_power_cycles, error_bound=bound
    )

    _accepted_candidates += 1

    Dprint("Optimization_metric: ", optimization_metric)

    Dprint(f"------------------------ iteration done ------------------------")

    itrPrint(
        (
            f"Bayesian Iteration: {cur_call}",
            f"> Error: {error}",
            f"> Power-cycle Remaining Ratio: {power_cycle_ratio}",
            f"> Power-cycle Reduction: {power_cycle_reduction}",
            f"> Optimization Metric: {optimization_metric}",
        )
    )

    return optimization_metric
