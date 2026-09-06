# Why `CheckMate + operators` is not `CHAI + operators`

> **Status.** The measured results previously quoted at the bottom of this
> document are **superseded**. They were produced before the identity gate,
> the dominance gate, the corrected error metrics, the frozen target lists,
> and the corrected objective existed, and they are not comparable with
> current runs. See [`EXPERIMENTAL_PROTOCOL.md`](EXPERIMENTAL_PROTOCOL.md)
> for the protocol that replaces them, and re-run
> `scripts/run_ablation_campaign.sh` for numbers that can be cited.

## Separating the two claims

Adding operators changes the **catalogue**. CHAI changes the **decision and
enforcement loop** around that catalogue. A two-arm comparison cannot tell
these apart, so the campaign is a 2x2:

| Arm | Catalogue | Control plane | Isolates |
| --- | --- | --- | --- |
| A `checkmate_baseline` | T1-T20 | off | reference |
| B `operator_only_plus10` | T1-T30 | off | B - A = operator effect |
| C `chai_plus10` | T1-T30 | on | C - B = control-plane effect |
| D `chai_baseline_ops` | T1-T20 | on | D - A = control plane alone |

The claim "it is not just the ten operators" holds only if `C - B` exceeds
`B - A`. `scripts/analyze_ablation.py` reports exactly that decomposition.

## Precise comparison claim

Adding approximation operators changes the **catalogue**: it gives a planner
more possible transformations.  CHAI changes the **decision and enforcement
loop** around that catalogue.  Therefore, the comparison is not based on the
number of operators alone.

This document uses `CheckMate + operators` as the *operator-only baseline*:
an expanded static transformation menu supplied to an LLM.  It does not make
a historical claim about features of a specific upstream CheckMate revision.

| Dimension | Operator-only baseline | CHAI |
| --- | --- | --- |
| Input to planning | Application + raw hardware profile + a static, expanded operator menu | Application + the same hardware profile + hardware-conditioned feasible menu |
| Before the LLM | Every operator remains selectable | Gatekeeper removes inapplicable, FPU/SIMD-incompatible, or memory-infeasible operators |
| Operator ordering | Fixed list | Hardware-specific cost ranking of feasible operators |
| If the LLM chooses an invalid operator | No enforcement implied by adding operators | Validator rejects the reference and requests one constrained retry |
| If invalid data reaches the JSON plan | No final safety mechanism implied | Hard clamp deletes infeasible plan entries before evaluation |
| Evidence | Typically only generated code / final result | Structured pre-LLM exclusions, rejected model references, discarded JSON plan entries, and final accepted IDs |
| Operators offered to one function | The whole app-wide menu | Only operators whose structure exists in that function (per-function narrowing, capped at `--max_operators_per_function`) |
| Relationship between plan and code | Plan is free prose; provenance scraped from the generated code | Plan is a structured contract (operator, region, knob schema with identity, invariant); generated code must implement exactly it |
| Acceptance gate | Host `gcc ... -lm` | Host build **and** MSP430 cross-build + FRAM/RAM fit |
| Operator label vs implementation | Accepted on the label | Rejected unless the operator's distinctive structure is present and knob-controlled |
| Porting to another target | Manually redesign the prompt/menu | Supply a hardware profile; the same registry is filtered and ranked again |

The important distinction is **control plane versus data plane**:

```text
Operator-only baseline
application + raw hardware profile
              |
              v
       [expanded operator catalogue] --> LLM --> approximate code

CHAI
application + hardware profile
              |
              v
       [feasibility gate + hardware cost ranking]
              |
              v
             LLM --> [post-LLM validation / constrained retry]
                         |
                         v
                  [JSON-plan hard clamp]
                         |
                         v
                compile, validate, and tune against Fused
```

New operators live in the data plane.  CHAI makes every current and future
operator pass through the same hardware-aware control plane.  In fact,
adding an operator without this control plane can make selection *less* safe:
it enlarges the set of infeasible choices exposed to the model.

## What makes the implementation hardware-aware

CHAI represents a target in a JSON hardware profile with RAM, FPU, SIMD,
energy budget, and clock fields.  The gatekeeper uses the capability and RAM
constraints to produce the allowed list before prompt construction.  The cost
model then ranks that surviving list for the target.  The validator repeats
the feasible-set check after model generation, and the final JSON clamp is a
non-negotiable last line of defence.

This means the same technique registry and the same benchmark can yield
different menus on an MSP430 (no FPU/SIMD, 10 KB RAM) and an ARM A53
(FPU/SIMD available).  No operator-card edit is required to make that change.

## Reproducible evidence

The controlled Sobel run, measured outcomes, negative results, and validated
manual upper bound are documented in
[`SOBEL_HW_AWARE_ABLATION.md`](SOBEL_HW_AWARE_ABLATION.md). The measurements
show that CHAI's control plane is different and more selective, but the natural
Qwen 7B paired run does not support a claim that CHAI is faster.

Run the three-profile demonstration from the configured Docker environment:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace chai-dev:local `
  python3 scripts/demo_hardware_profiles.py
```

It writes a structured rejection file for each profile under `outputs/`.
For an end-to-end CHAI run with `--hardware`, CHAI additionally writes:

```text
outputs/approximation_plan_decisions_<app>_<hardware>.json
outputs/approximation_plan_decisions_<app>_<hardware>.md
```

Those files distinguish the following states, which must not be counted as
the same thing:

1. **Pre-LLM catalogue exclusions**: techniques removed before the model is
   prompted, with their capability or memory reason.
2. **Post-LLM rejected references**: infeasible technique IDs the model
   mentioned; these enter the constrained retry path.
3. **Final clamped plan entries**: concrete `apx_all.json` objects removed
   before compilation/evaluation.  The JSON audit retains the original entry
   and the exact rejection reason.

This audit is the evidence for reported infeasible/discarded approximation
plans.  A zero in categories 2 and 3 is a valid result: it demonstrates that
the pre-LLM gate prevented invalid choices rather than indicating missing
instrumentation.

## Slide-ready wording

> CHAI is not CheckMate with a larger operator list.  It is a
> hardware-conditioned planning and enforcement layer around the operator
> registry: profile -> gate -> rank -> LLM -> validate/retry -> clamp ->
> evaluate.  The same operators therefore produce different *feasible*
> search spaces on different targets, and every discarded candidate or plan is
> reported with a constraint-based reason.

### On quoting the old Sobel number

The previously quoted "49x fewer checkpoints" line must not be reused. Three
things were wrong with it:

1. "Checkpoints" is Fused's power-on-reset count, not a checkpoint count.
2. The 1-cycle result came from a deterministic fixed-point backend installed
   specifically for `sobel_filtering`
   (`utils/operator_conformance.py`), not from the planner. The *natural*
   paired run favoured the operator-only arm.
3. Sobel's reported error was half the dissimilarity its own evaluator
   measured.

A backend-replaced result is a legitimate system-level finding, but it must be
reported in its own row, separately from planner-generated results, and it
cannot be generalised to the other five benchmarks unless equivalent backends
exist for them.
