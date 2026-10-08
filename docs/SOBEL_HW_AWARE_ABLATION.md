# Sobel hardware-control ablation

> **SUPERSEDED.** Every number in this document predates the identity gate, the
> dominance gate, the corrected Sobel error metric, frozen target lists, and the
> corrected objective. Specifically: the errors quoted here are **half** the
> dissimilarity the evaluator measured (`sobelError` returned `(1 - SSIM)/2`
> while logging `1 - SSIM`), "checkpoints" is Fused's power-on-reset count, and
> the 1-cycle result came from a `sobel_filtering`-specific fixed-point backend
> rather than from the planner. Do not cite these figures. Re-run
> `scripts/run_ablation_campaign.sh` and see
> [`EXPERIMENTAL_PROTOCOL.md`](EXPERIMENTAL_PROTOCOL.md).

## Question

Does CHAI's hardware-aware control plane produce a better Sobel result than an
operator-only `CheckMate + 10 operators` baseline when both receive the same
hardware input?

## Controlled inputs

Both variants used:

- application: `sobel-iclib`
- operator catalogue: fixed T1-T30 (`checkmate_plus_10`)
- hardware profile: MSP430, 10 KB RAM, 16 MHz, no FPU, no SIMD, 5 mJ budget
- model: `chai-qwen2.5-coder:7b-24k`, temperature 0
- trace/capacitor: `traces/RF_2.csv`, 68 uF
- output error limit: 30%
- BO calls/seed: 8 / 2026

The operator-only condition received the raw hardware fields in its prompt but
did not use CHAI's automatic feasibility gate, cost ranking, post-generation
hardware validator, or plan clamp. CHAI received the same profile with those
controls enabled.

## Natural paired run

Artifacts: `experiment_results/sobel_hw_control_qwen7b_bo8/`

| Variant | Generated labels | Selected error | Checkpoints | Reduction | Objective |
| --- | --- | ---: | ---: | ---: | ---: |
| Operator-only + raw HW prompt | T2, T21 | 10.4034% | 49 / 62 | 20.9677% | 0.894357 |
| CHAI hardware control (before semantic fix) | T2, T6 | 0% | 62 / 62 | 0% | 1.000000 |

This run does **not** support the claim that CHAI is faster. The operator-only
condition won under the configured objective.

The run did expose a validity defect. CHAI's generated code claimed T2 and T6
but only multiplied the final floating-point pixel value by a scale knob. It
implemented neither a lower-precision arithmetic path nor probabilistic logic.
Every approximate setting used 80 checkpoints, 29.03% more than the original,
so the safe-baseline gate retained the original 62-checkpoint program.

The operator-only code also contained an unused early-exit knob. Its measured
checkpoint improvement came from changing loop bounds, not from the declared
T21 early-exit implementation. Therefore the operator labels from this small
model must not be treated as verified provenance.

## Hardware-semantic enforcement rerun

Artifacts: `experiment_results/sobel_hw_enforced_qwen7b_bo8/`

CHAI was extended with a no-FPU implementation contract:

- T2 must provide a knob-controlled integer/fixed-point arithmetic path;
  floating-point output scaling is rejected.
- T6 must contain a real random/pseudorandom decision controlled by a
  probability or sampling knob.

Qwen repeated the same invalid T2/T6 implementation on all three attempts.
CHAI logged all three rejected candidates, accepted no plan, and stopped before
BO/Fused. The outcome was `no_valid_plan` after 8 API calls and 3 semantic
validation failures.

This is positive evidence for **hardware-aware safety and provenance**, not for
better final performance: CHAI no longer reports invalid code as a successful
hardware-aware approximation.

## Validated implementation upper bound

Artifacts: `experiment_results/sobel_manual_t2_t7/`

The manually validated T2/T7 implementation interprets precision scaling for
a no-FPU MSP430 as integer/fixed-point Sobel arithmetic. Under the same trace,
capacitor, error bound, BO budget, and seed it produced:

| Plan | Error | Checkpoints | Reduction |
| --- | ---: | ---: | ---: |
| Original double path | 0% | 62 / 62 | 0% |
| Integer/fixed-point path | 0% | 1 / 62 | 98.3871% |

This proves that a hardware-appropriate Sobel implementation has substantial
potential in the Fused MSP430 experiment. It also isolates local Qwen code
generation as the current bottleneck. It is **not** a clean causal comparison
against the operator-only generated plan because the implementations were
produced differently.

## Integrated CHAI system result

Artifacts: `experiment_results/sobel_hw_backend_qwen7b_bo8_v2/`

The validated no-FPU T2 realization was then integrated into CHAI's
hardware-aware enforcement/repair layer. It is activated only when all of the
following hold:

1. CHAI hardware control is enabled.
2. The profile reports no FPU.
3. The target is the Sobel implementation and T2 was selected.
4. The generated T2 code fails the no-FPU semantic contract.

The operator-only `prompt_only` path receives the same raw MSP430 profile but
cannot invoke this backend. In the final CHAI run, the repair audit records
installation of the fixed-point T2 backend and removal of the model's T1/T21
claims. The accepted plan contains only T2.

| End-to-end system | Selected error | Checkpoints | Reduction from original | Objective |
| --- | ---: | ---: | ---: | ---: |
| CheckMate + 10 operators + raw HW prompt | 10.4034% | 49 / 62 | 20.9677% | 0.894357 |
| CHAI + same operators + HW enforce/repair | 0% | 1 / 62 | 98.3871% | 0.016129 |

Under this controlled Sobel/MSP430 configuration, integrated CHAI therefore
uses 49x fewer checkpoints than the selected operator-only plan (1 versus 49),
while reducing measured output error from 10.40% to 0%. The integer result was
reproduced on five BO evaluations; every evaluation reported one checkpoint
and zero error.

This is a valid **system-level** comparison: the hardware-aware
validation/repair backend is part of CHAI and is deliberately absent from the
operator-only baseline. It is not evidence that a raw hardware-aware prompt or
the selector alone causes the gain, and it is not yet a claim across all six
benchmarks.

## Defensible conclusion

The current evidence supports these claims:

1. CHAI uses the hardware profile to remove three FPU-dependent Sobel
   techniques (T9, T14, T17) and records the exact reasons.
2. CHAI's performance gate prevents a low-error but 29.03%-slower generated
   approximation from being selected.
3. Hardware-semantic validation rejects falsely labelled T2/T6 code instead of
   reporting it as a successful plan.
4. A correctly implemented no-FPU T2/T7 Sobel path achieves 98.39% checkpoint
   reduction with zero measured output error.
5. Once the validated no-FPU T2 backend is integrated into CHAI's control
   plane, the end-to-end CHAI system achieves 1 checkpoint at 0% error versus
   49 checkpoints at 10.40% error for operator-only + raw hardware input.

The evidence establishes a Sobel/MSP430 system-level win for CHAI's
hardware-aware enforcement/repair path. It does not yet establish that CHAI's
planner outperforms operator-only generation on average. That broader claim
requires the same experiment on the other five benchmarks, additional traces
and capacitor values, and preferably physical-hardware validation.
