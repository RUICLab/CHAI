# Frozen experimental protocol

This protocol must be fixed **before** results are inspected. Everything here
is enforced in code; the file records why each choice was made.

## 1. What is being compared

"CHAI is better" is ambiguous between two different claims:

1. **Data plane** — CHAI offers a larger operator catalogue (T1–T30 rather than
   T1–T20).
2. **Control plane** — CHAI conditions planning on the hardware profile and
   enforces the result: feasibility gate → cost ranking → per-function
   narrowing → structured plan contract → post-generation validation →
   target-build gate → plan clamp.

A two-arm comparison cannot separate them. The campaign therefore runs a 2×2:

| Arm | Catalogue | Control plane | Purpose |
| --- | --- | --- | --- |
| **A** `checkmate_baseline` | T1–T20 | off (`prompt_only`) | CheckMate-style reference |
| **B** `operator_only_plus10` | T1–T30 | off (`prompt_only`) | **B − A = operator effect** |
| **C** `chai_plus10` | T1–T30 | on (`chai`) | **C − B = control-plane effect** |
| **D** `chai_baseline_ops` | T1–T20 | on (`chai`) | **D − A = control plane alone** |

The claim *"it is not just the ten operators"* is supported only if
`C − B` is a larger improvement than `B − A`, and corroborated by `D − A`.

Run it with:

```bash
bash scripts/run_ablation_campaign.sh
python3 scripts/analyze_ablation.py experiment_results/<run_id>
```

## 2. Naming

The operator-only arms are **not** an upstream CheckMate implementation. They
are a constructed operator-only LLM baseline. In any write-up they must be
called:

> Operator-only LLM baseline with the CheckMate+10 catalogue

Calling them "CheckMate" without that qualifier is misleading.

## 3. Held constant across every arm

| Variable | Value | Enforced by |
| --- | --- | --- |
| Target functions | frozen once per app, shared by all arms | `--target_functions` |
| Error bound | 0.30 | `config.config.ERROR_BOUND` |
| Objective | see §4 | `utils/optimization_metrics.py` |
| Trace / capacitor | `RF_2.csv` / 68 µF | runner |
| Hardware profile | `config/hardware_msp430.json` | runner |
| BO budget | 40 calls | `CHAI_BO_CALLS` |
| Seeds | 2026, 2027, 2028 | `CHAI_BO_SEED` |
| Model | `chai-qwen2.5-coder:7b-24k`, temperature 0 | runner |

Target selection was previously an independent LLM call in each run. That is
how the AR comparison ended up approximating **7 functions in one arm and 9 in
the other**, which makes a performance comparison uncontrolled. It is now
frozen once per application and passed to every arm.

## 4. The objective

```
reject   error > 0.30                          (hard accuracy gate)
reject   power_cycle_ratio > 1.0               (hard dominance gate)
minimize 1.0 * (error / 0.30) + 1.0 * (power_cycles / original_power_cycles)
```

Two changes from the previous formulation, both necessary:

* **Error is normalised by the bound.** The old objective summed a raw error in
  `[0, 0.30]` with a ratio in `[0, ∞)`. That silently weighted the hardware term
  roughly 3× more than accuracy while the code claimed the two were equal.
  Dividing by the bound puts both terms on a comparable `[0, 1]` scale inside
  the feasible region, so the weights mean what they say. Both are configurable
  (`CHAI_OBJECTIVE_ERROR_WEIGHT`, `CHAI_OBJECTIVE_CYCLE_WEIGHT`) and must be
  fixed before results are seen.
* **A dominance gate rejects candidates worse than doing nothing.** The ratio
  was previously unbounded above 1, and selected plans ran at **1.89×** (AR,
  CHAI) and **2.21×** (AR, Codex repair) the original program's power cycles
  while being reported as results. Disable only for a deliberate ablation, via
  `CHAI_REQUIRE_IMPROVEMENT=0`.

## 5. The identity contract is measured, not assumed

The generation prompt requires that every knob range contain a setting at which
the function reproduces the original exactly. Nothing used to check it: the
baseline gate only required `error ≤ 0.30`, so a plan whose declared reference
point was already **18.27% wrong** (AR, Codex repair) was accepted and every
downstream number was measured against a broken reference.

`runBayesOpt` now measures the identity point and discards the plan when the
declared safe baseline is not actually the identity
(`CHAI_IDENTITY_TOLERANCE`, default `1e-6`).

## 6. Outcomes are distinguishable

`gp_minimize` returns the 100.0 penalty when every candidate is rejected.
Comparing the baseline against that penalty reported "the baseline won" for a
campaign in which nothing worked. `bo_outcome` now records which happened:

| `bo_outcome` | Meaning |
| --- | --- |
| `approximation_selected` | a real approximate configuration won |
| `no_improvement_found` | nothing beat the unapproximated program |
| `baseline_selected` | the identity point had the best objective |
| `baseline_not_identity` | plan discarded: declared safe point was not the identity |
| `baseline_invalid` | plan discarded: safe point failed to build or run |
| `no_valid_plan` | generation never produced an acceptable plan |

Both `no_improvement_found` and `no_valid_plan` are legitimate findings and
must be reported, not dropped.

## 7. Error metrics are comparable

A shared "30% bound" previously meant a different amount of damage on every
benchmark. Every evaluator now returns a normalised *fraction of output quality
lost* in `[0, 1]`; see `ERROR_METRIC_SPEC` in `utils/error_analyzer.py`.

| Benchmark | Metric | Fixed |
| --- | --- | --- |
| Sobel | `1 − SSIM` | was returning `(1 − SSIM)/2` while logging `1 − SSIM` — every reported error was half the damage measured |
| AR | per-window normalised MAE | was comparing only column totals, so an output mislabelling every window could score ≈0 |
| FFT | `mean\|ΔX\| / mean\|X_ref\|` | per-bin division mapped legitimately-zero reference bins to a *perfect* score via `nan_to_num` |
| LQI | zero-safe relative error | divided by the ground truth with no zero guard |
| Bitcount | elementwise normalised error | `/(ref + 1e-10)` turned any error on a zero reference into ≈1e10 |
| String search | `1 − F1` | unchecked build, stale output |

Every evaluator now also checks the build and run exit status and deletes stale
outputs first. Sobel previously ran `./main && mv img.pgm predicted.pgm` without
checking the status, so a candidate that failed to run was scored against the
*previous* candidate's image.

## 8. Terminology

Fused writes `getPowerOnResetCount()` to `cycles.dump`
(`fused-checkmate/main.cpp`). That is a count of **simulated power-on resets /
power cycles**, not checkpoints. Logs, CSV headers, and write-ups use
`power_cycles`. The `checkpoint_*` aliases remain for backwards compatibility.

## 9. Acceptance gate uses the target toolchain

Generation previously accepted a candidate as soon as it compiled with **host
gcc**, using a fallback Makefile that links `-lm`. Under CHAI a candidate must
now also cross-compile for MSP430 and fit the device's memory regions
(RAM `0x1000`, FRAM `0xBF80`, from `eval-apps/support/msp430fr5994.ld`).

When the cross toolchain is unavailable the gate reports `skipped`, never
`passed`, so a run performed without it cannot be mistaken for a validated one.
This is a control-plane feature and is therefore CHAI-only.

## 10. Search budget

`CHAI_BO_CALLS=8` spent 1 call on the identity baseline and 5 on random initial
points, leaving **two** model-guided evaluations for search spaces up to 16
dimensions. Results at that budget are dominated by the random draw. The
campaign uses 40, and `utils/bo_config.py` warns below 10 model-guided calls.
Plan dimensionality is additionally capped by
`--max_operators_per_function` (default 4).

## 11. Reporting rules

* Report generated, repaired, and backend-replaced results **separately**.
  `logs/generation_repairs_<app>.jsonl` records every deterministic repair, and
  the Sobel/bitcount canonical backends replace the candidate entirely — those
  are system-level results, not evidence about the planner.
* Report `no_valid_plan` and `no_improvement_found` cells.
* Report the measured LLM call total (`api_calls_by_stage`), not the
  generation-loop count. Makefile generation alone can issue 10 calls per
  compile test, each with up to 5 retries.
* State the number of seeds and show the spread, not just the best run.

## 12. Compute setup

Inference and simulation are separated because they have different failure
modes.

**Inference (Ollama, `chai-qwen2.5-coder:7b-24k`).** Sustained 7B inference on
the MacBook thermally shuts the machine down mid-campaign. That does not fail
loudly: it leaves a partially written cell that is indistinguishable from a
legitimate failure, and it silently truncates the run. Inference therefore runs
on the lab GPU host, reached over an SSH tunnel:

```bash
# one-time, from your own terminal (installs the key; needs the password once)
ssh-copy-id -i ~/.ssh/id_rsa.pub nsola5@131.193.50.170

# then every campaign
OLLAMA_SSH_TUNNEL=nsola5@131.193.50.170 bash scripts/run_ablation_campaign.sh
```

The runner opens `ssh -N -L 11435:localhost:11434`, waits for the remote
Ollama to answer, points the container at `host.docker.internal:11435`, and
tears the tunnel down on exit. The manifest records `ollama_ssh_tunnel` so a
result can always be traced to where it was generated.

**Simulation (Docker, MSP430 toolchain, Fused).** Runs locally in
`chai-dev:local`. This is CPU work, not the thermal problem.

**Sleep.** The runner holds `caffeinate -i` for the duration of the campaign.
Without it, idle sleep kills the running container between cells.

**Durability.** A campaign outlives any single terminal only if it is launched
detached. Use:

```bash
OLLAMA_SSH_TUNNEL=nsola5@131.193.50.170 \
  nohup bash scripts/run_ablation_campaign.sh > campaign.log 2>&1 &
disown
```

(`setsid` does not exist on macOS; `nohup` plus `disown` is the portable
equivalent here. The runner's own `caffeinate -i` covers the sleep problem,
which was the actual cause of the earlier truncated run.)

A campaign launched from a foreground shell that is later closed will be
killed, and the partial `experiment_results/<run_id>/` left behind must be
discarded rather than analysed: the analyser will mark such cells invalid
(missing `status.json`), but do not reuse the run id.

**Model provenance.** The runner queries `/api/show` and records the model
digest and parameters in the manifest. If the GPU host's model differs from the
laptop's, that is a real confound and the digest is how you catch it. Verify
once before the first remote campaign:

```bash
ssh nsola5@131.193.50.170 'curl -s localhost:11434/api/show \
  -d "{\"model\":\"chai-qwen2.5-coder:7b-24k\"}" | head -c 400'
```

If the model is absent on the GPU host, create it there from
`config/ollama/Modelfile.qwen25-coder-7b-24k` before running anything.
