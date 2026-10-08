# Final campaign runbook

## Completed prerequisites

- T3, T5, T6, T13 and T19 no longer depend on identifier names.
- T3 and T13 have distinct structural contracts.
- T22's checker accepts anchor-plus-fill downsampling and rejects plain 2-D
  perforation that leaves holes.
- T25 is withdrawn; T27 is retired from the six-benchmark objective.
- All six target inventories are source-derived and checked in under
  `config/frozen_targets/`; the runner cannot regenerate them with an LLM.
- Direct caller snippets and a general innermost-loop invariance detector make
  exact reuse visible to planning.
- Every measured BO point is logged with a call number, enabling matched
  20/40/80/160/250 curves.
- Raw planner-pipeline responses and hashes are archived per repetition.
- Tunneled Ollama campaigns run a real one-token inference and require
  `size_vram > 0`, preventing a CUDA fault from silently becoming a CPU run.

## Development evidence (not final campaign evidence)

1. Qwen 7B saw the corrected `init_search` target but repeatedly returned
   `PLAN: none`, including the false claim that the function had no loops.
2. Qwen 32B produced the correct T3 plan and immutability precondition, then
   failed to implement it in three generation attempts.
3. The implementation followed that frozen plan exactly. Replay on MSP430 / RF_2 / 68 µF
   produced error 0.0 and 8 versus 26 power cycles: a 69.23% reduction.
4. After the exact-T3 sensitivity correction, Qwen 32B autonomously generated
   a valid `init_search` T3 implementation. An unchanged 20-call replay found
   error 0.0 and 1 versus 26 power cycles. The same artifact also contained a
   false T3 label on `strsearch`; that label is disabled at the best point and
   is now rejected by both call-site selection and the immediate-exit T3
   contract. Therefore this is strong development evidence for the
   `init_search` plan, but the complete cell must be regenerated under the
   tightened verifier before it can count in the final comparison.

The third result is an implementation ablation and must not be reported as
autonomous CHAI output. The fourth is autonomous development evidence, not a
final-campaign result, because its verifier changed after the run.

## Go/no-go sequence

1. Rotate the exposed Anthropic API key and remote login password. Do not put
   either value in chat or a tracked file.
2. Run a two-target Sonnet generation pilot (string search and Sobel). It must
   produce at least one contract-valid candidate for each before spending BO
   budget.
   If using remote Ollama for a development pilot, first confirm the host's
   NVIDIA driver is healthy; the launcher now refuses CPU fallback.
3. Run the 24-cell preflight (six apps × four arms × one fresh planner
   repetition) at 250 calls:

   ```bash
   CHAI_PROVIDER=Anthropic \
   MODEL=claude-sonnet-5 \
   CONFIRM_CREDENTIAL_ROTATION=1 \
   bash scripts/run_campaign_preflight.sh
   ```

4. Inspect `bo_budget_curve.csv`; freeze one common budget from
   20/40/80/160/250 before the final run.
5. Commit/freeze the exact code and require a clean worktree. Run six apps ×
   four arms × three fresh planner repetitions. Do not edit the mounted repo
   during the campaign.
6. Verify unique planner hashes, analyze all cells, and report medians,
   no-valid-plan/no-improvement rates, raw false claims, removed claims,
   accepted-artifact false claims, LLM calls/tokens, and target-build outcomes.

## Required interpretation

- Primary performance: minimum power cycles subject to the frozen error bound.
- T3 exact reuse is reported separately from approximate-error improvements.
- The six benchmarks informed operator repair/scoping; disclose this rather
  than describing them as a held-out evaluation set.
- A CHAI accepted-artifact false-claim count of zero is an enforcement outcome,
  not evidence that its underlying model made no mistakes.
