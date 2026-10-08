# Preflight status — 2026-09-13

This is development evidence from a dirty-tree run, pinned by each run's
configuration hash. It is not part of the final 72-cell campaign.

## String-search/T3

- Qwen32 autonomously selected T3 for `init_search`, planned exact keyed reuse,
  and generated code that passed conformance, compilation, identity, target
  build, and the output-invariant T3 handling.
- The unchanged two-function artifact was replayed for 20 BO calls under the
  constrained objective. Best knobs were `[1, 0, 100]`: useful `init_search`
  reuse enabled, spurious `strsearch` reuse disabled, and T21 at its exact
  setting.
- Result: error `0.0`, power cycles `1` versus `26`, reduction `96.1538%`.
- The same artifact revealed a verifier gap: the generated `strsearch` T3
  duplicated work in both branches. T3 now requires an immediate hit-path exit,
  and selection removes T3 when no stable-key repeated-call evidence exists.

Artifacts:

- Generation: `experiment_results/t3exact_qwen32_smoke_20260913`
- Replay: `experiment_results/t3exact_qwen32_replay20_20260913`

## Sobel/T22

- Qwen32 selected T22 in both completed selection attempts.
- The first plan was blocked by a parser bug that interpreted `pass` in
  `second pass (normalization)` as a fabricated C callee. The prose vocabulary
  and regression test are fixed.
- Subsequent code-generation attempts are infrastructure-invalid: remote GPU 0
  entered NVIDIA `Unknown Error`; Ollama then failed CUDA and silently loaded
  Qwen32 on CPU. GPU 1 remains visible to `nvidia-smi`, but Ollama and its direct
  CUDA backend report no available CUDA device because discovery is poisoned by
  the failed device.
- Required next action: administrator driver reset or host reboot, then rerun
  the single Sobel Qwen32 CHAI generation cell. Do not score the interrupted
  attempts as planner failures.

## Gate to the 24-cell preflight

Do not launch the 24-cell preflight until:

1. remote Qwen32 inference reports nonzero `size_vram`;
2. one Sobel/T22 candidate reaches all generation-time gates; and
3. the exposed Anthropic credential is rotated before any Sonnet run.
