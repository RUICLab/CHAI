# Codex full plan: FFT on MSP430

## Constraints used for planning

- MSP430 target: no FPU, no SIMD, 10 KB RAM, limited FRAM.
- The baseline FFT already fits closely; prior richer fixed-point repairs
  overflowed FRAM.
- The input size is eight, so memory-heavy memoization and model structures
  are not justified.

## Selected operators

1. **T1 Loop Perforation** — process a configurable fraction of butterfly
   indices.  `loop_perforation_percent=100` is the identity baseline.
2. **T4 Task Skipping** — optionally reuse the preceding butterfly result for
   odd indices.  Disabled at baseline.
3. **T7 Truncated Computations** — independently cap the butterfly fraction.
   `truncation_percent=100` is the identity baseline.

T2 is deliberately not selected: a full fixed-point FFT that satisfies the
no-FPU contract exceeded the target's FRAM in the prior repair experiment.
All controls are integers so the plan introduces no software floating-point
arithmetic.
