# AR Codex plan-repair condition

This directory contains a **separate** fixed plan for the AR experiment.  It
is not an Ollama output and must be invoked with `--no_llm --prerun_dir`.

The plan retains the operator IDs selected by the hardware-aware Qwen run.
The three Qwen-accepted functions are reproduced unchanged except for standard
knob-block spacing.  The six rejected functions are repaired implementations:

- `ACCEL_singleSample`: T2, T4, T11
- `sqrt16`: T2, T7
- `warmup_sensor`: T4
- `featurize`: T2, T7
- `recognize`: T1, T2
- `train`: T1, T2

Repairs are limited to valid C, preserving signatures and state, and using
integer/fixed-point operations for every T2 path (the MSP430 profile has no
FPU).  Where Qwen's rejected response omitted a usable T2 knob contract, the
repair uses a conservative integer scale/shift knob with its identity default;
this is an implementation repair, not a claim that Qwen supplied valid code.
