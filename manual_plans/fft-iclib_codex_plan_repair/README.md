# FFT Codex plan-repair condition

This is a third condition for the paired FFT experiment and makes no model
calls.  It starts with Qwen's rejected `fft_radix2` plan and retains exactly
its selected operator IDs: T1, T2, T3, T4, T7, T10, T12, and T13.

The raw plan's `precision_scaling_factor` was a floating-point value-scaling
knob and therefore failed the MSP430 no-FPU T2 contract.  The repair adds
`fixed_point_mode`: when enabled, each butterfly uses Q12 integer arithmetic
and an integer twiddle lookup table.  The original double-complex path remains
the safe default so the benchmark interface and reference behavior are
preserved.  Identity defaults are used for all other repaired knobs.

This measures a Codex implementation repair.  It must not be described as an
Ollama/Qwen generation result.
