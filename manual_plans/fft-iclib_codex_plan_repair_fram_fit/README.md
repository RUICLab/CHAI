# FFT Codex FRAM-fit repair

This is the executable Codex repair condition for FFT.  The original Qwen
operator selection (T1, T2, T3, T4, T7, T10, T12, T13) is retained, but its
floating-point knobs are represented by integer controls to fit MSP430 FRAM.

The safe default is the original FFT.  `fixed_point_mode=1` activates an
integer quantization fallback, controlled by `precision_bits`; it deliberately
avoids the double-complex butterfly so no FPU helper path is required.  The
fallback is intentionally lower fidelity and is evaluated under the same 30%
error bound rather than assumed safe.
