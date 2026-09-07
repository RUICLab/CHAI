# FFT Codex compact repair

The full Q12 fixed-point butterfly repair overflowed MSP430 FRAM by 3,030
bytes.  This compact repair retains Qwen's T1, T2, T3, T4, T7, T10, T12, and
T13 selection and the original safe floating-point path, but implements the
T2 alternative as a knob-controlled integer quantization path.  It avoids
duplicating the complete floating-point FFT and therefore fits the firmware.
