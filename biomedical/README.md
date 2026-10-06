# Biomedical benchmarks

This directory contains the dataset preparation, training, export, and
held-out evaluation code for the COUGHVID and PatchCamelyon (PCam) workloads.
The exported C applications live in `benchmark_applications/{coughvid,pcam}-iclib`
and are mirrored under `eval-apps/` for CHAI's build and evaluation path.

- `coughvid_detect_pilot.py`: trains the compact cough/not-cough CNN used by
  the embedded benchmark.
- `pcam_pilot.py`: trains the compact metastatic-tissue classifier and supports
  controlled input-resolution and channel-width experiments.
- `export_coughdetect_to_c.py` and `export_pcam_to_c.py`: fold BatchNorm into
  convolution weights and export frozen C headers and evaluation inputs.
- `evaluate_kernel_heldout.py`: evaluates an exported kernel over a held-out,
  multi-example set rather than a single baked-in input.
- `measure_power_cycles.py`: measures explicit CHAI-generated knob settings
  with the MSP430/Fused toolchain.

The older `coughvid_pilot.py` and `export_coughvid_to_c.py` remain as the
spectrogram-resolution pilot. They are not the compact embedded detector.

Reported classification metrics are calculated locally from frozen held-out
sets. COUGHVID labels follow the benchmark's documented distillation protocol;
neither workload is a clinical diagnostic system.

## Reproduce

```bash
python3 -m venv --system-site-packages .venv-biomedical
.venv-biomedical/bin/python -m pip install \
  'scikit-learn>=1.5' soundfile av gdown

.venv-biomedical/bin/python -m biomedical.coughvid_pilot
.venv-biomedical/bin/python -m biomedical.pcam_pilot
```

Data is stored under `data/biomedical/` and excluded from git.  The PCam pilot
uses the official validation split for development because the official 6.1 GB
training file may be rate-limited; its official test split remains untouched.
That split choice must be stated with every pilot result and replaced for a
final campaign.

Run the repository tests with the biomedical environment so NumPy and its
compiled scientific-Python dependencies come from one compatible environment:

```bash
.venv-biomedical/bin/python -m pytest -q
```
