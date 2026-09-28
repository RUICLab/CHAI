# Biomedical pilot benchmarks

These scripts establish two dataset-backed inference workloads before they are
exported to C and connected to CHAI/Fused:

- `coughvid_pilot.py`: self-reported COVID-19 versus healthy/symptomatic cough;
  approximation increases the STFT hop while keeping one frozen classifier.
- `pcam_pilot.py`: metastatic tissue versus no metastatic tissue;
  approximation reduces the sampling resolution of a frozen tiny CNN.

The pilot reports balanced accuracy and AUROC, plus disagreement from the exact
level and an operation-count proxy.  These are locally measured results, not
numbers copied from the dataset papers.  Fused power cycles must not be reported
until the trained kernels and their frozen weights are exported to C.

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

