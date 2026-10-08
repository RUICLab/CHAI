# Biomedical dataset pilot results

Three seeds (2026–2028). Values are medians with the seed range in parentheses. 
Decision thresholds are selected only on validation data. These are host-side 
classification/compute results; neither model has yet been exported to C or measured 
for MSP430 power cycles in Fused.

## COUGHVID — self-reported COVID-19 vs healthy/symptomatic

| STFT hop | frame reduction | balanced accuracy | AUROC | disagreement vs L0 |
|---:|---:|---:|---:|---:|
| 1× | 0.00% | 57.50% (53.50%–58.00%) | 58.89% (50.14%–60.71%) | 0.00% (0.00%–0.00%) |
| 2× | 49.83% | 56.50% (50.00%–56.50%) | 58.64% (49.76%–59.15%) | 32.50% (25.00%–60.50%) |
| 4× | 74.75% | 51.50% (51.00%–53.00%) | 52.95% (51.75%–55.16%) | 40.50% (34.50%–60.50%) |
| 8× | 87.21% | 53.00% (49.50%–53.00%) | 50.64% (48.30%–56.51%) | 50.00% (34.50%–53.00%) |

The exact-level classifier is not stable across seeds, so this is not yet a defensible 
medical-classification result. The 2× hop level is only a compute-preservation signal: 
it halves STFT frames, but a stronger frozen base classifier is required before CHAI 
optimization or power measurements are meaningful.

## PatchCamelyon — tumor vs non-tumor

| center resolution | MAC reduction | balanced accuracy | AUROC | disagreement vs L0 |
|---:|---:|---:|---:|---:|
| 64×64 | 0.00% | 75.55% (73.92%–76.54%) | 83.95% (83.80%–84.59%) | 0.00% (0.00%–0.00%) |
| 48×48 | 43.75% | 75.45% (74.91%–77.08%) | 83.81% (83.63%–85.18%) | 8.90% (8.16%–9.37%) |
| 32×32 | 75.00% | 75.50% (73.81%–76.77%) | 83.62% (83.10%–85.54%) | 9.88% (9.76%–12.28%) |
| 24×24 | 85.93% | 73.15% (72.91%–75.84%) | 81.97% (81.48%–84.16%) | 14.57% (12.76%–17.94%) |
| 16×16 | 93.75% | 71.83% (71.22%–72.98%) | 80.02% (79.13%–81.69%) | 18.60% (16.80%–21.79%) |

PCam is the promising workload. The useful operating point is the smallest resolution 
whose seed range retains the 64×64 reference accuracy; that decision must be based on 
the complete table rather than a single seed.

## Protocol qualifications

- COUGHVID uses 500 high-confidence cough recordings per class and a 60/20/20 
  stratified split per seed. Most COVID status labels are self-reported.
- PCam uses 24,576 images from the official validation file for development, holds out 
  the remaining 8,192 for model/threshold selection, and evaluates all 32,768 images 
  in the separately distributed test file. This is a pilot split because the official 
  6.1 GB training file was rate-limited.
- The official PCam repository documents 327,680 balanced 96×96 patches and labels 
  based on tumor in the center 32×32 region: https://github.com/basveeling/pcam
- COUGHVID source and metadata description: https://zenodo.org/records/4048312
