# Plan-quality reports

Generated from `experiment_results/planquality_20260906_164509`
(6 benchmarks x 4 arms x 3 repeats = 72 generation-only cells).

Open the two HTML files in a browser; everything else is plain text/JSON.

| File | What it is |
| --- | --- |
| `figures/` | **Start here for the paper.** Vector PDF + 400 dpi PNG of every figure, plus `all_figures.pdf`. See `figures/README.md`. |
| `how-chai-plans.html` | Same charts as a web page, with the mechanism written out. |
| `plan-quality-evidence.html` | Detailed evidence board: the FPU finding, per-function menus, the inert-knob code, full caveats. |
| `plan_quality_table.txt` | The same numbers as terminal tables. |
| `plan_quality_data.json` | Machine-readable, including per-function detail behind every count. |
| `ablation_analysis.txt` | Cell validity, provenance, and the (empty) 2x2 effect table. |

## Regenerating

```bash
python3 scripts/plan_quality_report.py experiment_results/<run_id> --json out.json
python3 scripts/analyze_ablation.py    experiment_results/<run_id>
```

To run a fresh sweep (needs Docker + the SSH tunnel to the GPU host):

```bash
GENERATION_ONLY=1 \
APPS="ar-iclib sobel-iclib stringsearch-iclib fft-iclib lqi-iclib bc-iclib" \
SEEDS="1 2 3" OLLAMA_SSH_TUNNEL=nsola5@131.193.50.170 \
  bash scripts/run_ablation_campaign.sh
```

## Headline numbers

False operator labels per accepted plan, pooled over all benchmarks and repeats:

| Arm | Plans | False | Per plan |
| --- | ---: | ---: | ---: |
| A  CheckMate T1-T20, control off | 26 | 13 | 0.50 |
| B  CheckMate+10 T1-T30, control off | 20 | 10 | 0.50 |
| C  CHAI T1-T30, control ON | 13 | 0 | **0.00** |
| D  CHAI T1-T20, control ON | 12 | 0 | **0.00** |

Arm D uses the original T1-T20 catalogue, so its result isolates the control
plane from the ten added operators.

## What these reports do NOT show

* Nothing about speed. No arm beat the unapproximated program on any benchmark
  measured; this sweep skipped search and simulation entirely.
* Only `ar-iclib` produced plans in all four arms. Five benchmarks contribute
  little or nothing, and every failure there was verified as a correct
  rejection caused by the 7B planner, not by the framework.
* CHAI accepts about half as many plans, which is why the headline is a rate
  rather than a count.
* False provenance is a lower bound: a checker can show an operator is absent,
  never prove one is present.
