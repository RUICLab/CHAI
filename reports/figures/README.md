# Paper figures

Regenerate with:

```bash
python3 scripts/make_paper_figures.py reports/plan_quality_data.json
```

| File | Content | Suggested placement |
| --- | --- | --- |
| `fig1_false_provenance.{pdf,png}` | (a) false operator labels per accepted plan, pooled over 6 benchmarks; (b) the three individual repeats behind it on ar-iclib | full width (7.0 in) |
| `fig3_coverage.{pdf,png}` | Coverage matrix: repeats yielding an accepted plan, 6 benchmarks x 4 arms | single column (3.4 in) |
| `fig4_specialisation.{pdf,png}` | Distinct per-function operator menus | single column (3.4 in) |
| `fig5_model_scale.{pdf,png}` | **The headline.** False provenance per plan at two model scales: qwen2.5-coder 7B vs Claude Sonnet 5 | full width (7.0 in) |
| `all_figures.pdf` | All of the above, one page each | review copy |

PDF is vector with embedded TrueType text (searchable, and it passes the usual
publisher font checks). PNG is 400 dpi for slides or a quick preview.

## Arms

| | Catalogue | Control plane |
| --- | --- | --- |
| A | CheckMate T1–T20 | off |
| B | CheckMate+10 T1–T30 | off |
| C | CHAI T1–T30 | **on** |
| D | CHAI T1–T20 | **on** |

**D is the arm that isolates the contribution.** It uses the original T1–T20
catalogue, so any difference between D and A is the control plane rather than
the ten added operators.

## Reading notes for a caption

* Colour is never the only encoding: the two unenforced arms also carry
  distinct hatching, so Fig. 1(a) survives greyscale printing and colour-vision
  deficiency. The palette was checked for CVD separation (worst adjacent pair
  ΔE 9.2 deutan, 27.6 normal vision).
* Fig. 1 is a **rate**, not a count, because the enforced arms accept roughly
  half as many plans; a raw count would flatter them.
* Fig. 1(b) exists because the planner is not reproducible at temperature 0.
  A single run fixes no magnitude, so the individual repeats are shown rather
  than a mean with an error bar that three points cannot support.
* Fig. 3 is the honest limit of this dataset: only ar-iclib has data in all
  four arms. Every zero was verified as a correct rejection (code that did not
  compile, an identity setting measuring 78% error, a knob declared and never
  used), so the constraint is the 7B planner, not the framework.

## Figure 5 — the model-scale control

The obvious objection to the plan-quality result is that it only measures a
small model being sloppy. Figure 5 answers it directly.

| arm | 7B | Sonnet 5 |
| --- | ---: | ---: |
| A  CheckMate T1-T20, control off | 0.43 (10/23) | 0.25 (6/24) |
| B  CheckMate+10, control off | 0.47 (9/19) | 0.48 (10/21) |
| C  CHAI T1-T30, control **on** | **0.00** (0/13) | **0.00** (0/13) |
| D  CHAI T1-T20, control **on** | **0.00** (0/12) | not run (billing) |

Same benchmark, same frozen target list, three repeats per cell. A frontier
model that writes valid, compiling C -- code that passes the identity gate and
the knob-sensitivity check -- still declares operators its own code does not
implement, at essentially the same rate as a 7B. Enforcement removes them at
both scales.

Arm D is the one that isolates the control plane from the ten added operators:
original T1-T20 catalogue, control plane on, 0.00. It has 7B data only; the
Sonnet 5 cell failed on API billing, not on the planner.

## What these figures do NOT show

Performance. Replaying the same Sonnet 5 plans through BayesOpt and the Fused
simulator at CheckMate's 0.50 error bound gave the opposite ordering:

| arm | best power-cycle reduction across 3 seeds |
| --- | --- |
| checkmate_baseline | **+84.2%** (3/19 cycles, 34.1% error) |
| operator_only_plus10 | none -- identity selected in all three |
| chai_plus10 | none -- identity selected in all three |

The unenforced baseline found large savings and CHAI found none. CHAI accepts
fewer functions, its surviving plans stay at their identity settings, and its
scaffolding costs ~5% (identity measures 20 cycles against the original 19).
These figures support a claim about plan quality only; they do not support
"CHAI is faster", and the measured evidence currently contradicts it.
