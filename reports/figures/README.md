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
