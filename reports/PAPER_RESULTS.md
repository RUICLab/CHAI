# Results — draft, numbers pulled from committed artifacts

Every figure below regenerates from a script in `adaptive/` or `scripts/`, and
every number is traceable to a file in `reports/`. Where a claim was withdrawn
during analysis, the withdrawal is noted rather than quietly dropped.

**Scope, stated once and applying throughout.** All results are from simulation
(Fused SystemC, MSP430FR5994); nothing ran on hardware. Approximation error per
level is measured on a **single 32×32 test image**, so error figures are point
estimates for that input, not distributions over a representative input set.
Approximation levels are hand-written canonical implementations, not planner
output — they bound what the transformation can do, not what CHAI's planner
produces.

---

## 1. The control plane removes false operator labels

Arm B (CheckMate + the same 10 added operators, no control plane) against arm C
(CHAI, identical 30-operator catalogue, control plane on). Both draw from the
same menu, so the difference is attributable to the control plane.

| | B: CheckMate + 10 ops | C: CHAI, same 30 ops |
|---|---|---|
| Operators offered to the planner | 32 (all) | 16 of 32 (hardware-filtered) |
| Distinct per-function menus (ar-iclib) | 1 | 7 |
| **False operator labels the model made** | **19 of 42 attempts** | **22 of 42 attempts** |
| **False labels shipped to the user** | **10** | **0** |
| Plans carrying at least one false label | 10 of 20 (50%) | 0 of 13 (0%) |
| Hardware-impossible operators delivered | 1 | 0 |

6 benchmarks × 3 repeats. Per-seed mislabel counts are B {5, 7, 7} and
C {7, 7, 8}: the ranges overlap, so **this run cannot separate the two planners'
accuracy, and CHAI's model is if anything marginally worse.** The claim is
therefore about *enforcement*, not model quality:

> CHAI detects and removes the false operator labels that CheckMate ships.

Never "CHAI's model makes fewer mistakes." The zero is true by construction —
`prune_unimplemented_operator_claims` strips disproven labels before the accepted
artifact is written, and it is gated CHAI-only.

*Source:* `scripts/operator_label_audit.py`, `reports/operator_label_audit.json`.
Counts are distinct `(benchmark, seed, function, operator id)` tuples over
`functions_attempted`, which is 42 in every arm. An earlier version reported
35 vs 37 from `operator_conformance_violation_count`; that counts violation
*events*, and because only CHAI enforces the check it retries more (2.47 attempts
per violation against 1.84), so the event count penalised the arm doing the
checking.

---

## 2. A runtime dispatcher selects execution level from measured energy

CHAI generates one configurable program; characterization produces a
quality–energy table; at run time the device reads its capacitor, computes
available energy, and fires the least-approximate affordable level.

```
E_available = ½·C·(V² − V_release²)
pick the least-approximate level with E_required·(1+margin) ≤ E_available
else defer
```

**Functional validation** (sobel, 4700 µF, 10% margin):

| V | available | fires | required (×1.1) |
|---|---|---|---|
| 3.59 / 3.50 | 18.04 / 16.54 mJ | L0 | 16.32 mJ |
| 3.45 → 2.60 | 15.72 → 3.64 mJ | L1 | 3.42 mJ |
| 2.55 / 2.50 | 3.03 / 2.44 mJ | L2 | 2.11 mJ |
| 2.45 → 2.30 | 1.86 → 0.18 mJ | DEFER | — |

All four branches fire, degradation is monotone in supply voltage, and there are
**zero safety violations**: no decision selected a level whose requirement
exceeded available energy including margin.

**Timeline** over a controlled high→medium→low→none→recovery profile shows
`L0 → L1 → DEFER → recovery → L0`; 15 of 19 periods complete, and each of the 4
deferrals occurs where the capacitor genuinely could not fund a task.

*Source:* `adaptive/dispatcher_validation.py`,
`reports/adaptive/dispatcher_validation.txt`.

---

## 3. Capacitor size sets the operating envelope

Holding a whole task requires 499 µF for L2, 811 µF for L1, 3866 µF for L0 — so
capacitance determines *which rows of the table can ever fire*, and is a design
parameter rather than a tuning knob.

Level mix on five measured RF recordings plus indoor solar, 15 s deadline:

| capacitance | L0 | L1 | L2 | defer | completion | mean error |
|---|---|---|---|---|---|---|
| 1000 µF | 0 | 15–17 | 0–1 | 2–3 | 84–89% | 0.019–0.035 |
| 4700 µF | 1–5 | 7–8 | 0 | 5–8 | 42–68% | 0.012–0.017 |

A larger buffer funds the exact level and roughly halves mean error, at the cost
of deferring more often while it refills.

**Startup cost is one-off and must be separated.** At 4700 µF a cold start
(0 V) spends 104.6 s filling the capacitor and completes 23 tasks; warm-started
it completes 36 — identical to a 500 µF device. An earlier analysis attributed
that gap to the architecture; it is a first-fill cost that amortises over a
deployment of hours.

*Source:* `adaptive/capacitor_sweep.py`, `adaptive/coldstart_study.py`.

---

## 4. When does runtime adaptation pay?

Dynamic-quality tiering — spend a large budget on exactness, a small one on a
completed-but-rougher result — beats the best fixed level in **7 of 24 cells**,
and the wins are concentrated where energy is scarce:

| cell | best fixed | oracle tier | gain |
|---|---|---|---|
| sobel / solar / 8 s | 25.9 | **32.4** | +25.1% |
| sobel / RF_6 / 8 s | 19.4 | **23.6** | +21.6% |
| sobel / RF_2 / 8 s | 25.5 | **30.1** | +18.0% |
| fft / RF_6 / 8 s | 27.8 | **32.0** | +15.1% |

The headline cell: fixed-L1 completes 64% at error 0.019; fixed-L2 completes 100%
at 0.280; the tier policy completes **100% at 0.099** — the cheap level's
deadline record at 2.8× its accuracy.

**Every win is at the 8 s deadline.** At 15 s and beyond the cheaper level is
nearly always affordable, the tier collapses onto a constant, and the gain is
0.0%. The opportunity exists only where the energy budget straddles a tier
boundary often enough to change the answer between periods.

*Quality* is Σ(1 − error) over tasks finished inside their deadline; completion
alone rewards always-cheapest and mean error alone rewards always-exact.

*Source:* `adaptive/tier_study.py`, `reports/adaptive/tier_study.txt`.

---

## 5. Why a deployable estimator does not capture it

This is an **oracle** ceiling — it reads the true future. Three causal
alternatives were tested, and none captures it:

**Forecast the future.** Harvest autocorrelation decays to ~0 by 5–8 s while
deadlines are 8–40 s. Measured forecast error is 27–50%, against a 25–38% gap
between adjacent levels: the noise is wider than the boundary it must resolve.
Chronos-Bolt zero-shot beats persistence but loses to a two-line EWMA
(α = 0.02) at 3 of 4 horizons, and cannot run on a 10 KB MCU at any size.

**Read stored energy.** Capacitor voltage measured 3.583 V at *every* decision
instant but the cold start — a constant, carrying no information. A full 68 µF
capacitor holds 261 µJ while the cheapest task needs 1436 µJ, so the device never
banks a task's worth of energy.

**Table on measured harvest rate.** Thresholds grid-searched on held-out data
collapse onto grid extremes — the learned table has one entry. There is no
harvest rate at which a different level is the better choice.

*Source:* `adaptive/forecast_study.py`, `adaptive/voltage_table_study.py`,
`adaptive/rate_table_study.py`, `adaptive/chronos_eval.py`.

---

## Claims withdrawn during analysis

Recorded because each was reported before being corrected.

- **"Reactive beats the best fixed level."** Compared against fixed-L1 when
  fixed-L0 dominated. Withdrawn.
- **"Adaptation has nothing to decide."** An artifact of evaluating at a 0.05
  error bound, which excludes L2 and leaves only one in-bound choice. The
  deployment bound is 0.30.
- **"The no-exact-level ladder gains 17%."** The entire gain came from a 0.344
  rung that is out of specification at 0.30. In-spec the two ladders are
  identical.
- **"Large capacitors are self-defeating."** A cold-start artifact; warm-started
  they match.
- **"Server forecasting fails on radio economics."** It loses at 0 µJ too, so
  the cause is the decision objective and forecast quality, not link cost.

---

## Limitations

1. **Single input.** All error figures come from one 32×32 image. L2 measured
   0.280 against a 0.30 bound — 6.7% headroom on n = 1. State L2 results as
   *"measured 0.280 on our test input"*, not as satisfying the bound in general.
2. **Simulation only.** No MSP430 was flashed; no radio packet was sent.
3. **Hand-written levels.** The planner-to-dispatcher join is not demonstrated
   end to end.
4. **Trace length.** Measured RF recordings are 25–52 s and are looped to fill a
   run; only the 300 s solar recording supports unlooped analysis. Two scenarios
   ("varying", "mixed") are synthesised by scaling and concatenating real traces
   and are labelled as such.
5. **Task energy is pinned** at its 68 µF measurement; a larger capacitor changes
   brown-out frequency and checkpoint overhead.
