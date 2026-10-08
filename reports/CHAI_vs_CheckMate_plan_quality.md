# Does CHAI plan better than CheckMate + the same operators?

**Short answer: yes, on plan quality — and the gain comes from the control
plane, not from the extra operators.**

Target: MSP430FR5994 (no FPU, no SIMD, 10 KB RAM, 16 MHz).
Campaign: 6 benchmarks x 3 repeats, 7B planner, temperature 0.
Source data: `reports/plan_quality_data.json`, run `planquality_20260906_164509`.

---

## The comparison that matters

The obvious objection to CHAI is *"you just added ten more operators."*
Arm **B** rules that out: it is CheckMate with **the identical 30-operator
catalogue** CHAI uses, and no control plane. Arm **C** is CHAI with the same
catalogue and the control plane on.

**Both arms choose from exactly the same menu. Any difference is the control
plane.**

| | **B** — CheckMate + 10 operators | **C** — CHAI, same 30 operators |
|---|---:|---:|
| Control plane | off | **on** |
| Operator catalogue | T1–T30 (32 incl. auto-generated) | identical |
| Operators offered to the planner | 32 (all) | **16 of 32** (median; 15–17) |
| Distinct per-function menus (ar-iclib) | 1 | **7** |
| Accepted plans | 20 | 13 |
| Hardware-impossible operators delivered | 1 | **0** |
| False operator labels the planner made (distinct IDs, 42 function attempts) | 19 | 22 |
| Unsupported labels *shipped* in the delivered plan | 10 | **0** |
| Plans carrying at least one false label | **10 of 20 (50%)** | **0 of 13 (0%)** |

---

## The three findings

### 1. Neither planner labels more accurately. Only one arm catches it.

Counting *distinct* false operator labels — one per (benchmark, seed, function,
operator id) that ever failed conformance — over a denominator both arms record:
**19 (arm B) vs 22 (arm C)** in 42 function attempts each, 0.45 vs 0.52 per
attempt. Per-seed counts are B {5, 7, 7} and C {7, 7, 8}: the ranges overlap, so
this run cannot separate the two planners. If anything CHAI's model is
marginally worse. It is certainly not better.

An earlier version of this table reported **35 vs 37** here and called it "the
same mislabelling rate". Both halves were wrong.
`operator_conformance_violation_count` counts violation *events*, so one bad
operator re-checked after each retry counts once per attempt — and the arms do
not retry equally. Conformance is enforced only in the CHAI arms, which
re-prompt after a failed check: 2.47 attempts per violation in arm C against
1.84 in arm B. The event count therefore penalised the arm doing the checking.
There was also no denominator: 35 and 37 were bare totals.

What differs is what survives to the delivered artifact: **10 vs 0**. Exactly
half of arm B's plans ship at least one operator label the code does not
implement. CHAI detects the same mistakes and removes them before acceptance.

> **Important caveat, state it plainly:** CHAI's zero is an *enforcement
> outcome*, not evidence of a better planner. CHAI prunes disproven labels
> before writing the accepted artifact, so the zero is true by construction.
> The honest claim is *"CHAI detects and removes the false labels that
> CheckMate ships"* — never *"CHAI's model makes fewer mistakes."* The 19-vs-22
> counts are shown precisely so the figure cannot be misread; they are
> recomputed by `scripts/operator_label_audit.py`, never typed by hand.

### 2. The hardware gate removes half the menu before the model sees it.

CHAI filters the catalogue against the target's real capabilities: 32
operators down to a median of 16. Arm B sees all 32, including operators the
chip physically cannot execute.

**Concrete failure:** on `fft-iclib`, arm B selected **T24 "Magnitude-Only
FFT" — which requires an FPU — on a chip with no FPU.** The same plan also
declared T24 without implementing it. One operator, two independent defects,
shipped in a delivered plan. CHAI never offered T24 for this target.

### 3. CHAI plans per function; CheckMate plans once.

On `ar-iclib`, arm B gave every function the *identical* operator menu
(1 distinct menu, all 3 repeats). CHAI produced **7 distinct menus** across the
same functions — the menu is derived per function from what that function
actually does.

---

## What this does NOT show

Be explicit about this when forwarding — it is the first thing a reviewer will
ask.

- **No performance claim.** These are properties of the *plan*, not of measured
  energy or power cycles. A separate replay found the unenforced baseline
  reaching larger cycle savings than CHAI on ar-iclib. Plan quality and
  end-to-end savings are currently separate results, and only the first is
  settled.
- **Coverage is limited.** The 7B planner produced usable plans on ar-iclib
  (both arms) and fft-iclib (arm B only). The other four benchmarks yielded no
  accepted plan in any arm; all were verified as correct rejections, not
  crashes.
- **"Unsupported labels" is a lower bound** from a static + behavioural checker.
  It proves a declared transformation is *absent*; it cannot prove one is
  genuinely present.
- **Arm D is not yet run at scale.** D (CHAI restricted to CheckMate's original
  T1–T20) is what fully separates the control plane from the ten added
  operators. The 7B data has it (12 plans, 0 false labels); the larger-model
  confirmation is still pending.

---

## Figures

| File | Use |
|---|---|
| `figures/chai_vs_checkmate_provenance.{png,pdf}` | Finding 1 — the headline |
| `figures/chai_vs_checkmate_hardware.{png,pdf}` | Findings 2 and 3 |

Regenerate with `python scripts/make_colleague_figures.py`.
