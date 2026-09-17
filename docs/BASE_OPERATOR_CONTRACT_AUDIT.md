# T1–T20 conformance audit

This audit records what the static checker can establish. It is a structural
floor, not proof that a candidate is faster or semantically useful.

| ID | Current structural evidence | Confidence | Audit result |
|---:|---|---|---|
| T1 | knob-derived value in the target loop's condition/increment | high | structural and name-independent |
| T2 | knob-controlled integer shift, mask, or narrowing cast | high | structural and name-independent |
| T3 | knob-enabled key equality; hit exits; miss computes then updates key | high | repaired; no cache/memo naming rule |
| T4 | knob-controlled branch/skip | medium | broad by definition; overlaps T3/T7, resolved by plan-label rules |
| T5 | knob-controlled index reduction into compact indexed bit storage | medium | repaired; no bloom/hash naming rule |
| T6 | random/PRNG production feeding a knob-controlled decision | medium | repaired; no RNG/seed naming rule |
| T7 | exit reachable from a knob-controlled iterative condition | high | structural and name-independent |
| T8 | knob-selected alternative path | low | presence is checkable; relative simplicity is not statically proven |
| T9 | knob-selected numeric representation operation | medium | structural; branch-locality remains conservative |
| T10 | knob-controlled shift/mask/divide-scale | medium | detects quantization shape, not numerical quality |
| T11 | knob plus reduced-work control structure | low | energy-awareness is an objective property, not a unique C transform |
| T12 | knob-controlled reduction of indexed/accumulated data | low | overlaps T1/T10 and needs manual provenance review |
| T13 | value recomputed under a knob guard and reused outside it | high | repaired; no prev/last/cached naming rule |
| T14 | knob-selected if/else fidelity paths | medium | path cost is measured later, not inferred here |
| T15 | distributed relaxed consistency | n/a | disabled for all six single-core benchmarks |
| T16 | zero/negligible-element conditional skip plus knob | medium | zero-test structure visible; actual sparsity benefit is measured later |
| T17 | neural-network pruning | n/a | disabled: no neural network in the suite |
| T18 | approximate database join | n/a | disabled: no database workload in the suite |
| T19 | knob-selected arithmetic/table alternative to an expensive call | high | repaired; no approx/surrogate naming rule |
| T20 | knob-controlled input-adaptive branch | low | structural adaptation visible; overlap with T8 remains |

The identifier-name defects found in T3 and T13 also existed in T5, T6 and
T19. All five were removed before the preflight. T8, T11, T12 and T20 remain
broad but are permissive rather than style-sensitive: they can over-accept a
label, but they do not reject a valid implementation because its variables use
unexpected names. Their labels therefore require manual review in the final
plan-quality table.
