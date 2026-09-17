# WITHDRAWN — do not cite

Everything in this directory is retained for provenance only. The headline
result does not survive review.

| file | status |
|---|---|
| `energy_adaptive_system.pdf` | **withdrawn** — central claim unsupported |
| `results_sobel.json`, `results_fft.json` | **withdrawn** — 480 runs produced by a simulator that does not match the energy model |
| `flow.png` | diagram only; the architecture it shows is still accurate |

## Why

**1. The simulator does not use the validated power model.**
`adaptive/selector.py` reimplemented the power loop instead of calling
`adaptive/power_model.py`, and the copy omits `KEEP_ALIVE`. The simulator
therefore powers the device off at 3.4 V, while `build_levels` prices each task
assuming discharge to 2.283 V — about a **10x mismatch** between the energy
delivered and the energy charged. Every run in the two JSON files inherits it.

**2. The comparison used the wrong baseline.**
The report calls `fixed-L1` "the best fixed level". It is not. On measured RF at
a 40 s deadline:

    fixed-L0    100%   error 0.0000   spill 62.82 mJ
    reactive    100%   error 0.0079   spill 70.30 mJ

`fixed-L0` dominates reactive on both axes and equals the oracle exactly. The
same holds on measured solar at 25 s. At loose deadlines the optimal policy is
simply "always exact" — which is precisely where adaptation was claimed to win.
The 2.4x-error / half-waste claim is an artifact of the chosen comparator.

**3. Mid-task level switching assumes resumability that was never established.**
The controller changes stride or term count partway through a task, keeps the
work already done, and records the final level's error. For a Sobel convolution
that produces a hybrid image belonging to neither level. The module's own
docstring states the device cannot switch mid-task.

**4. Sobel L0 is not the declared identity.**
`levels.py` selects the cheapest configuration with minimum observed error,
which picked `[5,1]` rather than the declared identity `[1,1]`. Its zero error
was measured on a single image and is image-dependent, yet the report labels it
"exact" and claims a 27.4% saving at the exact level.

## What remains usable

`adaptive/power_model.py` itself is validated against Fused's own log and is not
in question. The level *measurements* from Fused (`experiment_results/
sobel_t22_wide`, `fft_taylor`) are also unaffected — the defects are in the
selector simulation and in how levels were labelled and compared.
