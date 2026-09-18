"""Predeclared comparison of cheap forecasters, on honest evaluation windows.

    PYTHONPATH=. python3 -m adaptive.forecast_study

Method list, horizons, and the decision rule below were fixed BEFORE the numbers
were produced. This is deliberately a selection study, not a campaign: nothing
here reruns the 512-cell matrix.

Three methodological rules, each fixing a specific way the earlier analysis
could have fooled itself:

1. NO LOOPING. `simulate()` tiles a 25-52 s recording to fill a 360 s run, so a
   60 s average would be reading repeated copies of its own past and could look
   accurate by exploiting periodicity that does not exist in deployment. Every
   window here comes from the raw recording, used once.

2. NON-OVERLAPPING TARGETS. Sliding a window one second at a time over 300 s of
   solar yields 284 "samples" that share almost all their data. Targets here are
   stepped by the full horizon, so no two share a second.

3. HELD-OUT SELECTION. The averaging window is chosen on one portion of the data
   and scored on another. Picking the best window and reporting its error on the
   same trace measures nothing.

The honest limit, stated before any result: pooled across all five RF
recordings, non-overlapping targets number 11 at H=8s, 4 at H=15s, 1 at H=25s
and 0 at H=40s. RF cannot support conclusions at the horizons the controller
actually uses. Only the 300 s solar recording has even a modest sample.
"""

from __future__ import annotations

import numpy as np

from adaptive.power_model import PowerParams, load_trace

RF_TRACES = ["RF_1", "RF_2", "RF_6", "RF_7", "RF_9"]
HORIZONS = [8, 15, 25, 40]
MIN_CONTEXT_S = 15          # require this much history before scoring a window


# ---- the predeclared method list -------------------------------------------

def m_persistence(hist):
    return float(hist[-1])


def _ewma(hist, alpha):
    w = (1.0 - alpha) ** np.arange(len(hist))[::-1]
    return float((w * hist).sum() / w.sum())


def m_ewma_fast(hist):
    """The controller's current forecaster. alpha=0.3 gives an effective memory
    of only a few seconds despite 64 s of available history."""
    return _ewma(hist, 0.3)


def m_ewma_slow_05(hist):
    return _ewma(hist, 0.05)


def m_ewma_slow_02(hist):
    return _ewma(hist, 0.02)


def _mean_window(hist, seconds):
    return float(hist[-seconds:].mean()) if len(hist) else 0.0


def m_mean15(hist):
    return _mean_window(hist, 15)


def m_mean30(hist):
    return _mean_window(hist, 30)


def m_mean60(hist):
    return _mean_window(hist, 60)


METHODS = {
    "persistence": m_persistence,
    "ewma a=0.3 (current)": m_ewma_fast,
    "ewma a=0.05": m_ewma_slow_05,
    "ewma a=0.02": m_ewma_slow_02,
    "mean 15s": m_mean15,
    "mean 30s": m_mean30,
    "mean 60s": m_mean60,
}


# ---- evaluation -------------------------------------------------------------

def to_bins(trace, params):
    """One-second bins of harvest current, from the RAW recording (no looping)."""
    spb = 1000                                  # trace is 1 sample per ms
    n = (len(trace) // spb) * spb
    return trace[:n].reshape(-1, spb).mean(axis=1) / params.load_resistance


def windows(binned, horizon, start=0, stop=None):
    """Non-overlapping (start_index, target_mean) pairs.

    Targets step by the full horizon, so no two targets share a second.
    """
    stop = len(binned) if stop is None else stop
    i = max(start, MIN_CONTEXT_S)
    out = []
    while i + horizon <= stop:
        target = float(binned[i:i + horizon].mean())
        if target > 0:
            out.append((i, target))
        i += horizon
    return out


def score(binned, horizon, start=0, stop=None):
    """Relative absolute error per method over the non-overlapping windows."""
    errs = {name: [] for name in METHODS}
    for i, target in windows(binned, horizon, start, stop):
        hist = binned[:i]                       # causal: only what has been seen
        for name, fn in METHODS.items():
            pred = fn(hist)
            errs[name].append(abs(pred - target) / target)
    return {k: (np.mean(v) if v else float("nan")) for k, v in errs.items()}, \
           len(next(iter(errs.values())))


def main():
    params = PowerParams()
    print(__doc__.split("The honest limit")[0].strip())
    print("=" * 100)

    solar = to_bins(load_trace("traces/Solar_Indoor_Moving.csv"), params)
    rf = {n: to_bins(load_trace(f"traces/{n}.csv"), params) for n in RF_TRACES}

    # ---------------------------------------------------------------- sample sizes
    print("\nINDEPENDENT (non-overlapping) TARGETS AVAILABLE")
    print(f"  {'horizon':<10}{'solar':>8}{'RF pooled':>12}   per-RF-trace")
    for H in HORIZONS:
        per = {n: len(windows(b, H)) for n, b in rf.items()}
        print(f"  {str(H) + 's':<10}{len(windows(solar, H)):>8}{sum(per.values()):>12}   "
              + "  ".join(f"{n}:{c}" for n, c in per.items()))
    print("\n  RF is underpowered at every horizon the controller uses. Anything")
    print("  below ~10 windows is reported but must not be used to choose a method.")

    # ---------------------------------------------------------------- solar, held out
    half = len(solar) // 2
    print("\n" + "=" * 100)
    print(f"SOLAR, HELD OUT IN TIME: select on 0-{half}s, evaluate on {half}-{len(solar)}s")
    print("  (a genuine session split -- the evaluation half is never seen during selection)")
    for H in HORIZONS:
        sel, n_sel = score(solar, H, 0, half)
        evl, n_evl = score(solar, H, half, len(solar))
        if not n_sel or not n_evl:
            print(f"\n  H={H}s: too few windows (select {n_sel}, evaluate {n_evl})")
            continue
        best = min((k for k in sel if not np.isnan(sel[k])), key=lambda k: sel[k])
        print(f"\n  H={H}s   select n={n_sel}   evaluate n={n_evl}   "
              f"CHOSEN ON SELECTION HALF: {best}")
        print(f"    {'method':<24}{'select':>10}{'evaluate':>11}")
        for name in METHODS:
            mark = "  <-- chosen" if name == best else ""
            print(f"    {name:<24}{sel[name] * 100:9.1f}%{evl[name] * 100:10.1f}%{mark}")

    # ---------------------------------------------------------------- RF, leave one out
    print("\n" + "=" * 100)
    print("RF, LEAVE-ONE-RECORDING-OUT  (reported for completeness; underpowered)")
    for H in HORIZONS:
        rows = []
        for held in RF_TRACES:
            others = [n for n in RF_TRACES if n != held]
            pooled = {name: [] for name in METHODS}
            n_train = 0
            for n in others:
                s, c = score(rf[n], H)
                n_train += c
                for k, v in s.items():
                    if not np.isnan(v):
                        pooled[k].append(v)
            held_scores, n_held = score(rf[held], H)
            if not n_held or not n_train:
                continue
            valid = {k: np.mean(v) for k, v in pooled.items() if v}
            if not valid:
                continue
            best = min(valid, key=lambda k: valid[k])
            rows.append((held, n_train, n_held, best, held_scores[best]))
        if not rows:
            print(f"\n  H={H}s: no usable leave-one-out split (not enough windows)")
            continue
        print(f"\n  H={H}s")
        print(f"    {'held out':<10}{'n_train':>9}{'n_held':>8}  "
              f"{'chosen on others':<24}{'error on held-out':>18}")
        for held, n_tr, n_he, best, err in rows:
            print(f"    {held:<10}{n_tr:>9}{n_he:>8}  {best:<24}"
                  f"{err * 100:17.1f}%")

    print("\n" + "=" * 100)
    print("READ THIS BEFORE USING THE NUMBERS")
    print("  * A method winning here has lower FORECAST error. That is not the same")
    print("    as improving the controller. False-deferral rate, false-run rate and")
    print("    per-offered efficiency decide whether it belongs in the policy.")
    print("  * RF rows are shown for completeness and are underpowered; do not")
    print("    select a method on them.")
    print("  * Nothing here licenses a claim that harvested energy is unpredictable.")
    print("    These traces show little LINEAR, HISTORY-ONLY predictability at these")
    print("    horizons. Nonlinear structure, capacitor state and external signals")
    print("    are untested.")


if __name__ == "__main__":
    main()
