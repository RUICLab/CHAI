"""Compare forecasters on the question the selector actually asks.

Two metrics per window:

* ``mae``     - mean absolute error of the per-bin harvest current [A].
* ``energy``  - relative error of the *total* current summed over the horizon,
                which is what decides how much work the next cycle can afford.
                This is the metric that matters; per-bin shape does not.

Chronos-Bolt is probabilistic, so we also record ``q10``: the 10th-percentile
energy forecast. A selector that plans against q10 is being deliberately
pessimistic, which is the right bias when browning out mid-task wastes all the
energy already spent on it.
"""

from __future__ import annotations

import argparse

import numpy as np

from adaptive.baselines import BASELINES
from adaptive.forecast_data import horizon_bins, load_all, windows


def score(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - target)))
    actual = float(target.sum())
    rel = abs(float(pred.sum()) - actual) / actual if actual > 0 else float("nan")
    return mae, rel


def collect_windows(traces, context, horizon, stride):
    data = {}
    for trace in traces:
        pairs = list(windows(trace, context, horizon, stride))
        if pairs:
            data[trace.name] = (np.stack([c for c, _ in pairs]),
                                np.stack([t for _, t in pairs]))
    return data


def run_baselines(data, horizon):
    rows = []
    for name, fn in BASELINES.items():
        for trace_name, (ctx, tgt) in data.items():
            scores = [score(fn(ctx[i], horizon), tgt[i]) for i in range(len(ctx))]
            mae = float(np.mean([s[0] for s in scores]))
            rel = float(np.nanmean([s[1] for s in scores]))
            rows.append((name, trace_name, mae, rel, len(ctx)))
    return rows


def run_chronos(data, horizon, model="amazon/chronos-bolt-small", batch=32, device="cpu"):
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model, device_map=device,
                                               torch_dtype=torch.float32)
    rows = []
    for trace_name, (ctx, tgt) in data.items():
        maes, rels, q10s = [], [], []
        for i in range(0, len(ctx), batch):
            chunk = ctx[i:i + batch]
            # Chronos works on unitless series; scale to microamps for stability.
            tensor = torch.tensor(chunk * 1e6, dtype=torch.float32)
            quantiles, _ = pipe.predict_quantiles(
                tensor, prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9])
            q = quantiles.numpy() / 1e6                    # [batch, horizon, 3]
            for j in range(len(chunk)):
                target = tgt[i + j]
                mae, rel = score(q[j, :, 1], target)
                maes.append(mae)
                rels.append(rel)
                actual = target.sum()
                if actual > 0:
                    q10s.append(float(q[j, :, 0].sum() / actual))
        rows.append(("chronos-bolt", trace_name, float(np.mean(maes)),
                     float(np.nanmean(rels)), len(ctx)))
        print(f"    {trace_name}: median q10/actual energy ratio = {np.median(q10s):.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--skip-chronos", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    traces = load_all(bin_ms=args.bin_ms)
    horizon = horizon_bins(args.bin_ms)
    data = collect_windows(traces, args.context, horizon, args.stride)
    total = sum(len(c) for c, _ in data.values())
    print(f"horizon {horizon} bins ({horizon * args.bin_ms:.0f} ms), "
          f"context {args.context} bins, {total} windows over {len(data)} traces\n")

    rows = run_baselines(data, horizon)
    if not args.skip_chronos:
        print("  running chronos-bolt ...")
        rows += run_chronos(data, horizon, device=args.device)

    print(f"\n{'method':<16}{'trace':<8}{'MAE [uA]':>10}{'energy err':>12}{'windows':>9}")
    print("-" * 56)
    for name, trace_name, mae, rel, n in rows:
        print(f"{name:<16}{trace_name:<8}{mae * 1e6:>10.2f}{rel * 100:>11.1f}%{n:>9}")
    print(f"\n{'method':<16}{'MEAN energy err':>18}")
    print("-" * 36)
    agg = {}
    for name, _t, _m, rel, n in rows:
        agg.setdefault(name, []).append(rel)
    for name, vals in sorted(agg.items(), key=lambda kv: np.mean(kv[1])):
        print(f"{name:<16}{np.mean(vals) * 100:>17.1f}%")


if __name__ == "__main__":
    main()
