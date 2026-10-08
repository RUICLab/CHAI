#!/usr/bin/env python3
"""Export the trained TinyCough model to a C header CHAI can operate on.

Mirrors biomedical/export_pcam_to_c.py: folds BatchNorm into the preceding
Conv2d (a trained BN layer in eval mode is a fixed per-channel affine
transform, absorbed algebraically into the conv's weight and bias -- this
changes nothing numerically, it only removes ops an MCU would never run),
bakes in ONE real test-split example, and records the trained model's own
prediction on it so the C port can be checked for numerical correctness
before anything else is done with it.

Two things differ from the PCam export, both because the input is audio:

1. The baked-in input is the *log-mel spectrogram*, not the raw waveform.
   The STFT/mel front end is deliberately NOT ported to C here. Porting it
   would mean reimplementing torchaudio's MelSpectrogram (window function,
   mel filterbank construction, power/log conventions) with its own
   correctness risk, and it is not needed to test what this export is for:
   CHAI's planning and code generation on the CNN kernel. The front end is
   the more interesting CHAI target (it is where the FFT operators T24/T29
   apply, and where the pilot's own hop-length knob acts) and is the natural
   follow-up, but it is a separate piece of work with separate validation.

2. The spectrogram is normalised by feature_mean/feature_std, which are
   derived from the *training split*, not constants. They are read from the
   checkpoint rather than recomputed, because recomputing them from a
   different set of records would feed the network a different input than it
   was trained on.

Note on hardware: at the trained operating point (hop_multiplier=1, 297
frames) this kernel does not fit the MSP430 profile at all -- the first
conv's output alone is 380160 bytes against a real 4096-byte RAM region.
That is not worked around here. The export is faithful to the trained model;
the hardware verdict is a measurement, not something to engineer away.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from biomedical.coughvid_pilot import (
    TinyCough,
    audio_spectrogram,
    decode_audio,
    extract_archive,
    load_metadata,
    make_transform,
    select_records,
)


def fold_batchnorm(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """Return (weight, bias) for a single conv equivalent to conv then bn in
    eval mode. TinyCough's convs are bias=False, so folding only introduces a
    bias term; it does not need to remove one.
    """
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    bias = bn.bias.detach() - bn.running_mean * scale
    return weight.numpy().astype(np.float32), bias.numpy().astype(np.float32)


def c_float_literal(v: float) -> str:
    """`%.9g` collapses exact integers to a bare digit string with no '.' or
    'e', and a C float suffix on a plain integer constant ("0f") is a lexer
    error, not a float. Weights of exactly 0 do occur, so this bites.
    """
    text = f"{v:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def _nested_initializer(arr: np.ndarray) -> str:
    """Brace-nest to match the array's dimensionality. A flat unbraced list is
    legal C but triggers -Wmissing-braces, and the MSP430 cross-compiler (GCC
    8.2.0) crashes with an internal compiler error while printing that
    warning's hint on a very long line.
    """
    if arr.ndim == 1:
        return "{" + ", ".join(c_float_literal(v) for v in arr) + "}"
    return "{" + ", ".join(_nested_initializer(row) for row in arr) + "}"


def c_array(name: str, arr: np.ndarray) -> str:
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const float {name}{dims} = {_nested_initializer(arr)};"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("reports/biomedical/coughvid_pilot_seed2026.pt"),
    )
    parser.add_argument(
        "--archive", type=Path,
        default=Path("data/biomedical/coughvid/public_dataset.zip"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/biomedical/coughvid/extracted"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_applications/coughvid-iclib/coughvid_weights.h"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument(
        "--test-position", type=int, default=0,
        help="which example of the test split to bake in (0 = first)",
    )
    parser.add_argument(
        "--hop-multiplier", type=int, default=1,
        help="STFT hop multiplier; 1 is the trained operating point. Larger "
             "values produce fewer frames (a smaller input) but the pilot's "
             "own sweep measured accuracy falling to chance by 4 and below "
             "chance at 8, so this is not a free size reduction.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyCough()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_mean = float(checkpoint["feature_mean"])
    feature_std = float(checkpoint["feature_std"])

    layers = list(model.features)
    # Sequential is [Conv,BN,ReLU,Pool, Conv,BN,ReLU,Pool, Conv,BN,ReLU]
    convs_bns = [(layers[i], layers[i + 1]) for i in (0, 4, 8)]
    folded = [fold_batchnorm(conv, bn) for conv, bn in convs_bns]

    classifier_weight = model.classifier.weight.detach().numpy().astype(np.float32)  # (1,24)
    classifier_bias = model.classifier.bias.detach().numpy().astype(np.float32)      # (1,)

    # Reproduce the pilot's own record selection and split, so the baked-in
    # example is genuinely from the test split and not one the model trained on.
    extract_archive(args.archive, args.data)
    records = select_records(
        load_metadata(args.data), args.data, args.seed, args.max_per_class,
    )
    indices = np.arange(len(records))
    _train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=records["label"],
    )

    chosen = int(test_indices[args.test_position])
    row = records.iloc[chosen]
    label = int(row["label"])
    waveform = decode_audio(Path(row["audio_path"]))

    transform = make_transform(args.hop_multiplier)
    spectrogram = audio_spectrogram(waveform, transform)          # (n_mels, frames)
    normalised = (spectrogram - feature_mean) / feature_std
    tensor = normalised.unsqueeze(0).unsqueeze(0)                  # (1,1,n_mels,frames)

    with torch.inference_mode():
        logit = model(tensor).item()
    probability = float(torch.sigmoid(torch.tensor(logit)))

    n_mels, frames = normalised.shape
    chw = normalised.numpy().astype(np.float32)[None, :, :]        # (1,n_mels,frames)

    lines = [
        "/* coughvid_weights.h -- exported from",
        " * reports/biomedical/coughvid_pilot_seed2026.pt by",
        " * biomedical/export_coughvid_to_c.py.",
        " *",
        " * BatchNorm folded into the preceding conv. One real COUGHVID",
        " * test-split example baked in as an already-computed, already-",
        " * normalised log-mel spectrogram (the STFT/mel front end is NOT",
        " * ported to C -- see the export script's docstring), with the",
        " * PyTorch model's own prediction on it recorded so the C port can be",
        " * checked numerically before anything is done with CHAI.",
        " */",
        "#pragma once",
        "",
        f"#define COUGHVID_TEST_POSITION {args.test_position}",
        f"#define COUGHVID_TRUE_LABEL {label}  /* 1 = self-reported covid-19 */",
        f"#define COUGHVID_REFERENCE_LOGIT {logit:.9g}f",
        f"#define COUGHVID_REFERENCE_PROBABILITY {probability:.9g}f",
        f"#define COUGHVID_HOP_MULTIPLIER {args.hop_multiplier}",
        f"#define COUGHVID_N_MELS {n_mels}",
        f"#define COUGHVID_FRAMES {frames}",
        "",
    ]
    for i, (weight, bias) in enumerate(folded):
        lines.append(c_array(f"conv{i+1}_weight", weight))
        lines.append(c_array(f"conv{i+1}_bias", bias))
    lines.append(c_array("classifier_weight", classifier_weight[0]))
    lines.append(c_array("classifier_bias", classifier_bias))
    lines.append("")
    lines.append(c_array("input_spectrogram", chw))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    weight_elements = sum(w.size + b.size for w, b in folded) + classifier_weight.size + classifier_bias.size
    print(f"wrote {args.output}")
    print(f"hop_multiplier {args.hop_multiplier}: spectrogram {n_mels} mels x {frames} frames")
    print(f"test-split position {args.test_position} (record {chosen}), true label {label}")
    print(f"PyTorch reference: logit={logit:.6f} probability={probability:.6f}")
    print(f"weights: {weight_elements} elements ({weight_elements * 4} bytes float32)")
    print(f"input: {n_mels * frames} elements ({n_mels * frames * 4} bytes float32)")


if __name__ == "__main__":
    main()
