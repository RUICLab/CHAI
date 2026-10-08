#!/usr/bin/env python3
"""Export the trained TinyCoughDetect model to a C header CHAI can operate on.

Folds BatchNorm into the preceding Conv2d (a trained BN layer in eval mode is
a fixed per-channel affine transform, absorbed algebraically into the conv's
weight and bias -- numerically identical, it only removes ops an MCU would
never run), bakes in ONE real test-split example as an already-computed,
already-normalised log-mel spectrogram, and records the model's own prediction
so the C port can be checked numerically before anything else is done.

The STFT/mel front end is deliberately NOT ported to C. Reimplementing
torchaudio's MelSpectrogram (window function, filterbank construction,
power/log conventions) carries its own correctness risk and is not needed to
test what this export is for: CHAI's planning and code generation on the CNN
kernel. It is the natural follow-up, with its own validation.

Label caveat carried from the pilot: the model is distilled from the dataset's
published XGBoost cough detector, so any accuracy claim is "agrees with the
published detector", never "detects coughs correctly".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from biomedical.coughvid_detect_pilot import (
    DEFAULT_CHANNELS,
    TinyCoughDetect,
    decode_audio,
    make_transform,
    select_records,
    spectrogram_of,
)
from biomedical.coughvid_pilot import extract_archive, load_metadata


def fold_batchnorm(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    bias = bn.bias.detach() - bn.running_mean * scale
    return weight.numpy().astype(np.float32), bias.numpy().astype(np.float32)


def c_float_literal(v: float) -> str:
    """A C float suffix on a bare integer constant ("0f") is a lexer error, and
    `%.9g` collapses exact integers to a bare digit string. Weights of exactly
    0 do occur, so this matters."""
    text = f"{v:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def _nested_initializer(arr: np.ndarray) -> str:
    """Brace-nest to match dimensionality. A flat unbraced list is legal C but
    trips -Wmissing-braces, and the MSP430 cross-compiler (GCC 8.2.0) crashes
    with an internal compiler error printing that warning on a long line."""
    if arr.ndim == 1:
        return "{" + ", ".join(c_float_literal(v) for v in arr) + "}"
    return "{" + ", ".join(_nested_initializer(row) for row in arr) + "}"


def c_array(name: str, arr: np.ndarray) -> str:
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const float {name}{dims} = {_nested_initializer(arr)};"


#: Q-format fractional bits for the integer export. 2^11 = 2048 keeps weights
#: and normalised activations (both order 1-10) inside int16 while leaving
#: headroom, and an int32 accumulator cannot overflow for the 9*in_ch terms
#: accumulated here.
FP_BITS = 11
FP_SCALE = 1 << FP_BITS


def c_int16_array(name: str, arr: np.ndarray) -> str:
    """Emit a Q11 int16 array.

    The point of exporting integers is NOT smaller storage (though it halves
    it). It is that a no-FPU MSP430 turns every float multiply-accumulate into
    a call to __mspabi_mpyf (592 bytes of soft-float code) plus __mspabi_addf
    with __unpack_f/__pack_f behind it -- more library code per MAC than the
    whole convolution function. Keeping the operands as integers in memory is
    what removes those calls from the inner loop; quantising float operands
    inside the loop would not, because each conversion is itself a soft-float
    call.
    """
    q = np.clip(np.rint(arr.astype(np.float64) * FP_SCALE), -32768, 32767)
    q = q.astype(np.int16)

    def nest(a):
        if a.ndim == 1:
            return "{" + ", ".join(str(int(v)) for v in a) + "}"
        return "{" + ", ".join(nest(r) for r in a) + "}"

    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const int16_t {name}{dims} = {nest(q)};"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("reports/biomedical/coughvid_detect_seed2026.pt"))
    parser.add_argument("--archive", type=Path,
                        default=Path("data/biomedical/coughvid/public_dataset.zip"))
    parser.add_argument("--data", type=Path,
                        default=Path("data/biomedical/coughvid/extracted"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark_applications/coughvid-iclib/coughvid_weights.h"))
    parser.add_argument("--test-position", type=int, default=0)
    parser.add_argument(
        "--eval-examples", type=int, default=0,
        help="bake in this many spectrograms as the CALIBRATION set. The error "
             "metric scores decision disagreement across these during search, "
             "so they are the inputs the optimiser is allowed to see. The "
             "metric needs more than one: a single input cannot separate a "
             "uniform bias (fatal, the threshold is a fixed 0) from harmless "
             "noise. 0 keeps the header single-input.",
    )
    parser.add_argument(
        "--test-examples", type=int, default=0,
        help="dump this many further spectrograms to an .npz as an UNTOUCHED "
             "test set, disjoint from the calibration set. Deliberately not "
             "compiled into the header, so the search cannot see them and "
             "final accuracy is not selection-biased by the metric that chose "
             "the configuration.",
    )
    parser.add_argument(
        "--eval-output", type=Path,
        default=Path("reports/biomedical/coughvid_heldout_test.npz"),
        help="where the untouched test set is written",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # The architecture and the feature settings travel with the weights: loading
    # them into a different shape, or normalising with recomputed constants,
    # produces plausible numbers from the wrong network.
    channels = tuple(checkpoint.get("channels") or DEFAULT_CHANNELS)
    n_mels = int(checkpoint["n_mels"])
    frames = int(checkpoint["frames"])
    feature_mean = float(checkpoint["feature_mean"])
    feature_std = float(checkpoint["feature_std"])
    seed = int(checkpoint.get("seed", 2026))

    model = TinyCoughDetect(channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    layers = list(model.features)
    convs_bns = [(layers[i], layers[i + 1]) for i in range(0, 3 * len(channels), 3)]
    folded = [fold_batchnorm(conv, bn) for conv, bn in convs_bns]
    classifier_weight = model.classifier.weight.detach().numpy().astype(np.float32)
    classifier_bias = model.classifier.bias.detach().numpy().astype(np.float32)

    # Reproduce the pilot's own selection and split so the baked-in example is
    # genuinely from the test split, with the same seed and thresholds.
    extract_archive(args.archive, args.data)
    records = select_records(load_metadata(args.data), args.data, seed, 3000)
    labels = records["label"].to_numpy(dtype=np.int64)
    indices = np.arange(len(records))
    _train, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )

    chosen = int(test_idx[args.test_position])
    row = records.iloc[chosen]
    label = int(row["label"])
    published_score = float(row["cough_detected"])

    transform = make_transform(n_mels, frames)
    waveform = decode_audio(Path(row["audio_path"]))
    spectrogram = spectrogram_of(waveform, transform)
    normalised = (spectrogram - feature_mean) / feature_std
    tensor = normalised.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        logit = model(tensor).item()
    probability = float(torch.sigmoid(torch.tensor(logit)))

    actual_mels, actual_frames = normalised.shape
    chw = normalised.numpy().astype(np.float32)[None, :, :]

    lines = [
        "/* coughvid_weights.h -- exported from",
        " * reports/biomedical/coughvid_detect_seed2026.pt by",
        " * biomedical/export_coughdetect_to_c.py.",
        " *",
        " * Task: cough vs not-cough, DISTILLED from the COUGHVID dataset's own",
        " * published XGBoost cough detector. The label source is that",
        " * classifier's output, not human ground truth, so the defensible claim",
        " * is agreement with the published detector -- never that this detects",
        " * coughs correctly.",
        " *",
        " * BatchNorm folded into the preceding conv. One real test-split example",
        " * baked in as an already-computed, already-normalised log-mel",
        " * spectrogram, with the model's own prediction recorded so the C port",
        " * can be checked numerically before anything is done with CHAI.",
        " */",
        "#pragma once",
        "",
        f"#define COUGHVID_TEST_POSITION {args.test_position}",
        f"#define COUGHVID_TRUE_LABEL {label}  /* 1 = cough */",
        f"#define COUGHVID_PUBLISHED_SCORE {published_score:.9g}f  /* cough_detected */",
        f"#define COUGHVID_REFERENCE_LOGIT {logit:.9g}f",
        f"#define COUGHVID_REFERENCE_PROBABILITY {probability:.9g}f",
        f"#define COUGHVID_N_MELS {actual_mels}",
        f"#define COUGHVID_FRAMES {actual_frames}",
        f"#define COUGHVID_CHANNELS_C1 {channels[0]}",
        f"#define COUGHVID_CHANNELS_C2 {channels[1]}",
        f"#define COUGHVID_CHANNELS_C3 {channels[2]}",
        "",
    ]
    lines.append("#ifndef COUGHVID_CONV1_FIXED_POINT")
    lines.append(c_array("conv1_weight", folded[0][0]))
    lines.append("#endif")
    for i, (weight, bias) in enumerate(folded):
        if i > 0:
            lines.append(c_array(f"conv{i+1}_weight", weight))
        lines.append(c_array(f"conv{i+1}_bias", bias))
    lines.append(c_array("classifier_weight", classifier_weight[0]))
    lines.append(c_array("classifier_bias", classifier_bias))
    lines.append("")
    lines.append("#ifndef COUGHVID_CONV1_FIXED_POINT")
    lines.append(c_array("input_spectrogram", chw))
    lines.append("#endif")
    lines.append("")
    lines.append("/* Q%d fixed-point copies, for the integer convolution path. */" % FP_BITS)
    lines.append(f"#define PCAM_FP_BITS {FP_BITS}")
    lines.append(f"#define PCAM_FP_SCALE {FP_SCALE}")
    lines.append("#ifdef COUGHVID_CONV1_FIXED_POINT")
    lines.append(c_int16_array("conv1_weight_q", folded[0][0]))
    lines.append(c_int16_array("input_spectrogram_q", chw))
    lines.append("#endif")

    if args.eval_examples > 0:
        # Distinct from the single baked-in input: these exist so the error
        # metric can count changed DECISIONS rather than infer damage from one
        # logit. They are compiled only under LOCAL_RUN, so they never consume
        # device FRAM.
        wanted = test_idx[: args.eval_examples]
        eval_specs, eval_labels, eval_logits = [], [], []
        for position in wanted:
            eval_row = records.iloc[int(position)]
            try:
                eval_wave = decode_audio(Path(eval_row["audio_path"]))
            except Exception:
                continue  # crowdsourced recordings include unreadable files
            eval_spec = (spectrogram_of(eval_wave, transform) - feature_mean) / feature_std
            with torch.inference_mode():
                eval_logit = model(eval_spec.unsqueeze(0).unsqueeze(0)).item()
            eval_specs.append(eval_spec.numpy().astype(np.float32))
            eval_labels.append(int(eval_row["label"]))
            eval_logits.append(eval_logit)

        stacked = np.stack(eval_specs)[:, None, :, :]
        lines.append("")
        lines.append("#ifdef LOCAL_RUN")
        lines.append(f"#define COUGHVID_EVAL_COUNT {len(eval_specs)}")
        lines.append(c_array("input_spectrogram_set", stacked))
        # See the pcam exporter: the integer path reads a separate quantised
        # array, so the calibration loop needs a quantised set to switch to.
        lines.append("#ifdef COUGHVID_CONV1_FIXED_POINT")
        lines.append(c_int16_array("input_spectrogram_q_set", stacked))
        lines.append("#endif")
        lines.append(
            "static const int coughvid_eval_labels[] = {"
            + ", ".join(str(v) for v in eval_labels) + "};"
        )
        lines.append(
            "static const float coughvid_eval_reference[] = {"
            + ", ".join(c_float_literal(v) for v in eval_logits) + "};"
        )
        lines.append("#endif")
        print(f"calibration set: {len(eval_specs)} inputs, "
              f"{sum(eval_labels)} positive (test-split positions "
              f"0..{args.eval_examples - 1})")

    if args.test_examples > 0:
        held_positions = test_idx[
            args.eval_examples: args.eval_examples + args.test_examples
        ]
        held_specs, held_labels, held_logits = [], [], []
        for position in held_positions:
            held_row = records.iloc[int(position)]
            try:
                held_wave = decode_audio(Path(held_row["audio_path"]))
            except Exception:
                continue
            held_spec = (
                spectrogram_of(held_wave, transform) - feature_mean
            ) / feature_std
            with torch.inference_mode():
                held_logits.append(model(held_spec.unsqueeze(0).unsqueeze(0)).item())
            held_specs.append(held_spec.numpy().astype(np.float32))
            held_labels.append(int(held_row["label"]))
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.eval_output,
            spectrograms=np.stack(held_specs),
            labels=np.asarray(held_labels),
            reference_logits=np.asarray(held_logits, dtype=np.float32),
            positions=np.asarray(held_positions[: len(held_specs)]),
        )
        print(f"held-out test set: {len(held_specs)} inputs, "
              f"{sum(held_labels)} positive -> {args.eval_output}")
        print("  NOT compiled into the header: the search cannot see these")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    elements = sum(w.size + b.size for w, b in folded) + classifier_weight.size + classifier_bias.size
    print(f"wrote {args.output}")
    print(f"channels {channels}, spectrogram {actual_mels} mels x {actual_frames} frames")
    print(f"test-split position {args.test_position} (record {chosen}), label {label}, "
          f"published score {published_score:.4f}")
    print(f"PyTorch reference: logit={logit:.6f} probability={probability:.6f}")
    print(f"weights: {elements} elements ({elements * 4} B)  "
          f"input: {actual_mels * actual_frames} elements ({actual_mels * actual_frames * 4} B)")


if __name__ == "__main__":
    main()
