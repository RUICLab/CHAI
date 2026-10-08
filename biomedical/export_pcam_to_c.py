#!/usr/bin/env python3
"""Export the trained TinyPcam model to a C header CHAI can operate on.

Folds BatchNorm into the preceding Conv2d (standard practice: nobody runs
live batch statistics on an MCU; a trained BN layer in eval mode is just a
fixed per-channel affine transform, so it is algebraically absorbed into the
conv's weight and bias at export time -- this changes nothing numerically,
it only removes ops an embedded target would never actually execute).

Defaults to the FULL 64x64 resolution in float, matching the trained model
exactly, rather than pre-quantizing or pre-shrinking it -- at that size there
is nothing left for CHAI's hardware-aware gating to discover, and a real
MSP430 cross-build confirms it cannot fit regardless: the input tensor alone
(49152 bytes) is 4.8x the declared 10 KB RAM budget before any activation
buffer is even considered.

``--resolution`` re-runs the SAME trained weights (no retraining) at a
smaller spatial size, using pcam_pilot.py's own resize_batch(): the model's
global average pool makes it architecturally resolution-agnostic, and
pcam_pilot.py's own resolution-robustness sweep already measured that
resolution 32 is statistically indistinguishable in accuracy from 64
(balanced accuracy 0.7677 vs 0.7654, overlapping 95% CIs) while cutting
estimated MACs by 75%. This is the same downsampling pcam_pilot.py's
evaluate() already performs when swept over resolution in {64,48,32,24,16};
this script just bakes one resolution's weights+image into C instead of
measuring accuracy in Python.

Picks ONE real PCam test image (matching how every other CHAI benchmark in
this repo bakes in a single representative input for Fused's power-cycle
measurement, e.g. benchmark_applications/sobel-iclib/image_data.h) and
records the trained model's real prediction on it, so the C port can be
checked for numerical correctness before anything else is done with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from biomedical.pcam_pilot import DEFAULT_CHANNELS, TinyPcam, resize_batch


def fold_batchnorm(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """Return (weight, bias) for a single conv equivalent to conv followed by
    bn in eval mode. bn(conv(x)) = gamma * (conv(x) - mean) / sqrt(var+eps) + beta
                                  = (gamma/sqrt(var+eps)) * conv(x) + (beta - gamma*mean/sqrt(var+eps))
    conv has no bias (TinyPcam's convs are bias=False), so folding only
    introduces a bias term, it does not need to remove one.
    """
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    bias = bn.bias.detach() - bn.running_mean * scale
    return weight.numpy().astype(np.float32), bias.numpy().astype(np.float32)


def c_float_literal(v: float) -> str:
    """`%.9g` collapses exact integers (0, 1, -2, ...) to a bare digit string
    with no '.' or 'e', and a C floating-suffix on a plain integer-constant
    ("0f", "1f") is a lexer error ("invalid digit 'f' in decimal constant"),
    not a float -- unlike Python, C requires a decimal point or exponent
    before a float suffix is legal. Weight=0 and pixel=0.0/1.0 both occur in
    real data, so this bites immediately, not on some rare edge case.
    """
    text = f"{v:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def _nested_initializer(arr: np.ndarray) -> str:
    """Brace-nest an initializer to match arr's own dimensionality.

    A flat, unbraced list (``{v0, v1, ..., v215}``) is legal C for a
    multi-dimensional array -- it fills in row-major order -- but it triggers
    ``-Wmissing-braces``, and the MSP430 cross-compiler (GCC 8.2.0) has a
    diagnostics bug that crashes as an internal compiler error while trying to
    print that warning's hint on a single very long line. Properly nested
    braces are unambiguous C and never trigger the warning in the first place.
    """
    if arr.ndim == 1:
        return "{" + ", ".join(c_float_literal(v) for v in arr) + "}"
    return "{" + ", ".join(_nested_initializer(row) for row in arr) + "}"


def c_array(name: str, arr: np.ndarray) -> str:
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const float {name}{dims} = {_nested_initializer(arr)};"


#: Q-format fractional bits for the integer export. Weights and normalised
#: activations here are order 1-10, so 2^11 keeps them inside int16 with
#: headroom and an int32 accumulator cannot overflow over 9*in_ch terms.
FP_BITS = 11
FP_SCALE = 1 << FP_BITS


def c_int16_array(name: str, arr: np.ndarray) -> str:
    """Emit a Q11 int16 array.

    Halving storage is a side benefit. The point is that a no-FPU MSP430 turns
    every float multiply-accumulate into a call to __mspabi_mpyf (592 bytes of
    soft-float code) plus __mspabi_addf with __unpack_f/__pack_f behind it --
    more library code per MAC than the whole convolution function. Keeping the
    operands as integers IN MEMORY is what removes those calls from the inner
    loop; quantising float operands inside the loop would not, because each
    conversion is itself a soft-float call.

    The halved storage matters separately: it is what lets the full-width
    8-16-32 model fit the FRAM budget at all (13666 B against 13754 B, versus
    27332 B as float32).
    """
    q = np.clip(np.rint(arr.astype(np.float64) * FP_SCALE), -32768, 32767).astype(np.int16)

    def nest(a):
        if a.ndim == 1:
            return "{" + ", ".join(str(int(v)) for v in a) + "}"
        return "{" + ", ".join(nest(r) for r in a) + "}"

    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const int16_t {name}{dims} = {nest(q)};"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/biomedical/pcam_pilot_seed2026.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/biomedical/pcam"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_applications/pcam-iclib/pcam_weights.h"))
    parser.add_argument("--test-index", type=int, default=0, help="which test-set image to bake in")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="spatial size to resize the (already centre-cropped) image to. "
             "Defaults to the resolution recorded in the checkpoint. "
             "via pcam_pilot.resize_batch -- the same downsampling its own "
             "resolution-robustness sweep uses. The trained weights are "
             "unchanged; only the input's spatial size shrinks.",
    )
    parser.add_argument(
        "--calibration-examples", type=int, default=0,
        help="bake in this many images as the CALIBRATION set. The error "
             "metric scores decision disagreement across these during search, "
             "so they are the inputs the optimiser is allowed to see. 0 keeps "
             "the header single-input, which cannot support an accuracy claim: "
             "a single input cannot separate a uniform bias (fatal, since the "
             "decision threshold is a fixed 0) from harmless noise.",
    )
    parser.add_argument(
        "--test-examples", type=int, default=0,
        help="dump this many further images to an .npz as an UNTOUCHED test "
             "set, disjoint from the calibration set. Deliberately not "
             "compiled into the header: the search must not be able to see "
             "them, so final accuracy is not selection-biased by the metric "
             "that chose the configuration.",
    )
    parser.add_argument(
        "--eval-output", type=Path,
        default=Path("reports/biomedical/pcam_heldout_test.npz"),
        help="where the untouched test set is written",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # The architecture travels with the weights: a narrower model trained to
    # fit the device will not load into the default widths, and loading it
    # into the wrong shape is the kind of failure that produces plausible
    # numbers from the wrong network.
    channels = tuple(checkpoint.get("channels") or DEFAULT_CHANNELS)
    model = TinyPcam(channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if args.resolution is None:
        args.resolution = int(checkpoint.get("image_size") or 64)

    layers = list(model.features)
    # Sequential is [Conv,BN,ReLU] repeated once per channel width.
    convs_bns = [(layers[i], layers[i + 1]) for i in range(0, 3 * len(channels), 3)]
    folded = [fold_batchnorm(conv, bn) for conv, bn in convs_bns]

    classifier_weight = model.classifier.weight.detach().numpy().astype(np.float32)  # (1, 32)
    classifier_bias = model.classifier.bias.detach().numpy().astype(np.float32)      # (1,)

    # One real PCam test image, decompressed the same way pcam_pilot.py does.
    from biomedical.pcam_pilot import H5Pcam, decompress_if_needed  # noqa: E402

    test_set = H5Pcam(args.data, "test")
    test_set._open()
    image = np.asarray(test_set._images[args.test_index])  # (96, 96, 3) uint8
    label = int(np.asarray(test_set._labels[args.test_index]).reshape(-1)[0])

    # Match H5Pcam.__getitem__ then resize_batch exactly: normalise to [0,1]
    # first, then centre-crop to 64x64 and bilinear-resize to --resolution --
    # the identical path pcam_pilot.py's own resolution-robustness sweep uses,
    # so this bakes in one point on a curve that was already measured, not a
    # new, unvalidated transformation.
    raw = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)  # (1,3,96,96)
    tensor = resize_batch(raw, args.resolution)  # (1,3,resolution,resolution)

    with torch.inference_mode():
        logit = model(tensor).item()
    probability = float(torch.sigmoid(torch.tensor(logit)))
    cropped = tensor[0].permute(1, 2, 0).numpy()  # (resolution,resolution,3) in [0,1]

    lines = [
        "/* pcam_weights.h -- exported from reports/biomedical/pcam_pilot_seed2026.json",
        " * BatchNorm folded into the preceding conv (standard for deployment: no MCU",
        " * runs live BN statistics). One real PCam test image baked in as input, with",
        " * the PyTorch model's own prediction on it recorded for numerical validation",
        " * of the C port BEFORE anything is done with CHAI or Fused.",
        " */",
        "#pragma once",
        "",
        f"#define PCAM_TEST_INDEX {args.test_index}",
        f"#define PCAM_TRUE_LABEL {label}  /* 1 = tumor present */",
        f"#define PCAM_REFERENCE_LOGIT {logit:.9g}f",
        f"#define PCAM_REFERENCE_PROBABILITY {probability:.9g}f",
        f"#define PCAM_INPUT_RESOLUTION {args.resolution}",
        f"#define PCAM_CHANNELS_C1 {channels[0]}",
        f"#define PCAM_CHANNELS_C2 {channels[1]}",
        f"#define PCAM_CHANNELS_C3 {channels[2]}",
        "",
    ]
    # Only conv1's float weights are replaced by the integer path; conv2 and
    # conv3 still run in float, so their float weights are unconditional.
    # Carrying BOTH formats for every layer overflowed FRAM by 714 bytes.
    lines.append("#ifndef PCAM_CONV1_FIXED_POINT")
    lines.append(c_array("conv1_weight", folded[0][0]))
    lines.append("#endif")
    for i, (weight, bias) in enumerate(folded):
        if i > 0:
            lines.append(c_array(f"conv{i+1}_weight", weight))
        lines.append(c_array(f"conv{i+1}_bias", bias))
    lines.append(c_array("classifier_weight", classifier_weight[0]))
    lines.append(c_array("classifier_bias", classifier_bias))
    lines.append("")
    # Input image, channel-first (matches how the conv loops will index it),
    # normalised to [0,1] exactly as the PyTorch dataset does.
    chw = cropped.transpose(2, 0, 1)
    lines.append("#ifndef PCAM_CONV1_FIXED_POINT")
    lines.append(c_array("input_image", chw))
    lines.append("#endif")
    lines.append("")
    lines.append("/* Q%d fixed-point copies for the integer conv1 path." % FP_BITS)
    lines.append(" * Guarded to be MUTUALLY EXCLUSIVE with the float copies")
    lines.append(" * above: a device ships one weight format, and carrying")
    lines.append(" * both overflows FRAM. Emitting the integer arrays")
    lines.append(" * unconditionally broke the baseline build by 956 bytes and")
    lines.append(" * made the two builds incomparable. */")
    lines.append(f"#define PCAM_FP_BITS {FP_BITS}")
    lines.append(f"#define PCAM_FP_SCALE {FP_SCALE}")
    lines.append("#ifdef PCAM_CONV1_FIXED_POINT")
    lines.append(c_int16_array("conv1_weight_q", folded[0][0]))
    lines.append(c_int16_array("input_image_q", chw))
    lines.append("#endif")

    def prepared(index):
        """Return (chw_float_image, label) for a test-set index."""
        raw_image = np.asarray(test_set._images[index])
        raw_label = int(np.asarray(test_set._labels[index]).reshape(-1)[0])
        prepared_tensor = resize_batch(
            torch.from_numpy(raw_image).permute(2, 0, 1)
            .float().div_(255.0).unsqueeze(0),
            args.resolution,
        )
        return prepared_tensor, raw_label

    # The calibration set is compiled in; the held-out set deliberately is not.
    # Drawing them from disjoint index ranges after the baked-in image keeps the
    # optimiser's inputs and the reporting inputs separate by construction
    # rather than by convention.
    calibration_start = args.test_index + 1
    calibration_indices = list(
        range(calibration_start, calibration_start + args.calibration_examples)
    )
    heldout_start = calibration_start + args.calibration_examples
    heldout_indices = list(
        range(heldout_start, heldout_start + args.test_examples)
    )

    if calibration_indices:
        calibration_images, calibration_labels, calibration_logits = [], [], []
        for index in calibration_indices:
            prepared_tensor, raw_label = prepared(index)
            with torch.inference_mode():
                calibration_logits.append(model(prepared_tensor).item())
            calibration_images.append(
                prepared_tensor[0].numpy().astype(np.float32)
            )
            calibration_labels.append(raw_label)
        lines.append("")
        lines.append("/* Calibration set: the inputs the error metric scores")
        lines.append(" * during search. Compiled only under LOCAL_RUN, so the")
        lines.append(" * device image never carries them. */")
        lines.append("#ifdef LOCAL_RUN")
        lines.append(f"#define PCAM_EVAL_COUNT {len(calibration_images)}")
        lines.append(c_array("input_image_set", np.stack(calibration_images)))
        # The integer path reads a SEPARATE quantised array, so the
        # calibration loop must be able to switch that too. Emitting only the
        # float set leaves the integer path re-scoring the baked-in image for
        # every iteration, which makes the in-loop error metric meaningless.
        lines.append("#ifdef PCAM_CONV1_FIXED_POINT")
        lines.append(
            c_int16_array("input_image_q_set", np.stack(calibration_images))
        )
        lines.append("#endif")
        lines.append(
            "static const int pcam_eval_labels[] = {"
            + ", ".join(str(v) for v in calibration_labels) + "};"
        )
        lines.append("#endif")
        print(f"calibration set: {len(calibration_images)} images "
              f"({sum(calibration_labels)} positive), indices "
              f"{calibration_indices[0]}..{calibration_indices[-1]}")

    if heldout_indices:
        heldout_images, heldout_labels, heldout_logits = [], [], []
        for index in heldout_indices:
            prepared_tensor, raw_label = prepared(index)
            with torch.inference_mode():
                heldout_logits.append(model(prepared_tensor).item())
            heldout_images.append(prepared_tensor[0].numpy().astype(np.float32))
            heldout_labels.append(raw_label)
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.eval_output,
            images=np.stack(heldout_images),
            labels=np.asarray(heldout_labels),
            reference_logits=np.asarray(heldout_logits, dtype=np.float32),
            indices=np.asarray(heldout_indices),
        )
        print(f"held-out test set: {len(heldout_images)} images "
              f"({sum(heldout_labels)} positive) -> {args.eval_output}")
        print("  NOT compiled into the header: the search cannot see these, so "
              "accuracy reported on them is not selection-biased")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")
    print(f"resolution {args.resolution}x{args.resolution}")
    print(f"test image index {args.test_index}, true label {label}")
    print(f"PyTorch reference: logit={logit:.6f} probability={probability:.6f}")
    weight_bytes = sum(w.size for w, _ in folded) * 4 + classifier_weight.size * 4
    print(f"total weight elements: {weight_bytes // 4} ({weight_bytes} bytes as float32)")


if __name__ == "__main__":
    main()
