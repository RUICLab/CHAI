#!/usr/bin/env python3
"""Score a compiled C kernel on the UNTOUCHED held-out test set.

Why this exists as a separate step
----------------------------------
CHAI's in-loop error metric scores decision disagreement across the
*calibration* inputs baked into the exported header. Those are the inputs the
optimiser is allowed to see, so accuracy measured on them is selection-biased
by the very metric that chose the configuration. This script evaluates the
frozen configuration on a disjoint set that was never compiled in, which is
the only number that should be reported as accuracy.

Protocol this supports:

  1. search with CHAI's error constraint on the calibration set
  2. freeze the winning knob values
  3. run this script once on the held-out set
  4. report AUROC, balanced accuracy, sensitivity and specificity from here,
     and report calibration disagreement separately as a behavioural metric

It evaluates the REAL compiled artifact rather than a Python reimplementation,
by recompiling once per example with that example substituted into the header.
That is slow but it measures the thing that would actually be deployed.

Two traps this deliberately handles, both of which produced wrong numbers
before:

  * The integer/fixed-point kernels read a SEPARATE quantised input array.
    Substituting only the float array leaves every example scored against the
    same baked-in quantised input, which yields identical logits and a fake
    AUROC of exactly 0.5. Both arrays are substituted here.
  * A header carrying a calibration set makes the kernel loop over it. That
    would score the baked-in set instead of the example under test, so the
    loop is disabled for this harness and the first emitted logit is taken.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

#: Must match the Q-format the exporters use for the integer arrays.
FP_SCALE = 2048

_FLOAT_INPUT = {
    "pcam": re.compile(
        r"static const float input_image\[\d+\]\[\d+\]\[\d+\] = \{.*?\};", re.DOTALL
    ),
    "coughvid": re.compile(
        r"static const float input_spectrogram\[1\]\[\d+\]\[\d+\] = \{.*?\};",
        re.DOTALL,
    ),
}
#: Per-benchmark quantised input array. This was previously hardcoded to
#: coughvid's name, so for pcam the quantised input was never substituted and
#: every example scored against the same baked-in image -- yielding identical
#: logits and an AUROC of exactly 0.5000, which reads as a destroyed model
#: rather than a broken harness. Keyed by benchmark so the two cannot diverge.
_QUANT_INPUT = {
    "pcam": re.compile(
        r"static const int16_t input_image_q((?:\[\d+\])+) = \{.*?\};", re.DOTALL
    ),
    "coughvid": re.compile(
        r"static const int16_t input_spectrogram_q((?:\[\d+\])+) = \{.*?\};",
        re.DOTALL,
    ),
}
_EVAL_COUNT = re.compile(r"#define (?:PCAM|COUGHVID)_EVAL_COUNT \d+")


def c_float_literal(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def _nest(array: np.ndarray, render) -> str:
    if array.ndim == 1:
        return "{" + ", ".join(render(v) for v in array) + "}"
    return "{" + ", ".join(_nest(row, render) for row in array) + "}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True,
                        help="C source of the kernel variant to score")
    parser.add_argument("--header", type=Path, required=True,
                        help="exported weights header the kernel includes")
    parser.add_argument("--heldout", type=Path, required=True,
                        help=".npz written by the exporter's --test-examples")
    parser.add_argument("--benchmark", choices=sorted(_FLOAT_INPUT), required=True)
    parser.add_argument("--knobs", default="",
                        help="comma-separated name=value, e.g. 'row_stride=2'. "
                             "Empty scores the identity configuration.")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N examples (0 = all)")
    args = parser.parse_args()

    data = np.load(args.heldout)
    inputs = data["images"] if "images" in data else data["spectrograms"]
    labels = data["labels"]
    if args.limit:
        inputs, labels = inputs[: args.limit], labels[: args.limit]

    source = args.kernel.read_text()
    knobs = dict(
        pair.split("=") for pair in args.knobs.split(",") if pair
    )
    for name, value in knobs.items():
        source, count = re.subn(
            rf"int {re.escape(name)} = [0-9]+;", f"int {name} = {value};", source
        )
        if not count:
            raise SystemExit(f"knob {name!r} not found in {args.kernel}")

    header = _EVAL_COUNT.sub(
        "/* calibration loop disabled by evaluate_kernel_heldout.py */",
        args.header.read_text(),
    )
    float_pattern = _FLOAT_INPUT[args.benchmark]
    if not float_pattern.search(header):
        raise SystemExit(f"could not find the float input array in {args.header}")
    quant_pattern = _QUANT_INPUT[args.benchmark]
    has_quant = bool(quant_pattern.search(header))
    quant_name = ("input_image_q" if args.benchmark == "pcam"
                  else "input_spectrogram_q")
    if not has_quant:
        print(f"note: no {quant_name} in the header; float path only")

    work = Path(tempfile.mkdtemp())
    (work / "kernel.c").write_text(source)
    header_name = args.header.name

    # Reshape each example to the rank the header already declares rather than
    # assuming one. pcam stores (C,H,W) but coughvid stores (H,W) while its
    # header declares [1][H][W], so emitting the stored shape verbatim
    # produced input_spectrogram[16][16] and the kernel's [0][0][0] subscript
    # failed to compile.
    declared = re.search(
        r"static const float (?:input_image|input_spectrogram)"
        r"((?:\[\d+\])+) = ", header
    )
    if not declared:
        raise SystemExit("could not read the declared input dimensions")
    declared_shape = tuple(int(n) for n in re.findall(r"\[(\d+)\]", declared.group(1)))

    logits = []
    for example in inputs:
        example = example.reshape(declared_shape)
        dims = "".join(f"[{d}]" for d in declared_shape)
        name = ("input_image" if args.benchmark == "pcam"
                else "input_spectrogram")
        patched = float_pattern.sub(
            f"static const float {name}{dims} = "
            f"{_nest(example, c_float_literal)};",
            header, count=1,
        )
        if has_quant:
            quantised = np.clip(
                np.rint(example.astype(np.float64) * FP_SCALE), -32768, 32767
            ).astype(np.int16)
            patched = quant_pattern.sub(
                f"static const int16_t {quant_name}{dims} = "
                f"{_nest(quantised, lambda v: str(int(v)))};",
                patched, count=1,
            )
        (work / header_name).write_text(patched)
        build = subprocess.run(
            ["gcc", "-O2", "kernel.c", "-o", "kernel", "-DLOCAL_RUN", "-lm"],
            cwd=work, capture_output=True, text=True,
        )
        if build.returncode != 0:
            raise SystemExit("build failed:\n" + build.stderr[:2000])
        run = subprocess.run(["./kernel"], cwd=work, capture_output=True, text=True)
        logits.append(float(run.stdout.strip().splitlines()[0]))

    logits = np.asarray(logits)
    predictions = (logits >= 0.0).astype(int)

    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix, roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    reference = data["reference_logits"] if "reference_logits" in data else None

    print(f"held-out set: {len(labels)} examples, {int(labels.sum())} positive")
    print(f"knobs: {args.knobs or 'identity'}")
    print(f"  AUROC              {roc_auc_score(labels, logits):.4f}")
    print(f"  balanced accuracy  {balanced_accuracy_score(labels, predictions):.4f}")
    print(f"  accuracy           {(predictions == labels).mean():.4f}")
    print(f"  sensitivity        {tp / (tp + fn) if tp + fn else 0:.4f}  "
          f"({tp} of {tp + fn} positives)")
    print(f"  specificity        {tn / (tn + fp) if tn + fp else 0:.4f}  "
          f"({tn} of {tn + fp} negatives)")
    if reference is not None and len(reference) == len(logits):
        flips = int(((reference >= 0) != (logits >= 0)).sum())
        print(f"  decisions changed vs unapproximated: {flips}/{len(logits)} "
              f"({flips / len(logits):.4f})")
        print("  (reported separately: this is behavioural preservation, "
              "not accuracy)")


if __name__ == "__main__":
    main()
