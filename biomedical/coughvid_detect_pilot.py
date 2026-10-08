#!/usr/bin/env python3
"""COUGHVID cough-detection pilot, sized for the msp430fr5994 from the start.

Why this task and not COVID detection
-------------------------------------
The COVID task on COUGHVID does not support a usable model. Our own pilot and
the published literature agree: careful studies report AUC 0.58-0.63, and our
trained model emitted 0.5 +/- 0.016 on every test example, so there was no
margin for an approximation to preserve. Cough detection is the task the
dataset's own published benchmark addresses, and it is genuinely separable.

What the labels actually are -- read this before quoting any number
-------------------------------------------------------------------
The label comes from the ``cough_detected`` column, which is the OUTPUT of the
XGBoost classifier published with the dataset (Orlandic et al., Scientific
Data 2021, reported mean AUC 96.4%). Training against it is therefore
DISTILLATION of that classifier, not learning from human ground truth. The
correct claim for any result here is "agrees with the dataset's published
cough detector", never "detects coughs correctly".

That distinction is not pedantic. The dataset also carries expert pulmonologist
labels, and they do not corroborate the classifier on the negative class: of
the 820 expert-labelled recordings only 18 are marked ``no_cough``, and those
18 receive a MEAN ``cough_detected`` of 0.667 -- the published classifier
mostly calls them coughs. 18 negatives is far too few to train or evaluate
on, which is why they are not used here, but it does mean the published 96.4%
is the classifier's own cross-validation figure rather than an
expert-validated one.

Labels use only the confident tails of a strongly bimodal distribution
(3701 recordings below 0.1, 8868 above 0.9 out of 20072), so the middle is
dropped rather than forced into a class it does not belong to.

Sizing
------
The architecture is chosen to fit the real device, not shrunk afterwards.
The msp430fr5994 linker script provides a 4 KB RAM region (~3.3 KB once stack
headroom is reserved) and ~13.7 KB of usable FRAM after the managed-state
runtime. Unlike the COVID pilot's stride-1 convolutions with max-pooling --
whose first activation alone was 380 KB -- these convolutions are stride 2, so
the map shrinks immediately, and the spectrogram is 16 mels x 16 frames:

    conv1 out  6 x 8 x 8  = 1536 B
    conv2 out 12 x 4 x 4  =  768 B   peak live pair 2304 B
    conv3 out 16 x 2 x 2  =  256 B
    weights   ~2481 elems = 9924 B   + 1024 B input  = ~10.9 KB const
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from biomedical.common import (
    bootstrap_ci,
    binary_metrics,
    choose_device,
    seed_everything,
    write_json,
    youden_threshold,
)
from biomedical.coughvid_pilot import (
    extract_archive,
    index_audio,
    load_metadata,
)

SAMPLE_RATE = 16_000
#: Coughs are short (roughly 0.3-0.5 s), so a 1 s window keeps the event while
#: cutting the frame count fourfold against the COVID pilot's 3 s crop.
CROP_SECONDS = 1
N_FFT = 512

DEFAULT_N_MELS = 16
DEFAULT_FRAMES = 16
DEFAULT_CHANNELS = (6, 12, 16)

#: Only the confident tails of the published classifier's output become labels.
POSITIVE_THRESHOLD = 0.9
NEGATIVE_THRESHOLD = 0.1


def hop_for_frames(frames: int) -> int:
    """Hop length that yields *frames* STFT frames from a CROP_SECONDS window."""
    samples = SAMPLE_RATE * CROP_SECONDS
    return max(1, (samples - N_FFT) // max(1, frames - 1))


def make_transform(n_mels: int, frames: int) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=hop_for_frames(frames),
        n_mels=n_mels,
        f_min=40,
        f_max=7_600,
        power=2.0,
        center=False,
    )


def decode_audio(path: Path) -> torch.Tensor:
    """Decode to mono 16 kHz and take the highest-energy CROP_SECONDS window.

    Peak-energy cropping matches the COVID pilot's rationale: cough onset
    varies between recordings, so a fixed crop can miss the event entirely.
    """
    import av

    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout="mono", rate=SAMPLE_RATE
    )
    chunks = []
    for frame in container.decode(stream):
        for converted in resampler.resample(frame):
            chunks.append(np.asarray(converted.to_ndarray()).reshape(-1))
    for converted in resampler.resample(None):
        chunks.append(np.asarray(converted.to_ndarray()).reshape(-1))
    container.close()
    if not chunks:
        raise ValueError("audio stream decoded to zero samples")

    waveform = torch.from_numpy(np.concatenate(chunks).astype(np.float32))
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    desired = SAMPLE_RATE * CROP_SECONDS
    if len(waveform) > desired:
        squared = waveform ** 2
        stride = max(1, desired // 16)
        windows = 1 + (len(squared) - desired) // stride
        energies = torch.tensor(
            [squared[i * stride: i * stride + desired].sum() for i in range(windows)]
        )
        start = min(int(energies.argmax()) * stride, len(waveform) - desired)
        waveform = waveform[start: start + desired]
    elif len(waveform) < desired:
        waveform = torch.nn.functional.pad(waveform, (0, desired - len(waveform)))
    return waveform


def select_records(metadata: pd.DataFrame, root: Path, seed: int,
                   max_per_class: int) -> pd.DataFrame:
    """Label the confident tails of the published detector's output.

    The ambiguous middle band is DROPPED, not assigned to a class: a recording
    the published classifier scores 0.5 carries no usable supervision, and
    forcing it into a class would teach the student model to imitate a coin
    flip.
    """
    audio = index_audio(root)
    frame = metadata.copy()
    frame["cough_detected"] = pd.to_numeric(frame["cough_detected"], errors="coerce")
    frame = frame[frame["cough_detected"].notna()]

    positive = frame[frame["cough_detected"] >= POSITIVE_THRESHOLD].copy()
    negative = frame[frame["cough_detected"] <= NEGATIVE_THRESHOLD].copy()
    positive["label"] = 1
    negative["label"] = 0

    rng = np.random.default_rng(seed)
    balanced = []
    for part in (positive, negative):
        part = part[part["uuid"].astype(str).isin(audio)]
        if len(part) > max_per_class:
            keep = rng.choice(len(part), size=max_per_class, replace=False)
            part = part.iloc[np.sort(keep)]
        balanced.append(part)

    records = pd.concat(balanced, ignore_index=True)
    records["audio_path"] = [audio[str(uuid)] for uuid in records["uuid"]]
    return records.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TinyCoughDetect(nn.Module):
    """Stride-2 CNN whose activation footprint fits a 4 KB RAM region."""

    def __init__(self, channels: tuple[int, ...] = DEFAULT_CHANNELS):
        super().__init__()
        self.channels = tuple(channels)
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in self.channels:
            layers.append(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.channels[-1], 1)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        features = self.features(spectrogram)
        return self.classifier(features.mean(dim=(-2, -1))).squeeze(1)


def spectrogram_of(waveform: torch.Tensor, transform) -> torch.Tensor:
    return torch.log1p(transform(waveform)).float()


def predict(model: nn.Module, features: torch.Tensor, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            logits = model(features[start: start + 256].to(device))
            out.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("data/biomedical/coughvid/public_dataset.zip"))
    parser.add_argument("--data", type=Path,
                        default=Path("data/biomedical/coughvid/extracted"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/biomedical/coughvid_detect.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--channels", type=str, default=None,
                        help="comma-separated widths, e.g. '6,12,16'")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    channels = (
        tuple(int(p) for p in args.channels.split(","))
        if args.channels else DEFAULT_CHANNELS
    )

    extract_archive(args.archive, args.data)
    records = select_records(
        load_metadata(args.data), args.data, args.seed, args.max_per_class
    )

    transform = make_transform(args.n_mels, args.frames)
    features, kept, failures = [], [], []
    started = time.perf_counter()
    for index, row in records.iterrows():
        try:
            waveform = decode_audio(Path(row["audio_path"]))
            features.append(spectrogram_of(waveform, transform))
            kept.append(index)
        except Exception as error:  # crowdsourced recordings include broken files
            failures.append({"uuid": str(row["uuid"]), "error": str(error)})
    feature_seconds = time.perf_counter() - started
    if failures:
        records = records.iloc[kept].reset_index(drop=True)

    features = torch.stack(features)
    labels = records["label"].to_numpy(dtype=np.int64)

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=args.seed, stratify=labels[train_idx]
    )

    # Normalisation constants come from the TRAIN split only, and are saved with
    # the checkpoint: they are data-derived, not constants, so any consumer that
    # recomputes them feeds the network a different input than it was trained on.
    feature_mean = float(features[train_idx].mean())
    feature_std = float(features[train_idx].std().clamp_min(1e-6))
    features = ((features - feature_mean) / feature_std).unsqueeze(1)

    model = TinyCoughDetect(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positives = int(labels[train_idx].sum())
    negatives = int(len(train_idx) - positives)
    pos_weight = torch.tensor(negatives / max(positives, 1), device=device)

    loader = DataLoader(
        TensorDataset(features[train_idx], torch.from_numpy(labels[train_idx]).float()),
        batch_size=args.batch_size, shuffle=True,
    )
    history, best_state, best_auc = [], None, -1.0
    for epoch in range(args.epochs):
        model.train()
        total, seen = 0.0, 0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch), target, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        probabilities = predict(model, features[val_idx], device)
        metrics = binary_metrics(labels[val_idx], probabilities)
        history.append({"epoch": epoch + 1, "loss": total / seen, **metrics})
        print(f"epoch={epoch + 1} loss={total / seen:.4f} "
              f"val_bal_acc={metrics['balanced_accuracy']:.4f} "
              f"val_auc={metrics['auroc']:.4f}")
        if metrics["auroc"] > best_auc:
            best_auc = metrics["auroc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_prob = predict(model, features[val_idx], device)
    threshold = youden_threshold(labels[val_idx], val_prob)
    test_prob = predict(model, features[test_idx], device)
    test_metrics = binary_metrics(labels[test_idx], test_prob, threshold)

    checkpoint = args.output.with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "seed": args.seed,
            "channels": list(channels),
            "n_mels": args.n_mels,
            "frames": args.frames,
            "hop_length": hop_for_frames(args.frames),
            "crop_seconds": CROP_SECONDS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "selected_threshold": threshold,
        },
        checkpoint,
    )

    write_json(
        args.output,
        {
            "status": "pilot",
            "task": "COUGHVID cough-vs-not-cough, distilled from the dataset's "
                    "published XGBoost cough detector (cough_detected column)",
            "label_source": "cough_detected, the published classifier's OUTPUT; "
                            "this is distillation, not human ground truth",
            "label_caveat": "Of 820 expert-labelled recordings only 18 are "
                            "no_cough, and those receive mean cough_detected "
                            "0.667, so the published 96.4% AUC is that "
                            "classifier's own cross-validation figure and is "
                            "not corroborated by the expert labels.",
            "medical_use_warning": "research benchmark only; not a diagnostic model",
            "positive_threshold": POSITIVE_THRESHOLD,
            "negative_threshold": NEGATIVE_THRESHOLD,
            "seed": args.seed,
            "device": str(device),
            "checkpoint": str(checkpoint),
            "channels": list(channels),
            "n_mels": args.n_mels,
            "frames": args.frames,
            "hop_length": hop_for_frames(args.frames),
            "crop_seconds": CROP_SECONDS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "parameters": sum(p.numel() for p in model.parameters()),
            "selected_examples": len(records),
            "positive_examples": int(labels.sum()),
            "negative_examples": int((labels == 0).sum()),
            "train_examples": len(train_idx),
            "validation_examples": len(val_idx),
            "test_examples": len(test_idx),
            "decode_failures": failures[:20],
            "decode_failure_count": len(failures),
            "feature_seconds": feature_seconds,
            "loss_pos_weight": float(pos_weight),
            "history": history,
            "validation_selected_threshold": threshold,
            "test": test_metrics,
            "test_ci95": bootstrap_ci(
                labels[test_idx], test_prob, args.seed, threshold=threshold
            ),
        },
    )
    print(json.dumps({"test": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
