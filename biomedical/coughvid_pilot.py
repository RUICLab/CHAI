#!/usr/bin/env python3
"""COUGHVID COVID/non-COVID pilot with an approximable audio front end.

The model is intentionally compact: a logistic classifier over log-mel summary
statistics.  The approximation knob increases STFT hop length while preserving
the same classifier and feature dimensionality, so each level performs fewer
FFT frames rather than retraining a different model.

COUGHVID status is self-reported for most records.  These outputs are research
benchmark measurements, not diagnostic-performance claims.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import av
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


SAMPLE_RATE = 16_000
MAX_SECONDS = 3
N_FFT = 512
BASE_HOP = 160
N_MELS = 40


def extract_archive(archive: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)
    marker.touch()


def load_metadata(root: Path) -> pd.DataFrame:
    candidates = list(root.rglob("metadata_compiled.csv")) + list(root.rglob("compiled.csv"))
    if candidates:
        frame = pd.read_csv(candidates[0])
    else:
        rows = []
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            payload["uuid"] = payload.get("uuid", path.stem)
            rows.append(payload)
        frame = pd.DataFrame(rows)
    required = {"uuid", "status"}
    if not required.issubset(frame.columns):
        raise ValueError(f"COUGHVID metadata lacks {sorted(required - set(frame.columns))}")
    return frame


def index_audio(root: Path) -> dict[str, Path]:
    supported = {".wav", ".ogg", ".webm", ".mp3", ".m4a"}
    return {path.stem: path for path in root.rglob("*") if path.suffix.lower() in supported}


def decode_audio(path: Path) -> torch.Tensor:
    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    for frame in container.decode(stream):
        for converted in resampler.resample(frame):
            chunks.append(np.asarray(converted.to_ndarray()).reshape(-1))
    for converted in resampler.resample(None):
        chunks.append(np.asarray(converted.to_ndarray()).reshape(-1))
    container.close()
    if not chunks:
        raise ValueError("audio stream decoded to zero samples")
    waveform = torch.from_numpy(np.concatenate(chunks).astype(np.float32, copy=False))
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    desired = SAMPLE_RATE * MAX_SECONDS
    if len(waveform) > desired:
        # Select the highest-energy contiguous window rather than an arbitrary
        # start or centre crop, since cough onset varies across recordings.
        energy = torch.nn.functional.avg_pool1d(
            waveform.square()[None, None, :], kernel_size=desired, stride=max(1, desired // 16)
        ).flatten()
        start = int(energy.argmax()) * max(1, desired // 16)
        start = min(start, len(waveform) - desired)
        waveform = waveform[start : start + desired]
    elif len(waveform) < desired:
        waveform = torch.nn.functional.pad(waveform, (0, desired - len(waveform)))
    return waveform


def make_transform(hop_multiplier: int) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=BASE_HOP * hop_multiplier,
        n_mels=N_MELS,
        f_min=40,
        f_max=7_600,
        power=2.0,
        center=False,
    )


def audio_spectrogram(waveform: torch.Tensor, transform) -> torch.Tensor:
    return torch.log1p(transform(waveform)).float()


class TinyCough(nn.Module):
    """Small temporal-frequency CNN whose width may vary with STFT hop."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(24, 1)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        features = self.features(spectrogram)
        return self.classifier(features.mean(dim=(-2, -1))).squeeze(1)


def train_cnn(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: np.ndarray,
    seed: int,
    epochs: int,
) -> tuple[TinyCough, list[dict], str]:
    # Was hardcoded to "mps" or "cpu" and never checked CUDA at all, so on a
    # CUDA machine this silently fell back to CPU. choose_device() (already
    # used by pcam_pilot.py) tries CUDA, then MPS, then CPU.
    device = choose_device()
    model = TinyCough().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    data = DataLoader(TensorDataset(train_features, train_labels), batch_size=64, shuffle=True)
    history = []
    best_auc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for features, labels in data:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(features)
        probabilities = predict_cnn(model, validation_features, device)
        metrics = binary_metrics(validation_labels, probabilities)
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(train_features), **metrics})
        if metrics["auroc"] > best_auc:
            best_auc = metrics["auroc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    # str(): choose_device() returns a torch.device, which json cannot
    # serialize, and this function's own signature promises str here. The
    # caller writes this value straight into the report with write_json.
    return model, history, str(device)


def predict_cnn(model: TinyCough, features: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(features), 128):
            logits = model(features[start : start + 128].to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(probabilities)


def select_records(frame: pd.DataFrame, root: Path, seed: int, max_per_class: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["status"] = frame["status"].astype(str).str.strip().str.lower()
    frame = frame[frame["status"].isin(("covid-19", "healthy", "symptomatic"))]
    if "cough_detected" in frame:
        frame = frame[pd.to_numeric(frame["cough_detected"], errors="coerce").fillna(0) >= 0.8]
    frame["label"] = (frame["status"] == "covid-19").astype(int)
    audio = index_audio(root)
    frame["audio_path"] = frame["uuid"].astype(str).map(audio.get)
    frame = frame[frame["audio_path"].notna()]
    counts = frame["label"].value_counts()
    if len(counts) != 2 or counts.min() < 30:
        raise ValueError(f"not enough labeled audio after filtering: {counts.to_dict()}")
    per_class = min(max_per_class, int(counts.min()))
    return (
        frame.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=seed)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/biomedical/coughvid/public_dataset.zip"))
    parser.add_argument("--data", type=Path, default=Path("data/biomedical/coughvid/extracted"))
    parser.add_argument("--output", type=Path, default=Path("reports/biomedical/coughvid_pilot.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    seed_everything(args.seed)
    extract_archive(args.archive, args.data)
    records = select_records(load_metadata(args.data), args.data, args.seed, args.max_per_class)
    indices = np.arange(len(records))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=records["label"]
    )
    train_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.25,
        random_state=args.seed,
        stratify=records.iloc[train_indices]["label"],
    )

    waveforms = []
    kept_rows = []
    failures = []
    for row_index, row in records.iterrows():
        try:
            waveforms.append(decode_audio(Path(row["audio_path"])))
            kept_rows.append(row_index)
        except Exception as error:  # malformed crowdsourced recordings are expected
            failures.append({"uuid": str(row["uuid"]), "error": str(error)})
    if failures:
        # Rebuild the split after decode filtering so indices cannot drift.
        records = records.iloc[kept_rows].reset_index(drop=True)
        waveforms = [waveforms[index] for index in range(len(waveforms))]
        indices = np.arange(len(records))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=args.seed, stratify=records["label"]
        )
        train_indices, validation_indices = train_test_split(
            train_indices,
            test_size=0.25,
            random_state=args.seed,
            stratify=records.iloc[train_indices]["label"],
        )

    exact_transform = make_transform(1)
    exact_features = torch.stack([audio_spectrogram(waveform, exact_transform) for waveform in waveforms])
    labels = records["label"].to_numpy(dtype=np.int64)
    feature_mean = exact_features[train_indices].mean()
    feature_std = exact_features[train_indices].std().clamp_min(1e-6)
    exact_features = ((exact_features - feature_mean) / feature_std).unsqueeze(1)
    model, history, device = train_cnn(
        exact_features[train_indices],
        torch.from_numpy(labels[train_indices]).float(),
        exact_features[validation_indices],
        labels[validation_indices],
        args.seed,
        args.epochs,
    )
    validation_probabilities = predict_cnn(model, exact_features[validation_indices], device)
    validation_metrics = binary_metrics(labels[validation_indices], validation_probabilities)
    levels = []
    reference_predictions = None
    base_frames = 1 + (SAMPLE_RATE * MAX_SECONDS - N_FFT) // BASE_HOP
    for multiplier in (1, 2, 4, 8):
        transform = make_transform(multiplier)
        validation_features = torch.stack(
            [audio_spectrogram(waveforms[index], transform) for index in validation_indices]
        )
        validation_features = ((validation_features - feature_mean) / feature_std).unsqueeze(1)
        level_validation_probabilities = predict_cnn(model, validation_features, device)
        threshold = youden_threshold(labels[validation_indices], level_validation_probabilities)
        started = time.perf_counter()
        features = torch.stack([audio_spectrogram(waveforms[index], transform) for index in test_indices])
        features = ((features - feature_mean) / feature_std).unsqueeze(1)
        elapsed = time.perf_counter() - started
        probabilities = predict_cnn(model, features, device)
        predictions = probabilities >= threshold
        if reference_predictions is None:
            reference_predictions = predictions
        frames = 1 + (SAMPLE_RATE * MAX_SECONDS - N_FFT) // (BASE_HOP * multiplier)
        levels.append(
            {
                "hop_multiplier": multiplier,
                "validation_selected_threshold": threshold,
                **binary_metrics(labels[test_indices], probabilities, threshold),
                "prediction_disagreement_vs_l0": float(np.mean(predictions != reference_predictions)),
                "stft_frames": frames,
                "frame_reduction_vs_l0": 1.0 - frames / base_frames,
                "feature_seconds": elapsed,
                "ci95": bootstrap_ci(
                    labels[test_indices], probabilities, args.seed + multiplier, threshold=threshold
                ),
            }
        )
        print(json.dumps(levels[-1], sort_keys=True))

    write_json(
        args.output,
        {
            "status": "pilot",
            "task": "COUGHVID self-reported COVID-19 vs healthy/symptomatic cough",
            "medical_use_warning": "research benchmark only; not a diagnostic model",
            "seed": args.seed,
            "device": device,
            "selected_examples": len(records),
            "positive_examples": int(labels.sum()),
            "negative_examples": int((labels == 0).sum()),
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "test_examples": len(test_indices),
            "decode_failures": failures,
            "validation_l0": validation_metrics,
            "training_history": history,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "levels": levels,
        },
    )


if __name__ == "__main__":
    main()
