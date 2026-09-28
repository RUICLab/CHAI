#!/usr/bin/env python3
"""Train and evaluate a small, resolution-scalable PCam classifier.

This is deliberately a pilot rather than a claim about the official PCam
leaderboard.  When the 6.1 GB official training file is unavailable, the
official validation split is divided deterministically into development-train
and development-validation subsets; the official test split remains untouched.
The approximation knob is input resolution, which changes convolutional work
without changing model weights.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from biomedical.common import (
    bootstrap_ci,
    binary_metrics,
    choose_device,
    seed_everything,
    write_json,
    youden_threshold,
)


FILES = {
    "train": ("camelyonpatch_level_2_split_train_x.h5", "camelyonpatch_level_2_split_train_y.h5"),
    "valid": ("camelyonpatch_level_2_split_valid_x.h5", "camelyonpatch_level_2_split_valid_y.h5"),
    "test": ("camelyonpatch_level_2_split_test_x.h5", "camelyonpatch_level_2_split_test_y.h5"),
}


def decompress_if_needed(directory: Path, filename: str) -> Path:
    target = directory / filename
    if target.exists():
        return target
    archive = target.with_suffix(target.suffix + ".gz")
    if not archive.exists():
        raise FileNotFoundError(f"missing {target} (or {archive})")
    with gzip.open(archive, "rb") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return target


class H5Pcam(Dataset):
    def __init__(self, directory: Path, split: str):
        image_name, label_name = FILES[split]
        self.image_path = decompress_if_needed(directory, image_name)
        self.label_path = decompress_if_needed(directory, label_name)
        with h5py.File(self.image_path, "r") as handle:
            self.length = int(handle["x"].shape[0])
        self._images = None
        self._labels = None

    def __len__(self) -> int:
        return self.length

    def _open(self) -> None:
        if self._images is None:
            self._images = h5py.File(self.image_path, "r")["x"]
            self._labels = h5py.File(self.label_path, "r")["y"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        image = torch.from_numpy(np.asarray(self._images[index])).permute(2, 0, 1).float().div_(255.0)
        label = torch.tensor(float(np.asarray(self._labels[index]).reshape(-1)[0]))
        return image, label


class TinyPcam(nn.Module):
    """A 6.2K-parameter model small enough to make approximation interpretable."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.features(image)
        image = image.mean(dim=(-2, -1))
        return self.classifier(image).squeeze(1)


def resize_batch(images: torch.Tensor, resolution: int) -> torch.Tensor:
    # PCam's label is determined by the centre 32x32 region.  Use the same
    # centre 64x64 field of view at every level, then vary only sampling density.
    h, w = images.shape[-2:]
    top, left = (h - 64) // 2, (w - 64) // 2
    images = images[..., top : top + 64, left : left + 64]
    if images.shape[-1] == resolution:
        return images
    return F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)


def loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def evaluate(
    model: nn.Module, data: DataLoader, device: torch.device, resolution: int
) -> tuple[dict, list[float], list[int]]:
    labels: list[int] = []
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in data:
            images = resize_batch(images.to(device), resolution)
            logits = model(images)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.int().tolist())
    return binary_metrics(labels, probabilities), probabilities, labels


def estimate_macs(resolution: int) -> int:
    size = resolution
    total = 0
    channels = (3, 8, 16, 32)
    for input_channels, output_channels in zip(channels[:-1], channels[1:]):
        size = (size + 1) // 2
        total += size * size * input_channels * output_channels * 3 * 3
    return total + 32


def train_epoch(model: nn.Module, data: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    examples = 0
    for images, targets in data:
        # Histology has no canonical orientation.
        if torch.rand(()) < 0.5:
            images = images.flip(-1)
        if torch.rand(()) < 0.5:
            images = images.flip(-2)
        images, targets = resize_batch(images.to(device), 64), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(images)
        examples += len(images)
    return total_loss / examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/biomedical/pcam"))
    parser.add_argument("--output", type=Path, default=Path("reports/biomedical/pcam_pilot.json"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train", type=int, default=24576)
    parser.add_argument("--max-test", type=int, default=32768)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    development = H5Pcam(args.data, "valid")
    test = H5Pcam(args.data, "test")
    permutation = np.random.default_rng(args.seed).permutation(len(development))
    train_count = min(args.max_train, len(development) - 4096)
    train_set = Subset(development, permutation[:train_count].tolist())
    validation_set = Subset(development, permutation[train_count:].tolist())
    if args.max_test < len(test):
        test = Subset(test, list(range(args.max_test)))

    model = TinyPcam().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    best_state = None
    best_auc = -1.0
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader(train_set, args.batch_size, True), optimizer, device)
        metrics, _, _ = evaluate(model, loader(validation_set, args.batch_size), device, 64)
        history.append({"epoch": epoch + 1, "loss": loss, **metrics})
        print(f"epoch={epoch + 1} loss={loss:.4f} val_bal_acc={metrics['balanced_accuracy']:.4f} val_auc={metrics['auroc']:.4f}")
        if metrics["auroc"] > best_auc:
            best_auc = metrics["auroc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loader = loader(test, args.batch_size)
    validation_loader = loader(validation_set, args.batch_size)
    levels = []
    reference_predictions = None
    reference_macs = estimate_macs(64)
    for resolution in (64, 48, 32, 24, 16):
        _, validation_probabilities, validation_labels = evaluate(
            model, validation_loader, device, resolution
        )
        threshold = youden_threshold(validation_labels, validation_probabilities)
        started = time.perf_counter()
        _, probabilities, test_labels = evaluate(model, test_loader, device, resolution)
        elapsed = time.perf_counter() - started
        metrics = binary_metrics(test_labels, probabilities, threshold)
        predictions = np.asarray(probabilities) >= threshold
        if reference_predictions is None:
            reference_predictions = predictions
        macs = estimate_macs(resolution)
        levels.append(
            {
                "resolution": resolution,
                "validation_selected_threshold": threshold,
                **metrics,
                "prediction_disagreement_vs_l0": float(np.mean(predictions != reference_predictions)),
                "estimated_macs": macs,
                "mac_reduction_vs_l0": 1.0 - macs / reference_macs,
                "host_eval_seconds": elapsed,
                "ci95": bootstrap_ci(
                    test_labels,
                    probabilities,
                    args.seed + resolution,
                    threshold=threshold,
                    repetitions=200,
                ),
            }
        )
        print(json.dumps(levels[-1], sort_keys=True))

    checkpoint = args.output.with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "seed": args.seed}, checkpoint)
    write_json(
        args.output,
        {
            "status": "pilot",
            "task": "PCam metastasis present vs absent",
            "split_protocol": "official validation split used for development; official test untouched",
            "device": str(device),
            "seed": args.seed,
            "development_train_examples": len(train_set),
            "development_validation_examples": len(validation_set),
            "test_examples": len(test),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "levels": levels,
            "checkpoint": str(checkpoint),
        },
    )


if __name__ == "__main__":
    main()
