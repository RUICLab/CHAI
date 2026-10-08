#!/usr/bin/env python3
"""COUGHVID pilot using a PRETRAINED audio model as a frozen feature extractor.

The from-scratch pilot (coughvid_pilot.py) trains a tiny CNN on raw mel
spectrograms from only ~547 real covid-19 recordings and stays near chance
(balanced accuracy 0.53-0.58 across seeds). Two data-selection hypotheses for
why (merging symptomatic into the negative class; discarding 90% of available
negative data) were tested and both failed to help.

This tries a different kind of fix: instead of training a classifier from
scratch on so little data, use PANNs Cnn14 (Kong et al., "PANNs: Large-Scale
Pretrained Audio Neural Networks for Audio Pattern Recognition"), pretrained on
AudioSet's 527 sound classes -- one of which is literally "Cough" -- as a
FROZEN feature extractor, and train only a linear classifier on its 2048-dim
embedding. This is the standard "linear probe" protocol for evaluating whether
a pretrained representation carries more signal than a from-scratch model can
learn from this little labeled data.

Reuses extract_archive / load_metadata / index_audio / select_records from
coughvid_pilot.py UNCHANGED, so the same records are selected for the same seed
and results are directly comparable. Does NOT reuse decode_audio: PANNs Cnn14
was trained at 32 kHz (Cnn14(sample_rate=32000, ...) in panns_inference's own
source), while decode_audio hardcodes SAMPLE_RATE=16000 for the from-scratch
model. Feeding 16 kHz audio to a 32 kHz-trained model would not crash -- it
would silently halve the effective frequency range the model receives ---
so this file decodes at 32 kHz independently rather than resampling the
already-bandlimited 16 kHz signal upward after the fact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from biomedical.common import bootstrap_ci, binary_metrics, seed_everything, write_json, youden_threshold
from biomedical.coughvid_pilot import extract_archive, index_audio, load_metadata, select_records

PANNS_SAMPLE_RATE = 32_000
MAX_SECONDS = 3


def decode_audio_32k(path: Path) -> np.ndarray:
    """Same decode strategy as coughvid_pilot.decode_audio (peak-energy crop,
    peak normalisation) but at the 32 kHz PANNs expects instead of 16 kHz."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=PANNS_SAMPLE_RATE)
    chunks = []
    for frame in container.decode(stream):
        for converted in resampler.resample(frame):
            chunks.append(np.asarray(converted.to_ndarray()).reshape(-1))
    for converted in resampler.resample(None):
        chunks.append(np.asarray(converted.to_ndarray()).reshape(-1))
    container.close()
    if not chunks:
        raise ValueError("audio stream decoded to zero samples")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak
    desired = PANNS_SAMPLE_RATE * MAX_SECONDS
    if len(waveform) > desired:
        # Highest-energy contiguous window, matching decode_audio's rationale:
        # cough onset varies across recordings, so a fixed crop risks missing it.
        squared = waveform ** 2
        stride = max(1, desired // 16)
        n_windows = 1 + (len(squared) - desired) // stride
        energies = np.array(
            [squared[i * stride : i * stride + desired].sum() for i in range(n_windows)]
        )
        start = min(int(energies.argmax()) * stride, len(waveform) - desired)
        waveform = waveform[start : start + desired]
    elif len(waveform) < desired:
        waveform = np.pad(waveform, (0, desired - len(waveform)))
    return waveform


def extract_embeddings(audio_tagger, waveforms: list[np.ndarray]) -> np.ndarray:
    embeddings = []
    with torch.inference_mode():
        for waveform in waveforms:
            _clipwise, embedding = audio_tagger.inference(waveform[None, :])
            embeddings.append(embedding[0])
    return np.stack(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/biomedical/coughvid/public_dataset.zip"))
    parser.add_argument("--data", type=Path, default=Path("data/biomedical/coughvid/extracted"))
    parser.add_argument("--output", type=Path, default=Path("reports/biomedical/coughvid_embedding_pilot.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="save/reuse extracted embeddings + labels + splits here (.npz), "
             "since PANNs inference (~80ms/clip) dominates wall time and "
             "re-running it just to sweep classifier hyperparameters is wasteful",
    )
    parser.add_argument("--C", type=float, default=1.0, help="logistic regression inverse regularisation strength")
    args = parser.parse_args()

    seed_everything(args.seed)
    extract_archive(args.archive, args.data)
    # Unchanged call: same negative_statuses default ("healthy","symptomatic")
    # and negative_multiplier default (1.0, the verified-matching baseline
    # path) as coughvid_pilot.py, so the SAME records are selected for the
    # same seed and this is a fair, apples-to-apples comparison.
    records = select_records(load_metadata(args.data), args.data, args.seed, args.max_per_class)

    waveforms = []
    kept_rows = []
    failures = []
    for row_index, row in records.iterrows():
        try:
            waveforms.append(decode_audio_32k(Path(row["audio_path"])))
            kept_rows.append(row_index)
        except Exception as error:  # malformed crowdsourced recordings are expected
            failures.append({"uuid": str(row["uuid"]), "error": str(error)})
    if failures:
        records = records.iloc[kept_rows].reset_index(drop=True)

    labels = records["label"].to_numpy(dtype=np.int64)
    indices = np.arange(len(records))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_indices, validation_indices = train_test_split(
        train_indices, test_size=0.25, random_state=args.seed, stratify=labels[train_indices]
    )

    if args.embedding_cache and args.embedding_cache.exists():
        cached = np.load(args.embedding_cache)
        embeddings = cached["embeddings"]
        embedding_seconds = 0.0
        assert len(embeddings) == len(records), (
            f"cache has {len(embeddings)} embeddings for {len(records)} records "
            "-- stale cache, delete it and re-run"
        )
    else:
        from panns_inference import AudioTagging

        started = time.perf_counter()
        audio_tagger = AudioTagging(checkpoint_path=None, device="cpu")
        embeddings = extract_embeddings(audio_tagger, waveforms)
        embedding_seconds = time.perf_counter() - started
        if args.embedding_cache:
            args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.embedding_cache, embeddings=embeddings, labels=labels)

    classifier = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
    classifier.fit(embeddings[train_indices], labels[train_indices])

    # Reported alongside val/test so overfitting (high train, low val/test) is
    # visible rather than silently mistaken for "the embeddings carry no
    # signal" -- 2048 embedding dims against ~600 train examples is exactly the
    # regime where a too-weak C overfits.
    train_accuracy = float(classifier.score(embeddings[train_indices], labels[train_indices]))

    validation_probabilities = classifier.predict_proba(embeddings[validation_indices])[:, 1]
    validation_metrics = binary_metrics(labels[validation_indices], validation_probabilities)
    threshold = youden_threshold(labels[validation_indices], validation_probabilities)

    test_probabilities = classifier.predict_proba(embeddings[test_indices])[:, 1]
    test_metrics = binary_metrics(labels[test_indices], test_probabilities, threshold)

    write_json(
        args.output,
        {
            "status": "pilot",
            "task": "COUGHVID self-reported COVID-19 vs healthy/symptomatic cough, "
                    "PANNs Cnn14 (AudioSet-pretrained) frozen embedding + logistic probe",
            "medical_use_warning": "research benchmark only; not a diagnostic model",
            "feature_extractor": "PANNs Cnn14, checkpoint Cnn14_mAP=0.431, sample_rate=32000",
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_extraction_seconds": embedding_seconds,
            "seed": args.seed,
            "selected_examples": len(records),
            "positive_examples": int(labels.sum()),
            "negative_examples": int((labels == 0).sum()),
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "test_examples": len(test_indices),
            "decode_failures": failures,
            "logistic_regression_C": args.C,
            "train_accuracy": train_accuracy,
            "validation_l0": validation_metrics,
            "validation_selected_threshold": threshold,
            "test": test_metrics,
        },
    )
    print(json.dumps(
        {"seed": args.seed, "C": args.C, "train_accuracy": train_accuracy, **test_metrics},
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
