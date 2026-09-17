from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

from ..datasets.transformer_sgah import SGAH_CLASSES, SgahEvent, load_sgah_events
from ..learned.ad_tfm_at import (
    ADTFMConfig,
    DEFAULT_SGAH_LABELS,
    save_ad_tfm_classifier,
    train_ad_tfm_at,
)


def _split(events: list[SgahEvent]) -> tuple[list[SgahEvent], list[SgahEvent], list[SgahEvent]]:
    """Match the repository's frozen per-class whole-event 60/20/20 protocol."""
    n = len(events)
    train_end = max(1, int(np.floor(0.60 * n)))
    dev_end = max(train_end + 1, int(np.floor(0.80 * n))) if n >= 3 else n
    dev_end = min(dev_end, n)
    return events[:train_end], events[train_end:dev_end], events[dev_end:]


def _stack(events: list[SgahEvent]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([event.signal_matrix for event in events]).astype(np.float32)
    # SGAH class ids are 1..5; CrossEntropy expects 0..4.
    y = np.asarray([event.class_id - 1 for event in events], dtype=np.int64)
    return x, y


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(5), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "by_class": {
            DEFAULT_SGAH_LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(5)
        },
    }


def _transformer_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Class index 3 == SGAH class id 4 == main transformer fault.
    true_positive_class = y_true == 3
    pred_positive_class = y_pred == 3
    tp = int(np.sum(true_positive_class & pred_positive_class))
    fn = int(np.sum(true_positive_class & ~pred_positive_class))
    fp = int(np.sum(~true_positive_class & pred_positive_class))
    tn = int(np.sum(~true_positive_class & ~pred_positive_class))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    balanced = None if recall is None or specificity is None else 0.5 * (recall + specificity)
    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced,
    }


def run_sgah_ad_tfm_benchmark(
    data_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    time_dimensions: int = 4,
    frequency_dimensions: int = 4,
    omega0: float = 16.0,
    phase_augmentation: bool = True,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """Train/evaluate the paper-inspired AD-TFM-AT model on frozen SGAH events.

    Important protocol note: phase switching is applied to training events only.
    The paper's temporal-sliding augmentation is *not* applied here because this
    repository's pinned SGAH loader exposes already segmented 100-sample events;
    it does not preserve the longer pre/post-fault record needed to move a
    sampling window without synthesizing samples.
    """

    root = Path(data_dir)
    train_events: list[SgahEvent] = []
    dev_events: list[SgahEvent] = []
    test_events: list[SgahEvent] = []
    manifest: list[dict] = []

    for class_id, label in SGAH_CLASSES.items():
        events = load_sgah_events(root / f"{class_id}-data.csv", class_id)
        train, dev, test = _split(events)
        if not train or not test:
            raise ValueError(f"SGAH class {class_id} does not have enough whole events for frozen split")
        train_events.extend(train)
        dev_events.extend(dev)
        test_events.extend(test)
        manifest.append({
            "class_id": class_id,
            "label": label,
            "event_count": len(events),
            "train_count": len(train),
            "dev_count": len(dev),
            "test_count": len(test),
        })

    train_x, train_y = _stack(train_events)
    dev_x, dev_y = _stack(dev_events)
    test_x, test_y = _stack(test_events)

    config = ADTFMConfig(
        input_size=6,
        hidden_size=hidden_size,
        time_dimensions=time_dimensions,
        frequency_dimensions=frequency_dimensions,
        omega0=omega0,
        num_classes=5,
    )
    started = perf_counter()
    classifier, history = train_ad_tfm_at(
        train_x,
        train_y,
        val_x=dev_x,
        val_y=dev_y,
        config=config,
        class_names=DEFAULT_SGAH_LABELS,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        phase_augmentation=phase_augmentation,
        augmentation_mode="paper",
        seed=seed,
        device=device,
    )
    training_seconds = perf_counter() - started

    started = perf_counter()
    prediction = classifier.predict_batch(test_x)
    inference_seconds = perf_counter() - started
    test_pred = np.asarray(prediction["predicted_indices"], dtype=int)

    payload = {
        "source": "smartlab-hfut/SGAH-datasets (State Grid Corporation of China)",
        "method": "AD-TFM-AT adaptation of arXiv:2302.09332",
        "protocol": {
            "samples_per_event": 100,
            "channels": ["Ua", "Ub", "Uc", "Ia", "Ib", "Ic"],
            "split": "Per class, contiguous whole-event 60% train / 20% development / 20% frozen test.",
            "normalization": "Per-channel mean/std fitted on augmented training data only.",
            "augmentation": "Training-only phase switching (original + A<->B + A<->C).",
            "temporal_sliding": "Not used: current pinned SGAH loader exposes fixed 100-sample events, not longer records around the fault.",
            "model": {
                "hidden_size_D": hidden_size,
                "time_dimensions_K": time_dimensions,
                "frequency_dimensions_J": frequency_dimensions,
                "omega0": omega0,
                "attention": "trainable context-vector attention over all AD-TFM hidden states",
            },
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
        },
        "manifest": manifest,
        "training_seconds": float(training_seconds),
        "test_inference_seconds": float(inference_seconds),
        "history": history,
        "multiclass": _multiclass_metrics(test_y, test_pred),
        "main_transformer_fault_binary": _transformer_binary_metrics(test_y, test_pred),
        "test_predictions": [
            {
                "class_id": event.class_id,
                "event_id": event.event_id,
                "true_label": event.label,
                "predicted_label": DEFAULT_SGAH_LABELS[int(pred)],
                "probabilities": {
                    name: float(value)
                    for name, value in zip(DEFAULT_SGAH_LABELS, prediction["probabilities"][row])
                },
                "attention_peak_sample": int(np.argmax(prediction["attention"][row])),
            }
            for row, (event, pred) in enumerate(zip(test_events, test_pred))
        ],
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = save_ad_tfm_classifier(classifier, out / "ad_tfm_at_checkpoint.pt")
        payload["checkpoint"] = str(checkpoint)
        (out / "transformer_sgah_ad_tfm_benchmark.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload
