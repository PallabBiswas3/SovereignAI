from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..datasets.transformer_sgah import SGAH_CLASSES, SgahEvent, load_sgah_events
from ..learned.ad_tfm_ablation import (
    PAPER_ABLATION_VARIANTS,
    train_tfm_variant,
    trainable_parameter_count,
)
from ..learned.ad_tfm_at import ADTFMConfig, DEFAULT_SGAH_LABELS


def _split(events: list[SgahEvent]) -> tuple[list[SgahEvent], list[SgahEvent], list[SgahEvent]]:
    n = len(events)
    train_end = max(1, int(np.floor(0.60 * n)))
    dev_end = max(train_end + 1, int(np.floor(0.80 * n))) if n >= 3 else n
    dev_end = min(dev_end, n)
    return events[:train_end], events[train_end:dev_end], events[dev_end:]


def _stack(events: list[SgahEvent]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([event.signal_matrix for event in events]).astype(np.float32)
    y = np.asarray([event.class_id - 1 for event in events], dtype=np.int64)
    return x, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(5), zero_division=0
    )
    y_onehot = np.eye(5, dtype=float)[y_true]
    auc_by_class = {}
    for index, name in enumerate(DEFAULT_SGAH_LABELS):
        try:
            auc_by_class[name] = float(roc_auc_score(y_onehot[:, index], probabilities[:, index]))
        except ValueError:
            auc_by_class[name] = None
    try:
        macro_auc = float(roc_auc_score(y_onehot, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_ovr_auc": macro_auc,
        "by_class": {
            DEFAULT_SGAH_LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auc": auc_by_class[DEFAULT_SGAH_LABELS[i]],
            }
            for i in range(5)
        },
    }


def _main_transformer_binary(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    true_positive = y_true == 3
    pred_positive = y_pred == 3
    tp = int(np.sum(true_positive & pred_positive))
    fn = int(np.sum(true_positive & ~pred_positive))
    fp = int(np.sum(~true_positive & pred_positive))
    tn = int(np.sum(~true_positive & ~pred_positive))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    try:
        auc = float(roc_auc_score(true_positive.astype(int), probabilities[:, 3]))
    except ValueError:
        auc = None
    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "balanced_accuracy": None if recall is None or specificity is None else 0.5 * (recall + specificity),
    }


def run_sgah_tfm_ablation(
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
    fixed_scale: float = 1.0,
    fixed_shift: float = 0.0,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    root = Path(data_dir)
    train_events: list[SgahEvent] = []
    dev_events: list[SgahEvent] = []
    test_events: list[SgahEvent] = []
    manifest: list[dict] = []

    for class_id, label in SGAH_CLASSES.items():
        events = load_sgah_events(root / f"{class_id}-data.csv", class_id)
        train, dev, test = _split(events)
        if not train or not test:
            raise ValueError(f"SGAH class {class_id} does not have enough events")
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

    results: dict[str, dict] = {}
    for spec in PAPER_ABLATION_VARIANTS:
        started = perf_counter()
        classifier, history = train_tfm_variant(
            train_x,
            train_y,
            variant=spec,
            val_x=dev_x,
            val_y=dev_y,
            config=config,
            class_names=DEFAULT_SGAH_LABELS,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            phase_augmentation=phase_augmentation,
            fixed_scale=fixed_scale,
            fixed_shift=fixed_shift,
            seed=seed,
            device=device,
        )
        training_seconds = perf_counter() - started
        started = perf_counter()
        prediction = classifier.predict_batch(test_x)
        inference_seconds = perf_counter() - started
        probabilities = np.asarray(prediction["probabilities"], dtype=float)
        predicted = np.asarray(prediction["predicted_indices"], dtype=int)

        results[spec.name] = {
            "adaptive_wavelet": spec.adaptive_wavelet,
            "attention": spec.attention,
            "trainable_parameters": trainable_parameter_count(classifier.model),
            "training_seconds": float(training_seconds),
            "test_inference_seconds": float(inference_seconds),
            "history": history,
            "metrics": _metrics(test_y, predicted, probabilities),
            "main_transformer_fault_binary": _main_transformer_binary(test_y, predicted, probabilities),
        }

    ranking = sorted(
        (
            {
                "variant": name,
                "accuracy": row["metrics"]["accuracy"],
                "macro_f1": row["metrics"]["macro_f1"],
                "macro_ovr_auc": row["metrics"]["macro_ovr_auc"],
                "main_transformer_f1": row["main_transformer_fault_binary"]["f1"],
                "main_transformer_auc": row["main_transformer_fault_binary"]["auc"],
            }
            for name, row in results.items()
        ),
        key=lambda row: (row["macro_f1"], row["accuracy"]),
        reverse=True,
    )

    payload = {
        "source": "smartlab-hfut/SGAH-datasets",
        "method": "Paper-style TFM / AD-TFM / TFM-AT / AD-TFM-AT ablation",
        "protocol": {
            "split": "Per class contiguous whole-event 60/20/20 train/dev/frozen-test.",
            "phase_augmentation": bool(phase_augmentation),
            "temporal_sliding": "Not used because the current loader exposes fixed 100-sample events.",
            "shared_hyperparameters": {
                "D": hidden_size,
                "K": time_dimensions,
                "J": frequency_dimensions,
                "omega0": omega0,
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
            },
            "fixed_wavelet_definition": {
                "scale": fixed_scale,
                "translation": fixed_shift,
                "note": "The paper defines TFM as fixed-scale/fixed-translation but does not state their numeric constants in the paper text; these benchmark constants are explicit controlled choices, not claimed paper values.",
            },
        },
        "manifest": manifest,
        "results": results,
        "ranking": ranking,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "transformer_sgah_tfm_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
