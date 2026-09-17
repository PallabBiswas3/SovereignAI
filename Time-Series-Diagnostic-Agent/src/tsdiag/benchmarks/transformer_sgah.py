from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..datasets.transformer_sgah import SGAH_CHANNEL_NAMES, SGAH_CLASSES, SgahEvent, load_sgah_events
from ..domains.domain_steps import (
    transformer_corr,
    transformer_denoise,
    transformer_fuse,
    transformer_representation,
    transformer_spectral,
    transformer_sync,
    transformer_weights,
)
from ..domains.runners import TransformerDiagnosticPipeline


@dataclass
class SgahCase:
    class_id: int
    label: str
    event_id: int
    true_transformer_fault: bool
    predicted_transformer_fault: bool
    decision: str
    abstained: bool
    confidence: float
    verification_status: str | None
    runtime_seconds: float


def _split(events: list[SgahEvent]) -> tuple[list[SgahEvent], list[SgahEvent], list[SgahEvent]]:
    n = len(events)
    train_end = max(1, int(np.floor(0.60 * n)))
    dev_end = max(train_end + 1, int(np.floor(0.80 * n))) if n >= 3 else n
    dev_end = min(dev_end, n)
    return events[:train_end], events[train_end:dev_end], events[dev_end:]


def _feature_image(event: SgahEvent) -> np.ndarray:
    state = {
        "signal_matrix": event.signal_matrix,
        # SGAH does not publish a physical sampling frequency in the repository.
        # A normalized sample rate is sufficient here because the classifier uses
        # representation amplitudes/shape rather than physical frequency labels.
        "sampling_rate_hz": 1.0,
        "sensor_positions": list(SGAH_CHANNEL_NAMES),
    }
    for fn in (
        transformer_sync,
        transformer_denoise,
        transformer_corr,
        transformer_weights,
        transformer_fuse,
        transformer_spectral,
        transformer_representation,
    ):
        state.update(fn(state))
    return np.asarray(state["feature_image"], dtype=float)


class _FixedSgahClassifier:
    def __init__(self, estimator):
        self.estimator = estimator

    def __call__(self, feature_image):
        x = np.asarray(feature_image, dtype=float).reshape(1, -1)
        probability = float(self.estimator.predict_proba(x)[0, 1])
        return {
            "label": "main_transformer_fault" if probability >= 0.5 else None,
            "confidence": max(probability, 1.0 - probability),
            "probabilities": {
                "main_transformer_fault": probability,
                "not_transformer_fault": 1.0 - probability,
            },
        }


def _binary_metrics(rows: list[SgahCase]) -> dict:
    tp = sum(r.true_transformer_fault and r.predicted_transformer_fault for r in rows)
    fn = sum(r.true_transformer_fault and not r.predicted_transformer_fault for r in rows)
    fp = sum((not r.true_transformer_fault) and r.predicted_transformer_fault for r in rows)
    tn = sum((not r.true_transformer_fault) and not r.predicted_transformer_fault for r in rows)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    balanced = None if recall is None or specificity is None else 0.5 * (recall + specificity)
    covered = sum(not r.abstained for r in rows)
    supported = sum(r.verification_status == "SUPPORTED" for r in rows)
    return {
        "case_count": len(rows),
        "positive_count": tp + fn,
        "negative_count": tn + fp,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced,
        "coverage": None if not rows else covered / len(rows),
        "abstention_rate": None if not rows else 1.0 - covered / len(rows),
        "verification_support_rate": None if not rows else supported / len(rows),
        "mean_runtime_seconds": None if not rows else float(np.mean([r.runtime_seconds for r in rows])),
    }


def run_sgah_transformer_benchmark(data_dir: str | Path, *, output_dir: str | Path | None = None) -> dict:
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

    train_x = np.stack([_feature_image(event).reshape(-1) for event in train_events])
    train_y = np.asarray([event.class_id == 4 for event in train_events], dtype=int)
    estimator = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=-1,
    )
    estimator.fit(train_x, train_y)
    classifier = _FixedSgahClassifier(estimator)

    def raw_classifier_accuracy(events: list[SgahEvent]) -> float | None:
        if not events:
            return None
        x = np.stack([_feature_image(event).reshape(-1) for event in events])
        y = np.asarray([event.class_id == 4 for event in events], dtype=int)
        return float(np.mean(estimator.predict(x) == y))

    cases: list[SgahCase] = []
    failures: list[dict] = []
    pipeline = TransformerDiagnosticPipeline()
    for event in test_events:
        try:
            started = perf_counter()
            result = pipeline.run(
                event.signal_matrix,
                sampling_rate_hz=1.0,
                sensor_positions=list(SGAH_CHANNEL_NAMES),
                trained_image_model=classifier,
                anomaly_threshold=0.0,
            )
            runtime = perf_counter() - started
            predicted = bool(
                result.decision == "diagnose"
                and any(h.label == "main_transformer_fault" for h in result.hypotheses)
            )
            verification_status = None
            if result.verification:
                verification_status = str(result.verification[0].status)
            cases.append(SgahCase(
                class_id=event.class_id,
                label=event.label,
                event_id=event.event_id,
                true_transformer_fault=event.class_id == 4,
                predicted_transformer_fault=predicted,
                decision=result.decision,
                abstained=bool(result.abstained),
                confidence=float(result.confidence),
                verification_status=verification_status,
                runtime_seconds=float(runtime),
            ))
        except Exception as exc:
            failures.append({
                "class_id": event.class_id,
                "event_id": event.event_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

    normal_rows = [row for row in cases if row.class_id == 5]
    competing_rows = [row for row in cases if row.class_id in (1, 2, 3)]
    normal_fp = sum(row.predicted_transformer_fault for row in normal_rows)
    competing_fp = sum(row.predicted_transformer_fault for row in competing_rows)
    summary = _binary_metrics(cases)
    summary.update({
        "normal_false_positive_rate": None if not normal_rows else normal_fp / len(normal_rows),
        "competing_fault_false_positive_rate": None if not competing_rows else competing_fp / len(competing_rows),
    })
    by_class = {
        str(class_id): {
            "label": SGAH_CLASSES[class_id],
            **_binary_metrics([row for row in cases if row.class_id == class_id]),
        }
        for class_id in SGAH_CLASSES
    }

    payload = {
        "source": "smartlab-hfut/SGAH-datasets (State Grid Corporation of China)",
        "protocol": {
            "upstream_commit": "bbe1020e3fade83f7861657bb3eaea41c25ec0c9",
            "samples_per_event": 100,
            "split": "Per class, contiguous whole-event 60% train / 20% development / 20% frozen test; no row-level splitting.",
            "positive_class": "main_transformer_fault (class 4)",
            "negative_controls": "classes 1,2,3 competing grid faults plus class 5 normal",
            "classifier": "RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight=balanced_subsample, random_state=0, threshold=0.5)",
            "label_isolation": "Frozen test labels are used only for benchmark scoring. Classifier fitting uses training events only.",
            "sampling_rate": "normalized 1.0 sample unit; physical sample rate is not required for this representation benchmark",
            "public_execution": "Final test decisions are emitted by TransformerDiagnosticPipeline in classifier-led electrical mode; legacy impulsiveness remains evidence rather than a mandatory gate.",
        },
        "manifest": manifest,
        "development": {
            "train_raw_classifier_accuracy": raw_classifier_accuracy(train_events),
            "dev_raw_classifier_accuracy": raw_classifier_accuracy(dev_events),
        },
        "summary": summary,
        "by_class": by_class,
        "cases": [asdict(row) for row in cases],
        "failures": failures,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "transformer_sgah_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if cases:
            with (out / "transformer_sgah_cases.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(cases[0]).keys()))
                writer.writeheader()
                writer.writerows(asdict(row) for row in cases)
    return payload
