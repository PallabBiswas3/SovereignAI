from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from ..datasets.bearing_lenze import LenzeRecord, load_lenze_metadata, load_lenze_record
from ..datasets.bearing_paderborn import (
    PaderbornRecord,
    discover_paderborn_records,
    load_paderborn_record,
    nonoverlapping_multichannel_windows,
    paderborn_specimen_split,
)
from ..tools.bearing_hybrid import (
    BearingHybridClassifier,
    BearingHybridConfig,
    bearing_fault_frequencies,
    normalize_waveform_channels,
    speed_assisted_physics_features,
)


def _metrics(truth: list[str], prediction: list[str]) -> dict[str, Any]:
    labels = sorted(set(truth) | set(prediction))
    return {
        "count": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "per_class_recall": dict(zip(labels, map(float, recall_score(truth, prediction, labels=labels, average=None, zero_division=0)))),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def _aggregate(group_ids: list[str], truth: list[str], prediction: list[str]) -> tuple[list[str], list[str]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group, actual, predicted in zip(group_ids, truth, prediction):
        grouped[group].append((actual, predicted))
    actual_rows: list[str] = []
    predicted_rows: list[str] = []
    for rows in grouped.values():
        actual_rows.append(Counter(row[0] for row in rows).most_common(1)[0][0])
        predicted_rows.append(Counter(row[1] for row in rows).most_common(1)[0][0])
    return actual_rows, predicted_rows


def _build_arrays(
    records,
    loader: Callable,
    channel_builder: Callable[[dict], np.ndarray],
    *,
    group_key: str,
    window_samples: int,
    waveform_samples: int,
    max_windows_per_record: int,
    envelope_channel: int | None,
    progress_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    waveforms: list[np.ndarray] = []
    physics: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    total_records = len(records)
    for record_index, record in enumerate(records, start=1):
        loaded = loader(record)
        channels = channel_builder(loaded)
        windows = nonoverlapping_multichannel_windows(channels, window_samples, max_windows_per_record)
        for window in windows:
            waveforms.append(normalize_waveform_channels(window, waveform_samples))
            physics.append(speed_assisted_physics_features(
                window,
                loaded["sampling_rate_hz"],
                loaded["shaft_rate_hz"],
                fault_frequencies=loaded.get("fault_frequencies") or bearing_fault_frequencies(loaded["shaft_rate_hz"]),
                envelope_channel=envelope_channel,
            ))
            labels.append(str(loaded["label"]))
            groups.append(str(loaded[group_key]))
        if progress_prefix and (record_index == 1 or record_index == total_records or record_index % max(1, total_records // 10) == 0):
            print(
                f"[{progress_prefix}] prepared {record_index}/{total_records} records "
                f"({len(waveforms)} windows)",
                flush=True,
            )
    if not waveforms:
        raise ValueError("no usable signal windows were produced")
    return np.stack(waveforms), np.stack(physics), labels, groups


def _fit_ablation(
    train_arrays,
    test_arrays,
    *,
    modes: tuple[str, ...],
    epochs: int,
    waveform_samples: int,
    seed: int,
    model_dir: Path | None,
    progress_prefix: str = "bearing",
) -> dict[str, Any]:
    train_w, train_p, train_y, _ = train_arrays
    test_w, test_p, test_y, test_groups = test_arrays
    results: dict[str, Any] = {}
    for mode in modes:
        started = perf_counter()
        print(
            f"[{progress_prefix}:{mode}] training on {len(train_y)} windows; "
            f"testing on {len(test_y)} windows; epochs={epochs}",
            flush=True,
        )

        def show_epoch(row: dict) -> None:
            print(
                f"[{progress_prefix}:{mode}] epoch {row['epoch']}/{row['epochs']} "
                f"loss={row['loss']:.5f} time={row['epoch_seconds']:.1f}s "
                f"eta={row['eta_seconds']:.0f}s",
                flush=True,
            )

        classifier = BearingHybridClassifier(BearingHybridConfig(
            mode=mode, waveform_samples=waveform_samples, epochs=epochs, seed=seed,
        )).fit(train_w, train_p, train_y, progress_callback=show_epoch)
        prediction = classifier.predict(test_w, test_p).tolist()
        record_truth, record_prediction = _aggregate(test_groups, test_y, prediction)
        record_metrics = _metrics(record_truth, record_prediction)
        artifact = None
        if model_dir is not None:
            artifact = str(classifier.save(model_dir / f"bearing_{mode}.pt"))
        results[mode] = {
            "window_metrics": _metrics(test_y, prediction),
            "record_metrics": record_metrics,
            "fit_and_predict_seconds": perf_counter() - started,
            "model_artifact": artifact,
        }
        print(
            f"[{progress_prefix}:{mode}] complete "
            f"accuracy={record_metrics['accuracy']:.4f} "
            f"balanced_accuracy={record_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={record_metrics['macro_f1']:.4f}",
            flush=True,
        )
    return results


def run_paderborn_hybrid_benchmark(
    data_dir: str | Path,
    *,
    include_combined: bool = False,
    window_samples: int = 8192,
    waveform_samples: int = 4096,
    max_windows_per_record: int = 3,
    epochs: int = 15,
    seed: int = 17,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    records = discover_paderborn_records(data_dir, include_combined=include_combined)
    if not records:
        raise FileNotFoundError(f"no reviewed Paderborn .mat files found below {data_dir}")
    train, test = paderborn_specimen_split(records)
    if not train or not test:
        raise ValueError("Paderborn benchmark needs at least two bearing specimens per class")
    builder = lambda loaded: np.vstack([loaded["vibration"], loaded["current"]])
    train_arrays = _build_arrays(
        train, load_paderborn_record, builder, group_key="bearing_id", window_samples=window_samples,
        waveform_samples=waveform_samples, max_windows_per_record=max_windows_per_record, envelope_channel=0,
        progress_prefix="paderborn:train",
    )
    test_arrays = _build_arrays(
        test, load_paderborn_record, builder, group_key="bearing_id", window_samples=window_samples,
        waveform_samples=waveform_samples, max_windows_per_record=max_windows_per_record, envelope_channel=0,
        progress_prefix="paderborn:test",
    )
    out = None if output_dir is None else Path(output_dir)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "Paderborn University Bearing Dataset",
        "protocol": "bearing-specimen holdout; no window from a test bearing appears in training",
        "modalities": ["vibration", "motor_current", "speed_assisted_physics"],
        "train_bearings": sorted({row.bearing_id for row in train}),
        "test_bearings": sorted({row.bearing_id for row in test}),
        "configuration": {
            "window_samples": window_samples, "waveform_samples": waveform_samples,
            "max_windows_per_record": max_windows_per_record, "epochs": epochs, "seed": seed,
            "include_combined": include_combined,
        },
        "ablation": _fit_ablation(
            train_arrays, test_arrays, modes=("physics", "cnn", "fusion"), epochs=epochs,
            waveform_samples=waveform_samples, seed=seed, model_dir=out,
            progress_prefix="paderborn",
        ),
    }
    if out is not None:
        (out / "bearing_paderborn_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _lenze_condition_split(records: list[LenzeRecord]) -> tuple[list[LenzeRecord], list[LenzeRecord]]:
    rpms = sorted({record.rpm for record in records})
    if len(rpms) < 2:
        raise ValueError("Lenze validation needs at least two RPM operating conditions")
    held_out = rpms[-1]
    return [row for row in records if row.rpm != held_out], [row for row in records if row.rpm == held_out]


def run_lenze_drive_validation(
    dataset_root: str | Path,
    *,
    window_samples: int = 8192,
    waveform_samples: int = 4096,
    max_windows_per_record: int = 4,
    epochs: int = 15,
    seed: int = 17,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    records = [row for row in load_lenze_metadata(root / "Meta_Data.xlsx") if row.path.exists()]
    if not records:
        raise FileNotFoundError(f"Lenze Data/*.mat files are not present below {root}")
    train, test = _lenze_condition_split(records)
    out = None if output_dir is None else Path(output_dir)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    channel_variants = {
        "current_only": lambda loaded: np.vstack([np.zeros_like(loaded["current"]), loaded["current"]]),
        "speed_only": lambda loaded: np.vstack([loaded["speed"], np.zeros_like(loaded["speed"])]),
        "current_plus_speed": lambda loaded: np.vstack([loaded["speed"], loaded["current"]]),
    }
    for name, builder in channel_variants.items():
        train_arrays = _build_arrays(
            train, load_lenze_record, builder, group_key="record_id", window_samples=window_samples,
            waveform_samples=waveform_samples, max_windows_per_record=max_windows_per_record,
            envelope_channel=None,
            progress_prefix=f"lenze:{name}:train",
        )
        test_arrays = _build_arrays(
            test, load_lenze_record, builder, group_key="record_id", window_samples=window_samples,
            waveform_samples=waveform_samples, max_windows_per_record=max_windows_per_record,
            envelope_channel=None,
            progress_prefix=f"lenze:{name}:test",
        )
        variants[name] = _fit_ablation(
            train_arrays, test_arrays, modes=("cnn", "fusion"), epochs=epochs,
            waveform_samples=waveform_samples, seed=seed,
            model_dir=None if out is None else out / name,
            progress_prefix=f"lenze:{name}",
        )
    payload = {
        "dataset": "Lenze Motor Bearing Fault Dataset",
        "protocol": "held-out maximum-RPM operating condition; drive/encoder validation only",
        "held_out_rpm": max(row.rpm for row in records),
        "configuration": {
            "window_samples": window_samples, "waveform_samples": waveform_samples,
            "max_windows_per_record": max_windows_per_record, "epochs": epochs, "seed": seed,
        },
        "variants": variants,
    }
    if out is not None:
        (out / "bearing_lenze_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
