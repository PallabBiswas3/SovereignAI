from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def write_json_report(result: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain(result), indent=2, default=str), encoding="utf-8")
    return path


def write_fault_table(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_ablation_table(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write one row per fault and ablation variant."""
    return write_fault_table(rows, path)


def write_markdown_summary(
    summary: dict[str, Any],
    table_rows: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    ablation_rows: Iterable[dict[str, Any]] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Tennessee Eastman benchmark summary", ""]
    lines.append("## Aggregate metrics")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    rows = list(table_rows)
    lines.append("## Per-fault detection table")
    lines.append("")
    if rows:
        fields = list(rows[0].keys())
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    ablation = list(ablation_rows or [])
    if ablation:
        lines.append("")
        lines.append("## Root-cause ablation table")
        lines.append("")
        fields = list(ablation[0].keys())
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in ablation:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_detection_plots(table_rows: Iterable[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Create lightweight publication-draft benchmark plots.

    Matplotlib is imported lazily so the core package remains usable in minimal
    environments where plotting is not needed.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in table_rows if int(r.get("fault_id", 0)) != 0]
    if not rows:
        return []

    fault_labels = [f"IDV({int(r['fault_id']):02d})" for r in rows]
    detection = [float(r.get("post_fault_detection_rate") or 0.0) for r in rows]
    false_alarm = [float(r.get("pre_fault_false_alarm_rate") or 0.0) for r in rows]
    delay = [float(r.get("detection_delay_samples") or 0.0) for r in rows]

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(rows))
    ax.bar(x, detection, label="Post-fault detection rate")
    ax.plot(list(x), false_alarm, marker="o", label="Pre-fault false-alarm rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fault_labels, rotation=60, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("TEP detection rate and false-alarm rate by fault")
    ax.legend()
    fig.tight_layout()
    p = output_dir / "tep_detection_false_alarm_by_fault.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(fault_labels, delay)
    ax.set_xticks(range(len(fault_labels)))
    ax.set_xticklabels(fault_labels, rotation=60, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("TEP first-detection delay by fault")
    fig.tight_layout()
    p = output_dir / "tep_detection_delay_by_fault.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    return paths


def write_ablation_plots(ablation_rows: Iterable[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Plot root-cause ablation performance by variant."""
    import matplotlib.pyplot as plt

    rows = list(ablation_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []

    variants = sorted({str(r["variant"]) for r in rows})
    root_acc = []
    top3_acc = []
    fault_acc = []
    false_conf = []
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        root_cases = [r for r in group if r.get("root_hit") is not None]
        top3_cases = [r for r in group if r.get("top3_root_hit") is not None]
        fault_cases = [r for r in group if r.get("fault_id_hit") is not None]
        false_cases = [r for r in group if r.get("false_confident") is not None]
        root_acc.append(float(np.mean([bool(r["root_hit"]) for r in root_cases])) if root_cases else 0.0)
        top3_acc.append(float(np.mean([bool(r["top3_root_hit"]) for r in top3_cases])) if top3_cases else 0.0)
        fault_acc.append(float(np.mean([bool(r["fault_id_hit"]) for r in fault_cases])) if fault_cases else 0.0)
        false_conf.append(float(np.mean([bool(r["false_confident"]) for r in false_cases])) if false_cases else 0.0)

    paths: list[Path] = []
    x = np.arange(len(variants))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5 * width, root_acc, width, label="Root top-1")
    ax.bar(x - 0.5 * width, top3_acc, width, label="Root top-3")
    ax.bar(x + 0.5 * width, fault_acc, width, label="Fault-ID")
    ax.bar(x + 1.5 * width, false_conf, width, label="False-confident")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("TEP root-cause ablation by evidence source")
    ax.legend()
    fig.tight_layout()
    p = output_dir / "tep_root_cause_ablation.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


def write_fault_confusion_matrix(
    ablation_rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    variant: str = "topology_catalog",
) -> list[Path]:
    """Write CSV and heatmap for true IDV vs predicted catalog ID."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in ablation_rows if str(r.get("variant")) == variant]
    rows = [r for r in rows if r.get("expected_roots")]
    if not rows:
        return []

    true_ids = sorted({int(r["fault_id"]) for r in rows})
    pred_labels = sorted({"none" if r.get("predicted_fault_id") is None else f"IDV({int(r['predicted_fault_id']):02d})" for r in rows})
    true_labels = [f"IDV({i:02d})" for i in true_ids]
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    true_index = {label: i for i, label in enumerate(true_labels)}
    pred_index = {label: i for i, label in enumerate(pred_labels)}
    for r in rows:
        t = f"IDV({int(r['fault_id']):02d})"
        p = "none" if r.get("predicted_fault_id") is None else f"IDV({int(r['predicted_fault_id']):02d})"
        matrix[true_index[t], pred_index[p]] += 1

    csv_path = output_dir / f"tep_fault_confusion_{variant}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + pred_labels)
        for i, label in enumerate(true_labels):
            writer.writerow([label] + [int(v) for v in matrix[i]])

    fig, ax = plt.subplots(figsize=(max(7, len(pred_labels) * 0.65), max(6, len(true_labels) * 0.35)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_xticklabels(pred_labels, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(true_labels)))
    ax.set_yticklabels(true_labels)
    ax.set_xlabel("Predicted catalog fault")
    ax.set_ylabel("True fault")
    ax.set_title(f"TEP fault-ID confusion matrix: {variant}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    fig.tight_layout()
    png_path = output_dir / f"tep_fault_confusion_{variant}.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return [csv_path, png_path]
