#!/usr/bin/env python3
"""Validate, clean, merge, and group-split generated ControlPlane NLI JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_TO_ID = {"contradiction": 0, "entailment": 1, "neutral": 2}
ALLOWED_PROFILES = {
    "customer_support",
    "internal_assistant",
    "regulated_decision_support",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {
    "id",
    "dataset_version",
    "split",
    "batch_id",
    "source_family_id",
    "variant_id",
    "profile",
    "domain",
    "difficulty",
    "premise",
    "hypothesis",
    "label_text",
    "label",
    "claim_type",
    "importance",
    "phenomena",
    "gold_evidence",
    "rationale",
    "annotation_status",
    "generator",
}

# These five rows have label=1, supporting evidence, and entailment rationales,
# but the generator accidentally emitted label_text="neutral".
KNOWN_LABEL_OVERRIDES = {
    "controlplane-nli-v1-train-b006-000084": ("entailment", 1),
    "controlplane-nli-v1-train-b006-000088": ("entailment", 1),
    "controlplane-nli-v1-train-b006-000092": ("entailment", 1),
    "controlplane-nli-v1-train-b006-000096": ("entailment", 1),
    "controlplane-nli-v1-train-b006-000100": ("entailment", 1),
}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL files found in {input_dir}")
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.name}:{line_number}: {exc}") from exc
                row["_source_file"] = path.name
                row["_source_line"] = line_number
                rows.append(row)
    return rows, [path.name for path in files]


def validate_and_clean(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    corrections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        location = f"{row.get('_source_file')}:{row.get('_source_line')}"
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"{location}: missing fields {missing}")
            continue

        row_id = str(row["id"])
        if row_id in seen_ids:
            errors.append(f"{location}: duplicate id {row_id}")
            continue
        seen_ids.add(row_id)

        if row_id in KNOWN_LABEL_OVERRIDES:
            expected_text, expected_id = KNOWN_LABEL_OVERRIDES[row_id]
            old = {"label_text": row["label_text"], "label": row["label"]}
            row["label_text"] = expected_text
            row["label"] = expected_id
            corrections.append(
                {
                    "id": row_id,
                    "old": old,
                    "new": {"label_text": expected_text, "label": expected_id},
                    "reason": "Numeric label, evidence, and rationale consistently indicate entailment.",
                }
            )

        label_text = row["label_text"]
        if label_text not in LABEL_TO_ID:
            errors.append(f"{location}: invalid label_text {label_text!r}")
            continue
        if row["label"] != LABEL_TO_ID[label_text]:
            errors.append(
                f"{location}: label mismatch {label_text!r} != {row['label']!r}"
            )
            continue
        if row["profile"] not in ALLOWED_PROFILES:
            errors.append(f"{location}: invalid profile {row['profile']!r}")
            continue
        if row["difficulty"] not in ALLOWED_DIFFICULTIES:
            errors.append(f"{location}: invalid difficulty {row['difficulty']!r}")
            continue
        if not isinstance(row["variant_id"], int) or not 1 <= row["variant_id"] <= 4:
            errors.append(f"{location}: invalid variant_id {row['variant_id']!r}")
            continue
        if not isinstance(row["premise"], str) or not row["premise"].strip():
            errors.append(f"{location}: empty premise")
            continue
        if not isinstance(row["hypothesis"], str) or not row["hypothesis"].strip():
            errors.append(f"{location}: empty hypothesis")
            continue
        if not isinstance(row["gold_evidence"], list):
            errors.append(f"{location}: gold_evidence must be a list")
            continue
        if any(not isinstance(item, str) or item not in row["premise"] for item in row["gold_evidence"]):
            errors.append(f"{location}: gold evidence is not verbatim in the premise")
            continue

        pair_key = (normalized(row["premise"]), normalized(row["hypothesis"]))
        if pair_key in seen_pairs:
            errors.append(f"{location}: duplicate normalized premise/hypothesis pair")
            continue
        seen_pairs.add(pair_key)

        row.pop("_source_file", None)
        row.pop("_source_line", None)
        cleaned.append(row)

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cleaned:
        families[row["source_family_id"]].append(row)
    for family_id, items in sorted(families.items()):
        variants = sorted(item["variant_id"] for item in items)
        if len(items) != 4 or variants != [1, 2, 3, 4]:
            errors.append(
                f"{family_id}: expected four variants [1,2,3,4], got {variants}"
            )
        profiles = {item["profile"] for item in items}
        difficulties = {item["difficulty"] for item in items}
        if len(profiles) != 1:
            errors.append(f"{family_id}: family spans profiles {sorted(profiles)}")
        if len(difficulties) != 1:
            errors.append(f"{family_id}: family spans difficulties {sorted(difficulties)}")

    return cleaned, corrections, errors


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def choose_validation_families(
    rows: list[dict[str, Any]],
    *,
    validation_families: int,
    seed: int,
    iterations: int = 50_000,
) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_family_id"]].append(row)
    family_ids = sorted(grouped)
    if not 1 <= validation_families < len(family_ids):
        raise ValueError("validation_families must leave at least one family in each split")

    target_fraction = validation_families / len(family_ids)
    fields = ("label_text", "profile", "difficulty", "batch_id")
    targets = {
        field: {
            key: value * target_fraction
            for key, value in Counter(str(row[field]) for row in rows).items()
        }
        for field in fields
    }

    def score(selection: list[str]) -> float:
        selected_rows = [row for family_id in selection for row in grouped[family_id]]
        weights = {"label_text": 4.0, "profile": 2.0, "difficulty": 1.5, "batch_id": 1.0}
        total = 0.0
        for field in fields:
            counts = Counter(str(row[field]) for row in selected_rows)
            for key, target in targets[field].items():
                total += weights[field] * abs(counts.get(key, 0) - target)
        return total

    rng = random.Random(seed)
    best = family_ids[:validation_families]
    best_score = score(best)
    for _ in range(iterations):
        candidate = rng.sample(family_ids, validation_families)
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
            if best_score == 0:
                break
    return set(best)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_compact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "premise",
        "hypothesis",
        "label",
        "label_text",
        "source_family_id",
        "profile",
        "difficulty",
        "domain",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/controlplane/nli_finetune_v1/raw",
    )
    parser.add_argument(
        "--output-dir",
        default="data/controlplane/nli_finetune_v1/processed",
    )
    parser.add_argument("--validation-families", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, source_files = load_rows(input_dir)
    cleaned, corrections, errors = validate_and_clean(raw_rows)
    if errors:
        error_path = output_dir / "validation_errors.json"
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s); see {error_path}"
        )

    validation_family_ids = choose_validation_families(
        cleaned,
        validation_families=args.validation_families,
        seed=args.seed,
    )
    train_rows = [row for row in cleaned if row["source_family_id"] not in validation_family_ids]
    validation_rows = [row for row in cleaned if row["source_family_id"] in validation_family_ids]
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)

    write_jsonl(output_dir / "all_600_cleaned.jsonl", cleaned)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_compact_csv(output_dir / "train.csv", train_rows)
    write_compact_csv(output_dir / "validation.csv", validation_rows)

    summary = {
        "dataset_version": "controlplane-nli-v1",
        "annotation_status": "synthetic_candidate_unreviewed",
        "seed": args.seed,
        "source_files": source_files,
        "raw_rows": len(raw_rows),
        "accepted_rows": len(cleaned),
        "rejected_rows": len(raw_rows) - len(cleaned),
        "corrections": corrections,
        "all": {
            "rows": len(cleaned),
            "families": len({row["source_family_id"] for row in cleaned}),
            "labels": distribution(cleaned, "label_text"),
            "profiles": distribution(cleaned, "profile"),
            "difficulties": distribution(cleaned, "difficulty"),
            "batches": distribution(cleaned, "batch_id"),
        },
        "train": {
            "rows": len(train_rows),
            "families": len({row["source_family_id"] for row in train_rows}),
            "labels": distribution(train_rows, "label_text"),
            "profiles": distribution(train_rows, "profile"),
            "difficulties": distribution(train_rows, "difficulty"),
            "batches": distribution(train_rows, "batch_id"),
        },
        "validation": {
            "rows": len(validation_rows),
            "families": len(validation_family_ids),
            "labels": distribution(validation_rows, "label_text"),
            "profiles": distribution(validation_rows, "profile"),
            "difficulties": distribution(validation_rows, "difficulty"),
            "batches": distribution(validation_rows, "batch_id"),
        },
        "family_overlap": len(
            {row["source_family_id"] for row in train_rows}
            & {row["source_family_id"] for row in validation_rows}
        ),
        "label_mapping": LABEL_TO_ID,
        "notes": [
            "All examples remain synthetic_candidate_unreviewed.",
            "Five internally inconsistent batch-6 label_text values were corrected using their numeric labels, evidence, and rationales.",
            "The split is source-family separated; no family occurs in both training and validation.",
            "This structural audit does not replace human semantic label review.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
