#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


REPLACEMENTS = [
    {"Priya": "Amina", "priya@example.com": "amina@example.org", "2026": "2027"},
    {"Priya": "Elena", "priya@example.com": "elena@example.net", "$18": "$21"},
    {"Northstar": "Bluehaven", "Tesla": "Orion Motors", "$500": "$600"},
    {"Maria Chen": "Noah Williams", "Paris": "Lyon", "$900": "$950"},
    {"customer@example.com": "member@example.org", "2025": "2024"},
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replace_strings(value, replacements: dict[str, str]):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


def variants(base: list[dict], count: int, split: str) -> list[dict]:
    output = []
    for index in range(count):
        source = copy.deepcopy(base[index % len(base)])
        source_id = source["id"]
        source = replace_strings(source, REPLACEMENTS[index % len(REPLACEMENTS)])
        source["id"] = f"{split}-{index + 1:04d}-{source_id}"
        source["split"] = split
        source["annotation_status"] = "candidate_unreviewed"
        source["source_template_id"] = source_id
        source["variant_index"] = index + 1
        output.append(source)
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-candidate benchmark splits")
    parser.add_argument("--source", default="data/controlplane/evaluation_v1.jsonl")
    parser.add_argument("--output-dir", default="data/controlplane/benchmark_v1")
    parser.add_argument("--development", type=int, default=200)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=400)
    args = parser.parse_args()

    base = load_jsonl(Path(args.source))
    output_dir = Path(args.output_dir)
    development = variants(base, args.development, "development")
    validation = variants(list(reversed(base)), args.validation, "validation")
    test_labeled = variants(base[::2] + base[1::2], args.test, "test")

    write_jsonl(output_dir / "development_candidates.jsonl", development)
    write_jsonl(output_dir / "validation_candidates.jsonl", validation)

    label_keys = {
        "expected_action",
        "expected_categories",
        "expected_subtypes",
        "gold_claims",
        "gold_privacy_spans",
        "gold_bias",
    }
    test_inputs = [
        {key: value for key, value in row.items() if key not in label_keys}
        for row in test_labeled
    ]
    test_labels = [
        {"id": row["id"], **{key: row[key] for key in label_keys if key in row}}
        for row in test_labeled
    ]
    write_jsonl(output_dir / "test_inputs_blinded.jsonl", test_inputs)
    write_jsonl(output_dir / "reviewer_private" / "test_candidate_labels.jsonl", test_labels)
    manifest = {
        "status": "candidate_unreviewed",
        "source": args.source,
        "counts": {
            "development": len(development),
            "validation": len(validation),
            "test": len(test_inputs),
        },
        "warning": "Synthetic variants require independent human review and adjudication before use as gold labels.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
