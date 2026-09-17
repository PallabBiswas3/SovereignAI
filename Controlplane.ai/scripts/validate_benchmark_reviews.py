#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LABEL_FIELDS = ("expected_action", "expected_categories", "expected_subtypes", "gold_claims")


def normalized_label(review: dict) -> str:
    payload = {field: review.get(field, [] if field != "expected_action" else None) for field in LABEL_FIELDS}
    for field in ("expected_categories", "expected_subtypes"):
        payload[field] = sorted(payload[field])
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate independent benchmark reviews")
    parser.add_argument("reviews", help="JSONL containing reviewer annotations")
    parser.add_argument("--output", default="results/controlplane/review_validation.json")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.reviews).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        missing = [field for field in ("id", "reviewer_id", *LABEL_FIELDS) if field not in row]
        if missing:
            raise ValueError(f"Review is missing required fields {missing}: {row.get('id', '<unknown>')}")
        grouped[row["id"]].append(row)

    insufficient = []
    disagreements = []
    agreed = []
    for item_id, reviews in grouped.items():
        reviewers = {row["reviewer_id"] for row in reviews}
        if len(reviewers) < 2:
            insufficient.append(item_id)
            continue
        labels = {normalized_label(row) for row in reviews}
        if len(labels) == 1:
            agreed.append(item_id)
        else:
            disagreements.append(item_id)

    report = {
        "items": len(grouped),
        "agreed": len(agreed),
        "insufficient_independent_reviews": insufficient,
        "requires_adjudication": disagreements,
        "gold_ready": not insufficient and not disagreements,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["gold_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
