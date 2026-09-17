#!/usr/bin/env python3
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "batch_manifest.csv"

REQUIRED_TOP_LEVEL = {
    "id", "split", "batch_id", "source_family_id", "annotation_status",
    "difficulty", "tags", "interaction", "expected_action",
    "expected_categories", "expected_subtypes", "gold_claims",
    "gold_privacy_spans", "gold_bias",
    "expected_safe_response_behavior", "gold_rationale"
}

ALLOWED_ACTIONS = {"allow", "allow_with_warning", "redact", "human_review", "block"}
ALLOWED_CATEGORIES = {"privacy", "bias", "hallucination", "policy"}
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}
ALLOWED_BEHAVIOR = {
    "release_original", "release_with_warning", "redact_sensitive_spans",
    "withhold_for_review", "block_entire_response"
}

def pct(counter, key, total):
    return 100.0 * counter.get(key, 0) / total if total else 0.0

def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    all_ids = set()
    all_seeds = set()
    errors = []
    total_records = 0

    action_counts = Counter()
    profile_counts = Counter()
    difficulty_counts = Counter()
    split_counts = Counter()
    category_counts = Counter()

    # Manifest-level checks
    for row in rows:
        seed = int(row["seed"])
        if seed in all_seeds:
            errors.append(f"Duplicate seed in manifest: {seed}")
        all_seeds.add(seed)

    for row in rows:
        split = row["split"]
        batch_id = row["batch_id"]
        expected_count = int(row["count"])
        output = ROOT / row["output_file"]

        if not output.exists():
            errors.append(f"MISSING OUTPUT: {output.relative_to(ROOT)}")
            continue

        batch_count = 0
        with output.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                batch_count += 1
                total_records += 1
                try:
                    rec = json.loads(line)
                except Exception as e:
                    errors.append(f"{batch_id}:{line_no}: invalid JSON: {e}")
                    continue

                missing = REQUIRED_TOP_LEVEL - set(rec)
                if missing:
                    errors.append(f"{batch_id}:{line_no}: missing fields {sorted(missing)}")

                rid = rec.get("id")
                if rid in all_ids:
                    errors.append(f"{batch_id}:{line_no}: duplicate id {rid}")
                elif rid is not None:
                    all_ids.add(rid)

                if rec.get("split") != split:
                    errors.append(f"{batch_id}:{line_no}: split={rec.get('split')!r}, expected {split!r}")
                if rec.get("batch_id") != batch_id:
                    errors.append(f"{batch_id}:{line_no}: batch_id={rec.get('batch_id')!r}")

                act = rec.get("expected_action")
                if act not in ALLOWED_ACTIONS:
                    errors.append(f"{batch_id}:{line_no}: invalid expected_action={act!r}")
                else:
                    action_counts[act] += 1

                diff = rec.get("difficulty")
                if diff not in ALLOWED_DIFFICULTY:
                    errors.append(f"{batch_id}:{line_no}: invalid difficulty={diff!r}")
                else:
                    difficulty_counts[diff] += 1

                beh = rec.get("expected_safe_response_behavior")
                if beh not in ALLOWED_BEHAVIOR:
                    errors.append(f"{batch_id}:{line_no}: invalid safe behavior={beh!r}")

                cats = rec.get("expected_categories", [])
                if not isinstance(cats, list):
                    errors.append(f"{batch_id}:{line_no}: expected_categories must be a list")
                else:
                    bad = set(cats) - ALLOWED_CATEGORIES
                    if bad:
                        errors.append(f"{batch_id}:{line_no}: invalid categories {sorted(bad)}")
                    category_counts.update(cats)

                inter = rec.get("interaction", {})
                if not isinstance(inter, dict):
                    errors.append(f"{batch_id}:{line_no}: interaction must be an object")
                    inter = {}
                profile = inter.get("profile")
                if profile not in {"customer_support", "internal_assistant", "regulated_decision_support"}:
                    errors.append(f"{batch_id}:{line_no}: invalid profile={profile!r}")
                else:
                    profile_counts[profile] += 1

                # Gold-claim invariants from the prompt.
                response = inter.get("response", "")
                context = inter.get("context")
                for ci, claim in enumerate(rec.get("gold_claims", [])):
                    text = claim.get("text", "")
                    evidence = claim.get("evidence")
                    status = claim.get("status")
                    if text and text not in response:
                        errors.append(f"{batch_id}:{line_no}: claim[{ci}] not verbatim in response")
                    if evidence is not None:
                        if not isinstance(context, str) or evidence not in context:
                            errors.append(f"{batch_id}:{line_no}: evidence[{ci}] not verbatim in context")
                    if status not in {"supported", "contradicted", "unknown"}:
                        errors.append(f"{batch_id}:{line_no}: invalid claim status={status!r}")

                split_counts[split] += 1

        if batch_count != expected_count:
            errors.append(f"{batch_id}: {batch_count} records, expected {expected_count}")

    expected_total = sum(int(r["count"]) for r in rows)

    print("=" * 72)
    print("CONTROLPLANE DATASET VALIDATION")
    print("=" * 72)
    print(f"Manifest batches : {len(rows)}")
    print(f"Unique seeds     : {len(all_seeds)}")
    print(f"Expected records : {expected_total}")
    print(f"Observed records : {total_records}")
    print(f"Unique record IDs: {len(all_ids)}")
    print()

    print("Records by split:")
    for k in ["development", "validation", "test", "stress"]:
        print(f"  {k:12s} {split_counts[k]:6d}")
    print()

    if total_records:
        print("Profiles:")
        for k in ["customer_support", "internal_assistant", "regulated_decision_support"]:
            print(f"  {k:28s} {profile_counts[k]:6d} ({pct(profile_counts,k,total_records):5.1f}%)")
        print("Expected target: 35% / 35% / 30%")
        print()

        print("Expected actions:")
        for k in ["allow", "allow_with_warning", "redact", "human_review", "block"]:
            print(f"  {k:22s} {action_counts[k]:6d} ({pct(action_counts,k,total_records):5.1f}%)")
        print("Expected target: 25% / 20% / 15% / 30% / 10%")
        print()

        print("Difficulty:")
        for k in ["easy", "medium", "hard"]:
            print(f"  {k:8s} {difficulty_counts[k]:6d} ({pct(difficulty_counts,k,total_records):5.1f}%)")
        print("Expected target: 30% / 40% / 30%")
        print()

    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for e in errors[:100]:
            print(" -", e)
        if len(errors) > 100:
            print(f" ... and {len(errors)-100} more")
        sys.exit(1)
    else:
        print("PASS: no structural or invariant violations detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
