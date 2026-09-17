#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.risk.training import group_validation_split
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import TransformersNLIScorer
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6NLIConfig
from adaptivefact.verification.phase61_tuning import (
    SweepSelectionConfig,
    load_score_cache,
    save_score_cache,
    score_unknown_claims,
    select_operating_point,
    sweep_thresholds,
)


def read_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_validation_records(cfg: dict):
    data_cfg = cfg.get("data", {})
    validation_cfg = cfg.get("validation", {})
    loader = RAGTruthLoader(
        data_cfg.get("root", "data/raw/ragtruth"),
        include_low_quality=bool(data_cfg.get("include_low_quality", False)),
    )

    train_records = list(loader.load(split="train"))
    test_records = list(loader.load(split="test"))
    test_source_ids = {record.source_id for record in test_records if record.source_id}
    source_safe_train = [
        record
        for record in train_records
        if not record.source_id or record.source_id not in test_source_ids
    ]

    split = group_validation_split(
        source_safe_train,
        validation_size=float(validation_cfg.get("validation_fraction", 0.10)),
        random_state=int(validation_cfg.get("random_state", 6061)),
        candidates=int(validation_cfg.get("split_candidates", 30)),
    )
    records = [source_safe_train[i] for i in split.validation_indices]

    rng = np.random.default_rng(int(validation_cfg.get("random_state", 6061)))
    order = rng.permutation(len(records))
    records = [records[int(i)] for i in order]

    max_records = validation_cfg.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]

    metadata = {
        "source_split": "train",
        "official_test_labels_used_for_tuning": False,
        "official_test_sources_excluded": True,
        "train_records_total": len(train_records),
        "official_test_records": len(test_records),
        "train_records_after_test_source_exclusion": len(source_safe_train),
        "validation_records": len(records),
        "unique_validation_sources": len({r.source_id or r.id for r in records}),
        "random_state": int(validation_cfg.get("random_state", 6061)),
    }
    return records, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6.1: tune NLI thresholds on source-safe RAGTruth validation data")
    parser.add_argument("--config", default="configs/phase6_1_tuning.yaml")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg.get("directory", "results/phase6_1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / output_cfg.get("score_cache_file", "nli_score_cache.json")
    nli_cfg = cfg.get("nli", {})
    cache_signature = {
        "schema_version": 2,
        "validation": cfg.get("validation", {}),
        "extraction": cfg.get("extraction", {}),
        "phase5_verification": cfg.get("phase5_verification", {}),
        "retrieval": cfg.get("retrieval", {}),
        "nli_model": {
            "model_name": nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-small"),
            "max_length": int(nli_cfg.get("max_length", 512)),
        },
    }

    use_cache = False
    if cache_path.exists() and not args.rebuild_cache:
        records, scored_claims, cache_metadata = load_score_cache(cache_path)
        use_cache = cache_metadata.get("configuration_signature") == cache_signature
        if use_cache:
            print(f"Reusing compatible Phase-6.1 NLI scores from {cache_path}")
            scoring_runtime = cache_metadata.get("scoring_runtime", {})
            validation_metadata = cache_metadata.get("validation", {})
        else:
            print("Ignoring incompatible Phase-6.1 cache because scoring configuration changed.")

    if not use_cache:
        records, validation_metadata = build_validation_records(cfg)

        phase5 = Phase5Pipeline(
            ClaimExtractionConfig(**cfg.get("extraction", {})),
            DeterministicVerifierConfig(**cfg.get("phase5_verification", {})),
        )
        for idx, record in enumerate(records, start=1):
            phase5.process(record)
            if idx % 100 == 0:
                print(f"Phase 5 preprocessing: {idx}/{len(records)} validation responses...")

        unknown_claims = sum(
            claim.status.value == "unknown"
            for record in records
            for claim in record.atomic_claims
        )
        print(
            f"Loading NLI model {nli_cfg.get('model_name', 'cross-encoder/nli-deberta-v3-small')} "
            f"for {unknown_claims} Phase-5 UNKNOWN validation claims..."
        )
        nli = TransformersNLIScorer(
            model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-small"),
            device=nli_cfg.get("device"),
            batch_size=int(nli_cfg.get("batch_size", 16)),
            max_length=int(nli_cfg.get("max_length", 512)),
        )
        retrieval_cfg = EvidenceRetrieverConfig(**cfg.get("retrieval", {}))
        scored_claims, scoring_runtime = score_unknown_claims(records, nli, retrieval_cfg)

        cache_metadata = {
            "configuration_signature": cache_signature,
            "validation": validation_metadata,
            "scoring_runtime": scoring_runtime,
            "retrieval": cfg.get("retrieval", {}),
            "nli_model": {
                "model_name": nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-small"),
                "max_length": int(nli_cfg.get("max_length", 512)),
            },
        }
        save_score_cache(
            cache_path,
            records=records,
            scored_claims=scored_claims,
            metadata=cache_metadata,
        )
        print(f"Saved reusable NLI score cache to {cache_path}")

    sweep_cfg = cfg.get("sweep", {})
    rows = sweep_thresholds(
        records,
        scored_claims,
        entailment_thresholds=[float(x) for x in sweep_cfg.get("entailment_thresholds", [0.72, 0.80, 0.85, 0.90, 0.95])],
        contradiction_thresholds=[float(x) for x in sweep_cfg.get("contradiction_thresholds", [0.72, 0.80, 0.85, 0.90, 0.95])],
        decision_margins=[float(x) for x in sweep_cfg.get("decision_margins", [0.10, 0.15, 0.20, 0.25])],
    )

    selection = SweepSelectionConfig(**cfg.get("selection", {}))
    selected, selection_rule = select_operating_point(rows, selection)

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row["nli_supported_hallucination_overlap_rate"],
            -row["nli_contradiction_conflict_alignment"],
            -row["baseless_all_unknown_rate"],
            -row["conflict_span_detection_recall"],
            -row["nli_resolution_rate"],
        ),
    )

    csv_path = out_dir / output_cfg.get("sweep_csv_file", "threshold_sweep.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": cfg.get("experiment", {}).get("name", "phase6_1_nli_threshold_tuning"),
        "methodology": {
            "tuning_split": "RAGTruth train-derived, source-safe validation subset",
            "official_test_used_for_threshold_selection": False,
            "note": "RAGTruth span labels are used as tuning diagnostics, not as gold atomic-claim labels.",
        },
        "validation": validation_metadata,
        "scoring_runtime": scoring_runtime,
        "sweep": {
            "candidate_count": len(rows),
            "entailment_thresholds": sweep_cfg.get("entailment_thresholds"),
            "contradiction_thresholds": sweep_cfg.get("contradiction_thresholds"),
            "decision_margins": sweep_cfg.get("decision_margins"),
        },
        "selection_constraints": cfg.get("selection", {}),
        "selection_rule": selection_rule,
        "selected": selected,
        "top_candidates_safety_sorted": rows_sorted[:10],
    }

    summary_path = out_dir / output_cfg.get("summary_file", "phase6_1_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    artifact_cfg = cfg.get("artifact", {})
    threshold_path = Path(artifact_cfg.get("thresholds_file", "models/phase6/selected_nli_thresholds.json"))
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_payload = {
        "entailment_threshold": selected["entailment_threshold"],
        "contradiction_threshold": selected["contradiction_threshold"],
        "decision_margin": selected["decision_margin"],
        "min_contradiction_retrieval_score": Phase6NLIConfig().min_contradiction_retrieval_score,
        "evidence_conflict_threshold": Phase6NLIConfig().evidence_conflict_threshold,
        "min_contradiction_evidence_count": Phase6NLIConfig().min_contradiction_evidence_count,
        "selection_rule": selection_rule,
        "validation_metrics": selected,
        "source": str(summary_path),
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved threshold sweep to {csv_path}")
    print(f"Saved Phase 6.1 summary to {summary_path}")
    print(f"Saved selected NLI thresholds to {threshold_path}")
    print("Phase 6 can now load these thresholds without tuning on the official test set.")


if __name__ == "__main__":
    main()
