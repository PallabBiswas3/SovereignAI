#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import TransformersNLIScorer
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6NLIConfig, Phase6Pipeline, summarize_phase6


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: evidence retrieval + NLI for unresolved claims")
    parser.add_argument("--config", default="configs/phase6_ragtruth.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})

    loader = RAGTruthLoader(
        data_cfg.get("root", "data/raw/ragtruth"),
        include_low_quality=bool(data_cfg.get("include_low_quality", False)),
    )
    records = list(loader.load(split=exp_cfg.get("split", "test")))
    max_records = exp_cfg.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]

    phase5 = Phase5Pipeline(
        ClaimExtractionConfig(**cfg.get("extraction", {})),
        DeterministicVerifierConfig(**cfg.get("phase5_verification", {})),
    )

    phase5_timings = []
    for idx, record in enumerate(records, start=1):
        _, timing = phase5.process(record)
        phase5_timings.append(timing)
        if idx % 250 == 0:
            print(f"Phase 5 preprocessing: {idx}/{len(records)} responses...")

    phase5_status_counts = dict(
        Counter(
            claim.status.value
            for record in records
            for claim in record.atomic_claims
        )
    )

    nli_cfg = dict(cfg.get("nli", {}))
    thresholds_file = nli_cfg.get("thresholds_file")
    if thresholds_file:
        threshold_path = Path(thresholds_file)
        if threshold_path.exists():
            tuned = json.loads(threshold_path.read_text(encoding="utf-8"))
            nli_cfg["entailment_threshold"] = tuned["entailment_threshold"]
            nli_cfg["contradiction_threshold"] = tuned["contradiction_threshold"]
            nli_cfg["decision_margin"] = tuned["decision_margin"]
            nli_cfg["min_contradiction_retrieval_score"] = tuned.get(
                "min_contradiction_retrieval_score", 0.10
            )
            nli_cfg["evidence_conflict_threshold"] = tuned.get(
                "evidence_conflict_threshold", 0.85
            )
            nli_cfg["min_contradiction_evidence_count"] = tuned.get(
                "min_contradiction_evidence_count", 2
            )
            print(f"Loaded tuned NLI thresholds from {threshold_path}")
        else:
            print(f"Tuned threshold file not found at {threshold_path}; using thresholds from config.")

    print(
        f"Loading NLI model {nli_cfg.get('model_name', 'cross-encoder/nli-deberta-v3-small')} "
        f"for {phase5_status_counts.get('unknown', 0)} Phase-5 UNKNOWN claims..."
    )
    nli = TransformersNLIScorer(
        model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-small"),
        device=nli_cfg.get("device"),
        batch_size=int(nli_cfg.get("batch_size", 16)),
        max_length=int(nli_cfg.get("max_length", 512)),
    )

    retrieval = EvidenceRetrieverConfig(**cfg.get("retrieval", {}))
    decision = Phase6NLIConfig(
        entailment_threshold=float(nli_cfg.get("entailment_threshold", 0.72)),
        contradiction_threshold=float(nli_cfg.get("contradiction_threshold", 0.72)),
        decision_margin=float(nli_cfg.get("decision_margin", 0.10)),
        min_contradiction_retrieval_score=float(
            nli_cfg.get("min_contradiction_retrieval_score", 0.10)
        ),
        evidence_conflict_threshold=float(
            nli_cfg.get("evidence_conflict_threshold", 0.85)
        ),
        min_contradiction_evidence_count=int(
            nli_cfg.get("min_contradiction_evidence_count", 2)
        ),
    )
    phase6 = Phase6Pipeline(nli, retrieval_config=retrieval, nli_config=decision)

    processed, runtime = phase6.process_records(records)
    summary = summarize_phase6(
        processed,
        phase5_status_counts=phase5_status_counts,
        runtime=runtime,
    )
    summary["experiment"] = exp_cfg.get("name", "phase6")
    summary["phase5_preprocessing_latency_ms"] = {
        "mean_total_per_response": sum(x["total_ms"] for x in phase5_timings) / max(1, len(phase5_timings))
    }
    summary["config"] = {
        "retrieval": cfg.get("retrieval", {}),
        "nli": nli_cfg,
    }

    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg.get("directory", "results/phase6"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / output_cfg.get("summary_file", "phase6_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sample_limit = int(exp_cfg.get("save_enriched_max_records", 100))
    enriched_path = out_dir / output_cfg.get("enriched_file", "enriched_records_sample.jsonl")
    with enriched_path.open("w", encoding="utf-8") as f:
        for record in processed[:sample_limit]:
            f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved Phase 6 summary to {summary_path}")
    print(f"Saved enriched sample to {enriched_path}")


if __name__ == "__main__":
    main()
