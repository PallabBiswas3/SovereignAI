#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from adaptivefact.agents import AgentVerificationConfig, BoundedVerificationAgent
from adaptivefact.agents.tools import ContextSearchTool, CorpusDocument, CorpusSearchTool
from adaptivefact.benchmark.metrics import (
    classification_metrics,
    latency_percentiles,
    remediation_metrics,
    selective_triage_metrics,
)
from adaptivefact.benchmark.sampling import sample_records, sampling_summary
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.pipeline import (
    AdaptivePipelineConfig,
    AdaptiveVerificationComponent,
    AdaptiveVerificationPipeline,
)
from adaptivefact.risk.calibration import ProbabilityCalibrator
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.model import RiskModelBundle
from adaptivefact.routing.policy import ThresholdPolicy
from adaptivefact.routing.router import AdaptiveRouter
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import TransformersNLIScorer
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6NLIConfig, Phase6Pipeline


def load_corpus(path: str | None) -> list[CorpusDocument]:
    if not path:
        return []
    documents = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                documents.append(CorpusDocument.model_validate_json(line))
    return documents


def json_safe(value):
    """Replace non-finite metric values so output remains valid JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7/8: run the integrated adaptive verification pipeline")
    parser.add_argument("--config", default="configs/phase7_adaptive.yaml")
    parser.add_argument("--max-records", type=int, default=None, help="Override the config limit for a smoke run")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    artifacts = cfg["artifacts"]

    manifest = json.loads(Path(artifacts["phase2_manifest"]).read_text(encoding="utf-8"))
    model = RiskModelBundle.load(artifacts["risk_model"])
    extractor = RiskFeatureExtractor(FeatureExtractionConfig(**manifest["feature_config"]))
    calibrator = ProbabilityCalibrator.load(artifacts["calibrator"])
    threshold_policy = ThresholdPolicy.from_json(artifacts["routing_thresholds"])
    router = AdaptiveRouter(
        model,
        extractor,
        calibrator,
        tau1=threshold_policy.tau1,
        tau2=threshold_policy.tau2,
    )

    nli_cfg = cfg["nli"]
    nli = TransformersNLIScorer(
        model_name=nli_cfg["model_name"],
        device=nli_cfg.get("device"),
        batch_size=int(nli_cfg.get("batch_size", 16)),
        max_length=int(nli_cfg.get("max_length", 512)),
    )
    retrieval_config = EvidenceRetrieverConfig(**cfg.get("retrieval", {}))
    tuned = json.loads(Path(artifacts["nli_thresholds"]).read_text(encoding="utf-8"))
    phase6 = Phase6Pipeline(
        nli,
        retrieval_config=retrieval_config,
        nli_config=Phase6NLIConfig(
            entailment_threshold=float(tuned["entailment_threshold"]),
            contradiction_threshold=float(tuned["contradiction_threshold"]),
            decision_margin=float(tuned["decision_margin"]),
            min_contradiction_retrieval_score=float(
                tuned.get("min_contradiction_retrieval_score", 0.10)
            ),
            evidence_conflict_threshold=float(
                tuned.get("evidence_conflict_threshold", 0.85)
            ),
            min_contradiction_evidence_count=int(
                tuned.get("min_contradiction_evidence_count", 2)
            ),
        ),
    )
    phase5 = Phase5Pipeline(
        ClaimExtractionConfig(**cfg.get("extraction", {})),
        DeterministicVerifierConfig(**cfg.get("deterministic_verification", {})),
    )

    agent_cfg = dict(cfg.get("agent", {}))
    enabled = bool(agent_cfg.pop("enabled", True))
    corpus_path = agent_cfg.pop("approved_corpus_path", None)
    agent = None
    if enabled:
        tools = [ContextSearchTool(retrieval_config=retrieval_config)]
        corpus = load_corpus(corpus_path)
        if corpus:
            tools.append(CorpusSearchTool(corpus))
        agent = BoundedVerificationAgent(
            nli,
            tools,
            AgentVerificationConfig(**agent_cfg),
        )

    pipeline = AdaptiveVerificationPipeline(
        router,
        phase5,
        phase6,
        agent=agent,
        config=AdaptivePipelineConfig(**cfg.get("dispatcher", {})),
    )
    component = AdaptiveVerificationComponent(pipeline)

    dataset_cfg = cfg["dataset"]
    all_records = RAGTruthLoader(
        dataset_cfg["root"],
        include_low_quality=bool(dataset_cfg.get("include_low_quality", False)),
    ).load_list(split=cfg["experiment"].get("split", "test"))
    max_records = args.max_records if args.max_records is not None else cfg["experiment"].get("max_records")
    sampling_cfg = cfg.get("sampling", {})
    sampling_method = str(sampling_cfg.get("method", "stratified_random"))
    sampling_seed = int(sampling_cfg.get("seed", 42))
    records = sample_records(
        all_records,
        int(max_records) if max_records is not None else None,
        method=sampling_method,
        seed=sampling_seed,
    )

    y_true = []
    y_pred = []
    y_score = []
    decisions = []
    remediation_actions = []
    latencies = []
    route_counts = Counter()
    final_counts = Counter()
    total_search = total_retrieval = total_agent_claims = 0
    agent_response_budget_exhaustions = agent_claims_skipped = 0

    for index, record in enumerate(records, start=1):
        result = component.run(record)
        y_true.append(int(record.ground_truth_label.value == "hallucinated"))
        y_pred.append(result.prediction)
        y_score.append(float(result.confidence or result.prediction))
        decisions.append(record.verification_metadata.final_decision.value)
        remediation_actions.append(
            record.metadata["adaptive_pipeline"]["remediation"]["action"]
        )
        latencies.append(float(record.timing_metadata.total_latency_ms or 0.0))
        route_counts[record.verification_metadata.route.value] += 1
        final_counts[record.verification_metadata.final_decision.value] += 1
        total_search += result.search_calls
        total_retrieval += result.retrieval_calls
        agent_runtime = record.metadata["adaptive_pipeline"]["agent"]
        total_agent_claims += int(agent_runtime["claims"])
        agent_response_budget_exhaustions += int(
            agent_runtime.get("response_budget_exhausted", False)
        )
        agent_claims_skipped += int(
            agent_runtime.get("claims_skipped_due_response_budget", 0)
        )
        if index % 25 == 0:
            print(f"Processed {index}/{len(records)} responses...")

    safety_triage = classification_metrics(y_true, y_pred, y_score)
    selective = selective_triage_metrics(y_true, decisions, y_score)
    summary = {
        "experiment": cfg["experiment"].get("name", "phase7_adaptive_verification"),
        "n_records": len(records),
        "sampling": {
            "method": sampling_method,
            "seed": sampling_seed,
            "available_records": len(all_records),
            **sampling_summary(records),
        },
        "metrics": {
            "selective": selective,
            "safety_triage": safety_triage,
            "automatic_remediation": remediation_metrics(
                y_true,
                remediation_actions,
            ),
        },
        "route_counts": dict(route_counts),
        "final_decision_counts": dict(final_counts),
        "latency_ms": latency_percentiles(latencies),
        "nli_claims": total_retrieval,
        "agent_claims": total_agent_claims,
        "agent_search_calls": total_search,
        "agent_response_budget_exhaustions": agent_response_budget_exhaustions,
        "agent_claims_skipped_due_response_budget": agent_claims_skipped,
        "config": cfg,
        "notes": [
            "SELECTIVE metrics treat UNKNOWN and MIXED as pre-remediation abstentions requiring intervention.",
            "SAFETY_TRIAGE metrics treat every non-SUPPORTED decision as a safety intervention.",
            "AUTOMATIC_REMEDIATION reports human-free handling separately from useful answers.",
            "Do not tune thresholds on this test split; tune on validation data and rerun test once.",
        ],
    }

    output_dir = Path(cfg["experiment"].get("output_dir", "results/phase7"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json_safe(summary)
    (output_dir / "phase7_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    sample_limit = int(cfg["experiment"].get("save_enriched_max_records", 25))
    with (output_dir / "enriched_records_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in records[:sample_limit]:
            stream.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nSaved Phase 7/8 outputs to {output_dir}")


if __name__ == "__main__":
    main()
