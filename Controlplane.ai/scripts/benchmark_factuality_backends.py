#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from adaptivefact.agents.schema import AgentVerificationConfig
from adaptivefact.agents.tools.context import ContextSearchTool
from adaptivefact.agents.verifier import BoundedVerificationAgent
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from controlplane import ControlPlane
from controlplane.evaluation import evaluate_scenarios, load_scenarios
from controlplane.support_scorers import AlignScoreSupportScorer, MiniCheckSupportScorer
from controlplane.verification import AdaptiveFactVerificationService, LazyTransformersNLIScorer


def build_service(backend: str, args) -> AdaptiveFactVerificationService:
    nli = LazyTransformersNLIScorer(args.nli_model, device=args.nli_device or None)
    retrieval = EvidenceRetrieverConfig(min_score=args.retrieval_min_score)
    agent = BoundedVerificationAgent(
        nli,
        [ContextSearchTool(retrieval_config=retrieval)],
        AgentVerificationConfig(),
    )
    support = None
    if backend == "minicheck":
        support = MiniCheckSupportScorer(
            model_name=args.minicheck_model,
            cache_dir=args.minicheck_cache,
        )
    elif backend in {"align", "alignscore"}:
        if not args.align_checkpoint:
            raise RuntimeError("--align-checkpoint is required for the Align backend")
        support = AlignScoreSupportScorer(
            checkpoint_path=args.align_checkpoint,
            model=args.align_model,
            device=args.align_device,
            batch_size=args.align_batch_size,
        )
    elif backend != "deberta":
        raise ValueError(f"unknown backend {backend!r}")
    return AdaptiveFactVerificationService(
        nli=nli,
        agent=agent,
        support_scorer=support,
        retrieval_config=retrieval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the same ControlPlane v3 validation workload with DeBERTa-only "
            "verification versus optional MiniCheck/Align support scorers. DeBERTa "
            "remains the contradiction-capable path in every configuration."
        )
    )
    parser.add_argument("--scenarios", default="data/controlplane/generated/completed_pack_v1/controlplane_batch_generation/validation.jsonl")
    parser.add_argument("--output", default="results/controlplane/factuality_v3_backend_comparison.json")
    parser.add_argument("--backends", default="deberta,minicheck,alignscore")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--nli-device", default=os.getenv("CONTROLPLANE_NLI_DEVICE", ""))
    parser.add_argument("--retrieval-min-score", type=float, default=0.05)
    parser.add_argument("--minicheck-model", default="roberta-large")
    parser.add_argument("--minicheck-cache", default="./ckpts/minicheck")
    parser.add_argument("--align-checkpoint", default=os.getenv("CONTROLPLANE_ALIGNSCORE_CHECKPOINT", ""))
    parser.add_argument("--align-model", default="roberta-base")
    parser.add_argument("--align-device", default="cpu")
    parser.add_argument("--align-batch-size", type=int, default=16)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    requested = [item.strip().lower() for item in args.backends.split(",") if item.strip()]
    comparison: dict[str, object] = {
        "scenario_file": args.scenarios,
        "n_scenarios": len(scenarios),
        "nli_model": args.nli_model,
        "design_note": (
            "DeBERTa is retained in all configurations for explicit entailment/contradiction. "
            "MiniCheck and Align are optional support/alignment scorers; low support alone is not contradiction."
        ),
        "backends": {},
    }

    for backend in requested:
        started = perf_counter()
        try:
            checker = ControlPlane(audit_enabled=False, verification_service=build_service(backend, args))
            metrics = evaluate_scenarios(checker, scenarios)
            comparison["backends"][backend] = {
                "status": "completed",
                "wall_seconds": perf_counter() - started,
                "metrics": metrics,
            }
            compact = {
                "action_accuracy": metrics["action_accuracy"],
                "unsafe_allow_rate": metrics["unsafe_allow_rate"],
                "over_intervention_rate": metrics["over_intervention_rate"],
                "human_review_rate": metrics["human_review_rate"],
                "verification_depth_counts": metrics.get("verification_depth_counts", {}),
                "factuality_state_counts": metrics.get("factuality_state_counts", {}),
                "latency_ms": metrics["latency_ms"],
            }
            print(json.dumps({backend: compact}, indent=2))
        except Exception as exc:
            comparison["backends"][backend] = {
                "status": "skipped_or_failed",
                "wall_seconds": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"{backend}: skipped/failed: {type(exc).__name__}: {exc}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nComparison written to {output}")


if __name__ == "__main__":
    main()
