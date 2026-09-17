#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from adaptivefact.baselines import (
    LLMJudgeBaseline,
    NoVerificationBaseline,
    RetrievalNLIBaseline,
    SelfConsistencyBaseline,
)
from adaptivefact.benchmark.experiment import ExperimentConfig, run_experiment
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.generation.huggingface import HuggingFaceChatGenerator
from adaptivefact.verification.nli import TransformersNLIScorer
from adaptivefact.verification.retriever import TfidfRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 always-on baselines on RAGTruth")
    parser.add_argument("--config", default="configs/phase1_ragtruth.yaml")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    exp = raw["experiment"]
    ds = raw["dataset"]
    baseline_cfg = raw["baselines"]

    loader = RAGTruthLoader(ds["root"], include_low_quality=ds.get("include_low_quality", False))
    config = ExperimentConfig(
        name=exp["name"],
        split=exp.get("split", "test"),
        max_records=exp.get("max_records"),
        output_dir=exp.get("output_dir", "results"),
        tags={"phase": "1"},
    )

    components = {"no_verification": NoVerificationBaseline()}

    nli = None
    if baseline_cfg.get("retrieval_nli", {}).get("enabled", True) or baseline_cfg.get("self_consistency", {}).get("enabled", False):
        nli_cfg = raw["nli"]
        nli = TransformersNLIScorer(
            model_name=nli_cfg["model"],
            device=nli_cfg.get("device"),
            batch_size=nli_cfg.get("batch_size", 16),
            max_length=nli_cfg.get("max_length", 512),
        )

    if baseline_cfg.get("retrieval_nli", {}).get("enabled", True):
        cfg = baseline_cfg["retrieval_nli"]
        components["retrieval_nli"] = RetrievalNLIBaseline(
            nli=nli,
            retriever=TfidfRetriever(max_chunk_chars=cfg.get("max_chunk_chars", 900)),
            top_k=cfg.get("top_k", 2),
            unsupported_threshold=cfg.get("hallucination_threshold", 0.70),
            max_sentences=cfg.get("max_sentences", 20),
        )

    generator = None
    needs_generator = (
        baseline_cfg.get("llm_judge", {}).get("enabled", False)
        or baseline_cfg.get("self_consistency", {}).get("enabled", False)
    )
    if needs_generator:
        gen_cfg = raw["generator"]
        generator = HuggingFaceChatGenerator(
            gen_cfg["model"],
            device_map=gen_cfg.get("device_map", "auto"),
            torch_dtype=gen_cfg.get("torch_dtype", "auto"),
        )

    if baseline_cfg.get("llm_judge", {}).get("enabled", False):
        cfg = baseline_cfg["llm_judge"]
        components["llm_judge"] = LLMJudgeBaseline(
            generator,
            max_context_chars=cfg.get("max_context_chars", 12000),
            max_new_tokens=cfg.get("max_new_tokens", 120),
        )

    if baseline_cfg.get("self_consistency", {}).get("enabled", False):
        cfg = baseline_cfg["self_consistency"]
        components["self_consistency"] = SelfConsistencyBaseline(
            generator,
            nli,
            n_samples=cfg.get("n_samples", 3),
            temperature=cfg.get("temperature", 0.8),
            max_new_tokens=cfg.get("max_new_tokens", 256),
            max_context_chars=cfg.get("max_context_chars", 9000),
            hallucination_threshold=cfg.get("hallucination_threshold", 0.55),
            record_model_aliases=cfg.get("record_model_aliases", []),
            require_model_match=cfg.get("require_model_match", True),
        )

    run_experiment(config, loader, components)


if __name__ == "__main__":
    main()
