#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import yaml

from adaptivefact.verification.nli import NLIScorer, TransformersNLIScorer
from adaptivefact.verification.openai_nli import OpenAIEvidenceJudge


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prediction(score) -> str:
    values = {
        "supported": score.entailment,
        "contradicted": score.contradiction,
        "unknown": score.neutral,
    }
    return max(values, key=values.get)


def build_scorer(spec: dict) -> NLIScorer:
    if spec["kind"] == "transformers_nli":
        return TransformersNLIScorer(spec["model_name"])
    if spec["kind"] == "openai_evidence_judge":
        return OpenAIEvidenceJudge(spec["model_name"], reasoning_effort=spec.get("reasoning_effort", "low"))
    if spec["kind"] == "hybrid":
        return HybridScorer(
            TransformersNLIScorer(spec["primary_model"]),
            OpenAIEvidenceJudge(spec["escalation_model"], reasoning_effort=spec.get("reasoning_effort", "low")),
        )
    raise ValueError(f"Unknown scorer kind: {spec['kind']}")


class HybridScorer(NLIScorer):
    def __init__(self, primary: NLIScorer, escalation: NLIScorer, *, confidence_threshold: float = 0.80) -> None:
        self.primary = primary
        self.escalation = escalation
        self.confidence_threshold = confidence_threshold

    def score(self, premises: list[str], hypotheses: list[str]):
        primary = self.primary.score(premises, hypotheses)
        output = list(primary)
        indexes = [
            index for index, item in enumerate(primary)
            if max(item.entailment, item.contradiction) < self.confidence_threshold
            or item.neutral >= max(item.entailment, item.contradiction)
        ]
        if indexes:
            escalated = self.escalation.score(
                [premises[index] for index in indexes],
                [hypotheses[index] for index in indexes],
            )
            for index, item in zip(indexes, escalated):
                output[index] = item
        return output


def evaluate(spec: dict, examples: list[tuple[str, str, str]]) -> dict:
    scorer = build_scorer(spec)
    premises = [item[0] for item in examples]
    hypotheses = [item[1] for item in examples]
    gold = [item[2] for item in examples]
    started = perf_counter()
    predicted = [prediction(item) for item in scorer.score(premises, hypotheses)]
    latency_ms = (perf_counter() - started) * 1000.0
    contradictions = [index for index, value in enumerate(predicted) if value == "contradicted"]
    gold_contradictions = [index for index, value in enumerate(gold) if value == "contradicted"]
    gold_supported = [index for index, value in enumerate(gold) if value == "supported"]
    unsafe = sum(predicted[index] == "supported" and gold[index] != "supported" for index in range(len(gold)))
    tp_contradiction = sum(gold[index] == "contradicted" for index in contradictions)
    supported_retained = sum(predicted[index] == "supported" for index in gold_supported)
    usage_source = scorer.escalation if isinstance(scorer, HybridScorer) else scorer
    input_tokens = int(getattr(usage_source, "input_tokens", 0))
    output_tokens = int(getattr(usage_source, "output_tokens", 0))
    cost = (
        input_tokens * float(spec.get("estimated_input_per_million", 0.0))
        + output_tokens * float(spec.get("estimated_output_per_million", 0.0))
    ) / 1_000_000
    return {
        "id": spec["id"],
        "model_name": spec.get("model_name") or spec.get("escalation_model"),
        "n": len(gold),
        "accuracy": sum(a == b for a, b in zip(gold, predicted)) / max(1, len(gold)),
        "unsafe_release_rate": unsafe / max(1, sum(value != "supported" for value in gold)),
        "contradiction_precision": tp_contradiction / max(1, len(contradictions)),
        "contradiction_recall": tp_contradiction / max(1, len(gold_contradictions)),
        "supported_answer_retention": supported_retained / max(1, len(gold_supported)),
        "unknown_rate": sum(value == "unknown" for value in predicted) / max(1, len(predicted)),
        "latency_ms": latency_ms,
        "estimated_cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evidence-verification models on frozen gold claims")
    parser.add_argument("--config", default="configs/model_comparison.yaml")
    parser.add_argument("--include-api-models", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = load_rows(Path(config["dataset"]))
    examples = []
    for row in rows:
        context = row["interaction"].get("context") or ""
        for claim in row.get("gold_claims", []):
            examples.append((context, claim["text"], claim["status"]))
            if len(examples) >= int(config.get("max_claims", 100)):
                break
        if len(examples) >= int(config.get("max_claims", 100)):
            break
    reports = []
    for spec in config["models"]:
        if not spec.get("enabled", False):
            continue
        if spec["kind"] in {"openai_evidence_judge", "hybrid"} and not args.include_api_models:
            continue
        reports.append(evaluate(spec, examples))
    output = {"dataset": config["dataset"], "n_claims": len(examples), "models": reports}
    path = Path(config["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
