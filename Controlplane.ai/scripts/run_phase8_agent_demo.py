#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptivefact.agents import AgentVerificationConfig, BoundedVerificationAgent
from adaptivefact.agents.tools import CorpusDocument, CorpusSearchTool
from adaptivefact.data.schema import Claim, ClaimType, ResponseRecord, VerificationStatus
from adaptivefact.verification.nli import TransformersNLIScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled bounded-agent verification example")
    parser.add_argument("--corpus", default="data/controlplane/agent_corpus.jsonl")
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    documents = [
        CorpusDocument.model_validate_json(line)
        for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nli = TransformersNLIScorer(model_name=args.model, device=args.device)
    agent = BoundedVerificationAgent(
        nli,
        [CorpusSearchTool(documents)],
        AgentVerificationConfig(
            max_iterations=3,
            max_search_calls=3,
            max_nli_pairs=10,
            max_total_latency_ms=10_000,
            entailment_threshold=0.88,
            contradiction_threshold=0.88,
            decision_margin=0.20,
        ),
    )

    record = ResponseRecord(
        id="phase8-demo-response",
        dataset="controlled_agent_demo",
        query="What is the verified customer account balance?",
        context="The customer requested a callback. The balance must be checked in the account system.",
        generated_response="The customer account balance is $12,000.",
        metadata={"consequential": True},
    )
    claim = Claim(
        id="phase8-demo-claim",
        text="The customer account balance is $12,000.",
        type=ClaimType.NUMERIC,
        numbers=["$12,000"],
        status=VerificationStatus.UNKNOWN,
        risk_score=1.0,
    )

    result = agent.verify(record, claim)
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
