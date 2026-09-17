from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
import hashlib
import json

from adaptivefact.data.schema import ResponseRecord, VerificationStatus
from adaptivefact.verification.evidence import ContextTfidfIndex, EvidenceRetrieverConfig, merge_seed_evidence
from adaptivefact.verification.nli import NLIScorer, NLIScores
from adaptivefact.verification.phase6 import Phase6NLIConfig, decide_nli_evidence


@dataclass
class ScoredEvidence:
    text: str
    retrieval_score: float
    entailment: float
    contradiction: float
    neutral: float


@dataclass
class ScoredClaim:
    record_id: str
    claim_id: str
    record_index: int
    claim_index: int
    evidence: list[ScoredEvidence]


@dataclass
class SweepSelectionConfig:
    min_nli_resolved_claims: int = 25
    max_nli_supported_hallucination_overlap_rate: float = 0.03
    min_nli_contradiction_conflict_alignment: float = 0.10
    min_baseless_all_unknown_rate: float = 0.80


def score_unknown_claims(
    records: list[ResponseRecord],
    nli: NLIScorer,
    retrieval_config: EvidenceRetrieverConfig,
) -> tuple[list[ScoredClaim], dict[str, Any]]:
    premises: list[str] = []
    hypotheses: list[str] = []
    pair_metadata: list[tuple[int, int, float]] = []
    claim_meta: list[tuple[int, int, int, int]] = []
    no_evidence_claims = 0

    for record_index, record in enumerate(records):
        evidence_index = ContextTfidfIndex(record.context, retrieval_config)
        for claim_index, claim in enumerate(record.atomic_claims):
            if claim.status != VerificationStatus.UNKNOWN:
                continue

            retrieved = evidence_index.retrieve(claim.text, top_k=retrieval_config.top_k)
            seed = _seed_evidence(claim)
            evidence = merge_seed_evidence(seed, retrieved, top_k=retrieval_config.top_k)
            if not evidence:
                no_evidence_claims += 1
                continue

            start = len(premises)
            for item in evidence:
                premises.append(item.text)
                hypotheses.append(claim.text)
                pair_metadata.append((record_index, claim_index, float(item.score)))
            end = len(premises)
            claim_meta.append((record_index, claim_index, start, end))

    scores = nli.score(premises, hypotheses) if premises else []
    if len(scores) != len(premises):
        raise ValueError("NLI scorer returned a different number of scores than input pairs")

    scored_claims: list[ScoredClaim] = []
    for record_index, claim_index, start, end in claim_meta:
        record = records[record_index]
        claim = record.atomic_claims[claim_index]
        evidence_rows: list[ScoredEvidence] = []
        for pair_index in range(start, end):
            score = scores[pair_index]
            _, _, retrieval_score = pair_metadata[pair_index]
            evidence_rows.append(
                ScoredEvidence(
                    text=premises[pair_index],
                    retrieval_score=retrieval_score,
                    entailment=float(score.entailment),
                    contradiction=float(score.contradiction),
                    neutral=float(score.neutral),
                )
            )
        scored_claims.append(
            ScoredClaim(
                record_id=record.id,
                claim_id=claim.id,
                record_index=record_index,
                claim_index=claim_index,
                evidence=evidence_rows,
            )
        )

    return scored_claims, {
        "nli_pairs": len(premises),
        "nli_scored_claims": len(scored_claims),
        "no_evidence_claims": no_evidence_claims,
    }


def decide_scored_claim(scored: ScoredClaim, config: Phase6NLIConfig) -> VerificationStatus:
    if not scored.evidence:
        return VerificationStatus.UNKNOWN
    status, _, _, _ = decide_nli_evidence(
        [
            NLIScores(
                entailment=item.entailment,
                contradiction=item.contradiction,
                neutral=item.neutral,
            )
            for item in scored.evidence
        ],
        [item.retrieval_score for item in scored.evidence],
        config,
    )
    return status


def evaluate_thresholds(
    records: list[ResponseRecord],
    scored_claims: list[ScoredClaim],
    config: Phase6NLIConfig,
) -> dict[str, Any]:
    score_map = {(item.record_index, item.claim_index): item for item in scored_claims}
    status_map: dict[tuple[int, int], VerificationStatus] = {}
    nli_method_map: dict[tuple[int, int], str] = {}

    phase5_unknown = 0
    nli_supported = 0
    nli_contradicted = 0
    nli_unknown = 0
    final_counts = Counter()

    for record_index, record in enumerate(records):
        for claim_index, claim in enumerate(record.atomic_claims):
            key = (record_index, claim_index)
            if claim.status != VerificationStatus.UNKNOWN:
                status = claim.status
            else:
                phase5_unknown += 1
                scored = score_map.get(key)
                if scored is None:
                    status = VerificationStatus.UNKNOWN
                    nli_method_map[key] = "no_evidence"
                else:
                    status = decide_scored_claim(scored, config)
                    nli_method_map[key] = "nli"

                if status == VerificationStatus.SUPPORTED:
                    nli_supported += 1
                elif status == VerificationStatus.CONTRADICTED:
                    nli_contradicted += 1
                else:
                    nli_unknown += 1

            status_map[key] = status
            final_counts[status.value] += 1

    conflict_spans = 0
    detected_conflict_spans = 0
    baseless_spans = 0
    baseless_all_unknown = 0
    nli_supported_predictions = 0
    nli_supported_overlaps_hallucination = 0
    nli_contradiction_predictions = 0
    nli_contradiction_aligned = 0

    for record_index, record in enumerate(records):
        hallucination_ranges = [(span.start, span.end) for span in record.ground_truth_spans]
        conflict_ranges = [
            (span.start, span.end)
            for span in record.ground_truth_spans
            if span.normalized_type == VerificationStatus.CONTRADICTED
        ]

        for span in record.ground_truth_spans:
            span_range = (span.start, span.end)
            overlapping_indices = [
                claim_index
                for claim_index, claim in enumerate(record.atomic_claims)
                if _overlap(claim.span, span_range)
            ]
            if span.normalized_type == VerificationStatus.CONTRADICTED:
                conflict_spans += 1
                if any(
                    status_map[(record_index, claim_index)] == VerificationStatus.CONTRADICTED
                    for claim_index in overlapping_indices
                ):
                    detected_conflict_spans += 1
            elif span.normalized_type == VerificationStatus.UNKNOWN:
                baseless_spans += 1
                if overlapping_indices and all(
                    status_map[(record_index, claim_index)] == VerificationStatus.UNKNOWN
                    for claim_index in overlapping_indices
                ):
                    baseless_all_unknown += 1

        for claim_index, claim in enumerate(record.atomic_claims):
            key = (record_index, claim_index)
            if nli_method_map.get(key) != "nli":
                continue
            status = status_map[key]
            if status == VerificationStatus.SUPPORTED:
                nli_supported_predictions += 1
                if any(_overlap(claim.span, span_range) for span_range in hallucination_ranges):
                    nli_supported_overlaps_hallucination += 1
            elif status == VerificationStatus.CONTRADICTED:
                nli_contradiction_predictions += 1
                if any(_overlap(claim.span, span_range) for span_range in conflict_ranges):
                    nli_contradiction_aligned += 1

    nli_resolved = nli_supported + nli_contradicted
    return {
        "entailment_threshold": float(config.entailment_threshold),
        "contradiction_threshold": float(config.contradiction_threshold),
        "decision_margin": float(config.decision_margin),
        "phase5_unknown_claims": phase5_unknown,
        "nli_supported": nli_supported,
        "nli_contradicted": nli_contradicted,
        "nli_unknown": nli_unknown,
        "nli_resolved": nli_resolved,
        "nli_resolution_rate": nli_resolved / max(1, phase5_unknown),
        "final_status_counts": dict(final_counts),
        "conflict_span_detection_recall": detected_conflict_spans / max(1, conflict_spans),
        "n_conflict_spans": conflict_spans,
        "baseless_all_unknown_rate": baseless_all_unknown / max(1, baseless_spans),
        "n_baseless_spans": baseless_spans,
        "nli_supported_hallucination_overlap_rate": (
            nli_supported_overlaps_hallucination / max(1, nli_supported_predictions)
        ),
        "nli_supported_hallucination_overlap_count": nli_supported_overlaps_hallucination,
        "nli_supported_predictions": nli_supported_predictions,
        "nli_contradiction_conflict_alignment": (
            nli_contradiction_aligned / max(1, nli_contradiction_predictions)
        ),
        "nli_contradiction_aligned_count": nli_contradiction_aligned,
        "nli_contradiction_predictions": nli_contradiction_predictions,
    }


def sweep_thresholds(
    records: list[ResponseRecord],
    scored_claims: list[ScoredClaim],
    *,
    entailment_thresholds: list[float],
    contradiction_thresholds: list[float],
    decision_margins: list[float],
) -> list[dict[str, Any]]:
    results = []
    for entailment_threshold, contradiction_threshold, decision_margin in product(
        entailment_thresholds,
        contradiction_thresholds,
        decision_margins,
    ):
        results.append(
            evaluate_thresholds(
                records,
                scored_claims,
                Phase6NLIConfig(
                    entailment_threshold=float(entailment_threshold),
                    contradiction_threshold=float(contradiction_threshold),
                    decision_margin=float(decision_margin),
                ),
            )
        )
    return results


def select_operating_point(
    rows: list[dict[str, Any]],
    config: SweepSelectionConfig,
) -> tuple[dict[str, Any], str]:
    if not rows:
        raise ValueError("Threshold sweep produced no candidates")

    feasible = [row for row in rows if _is_feasible(row, config)]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["conflict_span_detection_recall"],
                row["nli_resolution_rate"],
                -row["nli_supported_hallucination_overlap_rate"],
                row["nli_contradiction_conflict_alignment"],
            ),
        )
        return selected, "feasible_safety_constraints_then_max_conflict_recall_and_resolution"

    selected = min(
        rows,
        key=lambda row: (
            _safety_penalty(row, config),
            -row["conflict_span_detection_recall"],
            -row["nli_resolution_rate"],
            row["entailment_threshold"] + row["contradiction_threshold"],
        ),
    )
    return selected, "fallback_minimum_safety_penalty_no_candidate_met_all_constraints"


def save_score_cache(
    path: str | Path,
    *,
    records: list[ResponseRecord],
    scored_claims: list[ScoredClaim],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "version": 2,
        "fingerprint": build_cache_fingerprint(records, metadata),
        "metadata": metadata,
        "records": [record.model_dump(mode="json") for record in records],
        "scored_claims": [
            {
                "record_id": item.record_id,
                "claim_id": item.claim_id,
                "record_index": item.record_index,
                "claim_index": item.claim_index,
                "evidence": [e.__dict__ for e in item.evidence],
            }
            for item in scored_claims
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_score_cache(path: str | Path) -> tuple[list[ResponseRecord], list[ScoredClaim], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [ResponseRecord.model_validate(item) for item in payload["records"]]
    scored_claims = []
    for item in payload["scored_claims"]:
        scored_claims.append(
            ScoredClaim(
                record_id=item["record_id"],
                claim_id=item["claim_id"],
                record_index=int(item["record_index"]),
                claim_index=int(item["claim_index"]),
                evidence=[ScoredEvidence(**e) for e in item["evidence"]],
            )
        )
    return records, scored_claims, payload.get("metadata", {})


def build_cache_fingerprint(records: list[ResponseRecord], metadata: dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "record_ids": [record.id for record in records],
            "metadata": metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _is_feasible(row: dict[str, Any], config: SweepSelectionConfig) -> bool:
    contradiction_ok = (
        row["nli_contradiction_predictions"] == 0
        or row["nli_contradiction_conflict_alignment"]
        >= config.min_nli_contradiction_conflict_alignment
    )
    return (
        row["nli_resolved"] >= config.min_nli_resolved_claims
        and row["nli_supported_hallucination_overlap_rate"]
        <= config.max_nli_supported_hallucination_overlap_rate
        and contradiction_ok
        and row["baseless_all_unknown_rate"] >= config.min_baseless_all_unknown_rate
    )


def _safety_penalty(row: dict[str, Any], config: SweepSelectionConfig) -> float:
    support_excess = max(
        0.0,
        row["nli_supported_hallucination_overlap_rate"]
        - config.max_nli_supported_hallucination_overlap_rate,
    )
    contradiction_shortfall = 0.0
    if row["nli_contradiction_predictions"] > 0:
        contradiction_shortfall = max(
            0.0,
            config.min_nli_contradiction_conflict_alignment
            - row["nli_contradiction_conflict_alignment"],
        )
    baseless_shortfall = max(
        0.0,
        config.min_baseless_all_unknown_rate - row["baseless_all_unknown_rate"],
    )
    resolved_shortfall = max(
        0,
        config.min_nli_resolved_claims - row["nli_resolved"],
    ) / max(1, config.min_nli_resolved_claims)
    return support_excess + contradiction_shortfall + baseless_shortfall + resolved_shortfall


def _seed_evidence(claim) -> list[str]:
    if (claim.verifier_used or "") in {
        "deterministic_numeric_conflict_candidate",
        "deterministic_date_conflict_candidate",
    }:
        return list(claim.evidence)
    return []


def _overlap(a: tuple[int, int] | None, b: tuple[int, int]) -> bool:
    if a is None:
        return False
    return max(a[0], b[0]) < min(a[1], b[1])
