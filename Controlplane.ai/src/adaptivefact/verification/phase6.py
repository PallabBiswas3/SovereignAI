from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from adaptivefact.data.schema import Claim, ResponseRecord, VerificationStatus
from adaptivefact.verification.evidence import (
    ContextTfidfIndex,
    EvidenceRetrieverConfig,
    merge_seed_evidence,
)
from adaptivefact.verification.nli import NLIScorer, NLIScores


@dataclass
class Phase6NLIConfig:
    entailment_threshold: float = 0.72
    contradiction_threshold: float = 0.72
    decision_margin: float = 0.10
    preserve_phase5_supported: bool = True
    min_contradiction_retrieval_score: float = 0.10
    evidence_conflict_threshold: float = 0.85
    min_contradiction_evidence_count: int = 2


@dataclass
class PendingClaim:
    record_index: int
    claim_index: int
    pair_start: int
    pair_end: int
    evidence_texts: list[str]
    evidence_scores: list[float]
    retrieval_ms: float


class Phase6Pipeline:
    """Retrieve evidence for Phase-5 UNKNOWN claims and verify them with NLI."""

    def __init__(
        self,
        nli: NLIScorer,
        *,
        retrieval_config: EvidenceRetrieverConfig | None = None,
        nli_config: Phase6NLIConfig | None = None,
    ) -> None:
        self.nli = nli
        self.retrieval_config = retrieval_config or EvidenceRetrieverConfig()
        self.nli_config = nli_config or Phase6NLIConfig()

    def process_records(
        self,
        records: list[ResponseRecord],
        *,
        claim_ids: set[str] | None = None,
    ) -> tuple[list[ResponseRecord], dict]:
        """Process UNKNOWN claims, optionally restricted to an explicit id set.

        The claim filter lets the Phase-7 dispatcher enforce different NLI
        budgets for lightweight and deep routes without mutating unselected
        claims or duplicating the Phase-6 implementation.
        """
        started = perf_counter()
        premises: list[str] = []
        hypotheses: list[str] = []
        pending: list[PendingClaim] = []
        retrieval_latencies: list[float] = []
        no_evidence_claims = 0

        for record_index, record in enumerate(records):
            index_started = perf_counter()
            evidence_index = ContextTfidfIndex(record.context, self.retrieval_config)
            index_ms = (perf_counter() - index_started) * 1000.0

            unresolved_claims = [
                (claim_index, claim)
                for claim_index, claim in enumerate(record.atomic_claims)
                if claim.status == VerificationStatus.UNKNOWN
                and (claim_ids is None or claim.id in claim_ids)
            ]

            if unresolved_claims:
                retrieval_latencies.append(index_ms)

            for claim_index, claim in unresolved_claims:
                retrieval_started = perf_counter()
                retrieved = evidence_index.retrieve(
                    claim.text,
                    top_k=self.retrieval_config.top_k,
                )
                seed = self._seed_evidence(claim)
                evidence = merge_seed_evidence(
                    seed,
                    retrieved,
                    top_k=self.retrieval_config.top_k,
                )
                retrieval_ms = (perf_counter() - retrieval_started) * 1000.0
                retrieval_latencies.append(retrieval_ms)

                if not evidence:
                    claim.verifier_used = "phase6_no_evidence"
                    no_evidence_claims += 1
                    continue

                pair_start = len(premises)
                for item in evidence:
                    premises.append(item.text)
                    hypotheses.append(claim.text)
                pair_end = len(premises)

                pending.append(
                    PendingClaim(
                        record_index=record_index,
                        claim_index=claim_index,
                        pair_start=pair_start,
                        pair_end=pair_end,
                        evidence_texts=[item.text for item in evidence],
                        evidence_scores=[float(item.score) for item in evidence],
                        retrieval_ms=retrieval_ms,
                    )
                )

        nli_started = perf_counter()
        scores = self.nli.score(premises, hypotheses) if premises else []
        nli_ms = (perf_counter() - nli_started) * 1000.0

        resolution_counts = Counter()
        for item in pending:
            claim = records[item.record_index].atomic_claims[item.claim_index]
            local_scores = scores[item.pair_start:item.pair_end]
            status, confidence, best_evidence, method, verification_scores = self._decide(
                local_scores,
                item.evidence_texts,
                item.evidence_scores,
            )
            previous_latency = claim.latency_ms or 0.0
            claim.status = status
            claim.confidence = confidence
            claim.evidence = best_evidence
            claim.verifier_used = method
            claim.verification_scores = verification_scores
            claim.latency_ms = previous_latency + item.retrieval_ms
            resolution_counts[status.value] += 1

        total_ms = (perf_counter() - started) * 1000.0
        return records, {
            "nli_pairs": len(premises),
            "nli_claims": len(pending),
            "no_evidence_claims": no_evidence_claims,
            "nli_resolution_counts": dict(resolution_counts),
            "retrieval_latency_ms": _percentiles(retrieval_latencies),
            "nli_batch_latency_ms": nli_ms,
            "phase6_total_ms": total_ms,
        }

    @staticmethod
    def _seed_evidence(claim: Claim) -> list[str]:
        method = claim.verifier_used or ""
        if method in {
            "deterministic_numeric_conflict_candidate",
            "deterministic_date_conflict_candidate",
        }:
            return list(claim.evidence)
        return []

    def _decide(
        self,
        scores: list[NLIScores],
        evidence_texts: list[str],
        evidence_scores: list[float],
    ) -> tuple[VerificationStatus, float | None, list[str], str, dict[str, float]]:
        status, confidence, evidence_index, method = decide_nli_evidence(
            scores,
            evidence_scores,
            self.nli_config,
        )
        best_evidence = [] if evidence_index is None else [evidence_texts[evidence_index]]
        verification_scores = {}
        if evidence_index is not None:
            selected = scores[evidence_index]
            verification_scores = {
                "entailment": float(selected.entailment),
                "contradiction": float(selected.contradiction),
                "neutral": float(selected.neutral),
            }
        return status, confidence, best_evidence, method, verification_scores


def decide_nli_evidence(
    scores: list[NLIScores],
    evidence_scores: list[float],
    config: Phase6NLIConfig,
) -> tuple[VerificationStatus, float | None, int | None, str]:
    """Make the same evidence-aware decision in runtime and threshold tuning."""
    if not scores:
        return VerificationStatus.UNKNOWN, None, None, "nli_no_scores"
    if len(scores) != len(evidence_scores):
        raise ValueError("NLI scores and evidence scores must have the same length")

    entail_index = max(range(len(scores)), key=lambda index: scores[index].entailment)
    contradiction_candidates = [
        index
        for index, retrieval_score in enumerate(evidence_scores)
        if retrieval_score >= config.min_contradiction_retrieval_score
    ]
    contra_index = (
        max(contradiction_candidates, key=lambda index: scores[index].contradiction)
        if contradiction_candidates
        else None
    )
    best_entailment = scores[entail_index].entailment
    best_contradiction = (
        scores[contra_index].contradiction if contra_index is not None else 0.0
    )
    corroborating_contradictions = [
        index
        for index in contradiction_candidates
        if scores[index].contradiction >= config.contradiction_threshold
    ]
    contradiction_corroborated = (
        len(corroborating_contradictions) >= config.min_contradiction_evidence_count
        # Phase-5 deterministic conflict candidates are inserted as seed
        # evidence with score 1.0 and may safely serve as the second signal.
        or any(evidence_scores[index] >= 0.999 for index in corroborating_contradictions)
    )

    if (
        contra_index is not None
        and contra_index != entail_index
        and best_entailment >= config.evidence_conflict_threshold
        and best_contradiction >= config.evidence_conflict_threshold
    ):
        return (
            VerificationStatus.UNKNOWN,
            float(max(best_entailment, best_contradiction)),
            contra_index,
            "nli_conflicting_evidence",
        )

    if (
        contra_index is not None
        and contradiction_corroborated
        and best_contradiction >= config.contradiction_threshold
        and best_contradiction - best_entailment >= config.decision_margin
    ):
        return (
            VerificationStatus.CONTRADICTED,
            float(best_contradiction),
            contra_index,
            "nli_contradiction",
        )

    if (
        best_entailment >= config.entailment_threshold
        and best_entailment - best_contradiction >= config.decision_margin
    ):
        return (
            VerificationStatus.SUPPORTED,
            float(best_entailment),
            entail_index,
            "nli_entailment",
        )

    best_index = max(
        range(len(scores)),
        key=lambda index: max(scores[index].entailment, scores[index].contradiction),
    )
    confidence = max(scores[best_index].entailment, scores[best_index].contradiction)
    return VerificationStatus.UNKNOWN, float(confidence), best_index, "nli_uncertain"


def summarize_phase6(
    records: list[ResponseRecord],
    *,
    phase5_status_counts: dict[str, int],
    runtime: dict,
) -> dict:
    claims = [claim for record in records for claim in record.atomic_claims]
    final_status_counts = Counter(claim.status.value for claim in claims)
    method_counts = Counter(claim.verifier_used or "none" for claim in claims)

    phase5_unknown = int(phase5_status_counts.get(VerificationStatus.UNKNOWN.value, 0))
    nli_resolved = sum(
        claim.verifier_used in {"nli_entailment", "nli_contradiction"}
        for claim in claims
    )
    nli_unknown = sum(claim.verifier_used in {"nli_uncertain", "phase6_no_evidence"} for claim in claims)

    gt_conflicts = 0
    gt_baseless = 0
    detected_conflicts = 0
    baseless_left_unknown = 0
    predicted_conflicts = 0
    aligned_predicted_conflicts = 0
    predicted_supported = 0
    supported_overlap_hallucination = 0

    for record in records:
        hallucination_ranges = [(span.start, span.end) for span in record.ground_truth_spans]
        conflict_ranges = [
            (span.start, span.end)
            for span in record.ground_truth_spans
            if span.normalized_type == VerificationStatus.CONTRADICTED
        ]

        for span in record.ground_truth_spans:
            span_range = (span.start, span.end)
            overlapping = [claim for claim in record.atomic_claims if _overlap(claim.span, span_range)]
            if span.normalized_type == VerificationStatus.CONTRADICTED:
                gt_conflicts += 1
                if any(claim.status == VerificationStatus.CONTRADICTED for claim in overlapping):
                    detected_conflicts += 1
            elif span.normalized_type == VerificationStatus.UNKNOWN:
                gt_baseless += 1
                if any(claim.status == VerificationStatus.UNKNOWN for claim in overlapping):
                    baseless_left_unknown += 1

        for claim in record.atomic_claims:
            if claim.status == VerificationStatus.CONTRADICTED:
                predicted_conflicts += 1
                if any(_overlap(claim.span, span_range) for span_range in conflict_ranges):
                    aligned_predicted_conflicts += 1
            elif claim.status == VerificationStatus.SUPPORTED:
                predicted_supported += 1
                if any(_overlap(claim.span, span_range) for span_range in hallucination_ranges):
                    supported_overlap_hallucination += 1

    return {
        "n_records": len(records),
        "n_claims": len(claims),
        "phase5_status_counts": phase5_status_counts,
        "final_status_counts": dict(final_status_counts),
        "final_status_fractions": {
            key: value / max(1, len(claims)) for key, value in final_status_counts.items()
        },
        "verifier_method_counts": dict(method_counts),
        "phase5_unknown_claims": phase5_unknown,
        "claims_sent_to_nli": int(runtime.get("nli_claims", 0)),
        "nli_pairs": int(runtime.get("nli_pairs", 0)),
        "no_evidence_claims": int(runtime.get("no_evidence_claims", 0)),
        "nli_resolution_rate_among_phase5_unknown": nli_resolved / max(1, phase5_unknown),
        "remaining_unknown_rate_among_phase5_unknown": nli_unknown / max(1, phase5_unknown),
        "overall_claim_resolution_rate": (
            final_status_counts.get(VerificationStatus.SUPPORTED.value, 0)
            + final_status_counts.get(VerificationStatus.CONTRADICTED.value, 0)
        ) / max(1, len(claims)),
        "span_diagnostics": {
            "n_conflict_spans": gt_conflicts,
            "conflict_span_detection_recall": detected_conflicts / max(1, gt_conflicts),
            "n_baseless_spans": gt_baseless,
            "baseless_span_unknown_rate": baseless_left_unknown / max(1, gt_baseless),
            "predicted_contradictions": predicted_conflicts,
            "predicted_contradiction_conflict_alignment": aligned_predicted_conflicts / max(1, predicted_conflicts),
            "predicted_supported": predicted_supported,
            "supported_claim_hallucination_overlap_rate": supported_overlap_hallucination / max(1, predicted_supported),
            "note": (
                "These are span-alignment diagnostics, not atomic-claim ground-truth accuracy. "
                "RAGTruth hallucination spans are localization annotations, not gold claim boundaries."
            ),
        },
        "latency_ms": {
            "retrieval": runtime.get("retrieval_latency_ms", {}),
            "nli_batch": float(runtime.get("nli_batch_latency_ms", 0.0)),
            "phase6_total": float(runtime.get("phase6_total_ms", 0.0)),
        },
    }


def _overlap(a: tuple[int, int] | None, b: tuple[int, int]) -> bool:
    if a is None:
        return False
    return max(a[0], b[0]) < min(a[1], b[1])


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }
