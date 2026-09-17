from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from adaptivefact.data.schema import ResponseRecord, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig, ClaimExtractor
from adaptivefact.verification.deterministic import DeterministicVerifier, DeterministicVerifierConfig


@dataclass
class Phase5Config:
    extraction: ClaimExtractionConfig
    verifier: DeterministicVerifierConfig


class Phase5Pipeline:
    def __init__(
        self,
        extraction_config: ClaimExtractionConfig | None = None,
        verifier_config: DeterministicVerifierConfig | None = None,
    ) -> None:
        self.extractor = ClaimExtractor(extraction_config)
        self.verifier = DeterministicVerifier(verifier_config)

    def process(self, record: ResponseRecord) -> tuple[ResponseRecord, dict[str, float]]:
        total_start = perf_counter()

        extraction_start = perf_counter()
        claims = self.extractor.extract(record)
        extraction_ms = (perf_counter() - extraction_start) * 1000.0

        verification_start = perf_counter()
        verified = [self.verifier.verify(claim, record.context) for claim in claims]
        verification_ms = (perf_counter() - verification_start) * 1000.0

        record.atomic_claims = verified
        total_ms = (perf_counter() - total_start) * 1000.0
        return record, {
            "extraction_ms": extraction_ms,
            "verification_ms": verification_ms,
            "total_ms": total_ms,
        }


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


def _overlap(a: tuple[int, int] | None, b: tuple[int, int]) -> bool:
    if a is None:
        return False
    return max(a[0], b[0]) < min(a[1], b[1])


def summarize_phase5(records: list[ResponseRecord], timings: list[dict[str, float]]) -> dict:
    claims = [claim for record in records for claim in record.atomic_claims]
    type_counts = Counter(claim.type.value for claim in claims)
    status_counts = Counter(claim.status.value for claim in claims)
    method_counts = Counter(claim.verifier_used or "none" for claim in claims)

    resolved = sum(
        claim.status in {VerificationStatus.SUPPORTED, VerificationStatus.CONTRADICTED}
        for claim in claims
    )
    unresolved = sum(claim.status == VerificationStatus.UNKNOWN for claim in claims)

    gt_spans = [span for record in records for span in record.ground_truth_spans]
    covered_spans = 0
    conflict_spans = 0
    detected_conflict_spans = 0
    baseless_spans = 0
    baseless_unknown = 0

    predicted_conflicts = 0
    predicted_conflicts_aligned = 0
    predicted_supported = 0
    supported_overlaps_hallucination = 0

    for record in records:
        for span in record.ground_truth_spans:
            span_range = (span.start, span.end)
            overlapping = [claim for claim in record.atomic_claims if _overlap(claim.span, span_range)]
            if overlapping:
                covered_spans += 1
            if span.normalized_type == VerificationStatus.CONTRADICTED:
                conflict_spans += 1
                if any(claim.status == VerificationStatus.CONTRADICTED for claim in overlapping):
                    detected_conflict_spans += 1
            elif span.normalized_type == VerificationStatus.UNKNOWN:
                baseless_spans += 1
                if any(claim.status == VerificationStatus.UNKNOWN for claim in overlapping):
                    baseless_unknown += 1

        hallucination_ranges = [(span.start, span.end) for span in record.ground_truth_spans]
        conflict_ranges = [
            (span.start, span.end)
            for span in record.ground_truth_spans
            if span.normalized_type == VerificationStatus.CONTRADICTED
        ]
        for claim in record.atomic_claims:
            if claim.status == VerificationStatus.CONTRADICTED:
                predicted_conflicts += 1
                if any(_overlap(claim.span, span_range) for span_range in conflict_ranges):
                    predicted_conflicts_aligned += 1
            if claim.status == VerificationStatus.SUPPORTED:
                predicted_supported += 1
                if any(_overlap(claim.span, span_range) for span_range in hallucination_ranges):
                    supported_overlaps_hallucination += 1

    extraction_latencies = [item["extraction_ms"] for item in timings]
    verification_latencies = [item["verification_ms"] for item in timings]
    total_latencies = [item["total_ms"] for item in timings]
    claim_latencies = [claim.latency_ms for claim in claims if claim.latency_ms is not None]

    return {
        "n_records": len(records),
        "n_claims": len(claims),
        "avg_claims_per_response": len(claims) / max(1, len(records)),
        "claim_type_counts": dict(type_counts),
        "status_counts": dict(status_counts),
        "status_fractions": {key: value / max(1, len(claims)) for key, value in status_counts.items()},
        "verifier_method_counts": dict(method_counts),
        "deterministic_resolution_rate": resolved / max(1, len(claims)),
        "phase6_unresolved_rate": unresolved / max(1, len(claims)),
        "span_diagnostics": {
            "n_ground_truth_hallucination_spans": len(gt_spans),
            "hallucination_span_claim_coverage": covered_spans / max(1, len(gt_spans)),
            "n_conflict_spans": conflict_spans,
            "conflict_span_detection_recall": detected_conflict_spans / max(1, conflict_spans),
            "n_baseless_spans": baseless_spans,
            "baseless_span_unknown_rate": baseless_unknown / max(1, baseless_spans),
            "predicted_contradictions": predicted_conflicts,
            "predicted_contradiction_conflict_alignment": predicted_conflicts_aligned / max(1, predicted_conflicts),
            "predicted_supported": predicted_supported,
            "supported_claim_hallucination_overlap_rate": supported_overlaps_hallucination / max(1, predicted_supported),
            "note": (
                "These are span-alignment diagnostics, not atomic-claim ground-truth accuracy. "
                "RAGTruth hallucination spans are localization annotations, not gold claim boundaries."
            ),
        },
        "latency_ms": {
            "extraction_per_response": _percentiles(extraction_latencies),
            "deterministic_verification_per_response": _percentiles(verification_latencies),
            "phase5_total_per_response": _percentiles(total_latencies),
            "per_claim_verifier": _percentiles(claim_latencies),
        },
    }
