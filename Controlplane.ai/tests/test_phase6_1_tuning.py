from adaptivefact.data.schema import (
    Claim,
    ClaimType,
    HallucinationSpan,
    ResponseRecord,
    VerificationStatus,
)
from adaptivefact.verification.nli import NLIScores
from adaptivefact.verification.phase6 import Phase6NLIConfig
from adaptivefact.verification.phase61_tuning import (
    ScoredClaim,
    ScoredEvidence,
    SweepSelectionConfig,
    decide_scored_claim,
    evaluate_thresholds,
    select_operating_point,
    sweep_thresholds,
)


def scored(entailment: float, contradiction: float, neutral: float = 0.0):
    return ScoredClaim(
        record_id="r",
        claim_id="c",
        record_index=0,
        claim_index=0,
        evidence=[
            ScoredEvidence(
                text="evidence",
                retrieval_score=1.0,
                entailment=entailment,
                contradiction=contradiction,
                neutral=neutral,
            )
        ],
    )


def make_record(span_type: VerificationStatus | None = None) -> ResponseRecord:
    spans = []
    if span_type is not None:
        spans = [
            HallucinationSpan(
                start=0,
                end=20,
                text="Example claim text",
                label_type="synthetic",
                normalized_type=span_type,
            )
        ]
    return ResponseRecord(
        id="r",
        dataset="synthetic",
        query="q",
        context="evidence",
        generated_response="Example claim text.",
        atomic_claims=[
            Claim(
                id="c",
                text="Example claim text.",
                type=ClaimType.FACTUAL,
                span=(0, 19),
                status=VerificationStatus.UNKNOWN,
            )
        ],
        ground_truth_spans=spans,
    )


def test_decision_respects_threshold_and_margin():
    item = scored(0.88, 0.05)
    assert decide_scored_claim(item, Phase6NLIConfig(0.80, 0.80, 0.10)) == VerificationStatus.SUPPORTED
    assert decide_scored_claim(item, Phase6NLIConfig(0.90, 0.80, 0.10)) == VerificationStatus.UNKNOWN


def test_sweep_reuses_scores_without_model_calls():
    record = make_record(VerificationStatus.CONTRADICTED)
    item = scored(0.02, 0.93)
    rows = sweep_thresholds(
        [record],
        [item],
        entailment_thresholds=[0.80, 0.95],
        contradiction_thresholds=[0.80],
        decision_margins=[0.10, 0.20],
    )
    assert len(rows) == 4
    assert all(row["nli_contradicted"] == 1 for row in rows)


def test_evaluation_reports_alignment_proxy():
    record = make_record(VerificationStatus.CONTRADICTED)
    row = evaluate_thresholds(
        [record],
        [scored(0.02, 0.94)],
        Phase6NLIConfig(0.80, 0.80, 0.10),
    )
    assert row["conflict_span_detection_recall"] == 1.0
    assert row["nli_contradiction_conflict_alignment"] == 1.0


def test_selection_prefers_feasible_candidate():
    base = {
        "conflict_span_detection_recall": 0.2,
        "nli_resolution_rate": 0.4,
        "nli_supported_hallucination_overlap_rate": 0.02,
        "nli_contradiction_conflict_alignment": 0.2,
        "baseless_all_unknown_rate": 0.9,
        "nli_resolved": 40,
        "nli_contradiction_predictions": 10,
        "entailment_threshold": 0.9,
        "contradiction_threshold": 0.9,
        "decision_margin": 0.2,
    }
    safer = dict(base)
    safer["conflict_span_detection_recall"] = 0.3
    unsafe = dict(base)
    unsafe["nli_supported_hallucination_overlap_rate"] = 0.08
    selected, rule = select_operating_point(
        [base, safer, unsafe],
        SweepSelectionConfig(),
    )
    assert selected["conflict_span_detection_recall"] == 0.3
    assert rule.startswith("feasible")
