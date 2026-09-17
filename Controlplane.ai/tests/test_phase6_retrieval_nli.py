from adaptivefact.data.schema import GenerationMetadata, ResponseRecord, VerificationStatus
from adaptivefact.verification.evidence import ContextTfidfIndex, EvidenceRetrieverConfig
from adaptivefact.verification.nli import NLIScorer, NLIScores
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import (
    Phase6NLIConfig,
    Phase6Pipeline,
    decide_nli_evidence,
    summarize_phase6,
)


class KeywordNLIScorer(NLIScorer):
    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        scores = []
        for premise, hypothesis in zip(premises, hypotheses):
            p = premise.lower()
            h = hypothesis.lower()
            if "joined tesla" in p and "founded tesla" in h:
                scores.append(NLIScores(entailment=0.03, contradiction=0.92, neutral=0.05))
            elif "paris is the capital" in p and "paris is the capital" in h:
                scores.append(NLIScores(entailment=0.95, contradiction=0.02, neutral=0.03))
            else:
                scores.append(NLIScores(entailment=0.20, contradiction=0.10, neutral=0.70))
        return scores


def make_record(response: str, context: str) -> ResponseRecord:
    return ResponseRecord(
        id="x",
        dataset="synthetic",
        query="q",
        context=context,
        generated_response=response,
        generation_metadata=GenerationMetadata(task_type="QA"),
    )


def test_context_index_retrieves_relevant_chunk():
    index = ContextTfidfIndex(
        "Paris is the capital of France. Berlin is the capital of Germany.",
        EvidenceRetrieverConfig(top_k=1, max_chunk_chars=45),
    )
    result = index.retrieve("Paris capital France", top_k=1)
    assert result
    assert "Paris" in result[0].text


def test_context_index_filters_zero_similarity_chunks():
    index = ContextTfidfIndex(
        "Paris is the capital of France.",
        EvidenceRetrieverConfig(top_k=3, min_score=0.05),
    )
    assert index.retrieve("quantum zucchini", top_k=3) == []


def test_low_relevance_chunk_cannot_confirm_contradiction():
    status, _, _, method = decide_nli_evidence(
        [NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01)],
        [0.02],
        Phase6NLIConfig(
            contradiction_threshold=0.90,
            min_contradiction_retrieval_score=0.10,
            min_contradiction_evidence_count=1,
        ),
    )
    assert status == VerificationStatus.UNKNOWN
    assert method == "nli_uncertain"


def test_disagreeing_high_confidence_evidence_remains_unknown():
    status, _, _, method = decide_nli_evidence(
        [
            NLIScores(entailment=0.96, contradiction=0.02, neutral=0.02),
            NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01),
        ],
        [0.70, 0.75],
        Phase6NLIConfig(evidence_conflict_threshold=0.85),
    )
    assert status == VerificationStatus.UNKNOWN
    assert method == "nli_conflicting_evidence"


def test_single_generic_contradiction_requires_corroboration():
    status, _, _, method = decide_nli_evidence(
        [NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01)],
        [0.75],
        Phase6NLIConfig(
            contradiction_threshold=0.90,
            min_contradiction_evidence_count=2,
        ),
    )
    assert status == VerificationStatus.UNKNOWN
    assert method == "nli_uncertain"


def test_two_relevant_contradictions_can_confirm_claim():
    status, _, _, method = decide_nli_evidence(
        [
            NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01),
            NLIScores(entailment=0.02, contradiction=0.96, neutral=0.02),
        ],
        [0.70, 0.65],
        Phase6NLIConfig(
            contradiction_threshold=0.90,
            min_contradiction_evidence_count=2,
        ),
    )
    assert status == VerificationStatus.CONTRADICTED
    assert method == "nli_contradiction"


def test_phase6_entails_unknown_entity_claim():
    record = make_record(
        "Paris is the capital of France.",
        "Paris is the capital of France. It is a major European city.",
    )
    Phase5Pipeline().process(record)
    assert record.atomic_claims[0].status == VerificationStatus.UNKNOWN

    pipeline = Phase6Pipeline(
        KeywordNLIScorer(),
        retrieval_config=EvidenceRetrieverConfig(top_k=2, max_chunk_chars=80),
        nli_config=Phase6NLIConfig(entailment_threshold=0.72, contradiction_threshold=0.72),
    )
    records, _ = pipeline.process_records([record])
    claim = records[0].atomic_claims[0]
    assert claim.status == VerificationStatus.SUPPORTED
    assert claim.verifier_used == "nli_entailment"


def test_phase6_can_confirm_contradiction():
    record = make_record(
        "Elon Musk founded Tesla.",
        "Elon Musk joined Tesla in 2004.",
    )
    Phase5Pipeline().process(record)
    pipeline = Phase6Pipeline(
        KeywordNLIScorer(),
        nli_config=Phase6NLIConfig(min_contradiction_evidence_count=1),
    )
    records, _ = pipeline.process_records([record])
    claim = records[0].atomic_claims[0]
    assert claim.status == VerificationStatus.CONTRADICTED
    assert claim.verifier_used == "nli_contradiction"


def test_phase6_keeps_uncertain_claim_unknown():
    record = make_record(
        "Alice Johnson founded Example Labs.",
        "Example Labs develops software products.",
    )
    Phase5Pipeline().process(record)
    pipeline = Phase6Pipeline(KeywordNLIScorer())
    records, _ = pipeline.process_records([record])
    assert records[0].atomic_claims[0].status == VerificationStatus.UNKNOWN


def test_phase6_summary_runs():
    record = make_record(
        "Paris is the capital of France.",
        "Paris is the capital of France.",
    )
    Phase5Pipeline().process(record)
    phase5_counts = {"unknown": 1}
    pipeline = Phase6Pipeline(KeywordNLIScorer())
    records, runtime = pipeline.process_records([record])
    summary = summarize_phase6(records, phase5_status_counts=phase5_counts, runtime=runtime)
    assert summary["n_records"] == 1
    assert summary["claims_sent_to_nli"] == 1
    assert summary["overall_claim_resolution_rate"] == 1.0
