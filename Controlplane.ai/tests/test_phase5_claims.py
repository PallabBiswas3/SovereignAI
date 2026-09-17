from adaptivefact.data.schema import GenerationMetadata, ResponseRecord, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig, ClaimExtractor
from adaptivefact.extraction.numeric_date import extract_dates, extract_numbers
from adaptivefact.verification.deterministic import DeterministicVerifier, DeterministicVerifierConfig
from adaptivefact.verification.phase5 import Phase5Pipeline, summarize_phase5


def record(response: str, context: str) -> ResponseRecord:
    return ResponseRecord(
        id="x",
        dataset="synthetic",
        query="q",
        context=context,
        generated_response=response,
        generation_metadata=GenerationMetadata(task_type="QA"),
    )


def test_number_normalization():
    values = extract_numbers("Revenue was $1.5 billion and margin was 12.5%.")
    keys = {(x.normalized, x.kind) for x in values}
    assert ("1500000000", "currency") in keys
    assert ("12.5", "percent") in keys


def test_date_extraction():
    values = extract_dates("The launch happened on January 5, 2024.")
    assert any(value.normalized == "2024-01-05" for value in values)


def test_claim_extractor_preserves_offsets():
    r = record("Revenue was $10 million. It launched in 2024.", "")
    claims = ClaimExtractor(ClaimExtractionConfig()).extract(r)
    assert len(claims) == 2
    assert r.generated_response[claims[0].span[0]:claims[0].span[1]].startswith("Revenue")


def test_numeric_exact_support():
    r = record("Revenue was $10 million.", "The company reported revenue of $10 million in the period.")
    claim = ClaimExtractor().extract(r)[0]
    verified = DeterministicVerifier().verify(claim, r.context)
    assert verified.status == VerificationStatus.SUPPORTED


def test_numeric_conflict_becomes_phase6_candidate():
    r = record("Revenue was $10 million.", "The company reported revenue of $12 million in the period.")
    claim = ClaimExtractor().extract(r)[0]
    verified = DeterministicVerifier().verify(claim, r.context)
    assert verified.status == VerificationStatus.UNKNOWN
    assert verified.verifier_used == "deterministic_numeric_conflict_candidate"
    assert verified.evidence


def test_unrelated_number_does_not_create_conflict():
    r = record("Revenue was $10 million.", "The company employs 12 million users worldwide.")
    claim = ClaimExtractor().extract(r)[0]
    verified = DeterministicVerifier().verify(claim, r.context)
    assert verified.status == VerificationStatus.UNKNOWN


def test_missing_entity_is_unknown_not_false():
    r = record("Alice Johnson founded Example Labs.", "The supplied context discusses a different topic.")
    claim = ClaimExtractor().extract(r)[0]
    verified = DeterministicVerifier().verify(claim, r.context)
    assert verified.status == VerificationStatus.UNKNOWN


def test_phase5_summary_runs():
    r = record("Revenue was $10 million.", "The company reported revenue of $10 million.")
    pipeline = Phase5Pipeline()
    enriched, timing = pipeline.process(r)
    summary = summarize_phase5([enriched], [timing])
    assert summary["n_records"] == 1
    assert summary["n_claims"] == 1
    assert summary["deterministic_resolution_rate"] == 1.0
