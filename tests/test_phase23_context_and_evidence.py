from app.core.config import Settings
from app.evidence.context import ContextCompiler
from app.evidence.models import Calculation, Measurement, Rule, RuleSource, SupportStatus, Claim
from app.rag.retrieval import RetrievedChunk


def _candidate(
    index: int,
    text: str,
    score: float,
    *,
    file: str = "Maintenance_SOP.md",
    revision: str | None = "Rev 3",
    document_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        document_id=document_id or f"doc-{index}",
        text=text,
        score=score,
        source={
            "file": file,
            "page": index,
            "section": "7.4",
            "revision": revision,
            "document_hash": f"hash-{index}",
        },
        scores={"dense": score, "sparse": None, "fusion": score, "reranker": None},
        retrieval_methods=["dense"],
        access_scope=["internal"],
    )


def test_context_compiler_deduplicates_and_honors_budget() -> None:
    relevant = "Pump-102 vibration shall not exceed 6.0 mm/s RMS under normal operation."
    candidates = [_candidate(1, relevant, 1.0), _candidate(2, relevant, 0.99)]
    candidates.extend(
        _candidate(index, f"Unrelated maintenance paragraph {index} " * 8, 0.01)
        for index in range(3, 21)
    )
    settings = Settings(
        context_max_fraction_of_window=0.60,
        context_output_reserve_tokens=20,
        context_max_evidence_tokens=70,
        context_max_evidence_chunks=5,
    )
    compiled = ContextCompiler(settings).compile(
        task="Assess Pump-102 vibration limit",
        evidence=candidates,
        selected_model="general",
        context_window=200,
        execution_mode="STANDARD",
    )
    assert compiled.metrics.raw_candidate_count == 20
    assert compiled.metrics.deduplicated_chunks >= 1
    assert compiled.metrics.final_evidence_count <= 5
    assert compiled.metrics.compiled_context_tokens <= 100
    assert any("6.0 mm/s RMS" in source.text for source in compiled.evidence)
    assert compiled.fragments[0].source_id == compiled.evidence[0].id


def test_context_preserves_source_provenance_and_technical_values() -> None:
    compiled = ContextCompiler(Settings(context_max_evidence_tokens=200)).compile(
        task="Check PU-102 Section 7.4",
        evidence=[_candidate(1, "SOP-MNT-42 Section 7.4 sets PU-102 to 6.0 mm/s RMS.", 0.9)],
        selected_model="general",
        context_window=4096,
        execution_mode="FAST",
    )
    source = compiled.evidence[0]
    assert source.file == "Maintenance_SOP.md"
    assert source.page == 1
    assert source.section == "7.4"
    assert source.revision == "Rev 3"
    assert "PU-102" in compiled.fragments[0].technical_identifiers
    assert any("6.0 mm/s" in value for value in compiled.fragments[0].numerical_values)


def test_context_surfaces_conflicting_revisions() -> None:
    candidates = [
        _candidate(1, "Normal vibration shall not exceed 6.0 mm/s RMS.", 0.9, revision="Rev 2", document_id="rev2"),
        _candidate(2, "Normal vibration shall not exceed 5.5 mm/s RMS.", 0.95, revision="Rev 3", document_id="rev3"),
    ]
    compiled = ContextCompiler().compile(
        task="What is the current vibration limit?",
        evidence=candidates,
        selected_model="general",
        context_window=4096,
        execution_mode="DEEP",
    )
    assert len(compiled.conflicts) == 1
    assert compiled.conflicts[0].type == "DOCUMENT_REVISION_CONFLICT"
    assert compiled.conflicts[0].status == "REQUIRES_REVIEW"
    assert len(compiled.conflicts[0].sources) == 2


def test_structured_measurement_rule_calculation_and_claim_validate() -> None:
    measurement = Measurement(
        id="M1", asset_id="Pump-102", metric="vibration",
        original_value=8.2, original_unit="mm/s", source_id="E1", confidence=0.96,
    )
    rule = Rule(
        id="R1", metric="vibration", operator="<=", threshold=6.0, unit="mm/s",
        rule_type="normal_limit", source=RuleSource(source_id="E2", section="7.4", revision="Rev 3"),
    )
    calculation = Calculation(
        id="CALC1", expression="8.2 <= 6.0", inputs=["M1", "R1"], result=False,
    )
    claim = Claim(
        id="CL1", text="Pump-102 exceeds its normal vibration limit.",
        claim_type="engineering_finding", evidence_ids=["M1", "R1"],
        calculation_ids=["CALC1"], support_status=SupportStatus.supported,
        support_score=1.0,
    )
    assert measurement.original_value == 8.2
    assert measurement.normalized_value == 8.2
    assert rule.source.section == "7.4"
    assert calculation.engine == "deterministic"
    assert claim.support_status == SupportStatus.supported
