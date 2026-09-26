from app.core.config import Settings
from app.evidence.context import ContextCompiler
from app.evidence.distiller import EvidenceDistiller
from app.rag.retrieval import RetrievedChunk


def chunk(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.8,
    file: str = "manual.pdf",
    section: str = "4.2",
    revision: str | None = None,
    page: int = 18,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        document_id=f"doc-{chunk_id}",
        source={
            "file": file,
            "section": section,
            "revision": revision,
            "page": page,
            "document_hash": f"hash-{chunk_id}",
        },
        scores={"dense": score, "sparse": None, "fusion": score, "reranker": score},
        retrieval_methods=["dense", "bm25"],
        access_scope=["internal"],
    )


def token_count(text: str) -> int:
    return ContextCompiler.estimate_tokens(text)


def test_distiller_keeps_query_relevant_technical_constraint():
    evidence = [
        chunk(
            "c1",
            "The maintenance team meets every Monday. "
            "Pump-102 maximum vibration limit is 7.1 mm/s RMS. "
            "The appendix contains historical procurement notes.",
        )
    ]
    result = EvidenceDistiller(estimate_tokens=token_count).distill(
        task="What is the vibration limit for Pump-102?",
        chunks=evidence,
        token_budget=18,
        chunk_cap=1,
    )

    assert result.selected
    selected_text = result.selected[0][1]
    assert "Pump-102" in selected_text
    assert "7.1 mm/s" in selected_text
    assert result.technical_sentence_count >= 1
    assert result.selected_tokens <= 18


def test_distiller_preserves_conflicting_revisions_even_below_chunk_cutoff():
    evidence = [
        chunk(
            "rev4",
            "Pump-102 maximum vibration limit is 7.1 mm/s RMS.",
            score=0.95,
            file="SOP-MNT-017.pdf",
            section="4.2",
            revision="Rev 4",
        ),
        chunk(
            "other",
            "Pump-102 inspection records require a bearing visual check.",
            score=0.90,
            file="inspection.pdf",
            section="bearing",
            revision="Rev 1",
        ),
        chunk(
            "rev3",
            "Pump-102 maximum vibration limit is 8.0 mm/s RMS.",
            score=0.70,
            file="SOP-MNT-017.pdf",
            section="4.2",
            revision="Rev 3",
        ),
    ]

    result = EvidenceDistiller(estimate_tokens=token_count).distill(
        task="What vibration limit applies to Pump-102?",
        chunks=evidence,
        token_budget=50,
        chunk_cap=2,
    )

    ids = {item.chunk_id for item, _text in result.selected}
    assert ids == {"rev4", "rev3"}
    assert result.conflict_preserved_chunks == 2
    texts = " ".join(text for _item, text in result.selected)
    assert "7.1 mm/s" in texts
    assert "8.0 mm/s" in texts


def test_context_compiler_distillation_is_feature_flagged_and_bounded():
    settings = Settings(
        context_distillation_enabled=True,
        context_distillation_target_tokens=24,
        context_distillation_sentence_redundancy_threshold=0.82,
        context_max_evidence_tokens=3000,
        context_max_evidence_chunks=8,
    )
    compiler = ContextCompiler(settings)
    evidence = [
        chunk(
            "c1",
            "General maintenance background is available. "
            "Pump-102 alarm threshold is 90 C. "
            "Above 90 C the procedure requires inspection. "
            "Procurement notes are unrelated to the current task.",
        )
    ]

    compiled = compiler.compile(
        task="What is the Pump-102 alarm threshold?",
        evidence=evidence,
        selected_model="qwen3:4b-instruct",
        context_window=4096,
        execution_mode="STANDARD",
    )

    assert compiled.metrics.distillation_applied is True
    assert compiled.metrics.selected_evidence_tokens <= 24
    assert compiled.budget.max_evidence_tokens == 24
    assert compiled.evidence
    assert "90 C" in compiled.evidence[0].text


def test_context_compiler_default_keeps_distillation_disabled():
    settings = Settings(context_distillation_enabled=False)
    compiler = ContextCompiler(settings)
    compiled = compiler.compile(
        task="Summarize Pump-102 guidance",
        evidence=[chunk("c1", "Pump-102 should be inspected after an alarm.")],
        selected_model="qwen3:4b-instruct",
        context_window=4096,
        execution_mode="STANDARD",
    )

    assert compiled.metrics.distillation_applied is False
