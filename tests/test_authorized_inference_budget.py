import json

from app.llm.ollama_provider import OllamaProvider


def _authorized_prompt(*, evidence_count: int = 8, broad: bool = False) -> str:
    request = (
        "Create a management briefing covering operations, maintenance, safety, and quality."
        if broad
        else "Summarize the authorized maintenance guidance for Pump-102."
    )
    context = {
        "asset_context": None,
        "authorized_document_evidence": [
            {
                "chunk_id": f"chunk-{index}",
                "document_id": f"doc-{index}",
                "text": f"Evidence {index}: " + ("x" * 2200),
                "source": {"file": f"doc-{index}.md", "page": index + 1},
            }
            for index in range(evidence_count)
        ],
        "context_policy": {"authorization_checked": True},
    }
    return (
        f"USER_REQUEST:\n{request}\n\n"
        "AUTHORIZED_CONTEXT_START\n"
        f"{json.dumps(context)}\n"
        "AUTHORIZED_CONTEXT_END\n\n"
        "Answer with citations."
    )


def _context_from_prompt(prompt: str) -> dict:
    raw = prompt.split("AUTHORIZED_CONTEXT_START\n", 1)[1].split("\nAUTHORIZED_CONTEXT_END", 1)[0]
    return json.loads(raw)


def test_general_prompt_keeps_existing_budget() -> None:
    prompt = "Explain preventive maintenance briefly."
    compacted, changed, num_ctx, num_predict = OllamaProvider._compact_authorized_prompt(prompt)
    assert compacted == prompt
    assert changed is False
    assert num_ctx == 8192
    assert num_predict == 1280


def test_routine_authorized_prompt_compacts_model_context_and_output_budget() -> None:
    prompt = _authorized_prompt()
    compacted, changed, num_ctx, num_predict = OllamaProvider._compact_authorized_prompt(prompt)
    context = _context_from_prompt(compacted)
    evidence = context["authorized_document_evidence"]

    assert changed is True
    assert num_ctx == 4096
    assert num_predict == 384
    assert len(evidence) == 4
    assert evidence[0]["chunk_id"] == "chunk-0"
    assert evidence[0]["document_id"] == "doc-0"
    assert evidence[0]["source"]["file"] == "doc-0.md"
    assert evidence[0]["text_truncated"] is True
    assert len(evidence[0]["text"]) <= 1002
    assert context["context_policy"]["model_context_compacted"] is True
    assert context["context_policy"]["model_evidence_limit"] == 4


def test_broad_authorized_prompt_retains_larger_bounded_budget() -> None:
    prompt = _authorized_prompt(broad=True)
    compacted, changed, num_ctx, num_predict = OllamaProvider._compact_authorized_prompt(prompt)
    context = _context_from_prompt(compacted)
    evidence = context["authorized_document_evidence"]

    assert changed is True
    assert num_ctx == 6144
    assert num_predict == 640
    assert len(evidence) == 6
    assert all(len(item["text"]) <= 1602 for item in evidence)
