from __future__ import annotations

from dataclasses import dataclass

from adaptivefact.data.schema import GenerationMetadata, ResponseRecord
from controlplane.schema import Interaction


@dataclass
class PreparedFactuality:
    """Canonical Phase-5 state shared across ControlPlane factuality stages.

    The hallucination detector prepares this once. Deeper AdaptiveFact routes
    reuse the same claims instead of extracting and deterministically verifying
    them a second time.
    """

    record: ResponseRecord
    phase5_runtime: dict[str, float]
    source: str = "controlplane_shared_phase5"


def render_grounding_context(interaction: Interaction) -> str | None:
    """Render legacy free-text and typed grounding evidence into one context."""

    parts: list[str] = []
    if interaction.context and interaction.context.strip():
        parts.append(interaction.context.strip())

    for item in interaction.grounding_evidence:
        label_bits = [f"evidence_id={item.evidence_id}"]
        if item.source_name:
            label_bits.append(f"source={item.source_name}")
        if item.document_id:
            label_bits.append(f"document_id={item.document_id}")
        if item.chunk_id:
            label_bits.append(f"chunk_id={item.chunk_id}")
        if item.page is not None:
            label_bits.append(f"page={item.page}")
        if item.revision:
            label_bits.append(f"revision={item.revision}")
        parts.append(f"[GroundingEvidence {' | '.join(label_bits)}]\n{item.text.strip()}")

    return "\n\n".join(part for part in parts if part).strip() or None


def build_response_record(
    interaction: Interaction,
    *,
    factuality_response: str | None = None,
) -> ResponseRecord:
    """Build the canonical factuality record without importing detector modules."""

    return ResponseRecord(
        id=interaction.id,
        dataset="controlplane_live",
        query=interaction.prompt,
        context=render_grounding_context(interaction),
        generated_response=(interaction.response if factuality_response is None else factuality_response),
        generation_metadata=GenerationMetadata(model=interaction.metadata.get("model")),
        metadata={
            "consequential": interaction.consequential,
            "grounding_evidence_count": len(interaction.grounding_evidence),
        },
    )
