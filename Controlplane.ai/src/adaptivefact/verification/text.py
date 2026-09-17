from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    return parts or [text]


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Build compact evidence chunks without introducing a heavy tokenizer dependency."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paragraphs:
        paragraphs = split_sentences(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            units = split_sentences(paragraph)
        else:
            units = [paragraph]

        for unit in units:
            extra = len(unit) + (1 if current else 0)
            if current and current_len + extra > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(unit)
            current_len += extra

    if current:
        chunks.append(" ".join(current))

    return chunks
