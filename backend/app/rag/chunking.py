from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextChunk:
    text: str
    index: int
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenanceChunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 120) -> None:
        if overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[TextChunk]:
        metadata = metadata or {}
        pages = re.split(r"^\s*\[PAGE\s+(\d+)\]\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
        page_segments: list[tuple[int | None, str]]
        if len(pages) > 1:
            page_segments = [(int(pages[index]), pages[index + 1]) for index in range(1, len(pages), 2)]
            if pages[0].strip():
                page_segments.insert(0, (1, pages[0]))
        else:
            page_segments = [(metadata.get("page"), text)]

        chunks: list[TextChunk] = []
        for page, page_text in page_segments:
            section: str | None = None
            start = 0
            while start < len(page_text):
                end = min(len(page_text), start + self.chunk_size)
                if end < len(page_text):
                    boundary = page_text.rfind("\n", start, end)
                    if boundary > start + self.chunk_size // 2:
                        end = boundary
                content = page_text[start:end].strip()
                heading = re.search(r"(?im)^#{1,4}\s+(?:Section\s+)?([\d.]+|.+)$|^Section\s+([\d.]+)", content)
                if heading:
                    raw_section = next((group for group in heading.groups() if group), section)
                    if raw_section:
                        numbered = re.match(r"([\d.]+)", raw_section)
                        section = numbered.group(1) if numbered else raw_section
                if content:
                    chunks.append(TextChunk(text=content, index=len(chunks), page=page, section=section, metadata=metadata.copy()))
                if end >= len(page_text):
                    break
                start = max(start + 1, end - self.overlap)
        return chunks
