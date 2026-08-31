from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.multimodal.ocr import DocumentTextExtractor
from app.multimodal.vision import VisionProvider
from app.tools.file_tools import extract_text


class EvidenceRecord(BaseModel):
    file: str
    media_type: str
    processor: str
    summary: str
    provenance: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    warning: str | None = None


class MultiFileEvidenceProcessor:
    """Routes each attachment by type and retains file-level provenance."""

    IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

    def __init__(self, vision: VisionProvider | None = None, extractor: DocumentTextExtractor | None = None) -> None:
        self.vision = vision
        self.extractor = extractor or DocumentTextExtractor()

    async def process(self, paths: list[Path], prompt: str) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        for path in paths:
            suffix = path.suffix.lower()
            provenance = {"file": path.name, "path": str(path), "type": suffix}
            if suffix in self.IMAGE_TYPES:
                records.append(await self._image(path, prompt, provenance))
            elif suffix == ".pdf":
                text, ocr = self.extractor.extract(path)
                records.append(EvidenceRecord(
                    file=path.name, media_type="application/pdf",
                    processor="tesseract-ocr" if ocr else "pypdf-text",
                    summary=text[:2_000], provenance=provenance,
                    metadata={"pages": len(ocr.pages) if ocr else text.count("[PAGE "), "ocr_confidence": ocr.mean_confidence if ocr else None},
                    warning=ocr.warning if ocr else None,
                ))
            elif suffix == ".csv":
                records.append(self._csv(path, provenance))
            elif suffix == ".xlsx":
                text = extract_text(path)
                records.append(EvidenceRecord(
                    file=path.name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    processor="openpyxl-structured", summary=text[:2_000], provenance=provenance,
                    metadata={"sheets": text.count("# Sheet:")},
                ))
            else:
                text = extract_text(path)
                records.append(EvidenceRecord(
                    file=path.name, media_type="text/document", processor=f"{suffix[1:] or 'text'}-extractor",
                    summary=text[:2_000], provenance=provenance,
                    metadata={"characters": len(text)},
                ))
        return records

    async def _image(self, path: Path, prompt: str, provenance: dict[str, Any]) -> EvidenceRecord:
        if not self.vision:
            return EvidenceRecord(
                file=path.name, media_type=f"image/{path.suffix.lower().lstrip('.')}", processor="vision-unavailable",
                summary="Image retained as evidence but no vision provider was configured.", provenance=provenance,
                warning="Vision analysis was not performed.",
            )
        result = await self.vision.analyze_image(path, prompt)
        return EvidenceRecord(
            file=path.name, media_type=f"image/{path.suffix.lower().lstrip('.')}", processor=f"local-vision:{result.model}",
            summary=result.description, provenance=provenance,
            metadata={"components": result.detected_components, "observations": result.observations, "confidence": result.confidence},
            warning=result.warning,
        )

    @staticmethod
    def _csv(path: Path, provenance: dict[str, Any]) -> EvidenceRecord:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = list(rows[0]) if rows else []
        missing = {column: sum(not str(row.get(column, "")).strip() for row in rows) for column in columns}
        sample = rows[:5]
        return EvidenceRecord(
            file=path.name, media_type="text/csv", processor="csv-structured-profiler",
            summary=f"CSV with {len(rows)} rows and {len(columns)} columns: {', '.join(columns)}",
            provenance=provenance, metadata={"rows": len(rows), "columns": columns, "missing": missing, "sample": sample},
        )
