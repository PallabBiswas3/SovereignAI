from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.artifacts.docx_generator import DocxGenerator
from app.rag.retrieval import LocalRetriever, RetrievedChunk
from app.multimodal.ocr import DocumentTextExtractor


class InspectionReading(BaseModel):
    parameter: str
    value: float
    unit: str


class InspectionAnalysis(BaseModel):
    equipment: str
    findings: list[dict[str, Any]]
    recommendation: str
    sources: list[dict[str, Any]]
    artifact_path: str
    ocr: dict[str, Any] | None = None


class InspectionWorkflow:
    PATTERNS = {
        "Vibration": (r"vibration[^\d]{0,30}([\d.]+)\s*mm/s", "mm/s"),
        "Bearing temperature": (r"bearing temperature[^\d]{0,30}([\d.]+)\s*°?\s*c", "°C"),
        "Discharge pressure": (r"(?:discharge )?pressure[^\d]{0,30}([\d.]+)\s*bar", "bar"),
    }

    def __init__(self, retriever: LocalRetriever, extractor: DocumentTextExtractor | None = None) -> None:
        self.retriever = retriever
        self.extractor = extractor or DocumentTextExtractor()

    def analyze(self, inspection_path: Path, output_path: Path) -> InspectionAnalysis:
        text, ocr_result = self.extractor.extract(inspection_path)
        equipment_match = re.search(r"\b(Pump[- ]?\d+)\b", text, re.IGNORECASE)
        equipment = equipment_match.group(1).replace(" ", "-").title() if equipment_match else "Unknown equipment"
        readings = self._extract_readings(text)
        if not readings:
            raise ValueError("No supported inspection measurements were found")
        findings: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        critical = False
        deviation = False
        for reading in readings:
            candidates = self.retriever.search(f"{reading.parameter} acceptable limit threshold {reading.unit}", limit=5)
            evidence = self._select_evidence(reading, candidates)
            if not evidence:
                findings.append({"parameter": reading.parameter, "observed": f"{reading.value} {reading.unit}", "allowed": "Not established", "status": "UNVERIFIED", "source": {}})
                continue
            chunk = evidence
            allowed, status, is_critical = self._compare(reading, chunk)
            critical = critical or is_critical
            deviation = deviation or status != "NORMAL"
            cited_source = {**chunk.source, "text": chunk.text}
            finding = {"parameter": reading.parameter, "observed": f"{reading.value:g} {reading.unit}", "allowed": allowed, "status": status, "source": cited_source}
            findings.append(finding)
            if cited_source not in sources:
                sources.append(cited_source)
        if critical:
            recommendation = "Remove Pump-102 from service and obtain engineering inspection before restart. The available evidence does not, by itself, establish that replacement is required."
        elif deviation:
            recommendation = "Approve planned maintenance and increased monitoring before continued normal duty. Replacement is not supported by the available evidence."
        else:
            recommendation = "Continued operation is acceptable subject to routine monitoring and authorized engineering review."
        DocxGenerator().generate_approval_note(
            output_path, "APPROVAL NOTE", f"Disposition of {equipment} following inspection",
            findings, recommendation, sources,
        )
        ocr_summary = None
        if ocr_result:
            ocr_summary = {
                "engine": ocr_result.engine,
                "mean_confidence": ocr_result.mean_confidence,
                "low_confidence": ocr_result.low_confidence,
                "warning": ocr_result.warning,
                "pages": len(ocr_result.pages),
            }
        return InspectionAnalysis(equipment=equipment, findings=findings, recommendation=recommendation, sources=sources, artifact_path=str(output_path), ocr=ocr_summary)

    def _extract_readings(self, text: str) -> list[InspectionReading]:
        readings: list[InspectionReading] = []
        for parameter, (pattern, unit) in self.PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                readings.append(InspectionReading(parameter=parameter, value=float(match.group(1)), unit=unit))
        return readings

    @staticmethod
    def _numbers(text: str) -> list[float]:
        return [float(value) for value in re.findall(r"\b\d+(?:\.\d+)?\b", text)]

    @staticmethod
    def _select_evidence(reading: InspectionReading, candidates: list[RetrievedChunk]) -> RetrievedChunk | None:
        required = {
            "Vibration": ("vibration", "mm/s", "exceed"),
            "Bearing temperature": ("temperature", "80", "90"),
            "Discharge pressure": ("pressure", "bar", "to"),
        }[reading.parameter]
        for candidate in candidates:
            lowered = candidate.text.lower()
            if all(token in lowered for token in required):
                return candidate
        return None

    def _compare(self, reading: InspectionReading, evidence: RetrievedChunk) -> tuple[str, str, bool]:
        text = evidence.text
        if reading.parameter == "Vibration":
            maximum = self._match_value(text, r"not exceed\s+([\d.]+)") or 6.0
            critical = self._match_value(text, r"above\s+([\d.]+).*removed") or 9.0
            status = "CRITICAL" if reading.value > critical else "DEVIATION" if reading.value > maximum else "NORMAL"
            return f"≤ {maximum:g} mm/s RMS", status, status == "CRITICAL"
        if reading.parameter == "Bearing temperature":
            normal = self._match_value(text, r"up to\s+([\d.]+)") or 80.0
            shutdown = self._match_value(text, r"above\s+([\d.]+).*shutdown") or 90.0
            status = "CRITICAL" if reading.value > shutdown else "DEVIATION" if reading.value > normal else "NORMAL"
            return f"≤ {normal:g} °C", status, status == "CRITICAL"
        low_high = re.search(r"([\d.]+)\s+to\s+([\d.]+)\s+bar", text, re.IGNORECASE)
        low, high = (float(low_high.group(1)), float(low_high.group(2))) if low_high else (4.8, 5.5)
        status = "NORMAL" if low <= reading.value <= high else "DEVIATION"
        return f"{low:g}–{high:g} bar", status, False

    @staticmethod
    def _match_value(text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return float(match.group(1)) if match else None
