from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.artifacts.docx_generator import DocxGenerator
from app.evidence.context import ContextCompiler
from app.evidence.executor import EvidenceFirstExecutor
from app.evidence.models import (
    EvidenceBundle,
    EvidenceConflict,
    EvidenceRequirement,
    EvidenceSource,
    Measurement,
    Rule,
    RuleSource,
    SupportStatus,
)
from app.multimodal.ocr import DocumentTextExtractor
from app.rag.retrieval import RetrievedChunk
from app.rag.decomposition import ModeAwareRetrievalPipeline
from app.resources.cache import CacheBackend, CacheKeyBuilder, CacheNamespace


class Retriever(Protocol):
    def collection_version(self) -> str: ...
    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]: ...


class InspectionReading(BaseModel):
    parameter: str
    metric: str
    value: float
    unit: str


class InspectionAnalysis(BaseModel):
    equipment: str
    findings: list[dict[str, Any]]
    recommendation: str
    sources: list[dict[str, Any]]
    artifact_path: str
    ocr: dict[str, Any] | None = None
    cache_hit: bool = False
    evidence_bundle: dict[str, Any] = Field(default_factory=dict)
    compiled_context: dict[str, Any] = Field(default_factory=dict)
    retrieval_metrics: list[dict[str, Any]] = Field(default_factory=list)


class InspectionWorkflow:
    PATTERNS = {
        "Vibration": ("vibration", r"vibration[^\d]{0,30}([\d.]+)\s*mm/s", "mm/s"),
        "Bearing temperature": ("bearing_temperature", r"bearing temperature[^\d]{0,30}([\d.]+)\s*(?:Â?°)?\s*c", "°C"),
        "Discharge pressure": ("discharge_pressure", r"(?:discharge )?pressure[^\d]{0,30}([\d.]+)\s*(Pa|kPa|MPa|bar)\b", "bar"),
    }

    def __init__(
        self,
        retriever: Retriever,
        extractor: DocumentTextExtractor | None = None,
        cache: CacheBackend | None = None,
        context_compiler: ContextCompiler | None = None,
        evidence_executor: EvidenceFirstExecutor | None = None,
        workcell_identity: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.extractor = extractor or DocumentTextExtractor()
        self.cache = cache
        self.context_compiler = context_compiler or ContextCompiler()
        self.evidence_executor = evidence_executor or EvidenceFirstExecutor()
        self.workcell_identity = workcell_identity

    def analyze(
        self,
        inspection_path: Path,
        output_path: Path,
        task: str = "Assess the inspection against applicable maintenance requirements.",
        *,
        selected_model: str = "general",
        context_window: int = 32768,
        execution_mode: str = "STANDARD",
    ) -> InspectionAnalysis:
        input_hashes = [
            hashlib.sha256(inspection_path.read_bytes()).hexdigest(),
            self.retriever.collection_version(),
            hashlib.sha256(task.encode("utf-8")).hexdigest(),
        ]
        workflow_version = "inspection-evidence-first-v2"
        if self.workcell_identity:
            workflow_version = f"{workflow_version}:{self.workcell_identity}"
        cache_key = CacheKeyBuilder.deterministic(input_hashes, workflow_version, "engineering-rules-v2")
        if self.cache:
            cached = self.cache.get(CacheNamespace.deterministic.value, cache_key)
            if isinstance(cached, dict):
                cached_analysis = InspectionAnalysis.model_validate({
                    **cached, "artifact_path": str(output_path), "cache_hit": True,
                })
                self._generate_artifact(output_path, cached_analysis)
                return cached_analysis

        text, ocr_result = self.extractor.extract(inspection_path)
        equipment_match = re.search(r"\b(Pump[- ]?\d+)\b", text, re.IGNORECASE)
        equipment = equipment_match.group(1).replace(" ", "-").title() if equipment_match else "Unknown equipment"
        readings = self._extract_readings(text)
        if not readings:
            raise ValueError("No supported inspection measurements were found")

        inspection_source = EvidenceSource(
            id="E-INSPECTION",
            file=inspection_path.name,
            document_hash=hashlib.sha256(inspection_path.read_bytes()).hexdigest(),
            text=text,
            access_scope=["internal"],
        )
        measurements = [
            Measurement(
                id=f"M{index}", asset_id=equipment, metric=reading.metric,
                original_value=reading.value, original_unit=reading.unit,
                source_id=inspection_source.id, confidence=0.95 if ocr_result else 1.0,
            )
            for index, reading in enumerate(readings, start=1)
        ]

        source_by_chunk: dict[str, EvidenceSource] = {}
        candidates_by_metric: dict[str, list[RetrievedChunk]] = {}
        all_candidates: dict[str, RetrievedChunk] = {}
        retrieval_metrics: list[dict[str, Any]] = []
        if execution_mode.upper() == "DEEP":
            pipeline = ModeAwareRetrievalPipeline(
                self.retriever, self.context_compiler.settings.max_retrieval_subqueries
            )
            deep_candidates = pipeline.search(task, execution_mode, 10)
            all_candidates.update({item.chunk_id: item for item in deep_candidates})
            retrieval_metrics.append({
                "query": task,
                "subqueries": pipeline.last_subqueries,
                "mode": "DEEP",
            })
        for reading in readings:
            query = f"{equipment} {reading.parameter} acceptable normal shutdown limit threshold {reading.unit}"
            candidates = self.retriever.search(query, limit=10)
            candidates_by_metric[reading.metric] = candidates
            all_candidates.update({item.chunk_id: item for item in candidates})
            telemetry = getattr(self.retriever, "last_telemetry", None)
            if telemetry is not None:
                retrieval_metrics.append({"query": query, **telemetry.to_dict()})
            for candidate in candidates:
                if not self._candidate_applies(reading, candidate):
                    continue
                if candidate.chunk_id not in source_by_chunk:
                    source_by_chunk[candidate.chunk_id] = EvidenceSource(
                        id=f"E{len(source_by_chunk) + 1}", document_id=candidate.document_id,
                        file=str(candidate.source.get("file", "unknown")),
                        page=candidate.source.get("page") if isinstance(candidate.source.get("page"), int) else None,
                        section=str(candidate.source["section"]) if candidate.source.get("section") is not None else None,
                        revision=str(candidate.source["revision"]) if candidate.source.get("revision") else None,
                        document_hash=str(candidate.source["document_hash"]) if candidate.source.get("document_hash") else None,
                        text=candidate.text,
                        retrieval_score=float(candidate.scores.get("fusion") or candidate.score),
                        reranker_score=float(candidate.scores["reranker"]) if isinstance(candidate.scores.get("reranker"), (int, float)) else None,
                        access_scope=candidate.access_scope or ["internal"],
                    )

        rules: list[Rule] = []
        for reading in readings:
            for candidate in candidates_by_metric[reading.metric]:
                source = source_by_chunk.get(candidate.chunk_id)
                if not source:
                    continue
                for rule in self._extract_rules(reading, candidate.text, source):
                    identity = (rule.metric, rule.rule_type, rule.operator, rule.threshold,
                                rule.lower_bound, rule.upper_bound, rule.unit, source.revision)
                    if any((existing.metric, existing.rule_type, existing.operator, existing.threshold,
                            existing.lower_bound, existing.upper_bound, existing.unit,
                            existing.source.revision) == identity for existing in rules):
                        continue
                    rule.id = f"R{len(rules) + 1}"
                    rules.append(rule)

        conflicts = self._detect_rule_conflicts(rules)
        requirements = [EvidenceRequirement(id="REQ-ASSET", type="asset_identity")]
        for reading in readings:
            requirements.extend([
                EvidenceRequirement(id=f"REQ-M-{reading.metric}", type="measurement", metric=reading.metric),
                EvidenceRequirement(
                    id=f"REQ-R-{reading.metric}", type="rule", metric=reading.metric,
                    rule_category="normal_limit" if reading.metric != "discharge_pressure" else "normal_range",
                ),
            ])
        if re.search(r"\b(latest|current|applicable revision)\b", task, re.I):
            requirements.append(EvidenceRequirement(id="REQ-REVISION", type="source_revision"))

        bundle = EvidenceBundle(
            sources=[inspection_source, *source_by_chunk.values()], measurements=measurements,
            rules=rules, requirements=requirements, conflicts=conflicts,
        )
        bundle = self.evidence_executor.execute(
            measurements=measurements, rules=rules, requirements=requirements,
            bundle=bundle, conflicts=conflicts,
        )
        findings = self._findings(readings, bundle)
        recommendation = self._recommendation(findings, bundle)
        sources = [source.model_dump(mode="json") for source in bundle.sources if source.id != inspection_source.id]
        compiled = self.context_compiler.compile(
            task=task, evidence=list(all_candidates.values()), selected_model=selected_model,
            context_window=context_window, execution_mode=execution_mode,
            measurements=bundle.measurements, rules=bundle.rules,
            calculations=bundle.calculations,
            open_questions=[item.id for item in requirements if not item.satisfied],
        )
        if compiled.conflicts:
            existing = {conflict.summary for conflict in bundle.conflicts}
            bundle.conflicts.extend(conflict for conflict in compiled.conflicts if conflict.summary not in existing)

        ocr_summary = None
        if ocr_result:
            ocr_summary = {
                "engine": ocr_result.engine, "mean_confidence": ocr_result.mean_confidence,
                "low_confidence": ocr_result.low_confidence, "warning": ocr_result.warning,
                "pages": len(ocr_result.pages), "cache_hit": ocr_result.cache_hit,
            }
        analysis = InspectionAnalysis(
            equipment=equipment, findings=findings, recommendation=recommendation,
            sources=sources, artifact_path=str(output_path), ocr=ocr_summary,
            evidence_bundle=bundle.model_dump(mode="json"),
            compiled_context=compiled.model_dump(mode="json"),
            retrieval_metrics=retrieval_metrics,
        )
        self._generate_artifact(output_path, analysis)
        if self.cache:
            self.cache.set(
                CacheNamespace.deterministic.value, cache_key,
                analysis.model_dump(mode="json"),
                metadata={"workflow_version": workflow_version,
                          "rule_version": "engineering-rules-v2", "input_hashes": input_hashes},
            )
        return analysis

    @staticmethod
    def _generate_artifact(output_path: Path, analysis: InspectionAnalysis) -> None:
        DocxGenerator().generate_approval_note(
            output_path, "APPROVAL NOTE", f"Disposition of {analysis.equipment} following inspection",
            analysis.findings, analysis.recommendation, analysis.sources,
        )

    def _extract_readings(self, text: str) -> list[InspectionReading]:
        readings: list[InspectionReading] = []
        for parameter, (metric, pattern, unit) in self.PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                observed_unit = match.group(2) if metric == "discharge_pressure" else unit
                readings.append(InspectionReading(parameter=parameter, metric=metric,
                                                  value=float(match.group(1)), unit=observed_unit))
        return readings

    @staticmethod
    def _candidate_applies(reading: InspectionReading, candidate: RetrievedChunk) -> bool:
        lowered = candidate.text.lower()
        required = {"vibration": ("vibration", "mm/s"),
                    "bearing_temperature": ("temperature",),
                    "discharge_pressure": ("pressure", "bar")}[reading.metric]
        return all(token in lowered for token in required)

    @staticmethod
    def _extract_rules(reading: InspectionReading, text: str, source: EvidenceSource) -> list[Rule]:
        source_ref = RuleSource(source_id=source.id, section=source.section, revision=source.revision)
        output: list[Rule] = []
        if reading.metric == "vibration":
            normal = InspectionWorkflow._match_value(text, r"not exceed\s+([\d.]+)")
            shutdown = InspectionWorkflow._match_value(text, r"above\s+([\d.]+).*?(?:removed|shutdown)")
            if normal is not None:
                output.append(Rule(id="pending", metric=reading.metric, operator="<=", threshold=normal,
                                   unit="mm/s", rule_type="normal_limit", source=source_ref))
            if shutdown is not None:
                output.append(Rule(id="pending", metric=reading.metric, operator="<=", threshold=shutdown,
                                   unit="mm/s", rule_type="shutdown_limit", source=source_ref))
        elif reading.metric == "bearing_temperature":
            normal = InspectionWorkflow._match_value(text, r"up to\s+([\d.]+)")
            shutdown = InspectionWorkflow._match_value(text, r"above\s+([\d.]+).*?shutdown")
            if normal is not None:
                output.append(Rule(id="pending", metric=reading.metric, operator="<=", threshold=normal,
                                   unit="°C", rule_type="normal_limit", source=source_ref))
            if shutdown is not None:
                output.append(Rule(id="pending", metric=reading.metric, operator="<=", threshold=shutdown,
                                   unit="°C", rule_type="shutdown_limit", source=source_ref))
        else:
            bounds = re.search(r"([\d.]+)\s*(?:to|[-–])\s*([\d.]+)\s*bar", text, re.IGNORECASE)
            if bounds:
                output.append(Rule(id="pending", metric=reading.metric, operator="between",
                                   lower_bound=float(bounds.group(1)), upper_bound=float(bounds.group(2)),
                                   unit="bar", rule_type="normal_range", source=source_ref))
        return output

    @staticmethod
    def _detect_rule_conflicts(rules: list[Rule]) -> list[EvidenceConflict]:
        groups: dict[tuple[str, str], list[Rule]] = {}
        for rule in rules:
            groups.setdefault((rule.metric, rule.rule_type), []).append(rule)
        conflicts: list[EvidenceConflict] = []
        for group in groups.values():
            values = {(rule.threshold, rule.lower_bound, rule.upper_bound, rule.unit) for rule in group}
            revisions = {rule.source.revision for rule in group if rule.source.revision}
            if len(values) > 1 and len(revisions) > 1:
                conflicts.append(EvidenceConflict(
                    id=f"CONFLICT-{len(conflicts) + 1}",
                    sources=sorted({rule.source.source_id for rule in group}),
                    summary=f"Conflicting {group[0].rule_type.replace('_', ' ')} values exist across document revisions.",
                    values=[{"rule_id": rule.id, "revision": rule.source.revision,
                             "threshold": rule.threshold, "lower_bound": rule.lower_bound,
                             "upper_bound": rule.upper_bound, "unit": rule.unit} for rule in group],
                ))
        return conflicts

    @staticmethod
    def _findings(readings: list[InspectionReading], bundle: EvidenceBundle) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        calculations = {tuple(item.inputs): item for item in bundle.calculations}
        sources = {item.id: item for item in bundle.sources}
        for reading, measurement in zip(readings, bundle.measurements):
            normal_type = "normal_range" if reading.metric == "discharge_pressure" else "normal_limit"
            normal = next((item for item in bundle.rules if item.metric == reading.metric and item.rule_type == normal_type), None)
            shutdown = next((item for item in bundle.rules if item.metric == reading.metric and item.rule_type == "shutdown_limit"), None)
            if not normal:
                findings.append({"parameter": reading.parameter, "observed": f"{reading.value:g} {reading.unit}",
                                 "allowed": "Not established", "status": "UNVERIFIED", "source": {}})
                continue
            normal_calc = calculations.get((measurement.id, normal.id))
            shutdown_calc = calculations.get((measurement.id, shutdown.id)) if shutdown else None
            conflicting = any(normal.source.source_id in conflict.sources for conflict in bundle.conflicts)
            if conflicting:
                status = "CONFLICTING_EVIDENCE"
            elif shutdown_calc and not bool(shutdown_calc.result):
                status = "CRITICAL"
            elif normal_calc and bool(normal_calc.result):
                status = "NORMAL"
            elif normal_calc:
                status = "DEVIATION"
            else:
                status = "UNVERIFIED"
            allowed = (f"{normal.lower_bound:g}–{normal.upper_bound:g} {normal.unit}"
                       if normal.operator == "between" else f"≤ {normal.threshold:g} {normal.unit}")
            source = sources.get(normal.source.source_id)
            findings.append({
                "parameter": reading.parameter,
                "observed": f"{measurement.normalized_value:g} {measurement.normalized_unit}",
                "original_observed": f"{measurement.original_value:g} {measurement.original_unit}",
                "allowed": allowed, "status": status,
                "source": source.model_dump(mode="json") if source else {},
                "measurement_id": measurement.id, "rule_id": normal.id,
                "calculation_id": normal_calc.id if normal_calc else None,
            })
        return findings

    @staticmethod
    def _recommendation(findings: list[dict[str, Any]], bundle: EvidenceBundle) -> str:
        statuses = {str(item["status"]) for item in findings}
        if "CONFLICTING_EVIDENCE" in statuses or bundle.conflicts:
            return "Applicable limits conflict across document revisions. Do not authorize a disposition until a human establishes the applicable revision."
        if "UNVERIFIED" in statuses or any(item.support_status in {SupportStatus.insufficient_evidence, SupportStatus.unsupported} for item in bundle.claims):
            return "Insufficient authorized evidence exists for a complete disposition. Obtain the missing applicable requirements before approval."
        if "CRITICAL" in statuses:
            return "Remove Pump-102 from service and obtain engineering inspection before restart. The available evidence does not, by itself, establish that replacement is required."
        if "DEVIATION" in statuses:
            return "Approve planned maintenance and increased monitoring before continued normal duty. Replacement is not supported by the available evidence."
        return "Continued operation is acceptable subject to routine monitoring and authorized engineering review."

    @staticmethod
    def _match_value(text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return float(match.group(1)) if match else None
