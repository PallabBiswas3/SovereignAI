from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptivefact.data.schema import ResponseRecord, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.model import RiskModelBundle
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.phase5 import Phase5Pipeline
from controlplane.detectors.base import Detector
from controlplane.detectors.privacy import mask_privacy_values
from controlplane.factuality import PreparedFactuality, build_response_record
from controlplane.schema import DetectorResult, Finding, FindingStatus, Interaction, RiskCategory, Severity


class HallucinationDetector(Detector):
    """Cheap factuality risk router plus deterministic verification.

    The trained risk score is *only* a compute-routing prior by default. It does
    not become a hallucination finding unless explicitly enabled for an ablation.
    """

    name = "hallucination"

    def __init__(self) -> None:
        self._risk_cache: tuple[RiskModelBundle, RiskFeatureExtractor] | None = None
        self._risk_cache_key: tuple[str, str] | None = None

    def prepare(self, interaction: Interaction, settings: dict[str, Any]) -> PreparedFactuality:
        record = build_response_record(interaction, factuality_response=mask_privacy_values(interaction.response))
        phase5 = Phase5Pipeline(
            ClaimExtractionConfig(**settings.get("extraction", {})),
            DeterministicVerifierConfig(**settings.get("verification", {})),
        )
        _, runtime = phase5.process(record)
        return PreparedFactuality(record=record, phase5_runtime=runtime)

    def detect(self, interaction: Interaction, settings: dict[str, Any]) -> DetectorResult:
        prepared = self.prepare(interaction, settings)
        return self.detect_prepared(interaction, settings, prepared)

    def detect_prepared(self, interaction: Interaction, settings: dict[str, Any], prepared: PreparedFactuality) -> DetectorResult:
        record = prepared.record
        findings: list[Finding] = []
        supported = 0
        conflict_candidates = 0
        unsupported_entity_claims = 0
        context_folded = (record.context or "").casefold()
        emit_unsupported_entities = bool(settings.get("emit_unsupported_entity_findings", False))

        for claim in record.atomic_claims:
            if claim.status == VerificationStatus.SUPPORTED:
                supported += 1
                continue
            method = claim.verifier_used or ""
            if method.endswith("conflict_candidate"):
                conflict_candidates += 1
                findings.append(
                    Finding(
                        category=RiskCategory.HALLUCINATION,
                        subtype="grounded_value_conflict",
                        severity=Severity.HIGH if interaction.consequential else Severity.MEDIUM,
                        confidence=0.68,
                        status=FindingStatus.SUSPECTED,
                        message="A number or date differs from strongly related source evidence.",
                        detector=self.name,
                        span=claim.span,
                        evidence=claim.evidence,
                        metadata={"claim_id": claim.id, "claim": claim.verification_text, "verifier": method},
                    )
                )
                continue

            missing_entities = [entity for entity in claim.entities if entity.casefold() not in context_folded]
            if record.context and missing_entities:
                unsupported_entity_claims += 1
                if emit_unsupported_entities:
                    findings.append(
                        Finding(
                            category=RiskCategory.HALLUCINATION,
                            subtype="unsupported_entity_claim",
                            severity=Severity.MEDIUM,
                            confidence=0.58,
                            status=FindingStatus.UNDECIDABLE,
                            message="A factual claim mentions entities not found verbatim in supplied evidence.",
                            detector=self.name,
                            span=claim.span,
                            evidence=claim.evidence,
                            metadata={
                                "claim_id": claim.id,
                                "claim": claim.verification_text,
                                "missing_entities": missing_entities,
                                "diagnostic_only_by_default": True,
                            },
                        )
                    )

        risk = self._trained_risk(record, settings)
        risk_source = "trained_model" if risk is not None else "deterministic_fallback"
        if risk is None:
            unresolved = max(0, len(record.atomic_claims) - supported)
            risk = min(1.0, (conflict_candidates * 0.5 + unresolved * 0.04) / max(1, len(record.atomic_claims)))

        threshold = float(settings.get("risk_threshold", 0.65))
        if bool(settings.get("emit_risk_finding", False)) and risk >= threshold:
            findings.append(
                Finding(
                    category=RiskCategory.HALLUCINATION,
                    subtype="elevated_response_risk",
                    severity=Severity.HIGH if risk >= 0.85 else Severity.MEDIUM,
                    confidence=float(risk),
                    status=FindingStatus.SUSPECTED,
                    message="The factuality risk prior exceeds the configured verification-routing threshold.",
                    detector=self.name,
                    metadata={"risk_source": risk_source, "threshold": threshold, "signal_role": "verification_router"},
                )
            )

        return DetectorResult(
            detector=self.name,
            findings=findings,
            metadata={
                "risk_score": float(risk),
                "risk_source": risk_source,
                "risk_role": "verification_router_only",
                "risk_threshold": threshold,
                "claims": len(record.atomic_claims),
                "deterministically_supported": supported,
                "conflict_candidates": conflict_candidates,
                "unsupported_entity_diagnostics": unsupported_entity_claims,
                "unsupported_entity_findings_enabled": emit_unsupported_entities,
                "evidence_available": bool(record.context),
                "phase5": prepared.phase5_runtime,
                "factuality_artifact_source": prepared.source,
            },
        )

    def _trained_risk(self, record: ResponseRecord, settings: dict[str, Any]) -> float | None:
        if not bool(settings.get("use_trained_risk", True)):
            return None
        model_path = Path(settings.get("risk_model", "models/phase2/selected_model.joblib"))
        manifest_path = Path(settings.get("risk_manifest", "models/phase2/manifest.json"))
        if not model_path.exists() or not manifest_path.exists():
            return None
        key = (str(model_path.resolve()), str(manifest_path.resolve()))
        try:
            if self._risk_cache is None or self._risk_cache_key != key:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                extractor = RiskFeatureExtractor(FeatureExtractionConfig(**manifest["feature_config"]))
                model = RiskModelBundle.load(model_path)
                self._risk_cache = (model, extractor)
                self._risk_cache_key = key
            model, extractor = self._risk_cache
            return float(model.predict_proba(extractor.transform([record]))[0])
        except Exception:
            return None
