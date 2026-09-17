from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptivefact.data.schema import GenerationMetadata, ResponseRecord, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.model import RiskModelBundle
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.phase5 import Phase5Pipeline
from controlplane.detectors.base import Detector
from controlplane.detectors.privacy import mask_privacy_values
from controlplane.schema import (
    DetectorResult,
    Finding,
    FindingStatus,
    Interaction,
    RiskCategory,
    Severity,
)


class HallucinationDetector(Detector):
    """Low-latency adapter around adaptivefact's risk and deterministic layers.

    This intentionally does not download or invoke an NLI model. Deep NLI remains
    an optional verification backend after policy routing.
    """

    name = "hallucination"

    def __init__(self) -> None:
        self._risk_cache: tuple[RiskModelBundle, RiskFeatureExtractor] | None = None
        self._risk_cache_key: tuple[str, str] | None = None

    def detect(self, interaction: Interaction, settings: dict[str, Any]) -> DetectorResult:
        factuality_response = mask_privacy_values(interaction.response)
        record = ResponseRecord(
            id=interaction.id,
            dataset="controlplane_live",
            query=interaction.prompt,
            context=interaction.context,
            generated_response=factuality_response,
            generation_metadata=GenerationMetadata(model=interaction.metadata.get("model")),
        )
        phase5 = Phase5Pipeline(
            ClaimExtractionConfig(**settings.get("extraction", {})),
            DeterministicVerifierConfig(**settings.get("verification", {})),
        )
        phase5.process(record)

        findings: list[Finding] = []
        supported = 0
        conflict_candidates = 0
        context_folded = (interaction.context or "").casefold()

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
                        metadata={"claim_id": claim.id, "claim": claim.text, "verifier": method},
                    )
                )
                continue

            missing_entities = [entity for entity in claim.entities if entity.casefold() not in context_folded]
            if interaction.context and missing_entities:
                findings.append(
                    Finding(
                        category=RiskCategory.HALLUCINATION,
                        subtype="unsupported_entity_claim",
                        severity=Severity.MEDIUM,
                        confidence=0.58,
                        status=FindingStatus.UNKNOWN,
                        message="A factual claim mentions entities not found in the supplied evidence.",
                        detector=self.name,
                        span=claim.span,
                        evidence=claim.evidence,
                        metadata={"claim_id": claim.id, "claim": claim.text, "missing_entities": missing_entities},
                    )
                )

        risk = self._trained_risk(record, settings)
        risk_source = "trained_model" if risk is not None else "deterministic_fallback"
        if risk is None:
            unresolved = max(0, len(record.atomic_claims) - supported)
            risk = min(1.0, (conflict_candidates * 0.5 + unresolved * 0.04) / max(1, len(record.atomic_claims)))

        threshold = float(settings.get("risk_threshold", 0.65))
        if risk >= threshold:
            severity = Severity.HIGH if risk >= 0.85 else Severity.MEDIUM
            findings.append(
                Finding(
                    category=RiskCategory.HALLUCINATION,
                    subtype="elevated_response_risk",
                    severity=severity,
                    confidence=float(risk),
                    status=FindingStatus.SUSPECTED,
                    message="The response-level hallucination risk exceeds the configured threshold.",
                    detector=self.name,
                    metadata={"risk_source": risk_source, "threshold": threshold},
                )
            )

        if not interaction.context and record.atomic_claims:
            findings.append(
                Finding(
                    category=RiskCategory.HALLUCINATION,
                    subtype="evidence_unavailable",
                    severity=Severity.MEDIUM if interaction.consequential else Severity.LOW,
                    confidence=0.50,
                    status=FindingStatus.UNKNOWN,
                    message="No grounding context was supplied, so factual claims could not be verified.",
                    detector=self.name,
                )
            )

        return DetectorResult(
            detector=self.name,
            findings=findings,
            metadata={
                "risk_score": float(risk),
                "risk_source": risk_source,
                "claims": len(record.atomic_claims),
                "deterministically_supported": supported,
                "conflict_candidates": conflict_candidates,
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
