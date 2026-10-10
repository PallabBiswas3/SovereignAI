from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature
import re
from time import perf_counter

from controlplane.actions import transform_response
from controlplane.audit import JsonlAuditStore
from controlplane.detectors import BiasDetector, ConversationRiskDetector, Detector, HallucinationDetector, PrivacyDetector
from controlplane.factuality import PreparedFactuality
from controlplane.policy import PolicyEngine, PolicyProfile, PolicyRepository
from controlplane.schema import CheckReport, DetectorResult, Finding, FindingStatus, Interaction, RiskCategory, Severity, VerificationDepth
from controlplane.verification import VerificationService


_DEPTH_RANK = {VerificationDepth.QUICK: 0, VerificationDepth.STANDARD: 1, VerificationDepth.DEEP: 2}
_PHYSICAL_ACTION = re.compile(r"\b(?:controlled\s+)?shutdown\b|\b(?:isolate|isolation|line[- ]break|bearing\s+replacement)\b", re.IGNORECASE)
_ACTION_OBLIGATION = re.compile(r"\b(?:required|must|should|initiate|perform)\b", re.IGNORECASE)
_NON_CURRENT_ACTION = re.compile(r"\b(?:historical|superseded|formerly|used\s+to|not\s+valid|do\s+not|must\s+not|should\s+not|cannot)\b", re.IGNORECASE)
_WORK_RECORD_ASSERTION = re.compile(
    r"\b(?:no\s+)?(?:physical|maintenance|repair)\s+work\s+(?:was|has\s+been)\s+(?:performed|completed)\b|"
    r"\b(?:no\s+)?(?:repair|replacement)\s+(?:was|has\s+been)\s+(?:performed|completed)\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ASSET = re.compile(r"\b[A-Za-z]+-\d+\b", re.IGNORECASE)


def infer_consequential_work_record(response: str) -> bool:
    """A categorical execution-history claim needs accountable evidence."""
    return bool(_WORK_RECORD_ASSERTION.search(response))


def _matching_work_record(interaction: Interaction) -> bool:
    assertion = _WORK_RECORD_ASSERTION.search(interaction.response)
    if assertion is None:
        return False
    requested_date = _DATE.search(interaction.response) or _DATE.search(interaction.prompt)
    requested_asset = _ASSET.search(interaction.response) or _ASSET.search(interaction.prompt)
    normalized_assertion = " ".join(assertion.group().casefold().split())
    for item in interaction.grounding_evidence:
        if str(item.metadata.get("document_status", "")).casefold() in {"superseded", "unverified", "draft"}:
            continue
        kind = str(item.metadata.get("document_type") or item.source_name or "").casefold()
        if not any(name in kind for name in ("work_order", "maintenance_history", "inspection_report", "operations_log")):
            continue
        if normalized_assertion not in " ".join(item.text.casefold().split()):
            continue
        if requested_date and requested_date.group() not in item.text:
            continue
        source_assets = {match.group().casefold() for match in _ASSET.finditer(item.text)}
        if requested_asset and requested_asset.group().casefold() not in source_assets:
            continue
        return True
    return False


def infer_consequential_action(question: str, response: str) -> bool:
    """Treat a present-tense physical-work instruction as consequential.

    The caller's consequential=True remains authoritative; this is a narrow
    upward-only fallback for callers that forgot to flag an actionable answer.
    """
    if not response.strip():
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", response):
        if _NON_CURRENT_ACTION.search(sentence):
            continue
        if _PHYSICAL_ACTION.search(sentence) and _ACTION_OBLIGATION.search(sentence):
            return True
    return False


class ControlPlane:
    """Execute enabled safety checks and enforce a versioned policy."""

    def __init__(self, *, policy_repository: PolicyRepository | None = None, detectors: dict[str, Detector] | None = None, policy_engine: PolicyEngine | None = None, audit_store: JsonlAuditStore | None = None, audit_enabled: bool = True, verification_service: VerificationService | None = None) -> None:
        self.policy_repository = policy_repository or PolicyRepository()
        self.detectors = detectors or {"privacy": PrivacyDetector(), "bias": BiasDetector(), "hallucination": HallucinationDetector(), "conversation": ConversationRiskDetector()}
        self.policy_engine = policy_engine or PolicyEngine()
        self.audit_store = audit_store or JsonlAuditStore()
        self.audit_enabled = audit_enabled
        self.verification_service = verification_service

    def check(self, interaction: Interaction, profile: PolicyProfile | None = None) -> CheckReport:
        started = perf_counter()
        inferred_action = infer_consequential_action(interaction.prompt, interaction.response)
        inferred_work_record = infer_consequential_work_record(interaction.response)
        if not interaction.consequential and (inferred_action or inferred_work_record):
            interaction = interaction.model_copy(update={
                "consequential": True,
                "metadata": {**interaction.metadata, "consequential_inferred": (
                    "physical_action_instruction" if inferred_action else "physical_work_record"
                )},
            })
        profile = profile or self.policy_repository.load(interaction.profile)
        enabled = {name: detector for name, detector in self.detectors.items() if name in profile.checks and profile.checks[name].enabled}
        results: dict[str, DetectorResult] = {}
        prepared_factuality: PreparedFactuality | None = None

        hallucination_detector = enabled.get("hallucination")
        if isinstance(hallucination_detector, HallucinationDetector) and interaction.response.strip():
            try:
                settings = profile.checks["hallucination"].settings
                prepared_factuality = hallucination_detector.prepare(interaction, settings)
                results["hallucination"] = hallucination_detector.detect_prepared(interaction, settings, prepared_factuality)
            except Exception as exc:
                results["hallucination"] = DetectorResult(detector="hallucination", error=f"{type(exc).__name__}: {exc}")

        parallel_enabled = {name: detector for name, detector in enabled.items() if name != "hallucination" or name not in results}
        if parallel_enabled:
            with ThreadPoolExecutor(max_workers=len(parallel_enabled), thread_name_prefix="controlplane") as executor:
                futures = {executor.submit(detector.run, interaction, profile.checks[name].settings): name for name, detector in parallel_enabled.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        results[name] = DetectorResult(detector=name, error=f"{type(exc).__name__}: {exc}")

        ordered_results = [results[name] for name in enabled]
        depth = self._hallucination_depth(profile, interaction, results.get("hallucination"))
        if interaction.response.strip():
            if self.verification_service is not None:
                try:
                    verify_parameters = signature(self.verification_service.verify).parameters
                    if "prepared" in verify_parameters:
                        verification_result = self.verification_service.verify(interaction, depth, prepared=prepared_factuality)
                    else:
                        verification_result = self.verification_service.verify(interaction, depth)
                    ordered_results.append(verification_result)
                except Exception as exc:
                    ordered_results.append(DetectorResult(detector="adaptivefact", error=f"{type(exc).__name__}: {exc}", metadata={"verification_depth": depth.value}))
            elif self._verification_required(interaction, depth, prepared_factuality):
                ordered_results.append(
                    DetectorResult(
                        detector="adaptivefact",
                        error="required factuality verification backend is unavailable",
                        metadata={
                            "verification_depth": depth.value,
                            "reason": "consequential_or_deep_factuality_check_required",
                        },
                    )
                )

        findings = [finding for result in ordered_results for finding in result.findings]
        if inferred_action:
            findings.append(Finding(
                category=RiskCategory.POLICY, subtype="physical_action_requires_review",
                severity=Severity.HIGH, confidence=1.0, status=FindingStatus.UNKNOWN,
                message="A current physical-action instruction requires accountable human review before release.",
                detector="controlplane",
            ))
        if inferred_work_record and not _matching_work_record(interaction):
            findings.append(Finding(
                category=RiskCategory.POLICY, subtype="unverified_operational_event",
                severity=Severity.HIGH, confidence=1.0, status=FindingStatus.UNKNOWN,
                message="A categorical work-execution claim lacks a matching authorized work record.",
                detector="controlplane",
            ))
        for result in ordered_results:
            if result.error:
                findings.append(
                    Finding(
                        category=RiskCategory.POLICY,
                        subtype="detector_failure",
                        severity=Severity.HIGH if interaction.consequential else Severity.MEDIUM,
                        confidence=1.0,
                        status=FindingStatus.UNKNOWN,
                        message=f"Required detector '{result.detector}' failed: {result.error}",
                        detector="controlplane",
                        metadata={"failed_detector": result.detector},
                    )
                )

        elapsed_before_policy = (perf_counter() - started) * 1000.0
        budget_exceeded = elapsed_before_policy > profile.latency_budget_ms
        decision = self.policy_engine.decide(interaction, findings, profile, latency_budget_exceeded=budget_exceeded, verification_depth=depth)
        final_response = transform_response(interaction.response, decision.action, findings, profile)
        report = CheckReport(interaction_id=interaction.id, policy_id=profile.id, policy_version=profile.version, original_response=interaction.response, final_response=final_response, decision=decision, findings=findings, detector_results=ordered_results, total_latency_ms=(perf_counter() - started) * 1000.0)
        if self.audit_enabled:
            self.audit_store.write(report, profile)
        return report

    @staticmethod
    def _verification_required(interaction: Interaction, depth: VerificationDepth, prepared: PreparedFactuality | None) -> bool:
        if prepared is None or not prepared.record.atomic_claims:
            return False
        return interaction.consequential or depth in {VerificationDepth.STANDARD, VerificationDepth.DEEP}

    @staticmethod
    def _hallucination_depth(profile: PolicyProfile, interaction: Interaction, hallucination_result: DetectorResult | None) -> VerificationDepth:
        check = profile.checks.get("hallucination")
        if check is None or not check.enabled:
            return VerificationDepth.QUICK
        maximum = check.depth
        if maximum == VerificationDepth.QUICK:
            return maximum
        if interaction.consequential:
            return maximum

        metadata = hallucination_result.metadata if hallucination_result is not None else {}
        risk = float(metadata.get("risk_score", 1.0))
        conflicts = int(metadata.get("conflict_candidates", 0))
        quick_threshold = float(check.settings.get("routing_standard_threshold", 0.30))
        deep_threshold = float(check.settings.get("routing_deep_threshold", 0.70))

        if conflicts > 0:
            desired = VerificationDepth.STANDARD
        elif risk < quick_threshold:
            desired = VerificationDepth.QUICK
        elif risk < deep_threshold:
            desired = VerificationDepth.STANDARD
        else:
            desired = VerificationDepth.DEEP

        if _DEPTH_RANK[desired] > _DEPTH_RANK[maximum]:
            return maximum
        return desired
