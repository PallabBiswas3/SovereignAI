from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature
from time import perf_counter

from controlplane.actions import transform_response
from controlplane.audit import JsonlAuditStore
from controlplane.detectors import BiasDetector, ConversationRiskDetector, Detector, HallucinationDetector, PrivacyDetector
from controlplane.factuality import PreparedFactuality
from controlplane.policy import PolicyEngine, PolicyProfile, PolicyRepository
from controlplane.schema import CheckReport, DetectorResult, Finding, FindingStatus, Interaction, RiskCategory, Severity, VerificationDepth
from controlplane.verification import VerificationService


_DEPTH_RANK = {VerificationDepth.QUICK: 0, VerificationDepth.STANDARD: 1, VerificationDepth.DEEP: 2}


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
