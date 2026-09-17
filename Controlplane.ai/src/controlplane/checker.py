from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from controlplane.actions import transform_response
from controlplane.audit import JsonlAuditStore
from controlplane.detectors import (
    BiasDetector,
    ConversationRiskDetector,
    Detector,
    HallucinationDetector,
    PrivacyDetector,
)
from controlplane.policy import PolicyEngine, PolicyProfile, PolicyRepository
from controlplane.schema import (
    CheckReport,
    DetectorResult,
    Finding,
    FindingStatus,
    Interaction,
    RiskCategory,
    Severity,
    VerificationDepth,
)
from controlplane.verification import VerificationService


class ControlPlane:
    """Execute enabled safety checks in parallel and enforce a versioned policy."""

    def __init__(
        self,
        *,
        policy_repository: PolicyRepository | None = None,
        detectors: dict[str, Detector] | None = None,
        policy_engine: PolicyEngine | None = None,
        audit_store: JsonlAuditStore | None = None,
        audit_enabled: bool = True,
        verification_service: VerificationService | None = None,
    ) -> None:
        self.policy_repository = policy_repository or PolicyRepository()
        self.detectors = detectors or {
            "privacy": PrivacyDetector(),
            "bias": BiasDetector(),
            "hallucination": HallucinationDetector(),
            "conversation": ConversationRiskDetector(),
        }
        self.policy_engine = policy_engine or PolicyEngine()
        self.audit_store = audit_store or JsonlAuditStore()
        self.audit_enabled = audit_enabled
        self.verification_service = verification_service

    def check(self, interaction: Interaction, profile: PolicyProfile | None = None) -> CheckReport:
        started = perf_counter()
        profile = profile or self.policy_repository.load(interaction.profile)
        enabled = {
            name: detector
            for name, detector in self.detectors.items()
            if name in profile.checks and profile.checks[name].enabled
        }

        results: dict[str, DetectorResult] = {}
        if enabled:
            with ThreadPoolExecutor(max_workers=len(enabled), thread_name_prefix="controlplane") as executor:
                futures = {
                    executor.submit(detector.run, interaction, profile.checks[name].settings): name
                    for name, detector in enabled.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:  # defensive boundary around executor failures
                        results[name] = DetectorResult(detector=name, error=f"{type(exc).__name__}: {exc}")

        ordered_results = [results[name] for name in enabled]
        if self.verification_service is not None and interaction.response.strip():
            depth = self._hallucination_depth(profile)
            try:
                ordered_results.append(self.verification_service.verify(interaction, depth))
            except Exception as exc:  # fail closed through existing detector-failure policy rules
                ordered_results.append(
                    DetectorResult(
                        detector="adaptivefact",
                        error=f"{type(exc).__name__}: {exc}",
                        metadata={"verification_depth": depth.value},
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
        decision = self.policy_engine.decide(
            interaction,
            findings,
            profile,
            latency_budget_exceeded=budget_exceeded,
        )
        final_response = transform_response(interaction.response, decision.action, findings, profile)
        report = CheckReport(
            interaction_id=interaction.id,
            policy_id=profile.id,
            policy_version=profile.version,
            original_response=interaction.response,
            final_response=final_response,
            decision=decision,
            findings=findings,
            detector_results=ordered_results,
            total_latency_ms=(perf_counter() - started) * 1000.0,
        )
        if self.audit_enabled:
            self.audit_store.write(report, profile)
        return report

    @staticmethod
    def _hallucination_depth(profile: PolicyProfile) -> VerificationDepth:
        check = profile.checks.get("hallucination")
        return check.depth if check is not None and check.enabled else VerificationDepth.QUICK
