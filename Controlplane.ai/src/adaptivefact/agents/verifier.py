from __future__ import annotations

from time import perf_counter

from adaptivefact.agents.schema import (
    AgentStep,
    AgentVerificationConfig,
    AgentVerificationResult,
    SearchResult,
)
from adaptivefact.agents.tools.base import SearchTool
from adaptivefact.data.schema import Claim, ResponseRecord, VerificationStatus
from adaptivefact.verification.nli import NLIScorer, NLIScores


class BoundedVerificationAgent:
    """Evidence-seeking claim verifier with explicit latency/call/cost limits."""

    def __init__(
        self,
        nli: NLIScorer,
        search_tools: list[SearchTool],
        config: AgentVerificationConfig | None = None,
    ) -> None:
        if not search_tools:
            raise ValueError("BoundedVerificationAgent requires at least one approved search tool")
        self.nli = nli
        self.search_tools = search_tools
        self.config = config or AgentVerificationConfig()

    def verify(self, record: ResponseRecord, claim: Claim) -> AgentVerificationResult:
        started = perf_counter()
        evidence = self._seed_evidence(record, claim)
        trace: list[AgentStep] = []
        search_calls = 0
        nli_pairs = 0
        total_cost = 0.0
        final_confidence: float | None = None
        final_evidence: list[SearchResult] = []
        stop_reason = "max_iterations_reached"

        for iteration in range(1, self.config.max_iterations + 1):
            step_started = perf_counter()
            elapsed = self._elapsed_ms(started)
            if elapsed >= self.config.max_total_latency_ms:
                stop_reason = "latency_budget_exhausted"
                break

            query = self._query(record, claim, iteration)
            new_results: list[SearchResult] = []
            tools_used: list[str] = []
            for tool in self.search_tools:
                if search_calls >= self.config.max_search_calls:
                    stop_reason = "search_budget_exhausted"
                    break
                next_cost = total_cost + self.config.search_cost_per_call
                if next_cost > self.config.max_cost:
                    stop_reason = "cost_budget_exhausted"
                    break
                if self._elapsed_ms(started) >= self.config.max_total_latency_ms:
                    stop_reason = "latency_budget_exhausted"
                    break
                results = tool.search(
                    query,
                    record=record,
                    claim=claim,
                    top_k=self.config.top_k_per_search,
                )
                search_calls += 1
                total_cost = next_cost
                tools_used.append(tool.name)
                new_results.extend(results)

            evidence = self._merge_evidence(evidence, new_results)
            if self._elapsed_ms(started) >= self.config.max_total_latency_ms:
                trace.append(
                    AgentStep(
                        iteration=iteration,
                        query=query,
                        tool_names=tools_used,
                        results=new_results,
                        latency_ms=(perf_counter() - step_started) * 1000.0,
                    )
                )
                stop_reason = "latency_budget_exhausted"
                break
            remaining_pairs = self.config.max_nli_pairs - nli_pairs
            eligible = [
                item for item in evidence
                if item.authority_score >= self.config.min_authority_score
                and item.retrieval_score >= self.config.min_retrieval_score
            ][: max(0, remaining_pairs)]

            if not eligible:
                trace.append(
                    AgentStep(
                        iteration=iteration,
                        query=query,
                        tool_names=tools_used,
                        results=new_results,
                        latency_ms=(perf_counter() - step_started) * 1000.0,
                    )
                )
                if search_calls >= self.config.max_search_calls:
                    stop_reason = "no_reliable_evidence"
                    break
                continue

            scores = self.nli.score(
                [item.text for item in eligible],
                [claim.text] * len(eligible),
            )
            nli_pairs += len(eligible)
            status, confidence, selected, best_entailment, best_contradiction, reason = self._decide(
                eligible,
                scores,
            )
            trace.append(
                AgentStep(
                    iteration=iteration,
                    query=query,
                    tool_names=tools_used,
                    results=new_results,
                    best_entailment=best_entailment,
                    best_contradiction=best_contradiction,
                    decision=status,
                    latency_ms=(perf_counter() - step_started) * 1000.0,
                )
            )
            final_confidence = confidence
            final_evidence = selected
            if status in {VerificationStatus.SUPPORTED, VerificationStatus.CONTRADICTED}:
                stop_reason = reason
                break
            if reason == "conflicting_evidence":
                stop_reason = reason
                break
            if nli_pairs >= self.config.max_nli_pairs:
                stop_reason = "nli_budget_exhausted"
                break

        status = trace[-1].decision if trace else VerificationStatus.UNKNOWN
        if stop_reason not in {"resolved_supported", "resolved_contradicted"}:
            status = VerificationStatus.UNKNOWN

        if self.config.require_citations and status != VerificationStatus.UNKNOWN:
            if not final_evidence or any(not item.source_name for item in final_evidence):
                status = VerificationStatus.UNKNOWN
                stop_reason = "citation_requirement_failed"

        return AgentVerificationResult(
            claim_id=claim.id,
            status=status,
            confidence=final_confidence,
            evidence=final_evidence,
            trace=trace,
            iterations=len(trace),
            search_calls=search_calls,
            nli_pairs=nli_pairs,
            total_cost=total_cost,
            latency_ms=self._elapsed_ms(started),
            stop_reason=stop_reason,
        )

    def apply(self, claim: Claim, result: AgentVerificationResult) -> Claim:
        claim.status = result.status
        claim.confidence = result.confidence
        claim.evidence = [item.text for item in result.evidence]
        claim.verifier_used = f"agent:{result.stop_reason}"
        claim.latency_ms = (claim.latency_ms or 0.0) + result.latency_ms
        claim.cost = (claim.cost or 0.0) + result.total_cost
        return claim

    def _decide(
        self,
        evidence: list[SearchResult],
        scores: list[NLIScores],
    ) -> tuple[VerificationStatus, float | None, list[SearchResult], float, float, str]:
        if not scores:
            return VerificationStatus.UNKNOWN, None, [], 0.0, 0.0, "no_nli_scores"

        entail_index = max(range(len(scores)), key=lambda index: scores[index].entailment)
        contradiction_index = max(range(len(scores)), key=lambda index: scores[index].contradiction)
        best_entailment = float(scores[entail_index].entailment)
        best_contradiction = float(scores[contradiction_index].contradiction)

        entail_ready = best_entailment >= self.config.entailment_threshold
        contradiction_ready = best_contradiction >= self.config.contradiction_threshold
        if (
            entail_ready
            and contradiction_ready
            and evidence[entail_index].source_id != evidence[contradiction_index].source_id
        ):
            return (
                VerificationStatus.UNKNOWN,
                max(best_entailment, best_contradiction),
                [evidence[entail_index], evidence[contradiction_index]],
                best_entailment,
                best_contradiction,
                "conflicting_evidence",
            )

        if (
            contradiction_ready
            and best_contradiction - best_entailment >= self.config.decision_margin
        ):
            return (
                VerificationStatus.CONTRADICTED,
                best_contradiction * evidence[contradiction_index].authority_score,
                [evidence[contradiction_index]],
                best_entailment,
                best_contradiction,
                "resolved_contradicted",
            )

        if entail_ready and best_entailment - best_contradiction >= self.config.decision_margin:
            return (
                VerificationStatus.SUPPORTED,
                best_entailment * evidence[entail_index].authority_score,
                [evidence[entail_index]],
                best_entailment,
                best_contradiction,
                "resolved_supported",
            )

        best_index = max(
            range(len(scores)),
            key=lambda index: max(scores[index].entailment, scores[index].contradiction),
        )
        return (
            VerificationStatus.UNKNOWN,
            max(scores[best_index].entailment, scores[best_index].contradiction)
            * evidence[best_index].authority_score,
            [evidence[best_index]],
            best_entailment,
            best_contradiction,
            "uncertain_evidence",
        )

    @staticmethod
    def _seed_evidence(record: ResponseRecord, claim: Claim) -> list[SearchResult]:
        return [
            SearchResult(
                source_id=f"{record.id}:seed:{index}",
                source_name="Earlier verification evidence",
                text=text,
                authority_score=0.75,
                retrieval_score=1.0,
                metadata={"tool": "phase6_seed"},
            )
            for index, text in enumerate(claim.evidence)
            if text.strip()
        ]

    def _merge_evidence(
        self,
        existing: list[SearchResult],
        new: list[SearchResult],
    ) -> list[SearchResult]:
        merged: list[SearchResult] = []
        seen: set[tuple[str, str]] = set()
        for item in [*existing, *new]:
            key = (item.source_id, " ".join(item.text.split()).casefold())
            if not item.text.strip() or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        merged.sort(
            key=lambda item: (item.authority_score, item.retrieval_score),
            reverse=True,
        )
        return merged[: self.config.max_sources]

    @staticmethod
    def _query(record: ResponseRecord, claim: Claim, iteration: int) -> str:
        if iteration == 1:
            return claim.text
        structured = " ".join([*claim.entities, *claim.numbers, *claim.dates]).strip()
        if iteration == 2:
            return " ".join(part for part in [structured, record.query] if part).strip() or claim.text
        return f"verified authoritative evidence {structured or claim.text}".strip()

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (perf_counter() - started) * 1000.0
