from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from time import perf_counter

import numpy as np

from adaptivefact.data.schema import Claim, ResponseRecord, VerificationStatus
from adaptivefact.verification.evidence import ContextTfidfIndex, EvidenceRetrieverConfig, merge_seed_evidence
from adaptivefact.verification.nli import NLIScorer, NLIScores
from adaptivefact.verification.text import split_sentences


_EVIDENCE_CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9:_-]{2,127})\]")
_RETRIEVAL_STOPWORDS = frozenset({
    "and", "are", "as", "at", "been", "for", "from", "has", "have", "into",
    "its", "per", "that", "the", "their", "these", "this", "those", "was",
    "were", "with",
})


@dataclass
class Phase6NLIConfig:
    entailment_threshold: float = 0.72
    contradiction_threshold: float = 0.72
    decision_margin: float = 0.10
    preserve_phase5_supported: bool = True
    min_contradiction_retrieval_score: float = 0.10
    evidence_conflict_threshold: float = 0.85
    min_contradiction_evidence_count: int = 2


@dataclass
class PendingClaim:
    record_index: int
    claim_index: int
    pair_start: int
    pair_end: int
    evidence_texts: list[str]
    evidence_scores: list[float]
    retrieval_ms: float


class Phase6Pipeline:
    """Retrieve evidence for claims and verify them with NLI.

    Structured ControlPlane/GraphRAG evidence is preserved as separate evidence
    units so independent supporting and contradictory sources are not flattened
    into one premise. Legacy free-text context retains the TF-IDF chunking path.
    """

    def __init__(self, nli: NLIScorer, *, retrieval_config: EvidenceRetrieverConfig | None = None, nli_config: Phase6NLIConfig | None = None) -> None:
        self.nli = nli
        self.retrieval_config = retrieval_config or EvidenceRetrieverConfig()
        self.nli_config = nli_config or Phase6NLIConfig()

    def process_records(
        self,
        records: list[ResponseRecord],
        *,
        claim_ids: set[str] | None = None,
        recheck_supported: bool = False,
    ) -> tuple[list[ResponseRecord], dict]:
        started = perf_counter()
        premises: list[str] = []
        hypotheses: list[str] = []
        pending: list[PendingClaim] = []
        retrieval_latencies: list[float] = []
        no_evidence_claims = 0

        for record_index, record in enumerate(records):
            structured_items = self._structured_items(record)
            index_started = perf_counter()
            evidence_index = None if structured_items else ContextTfidfIndex(record.context, self.retrieval_config)
            index_ms = (perf_counter() - index_started) * 1000.0
            eligible_statuses = {VerificationStatus.UNKNOWN}
            if recheck_supported or not self.nli_config.preserve_phase5_supported:
                eligible_statuses.add(VerificationStatus.SUPPORTED)
            candidate_claims = [
                (claim_index, claim)
                for claim_index, claim in enumerate(record.atomic_claims)
                if claim.status in eligible_statuses and (claim_ids is None or claim.id in claim_ids)
            ]
            if candidate_claims:
                retrieval_latencies.append(index_ms)

            for claim_index, claim in candidate_claims:
                retrieval_started = perf_counter()
                query = claim.verification_text
                if structured_items:
                    evidence_texts, evidence_scores = self._retrieve_structured(structured_items, query)
                else:
                    retrieved = evidence_index.retrieve(query, top_k=self.retrieval_config.top_k) if evidence_index is not None else []
                    seed = self._seed_evidence(claim)
                    evidence = merge_seed_evidence(seed, retrieved, top_k=self.retrieval_config.top_k)
                    evidence_texts = [item.text for item in evidence]
                    evidence_scores = [float(item.score) for item in evidence]
                retrieval_ms = (perf_counter() - retrieval_started) * 1000.0
                retrieval_latencies.append(retrieval_ms)

                if not evidence_texts:
                    if claim.status != VerificationStatus.SUPPORTED:
                        claim.verifier_used = "phase6_no_evidence"
                    no_evidence_claims += 1
                    continue

                pair_start = len(premises)
                hypothesis = _EVIDENCE_CITATION.sub("", query).strip()
                for text in evidence_texts:
                    premises.append(text)
                    hypotheses.append(hypothesis)
                pair_end = len(premises)
                pending.append(PendingClaim(record_index, claim_index, pair_start, pair_end, evidence_texts, evidence_scores, retrieval_ms))

        nli_started = perf_counter()
        scores = self.nli.score(premises, hypotheses) if premises else []
        nli_ms = (perf_counter() - nli_started) * 1000.0
        resolution_counts = Counter()
        for item in pending:
            claim = records[item.record_index].atomic_claims[item.claim_index]
            prior_status = claim.status
            prior_confidence = claim.confidence
            prior_evidence = list(claim.evidence)
            prior_method = claim.verifier_used
            local_scores = scores[item.pair_start:item.pair_end]
            status, confidence, best_evidence, method, verification_scores = self._decide(
                local_scores, item.evidence_texts, item.evidence_scores, claim.subject, claim.verification_text,
            )

            # Preserve a deterministic exact support only when semantic checking is
            # genuinely inconclusive. Explicit contradiction and CONFLICTING states
            # must always override the earlier Phase-5 support decision.
            if prior_status == VerificationStatus.SUPPORTED and status == VerificationStatus.UNKNOWN:
                claim.status = prior_status
                claim.confidence = prior_confidence
                claim.evidence = prior_evidence
                claim.verifier_used = prior_method
                claim.verification_scores = verification_scores
            else:
                claim.status = status
                claim.confidence = confidence
                claim.evidence = best_evidence
                claim.verifier_used = method
                claim.verification_scores = verification_scores
            claim.latency_ms = (claim.latency_ms or 0.0) + item.retrieval_ms
            resolution_counts[claim.status.value] += 1

        return records, {
            "nli_pairs": len(premises),
            "nli_claims": len(pending),
            "no_evidence_claims": no_evidence_claims,
            "nli_resolution_counts": dict(resolution_counts),
            "retrieval_latency_ms": _percentiles(retrieval_latencies),
            "nli_batch_latency_ms": nli_ms,
            "phase6_total_ms": (perf_counter() - started) * 1000.0,
        }

    @staticmethod
    def _structured_items(record: ResponseRecord) -> list[dict]:
        raw = record.metadata.get("structured_evidence") if isinstance(record.metadata, dict) else None
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict) and str(item.get("text") or "").strip()]

    def _retrieve_structured(self, items: list[dict], query: str) -> tuple[list[str], list[float]]:
        cited_ids = set(_EVIDENCE_CITATION.findall(query))
        query_tokens = self._tokens(_EVIDENCE_CITATION.sub(" ", query))
        ranked: list[tuple[bool, float, float, str]] = []
        for item in items:
            text = str(item.get("text") or "").strip()
            supplied = item.get("retrieval_score")
            supplied_score = float(supplied) if isinstance(supplied, (int, float)) else 0.0
            shared_terms = query_tokens & self._tokens(text)
            lexical = len(shared_terms) / max(1, len(query_tokens))
            cited = bool(cited_ids.intersection(
                str(item.get(key) or "") for key in ("evidence_id", "chunk_id", "document_id")
            ))
            # Superseded limits are not competing current authority. Retain a
            # deliberate citation (including misuse for the authority guard to
            # reject) and explicitly historical claims, but never let uncited
            # obsolete ranges manufacture a conflict with a current fact.
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            historical = bool(re.search(r"\b(?:historical|superseded|archived|formerly|previous|obsolete)\b", query, re.IGNORECASE))
            if str(metadata.get("document_status", "")).casefold() == "superseded" and not cited and not historical:
                continue
            # The supplied score ranks the original user query, not this atomic
            # claim. It may break ties but cannot make an unrelated chunk relevant.
            min_shared_terms = 1 if len(query_tokens) <= 2 else 2
            if not cited and (len(shared_terms) < min_shared_terms or lexical < self.retrieval_config.min_score):
                continue
            ranked.append((cited, lexical, supplied_score, text))
        ranked.sort(key=lambda pair: (pair[0], pair[1], pair[2]), reverse=True)
        selected = ranked[: max(1, self.retrieval_config.top_k)]
        # 1.0 means a deliberate citation. Exact lexical overlap across a whole
        # chunk must not grant an uncited source the same contradiction authority.
        return [text for _, _, _, text in selected], [1.0 if cited else min(0.99, float(lexical)) for cited, lexical, _, _ in selected]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = {token.casefold().strip(".,:;()[]{}\"'") for token in (text or "").split()}
        return {token for token in tokens if len(token) >= 3 and token not in _RETRIEVAL_STOPWORDS}

    @staticmethod
    def _seed_evidence(claim: Claim) -> list[str]:
        if (claim.verifier_used or "") in {"deterministic_numeric_conflict_candidate", "deterministic_date_conflict_candidate"}:
            return list(claim.evidence)
        return []

    def _decide(self, scores: list[NLIScores], evidence_texts: list[str], evidence_scores: list[float], subject: str | None, claim_text: str = ""):
        # NLI over a long chunk can combine an entity in one sentence with a
        # generic opposing rule in another. A contradiction needs a sentence
        # that actually names the claim subject; support remains NLI-scored.
        contradiction_scores = [
            score if (self._subject_local_to_sentence(subject, text)
                      and self._compatible_numeric_relation(claim_text, text)) else 0.0
            for score, text in zip(evidence_scores, evidence_texts)
        ]
        status, confidence, evidence_index, method = decide_nli_evidence(scores, contradiction_scores, self.nli_config)
        best_evidence = [] if evidence_index is None else [evidence_texts[evidence_index]]
        verification_scores = {}
        if evidence_index is not None:
            selected = scores[evidence_index]
            verification_scores = {"entailment": float(selected.entailment), "contradiction": float(selected.contradiction), "neutral": float(selected.neutral)}
        return status, confidence, best_evidence, method, verification_scores

    @staticmethod
    def _compatible_numeric_relation(claim: str, evidence: str) -> bool:
        """A recorded value and an operating band are not contradictory facts.

        This guard only distinguishes explicit numeric scalar/range relations
        for the same metric. It does not decide whether a reading is safe or
        within limits. Comparisons, ambiguous prose and mixed source relations
        retain the ordinary NLI contradiction path.
        """
        metrics = r"discharge\s+pressure|pressure|vibration|temperature|speed|load"
        units = r"bar|mm/s(?:\s+RMS)?|rpm|percent|%|°?C"
        metric = re.search(rf"\b({metrics})\b", claim, re.IGNORECASE)
        if metric is None or re.search(r"\b(?:within|outside|above|below|exceed|exceeds|threshold|limit)\b", claim, re.IGNORECASE):
            return True
        value = rf"\d+(?:\.\d+)?\s*(?:{units})(?=\W|$)"
        band = rf"\d+(?:\.\d+)?\s*(?:to|[-–])\s*{value}"
        def kind(text: str) -> str | None:
            if re.search(band, text, re.IGNORECASE):
                return "range"
            if re.search(value, text, re.IGNORECASE):
                return "scalar"
            return None
        relation = kind(_EVIDENCE_CITATION.sub("", claim))
        if relation is None:
            return True
        topic = metric.group(1)
        source_relations = {
            kind(sentence) for sentence in split_sentences(evidence)
            if re.search(rf"\b{re.escape(topic)}\b", sentence, re.IGNORECASE)
        } - {None}
        return not source_relations or relation in source_relations

    @classmethod
    def _subject_local_to_sentence(cls, subject: str | None, evidence: str) -> bool:
        if not subject:
            return True
        anchors = cls._tokens(subject)
        if not anchors:
            return True
        required = min(2, len(anchors))
        return any(len(anchors & cls._tokens(sentence)) >= required for sentence in split_sentences(evidence))


def decide_nli_evidence(scores: list[NLIScores], evidence_scores: list[float], config: Phase6NLIConfig):
    if not scores:
        return VerificationStatus.UNKNOWN, None, None, "nli_no_scores"
    if len(scores) != len(evidence_scores):
        raise ValueError("NLI scores and evidence scores must have the same length")
    entail_index = max(range(len(scores)), key=lambda index: scores[index].entailment)
    contradiction_candidates = [index for index, retrieval_score in enumerate(evidence_scores) if retrieval_score >= config.min_contradiction_retrieval_score]
    contra_index = max(contradiction_candidates, key=lambda index: scores[index].contradiction) if contradiction_candidates else None
    best_entailment = scores[entail_index].entailment
    best_contradiction = scores[contra_index].contradiction if contra_index is not None else 0.0
    corroborating = [index for index in contradiction_candidates if scores[index].contradiction >= config.contradiction_threshold]
    contradiction_corroborated = len(corroborating) >= config.min_contradiction_evidence_count or any(evidence_scores[index] >= 0.999 for index in corroborating)

    if contra_index is not None and contra_index != entail_index and best_entailment >= config.evidence_conflict_threshold and best_contradiction >= config.evidence_conflict_threshold:
        return VerificationStatus.CONFLICTING, float(max(best_entailment, best_contradiction)), contra_index, "nli_conflicting_evidence"
    if contra_index is not None and contradiction_corroborated and best_contradiction >= config.contradiction_threshold and best_contradiction - best_entailment >= config.decision_margin:
        return VerificationStatus.CONTRADICTED, float(best_contradiction), contra_index, "nli_contradiction"
    if best_entailment >= config.entailment_threshold and best_entailment - best_contradiction >= config.decision_margin:
        return VerificationStatus.SUPPORTED, float(best_entailment), entail_index, "nli_entailment"
    best_index = max(range(len(scores)), key=lambda index: max(scores[index].entailment, scores[index].contradiction))
    confidence = max(scores[best_index].entailment, scores[best_index].contradiction)
    return VerificationStatus.UNKNOWN, float(confidence), best_index, "nli_uncertain"


def summarize_phase6(records: list[ResponseRecord], *, phase5_status_counts: dict[str, int], runtime: dict) -> dict:
    claims = [claim for record in records for claim in record.atomic_claims]
    final_status_counts = Counter(claim.status.value for claim in claims)
    # Resolution means a positive or negative factual verdict, not an abstention
    # or conflicting evidence. Counts expose the denominator, including empty runs.
    resolved = sum(claim.status in {VerificationStatus.SUPPORTED, VerificationStatus.CONTRADICTED,
                                  VerificationStatus.UNSUPPORTED} for claim in claims)
    return {
        "n_records": len(records),
        "n_claims": len(claims),
        "resolved_claim_count": resolved,
        "overall_claim_resolution_rate": resolved / len(claims) if claims else 0.0,
        "phase5_status_counts": phase5_status_counts,
        "final_status_counts": dict(final_status_counts),
        "claims_sent_to_nli": int(runtime.get("nli_claims", 0)),
        "nli_pairs": int(runtime.get("nli_pairs", 0)),
        "no_evidence_claims": int(runtime.get("no_evidence_claims", 0)),
        "latency_ms": {"retrieval": runtime.get("retrieval_latency_ms", {}), "nli_batch": float(runtime.get("nli_batch_latency_ms", 0.0)), "phase6_total": float(runtime.get("phase6_total_ms", 0.0))},
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "mean": float(np.mean(arr))}
