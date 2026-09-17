from __future__ import annotations

"""Deterministic release gate for LLM summaries of DiagnosticResult payloads."""

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Iterable, Mapping


_ALLOWED_RESULT_ROOTS = {
    "confidence",
    "decision",
    "detection",
    "evidence",
    "hypotheses",
    "localization",
    "prognosis",
    "recommended_actions",
    "uncertainty",
}
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:%|\b)")
_SENTENCE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")


@dataclass(frozen=True)
class ClaimReference:
    claim_text: str
    source_ref: str
    claimed_value: Any
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClaimReference":
        return cls(
            claim_text=str(value.get("claim_text", "")),
            source_ref=str(value.get("source_ref", "")),
            claimed_value=value.get("claimed_value"),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", ())),
        )


@dataclass(frozen=True)
class SentenceClaims:
    sentence_index: int
    sentence: str
    claims: tuple[ClaimReference, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SentenceClaims":
        return cls(
            sentence_index=int(value["sentence_index"]),
            sentence=str(value["sentence"]),
            claims=tuple(
                ClaimReference.from_mapping(claim) for claim in value.get("claims", ())
            ),
        )


@dataclass(frozen=True)
class ClaimFinding:
    sentence_index: int
    claim_text: str
    source_ref: str
    status: str
    reason: str


@dataclass(frozen=True)
class GroundingReport:
    release_allowed: bool
    sentence_count: int
    claim_count: int
    findings: tuple[ClaimFinding, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sentences(summary: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(str(summary)) if part.strip()]


def _json_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("source_ref must be an absolute JSON pointer")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
    if not parts or parts[0] not in _ALLOWED_RESULT_ROOTS:
        raise ValueError(f"source_ref root is not releasable: {pointer}")
    value: Any = payload
    for part in parts:
        if isinstance(value, Mapping):
            if part not in value:
                raise KeyError(pointer)
            value = value[part]
        elif isinstance(value, (list, tuple)):
            try:
                value = value[int(part)]
            except (ValueError, IndexError) as exc:
                raise KeyError(pointer) from exc
        else:
            raise KeyError(pointer)
    return value


def _normalized(value: Any) -> str:
    return " ".join(str(value).replace("_", " ").strip().lower().split())


def _values_match(claimed: Any, actual: Any, tolerance: float) -> bool:
    if isinstance(claimed, bool) or isinstance(actual, bool):
        return claimed is actual
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(claimed), float(actual), rel_tol=tolerance, abs_tol=tolerance)
    return _normalized(claimed) == _normalized(actual)


def _numeric_claim_tokens(claims: Iterable[ClaimReference]) -> list[float]:
    values = []
    for claim in claims:
        if isinstance(claim.claimed_value, (int, float)) and not isinstance(
            claim.claimed_value, bool
        ):
            values.append(float(claim.claimed_value))
    return values


def verify_summary_claims(
    summary: str,
    sentence_claims: Iterable[SentenceClaims | Mapping[str, Any]],
    diagnostic_result: Mapping[str, Any],
    *,
    numeric_tolerance: float = 1e-6,
) -> GroundingReport:
    """Verify a structured, sentence-by-sentence LLM summary.

    The generating LLM must emit one annotation per sentence and a JSON pointer
    for every factual claim.  Free-form summaries without this companion
    structure fail closed. Authorization and document-citation checks remain
    responsibilities of the host control plane.
    """

    sentences = _sentences(summary)
    annotations = [
        value if isinstance(value, SentenceClaims) else SentenceClaims.from_mapping(value)
        for value in sentence_claims
    ]
    findings: list[ClaimFinding] = []
    by_index = {row.sentence_index: row for row in annotations}
    if len(by_index) != len(annotations):
        findings.append(ClaimFinding(-1, "", "", "REJECTED", "duplicate sentence annotations"))

    known_evidence = {
        str(row.get("evidence_id"))
        for row in diagnostic_result.get("evidence", [])
        if isinstance(row, Mapping) and row.get("evidence_id")
    }
    for index, sentence in enumerate(sentences):
        annotation = by_index.get(index)
        if annotation is None:
            findings.append(ClaimFinding(index, sentence, "", "REJECTED", "sentence is unannotated"))
            continue
        if annotation.sentence.strip() != sentence:
            findings.append(ClaimFinding(index, sentence, "", "REJECTED", "annotation text does not match summary sentence"))
            continue
        if not annotation.claims:
            findings.append(ClaimFinding(index, sentence, "", "REJECTED", "sentence has no traceable claims"))
            continue

        claimed_numbers = _numeric_claim_tokens(annotation.claims)
        for token in _NUMBER.findall(sentence):
            number = float(token.rstrip("%"))
            if token.endswith("%"):
                candidates = claimed_numbers + [value * 100.0 for value in claimed_numbers]
            else:
                candidates = claimed_numbers
            if not any(math.isclose(number, value, rel_tol=numeric_tolerance, abs_tol=numeric_tolerance) for value in candidates):
                findings.append(ClaimFinding(index, token, "", "REJECTED", "numeric statement has no claim reference"))

        for claim in annotation.claims:
            if not claim.claim_text.strip() or _normalized(claim.claim_text) not in _normalized(sentence):
                findings.append(ClaimFinding(index, claim.claim_text, claim.source_ref, "REJECTED", "claim text is not present in its sentence"))
                continue
            unknown_evidence = sorted(set(claim.evidence_ids) - known_evidence)
            if unknown_evidence:
                findings.append(ClaimFinding(index, claim.claim_text, claim.source_ref, "REJECTED", f"unknown evidence IDs: {unknown_evidence}"))
                continue
            try:
                actual = _json_pointer(diagnostic_result, claim.source_ref)
            except (KeyError, ValueError) as exc:
                findings.append(ClaimFinding(index, claim.claim_text, claim.source_ref, "REJECTED", str(exc)))
                continue
            if not _values_match(claim.claimed_value, actual, numeric_tolerance):
                findings.append(ClaimFinding(index, claim.claim_text, claim.source_ref, "REJECTED", f"claimed value does not match DiagnosticResult value {actual!r}"))
                continue
            findings.append(ClaimFinding(index, claim.claim_text, claim.source_ref, "SUPPORTED", "claim resolves to the request's DiagnosticResult"))

    unexpected = sorted(set(by_index) - set(range(len(sentences))))
    for index in unexpected:
        findings.append(ClaimFinding(index, by_index[index].sentence, "", "REJECTED", "annotation has no corresponding summary sentence"))
    release_allowed = bool(sentences) and bool(findings) and all(
        finding.status == "SUPPORTED" for finding in findings
    )
    return GroundingReport(
        release_allowed=release_allowed,
        sentence_count=len(sentences),
        claim_count=sum(len(row.claims) for row in annotations),
        findings=tuple(findings),
    )
