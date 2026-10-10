"""Conservative checks that source authority cannot be supplied by an NLI score.

These checks use only authorized, structured evidence passed by the caller. They
recognize a few explicit assertions; uncertain language remains with the normal
semantic verifier rather than being guessed into a contradiction.
"""

from __future__ import annotations

import re
from decimal import Decimal

from adaptivefact.data.schema import Claim, VerificationStatus
from controlplane.schema import GroundingEvidence, Interaction


_MEASUREMENT = re.compile(
    r"(?<![\w-])(?P<first>\d+(?:\.\d+)?)"
    r"(?:\s*(?:to|[-–])\s*(?P<second>\d+(?:\.\d+)?))?"
    r"\s*(?P<unit>mm/s(?:\s+rms)?|bar|percent|%|rpm|°?c(?:elsius)?)"
    r"(?=\W|$)",
    re.IGNORECASE,
)
_ASSET = re.compile(r"\b[A-Za-z]+-\d+\b", re.IGNORECASE)
_ABSENCE = re.compile(
    r"\b(?:does\s+not\s+(?:include|contain|provide|record)|"
    r"cannot\s+determine|can't\s+determine|"
    r"(?:has|have|contains?|includes?)\s+no|"
    r"(?:is|are|was|were)\s+not\s+(?:available|provided|recorded|documented))\b",
    re.IGNORECASE,
)
_TOPICS = {
    "load": (re.compile(r"\bload\b", re.IGNORECASE), {"%"}),
    "pressure": (re.compile(r"\b(?:discharge[- ]?)?pressure\b", re.IGNORECASE), {"bar"}),
    "vibration": (re.compile(r"\bvibration\b", re.IGNORECASE), {"mm/s"}),
    "temperature": (re.compile(r"\btemperature\b", re.IGNORECASE), {"c"}),
}
_POLICY_ASSERTION = re.compile(
    r"\b(?:system|developer)\s+(?:policy|instruction)\b|"
    r"\b(?:system|developer)\s+(?:is|was)\s+instructed\b",
    re.IGNORECASE,
)
_POLICY_DENIAL = re.compile(
    r"\b(?:not|never|no)\s+(?:a\s+|the\s+)?(?:system|developer)\s+(?:policy|instruction)\b|"
    r"\bnot\s+an\s+instruction\b",
    re.IGNORECASE,
)
_REPORTED_CLAIM = re.compile(r"\b(?:claims?\s+to\s+be|purports?\s+to\s+be|quotes?|alleges?)\b", re.IGNORECASE)
_ENDORSEMENT = re.compile(r"\b(?:must|should|needs?\s+to)\s+be\s+(?:followed|obeyed)\b", re.IGNORECASE)
_NOTE_REFERENCE = re.compile(r"\b(?:operator[- ]note|retrieved\s+note|source\s+note)\b", re.IGNORECASE)
_INSTRUCTION_VERBS = ("ignore", "disclose", "reveal", "override", "bypass")
_DIRECT_UNTRUSTED_INSTRUCTION = re.compile(
    r"^\s*(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"(?:disclose|reveal)\s+(?:restricted\s+|confidential\s+)?\w+\s+records)\b",
    re.IGNORECASE,
)
_BEARING_TYPE = re.compile(r"\b(?:drive[- ]end\s+)?bearing\s+(?:is\s+)?type\s+[A-Za-z0-9][A-Za-z0-9-]*\b", re.IGNORECASE)
_REFERENCE_DENIAL = re.compile(
    r"\b(?P<target>(?:the\s+)?(?:[A-Za-z][A-Za-z-]*\s+){1,7}(?:SOP|manual|procedure|log))\s+"
    r"is\s+not\s+(?:explicitly\s+)?referenced\b",
    re.IGNORECASE,
)
_REVISION_SCOPE_ASSERTION = re.compile(
    r"\bRev\s+(?P<revision>\d+)\b[^.;]{0,100}\bcurrent\s+(?P<topic>pressure|vibration|temperature)\s+decisions\b",
    re.IGNORECASE,
)


def _asset_compatible(question: str, item: GroundingEvidence) -> bool:
    """A fact about another named asset cannot refute a missing-fact claim."""
    requested = _ASSET.search(question)
    mentioned = _ASSET.search(item.text)
    return requested is None or mentioned is None or requested.group().casefold() == mentioned.group().casefold()


def _named_reference_present(denial: str, source: str) -> bool:
    match = _REFERENCE_DENIAL.search(denial)
    if match is None:
        return False
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", match.group("target")) if token.casefold() not in {"a", "the"}]
    if len(tokens) < 2:
        return False
    if _REFERENCE_DENIAL.search(source):
        return False
    pattern = r".{0,32}".join(rf"\b{re.escape(token)}\b" for token in tokens)
    return bool(re.search(pattern, source, re.IGNORECASE | re.DOTALL))


def _documented_fact_refuting_absence(
    claim: Claim, evidence: list[GroundingEvidence], question: str,
) -> GroundingEvidence | None:
    """Refute only narrow, directly witnessed denials of available evidence.

    This does not infer that a missing work order or a partial document set is
    complete. Historical sources can establish *historical* facts, but cannot
    supply a current operating limit.
    """
    text = claim.verification_text
    context = f"{question} {text}"
    missing = bool(_ABSENCE.search(text) or re.search(
        r"\b(?:no\s+(?:bearing\s+type|evidence)|not\s+explicitly\s+referenced|not\s+documented|not\s+available)\b",
        text, re.IGNORECASE,
    ))
    if not missing:
        return None
    for item in evidence:
        status = str(item.metadata.get("document_status", "")).casefold()
        if status in {"unverified", "draft"} or not _asset_compatible(question, item):
            continue
        source = item.text
        if status != "superseded":
            if re.search(r"\bbearing\s+type\b", context, re.IGNORECASE) and _BEARING_TYPE.search(source):
                if not re.search(r"\bdrive[- ]end\b", context, re.IGNORECASE) or re.search(r"\bdrive[- ]end\b", source, re.IGNORECASE):
                    return item
            if _named_reference_present(text, source):
                return item
        if re.search(r"\b(?:age|older|historical)\b", context, re.IGNORECASE) and re.search(r"\bthreshold\b", context, re.IGNORECASE):
            requested = {value for _, value in _measurements(question)}
            historical = bool(re.search(r"\b(?:historical|older|former)\b", source, re.IGNORECASE))
            if requested and historical and requested & _evidence_values(item):
                return item
    return None


def _cross_family_revision_scope(claim: Claim, evidence: list[GroundingEvidence]) -> GroundingEvidence | None:
    """A revision in one document family does not govern another measured topic."""
    assertion = _REVISION_SCOPE_ASSERTION.search(claim.verification_text)
    if assertion is None:
        return None
    revision = assertion.group("revision")
    topic = assertion.group("topic").casefold()
    if any(
        str(item.metadata.get("document_status", "")).casefold() == "current"
        and (str(item.revision or "") == revision or re.search(rf"\bRev\s+{re.escape(revision)}\b", item.text, re.IGNORECASE))
        and re.search(rf"\bcurrent\s+{topic}\s+decisions\b", item.text, re.IGNORECASE)
        and not re.search(rf"\bnot\s+valid\s+for\s+current\s+{topic}\s+decisions\b", item.text, re.IGNORECASE)
        for item in evidence
    ):
        return None
    supersession = re.compile(rf"\bsuperseded\s+by\s+Rev\s+{re.escape(revision)}\b", re.IGNORECASE)
    for item in evidence:
        family = item.metadata.get("document_family")
        if not family or not supersession.search(item.text) or re.search(rf"\b{topic}\b", item.text, re.IGNORECASE):
            continue
        other_family = any(
            peer.metadata.get("document_family") not in {None, family}
            and re.search(rf"\bcurrent\s+{topic}\s+decisions\b", peer.text, re.IGNORECASE)
            and re.search(r"\bsuperseded\s+by\s+Rev\s+\d+\b", peer.text, re.IGNORECASE)
            for peer in evidence
        )
        if other_family:
            return item
    return None


def _measurements(text: str) -> list[tuple[re.Match[str], tuple[str, str | None, str]]]:
    found = []
    for match in _MEASUREMENT.finditer(text):
        unit = match.group("unit").casefold().replace(" ", "")
        unit = "%" if unit in {"%", "percent"} else "mm/s" if unit.startswith("mm/s") else "c" if unit in {"°c", "c", "celsius"} else unit
        first = str(Decimal(match.group("first")).normalize())
        second = str(Decimal(match.group("second")).normalize()) if match.group("second") else None
        found.append((match, (first, second, unit)))
    return found


def _asserted_current_measurements(text: str) -> set[tuple[str, str | None, str]]:
    """Keep 'current X' / 'X is current', excluding historical or negated uses."""
    output = set()
    for match, value in _measurements(text):
        before = re.split(r"[.;]", text[:match.start()])[-1]
        after = re.split(r"[.;]", text[match.end():])[0]
        preceding = before[-95:]
        following = after[:50]
        positive_before = bool(re.search(r"\bcurrent\b", preceding, re.IGNORECASE))
        positive_after = bool(re.search(r"\b(?:is|are|remains?)\s+current\b", following, re.IGNORECASE))
        denied = bool(re.search(r"\b(?:not|never|no\s+longer)\s+(?:the\s+)?current\b|\b(?:cannot|can't)\s+be\s+used\b", before + following, re.IGNORECASE))
        historical = bool(re.search(r"\b(?:superseded|historical|formerly)\b", preceding, re.IGNORECASE))
        if (positive_before or positive_after) and not denied and not historical:
            output.add(value)
    return output


def _evidence_values(item: GroundingEvidence) -> set[tuple[str, str | None, str]]:
    return {value for _, value in _measurements(item.text)}


def _current_evidence_values(item: GroundingEvidence) -> set[tuple[str, str | None, str]]:
    """A current document may affirm a value without spelling out 'current'."""
    values = set()
    for segment in re.split(r";|\n|(?<=\.)\s+", item.text):
        if re.search(r"\b(?:old|historical|obsolete|superseded)\b|\bnot\s+valid\b", segment, re.IGNORECASE):
            continue
        values.update(value for _, value in _measurements(segment))
    return values


def _superseded_current_claim(claim: Claim, evidence: list[GroundingEvidence]) -> GroundingEvidence | None:
    values = _asserted_current_measurements(claim.verification_text)
    if not values:
        return None
    for old in evidence:
        if str(old.metadata.get("document_status", "")).casefold() != "superseded":
            continue
        overlap = values & _evidence_values(old)
        if not overlap:
            continue
        family = old.metadata.get("document_family")
        reaffirmed = any(
            str(item.metadata.get("document_status", "")).casefold() == "current"
            and family is not None and item.metadata.get("document_family") == family
            and bool(overlap & _current_evidence_values(item))
            for item in evidence
        )
        if not reaffirmed:
            return old
    return None


def _recorded_measurement_refuting_absence(claim: Claim, evidence: list[GroundingEvidence]) -> GroundingEvidence | None:
    text = claim.verification_text
    if not _ABSENCE.search(text):
        return None
    asset_match = _ASSET.search(text)
    if asset_match is None:
        return None
    asset = asset_match.group().casefold()
    topics = [(pattern, units) for pattern, units in _TOPICS.values() if pattern.search(text)]
    if not topics:
        return None
    for item in evidence:
        if str(item.metadata.get("document_status", "")).casefold() in {"superseded", "unverified"}:
            continue
        primary_asset = _ASSET.search(item.text)
        if primary_asset is None or primary_asset.group().casefold() != asset:
            continue
        for segment in re.split(r";|\n|(?<=\.)\s+", item.text):
            for pattern, units in topics:
                if not pattern.search(segment):
                    continue
                if any(value[2] in units for _, value in _measurements(segment)):
                    return item
    return None


def _untrusted_policy_promotion(claim: Claim, evidence: list[GroundingEvidence], response: str) -> GroundingEvidence | None:
    text = claim.verification_text
    parent = response[claim.parent_span[0]:claim.parent_span[1]] if claim.parent_span else text
    if _POLICY_DENIAL.search(text) or (_REPORTED_CLAIM.search(text) and not _ENDORSEMENT.search(parent)) or not (_POLICY_ASSERTION.search(text) or _DIRECT_UNTRUSTED_INSTRUCTION.search(text)):
        return None
    for item in evidence:
        kind = str(item.metadata.get("document_type") or item.source_name or "").casefold()
        trust = str(item.metadata.get("trust_level", "")).casefold()
        untrusted = trust == "untrusted" or "untrusted" in kind or "operator_note" in kind
        linked = _NOTE_REFERENCE.search(text) or any(
            re.search(rf"\b{verb}\b", text, re.IGNORECASE)
            and re.search(rf"\b{verb}\b", item.text, re.IGNORECASE)
            for verb in _INSTRUCTION_VERBS
        )
        if untrusted and linked:
            return item
    return None


def apply_evidence_authority_checks(interaction: Interaction, claims: list[Claim]) -> dict[str, int]:
    """Override semantic support only for explicit, provenance-checkable errors."""
    evidence = interaction.grounding_evidence
    counts = {"superseded_current": 0, "recorded_value_denied": 0, "untrusted_instruction_promotion": 0, "documented_fact_denied": 0, "cross_family_revision_scope": 0}
    if not evidence:
        return counts
    for claim in claims:
        checks = (
            ("untrusted_instruction_promotion", lambda candidate, sources: _untrusted_policy_promotion(candidate, sources, interaction.response)),
            ("superseded_current", _superseded_current_claim),
            ("recorded_value_denied", _recorded_measurement_refuting_absence),
            ("documented_fact_denied", lambda candidate, sources: _documented_fact_refuting_absence(candidate, sources, interaction.prompt)),
            ("cross_family_revision_scope", _cross_family_revision_scope),
        )
        for reason, check in checks:
            item = check(claim, evidence)
            if item is None:
                continue
            claim.status = VerificationStatus.CONTRADICTED
            claim.confidence = 1.0
            claim.verifier_used = f"evidence_authority:{reason}"
            claim.evidence = [item.text]
            claim.evidence_ids = [item.evidence_id]
            claim.verification_metadata.update({"authority_reason": reason, "authority_evidence_id": item.evidence_id})
            counts[reason] += 1
            break
    return counts
