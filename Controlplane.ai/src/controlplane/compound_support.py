"""High-precision support for a narrow two-source numeric range assertion.

This does not infer authority from retrieved prose: both cited source IDs must be
present, the range source must be current, and the recorded value must be in a
separate authorized observation for the same requested asset.
"""

from __future__ import annotations

import re
from decimal import Decimal

from adaptivefact.data.schema import Claim, VerificationStatus
from adaptivefact.verification.text import split_sentences
from controlplane.schema import GroundingEvidence, Interaction


_CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9:_-]{2,127})\]")
_ASSET = re.compile(r"\b[A-Za-z]+-\d+\b")
_ASSERTION = re.compile(
    r"(?:the\s+)?(?:recorded\s+)?"
    r"(?P<metric>discharge\s+pressure|pressure|bearing\s+temperature|temperature|speed|load|vibration)\s+"
    r"(?:of|was|is)\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>bar|mm/s(?:\s+RMS)?|°?C|rpm|%|percent)"
    r"(?:\s+on\s+(?P<timestamp>\d{4}-\d{2}-\d{2}T[0-9:]+Z))?(?:\s+recorded\s+in)?\s+"
    r"(?:is|was|falls?)\s+within\s+(?:the\s+)?current\s+(?:operating\s+)?(?:range|band)",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:within|outside|above|below|exceeds?|higher|lower)\b.{0,100}"
    r"\b(?:range|band|threshold|limit)\b", re.IGNORECASE,
)
_GLOBAL_ABSENCE = re.compile(
    r"\bno\s+(?:authorized\s+)?(?:evidence|document|procedure)s?\b|"
    r"\bnot\b.{0,40}\bany\s+(?:retrieved\s+|authorized\s+)?documents?\b", re.IGNORECASE,
)
_RANGE = re.compile(
    r"(?P<lower>\d+(?:\.\d+)?)\s*(?:to|[-–])\s*"
    r"(?P<upper>\d+(?:\.\d+)?)\s*(?P<unit>bar|mm/s(?:\s+RMS)?|°?C|rpm|%|percent)(?=\W|$)",
    re.IGNORECASE,
)


def _unit(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.casefold())
    return "%" if normalized == "percent" else normalized


def _cited_sources(claim: Claim, interaction: Interaction) -> list[GroundingEvidence] | None:
    ids = _CITATION.findall(claim.verification_text)
    if len(ids) != 2 or ids[0] == ids[1]:
        return None
    by_id = {item.evidence_id: item for item in interaction.grounding_evidence}
    if any(item_id not in by_id for item_id in ids):
        return None
    return [by_id[item_id] for item_id in ids]


def cited_numeric_range_support(claim: Claim, interaction: Interaction) -> list[GroundingEvidence] | None:
    """Return both sources only when every part of a comparison is witnessed."""
    sources = _cited_sources(claim, interaction)
    if sources is None:
        return None
    clean = re.sub(r"\s+", " ", _CITATION.sub("", claim.verification_text)).strip().rstrip(". ")
    clean = re.sub(r",?\s*(?:as per|per|according to|supported by)\s*$", "", clean, flags=re.IGNORECASE).strip()
    asserted_band = None
    explicit_band = re.search(r"\s+of\s+" + _RANGE.pattern + r"\s+specified\s+in$", clean, re.IGNORECASE)
    if explicit_band:
        asserted_band = (Decimal(explicit_band.group("lower")), Decimal(explicit_band.group("upper")), _unit(explicit_band.group("unit")))
        clean = clean[:explicit_band.start()]
    assertion = _ASSERTION.fullmatch(clean)
    asset_match = _ASSET.search(interaction.prompt)
    if assertion is None:
        return None
    if asset_match is None:
        shared_assets = set.intersection(*(
            {item.casefold() for item in _ASSET.findall(source.text)} for source in sources
        ))
        if len(shared_assets) != 1:
            return None
        asset = next(iter(shared_assets))
    else:
        asset = asset_match.group().casefold()
    asset_family = asset.rsplit("-", 1)[0] + "-"
    def matches_asset(text: str) -> bool:
        assets = {item.casefold() for item in _ASSET.findall(text)
                  if item.casefold().startswith(asset_family)}
        return assets == {asset}
    metric = re.sub(r"\s+", " ", assertion.group("metric").casefold())
    value = Decimal(assertion.group("value"))
    unit = _unit(assertion.group("unit"))
    timestamp = assertion.group("timestamp")
    metric_pattern = re.compile(rf"\b{re.escape(metric)}\b", re.IGNORECASE)

    for observation in sources:
        if str(observation.metadata.get("document_status", "")).casefold() != "current":
            continue
        if not matches_asset(observation.text):
            continue
        if not any(
            metric_pattern.search(sentence)
            and not re.search(r"\b(?:not|hypothetical|example|target|simulated)\b", sentence, re.IGNORECASE)
            and re.search(rf"(?<![\w.]){re.escape(assertion.group('value'))}\s*{re.escape(assertion.group('unit'))}(?!\w)", sentence, re.IGNORECASE)
            and (not timestamp or timestamp in sentence)
            and asset in sentence.casefold()
            for sentence in split_sentences(observation.text)
        ):
            continue
        for range_source in sources:
            if range_source.evidence_id == observation.evidence_id:
                continue
            if str(range_source.metadata.get("document_status", "")).casefold() != "current":
                continue
            if not matches_asset(range_source.text):
                continue
            eligible_sentences = [sentence for sentence in split_sentences(range_source.text)
                                  if metric_pattern.search(sentence)
                                  and re.search(r"\b(?:current|normal|operating)\b", sentence, re.IGNORECASE)
                                  and not re.search(r"\b(?:old|historical|obsolete|superseded|not|example|hypothetical)\b", sentence, re.IGNORECASE)]
            bands = {(Decimal(band.group("lower")), Decimal(band.group("upper")))
                     for sentence in eligible_sentences for band in _RANGE.finditer(sentence)
                     if _unit(band.group("unit")) == unit}
            if len(bands) != 1:
                continue
            for sentence in split_sentences(range_source.text):
                if not metric_pattern.search(sentence) or not re.search(r"\b(?:current|normal|operating)\b", sentence, re.IGNORECASE):
                    continue
                if re.search(r"\b(?:old|historical|obsolete|superseded)\b|\bnot\s+valid\b", sentence, re.IGNORECASE):
                    continue
                for band in _RANGE.finditer(sentence):
                    if _unit(band.group("unit")) != unit:
                        continue
                    if asserted_band is not None and asserted_band != (
                        Decimal(band.group("lower")), Decimal(band.group("upper")), _unit(band.group("unit")),
                    ):
                        continue
                    if Decimal(band.group("lower")) <= value <= Decimal(band.group("upper")):
                        return [observation, range_source]
    return None


def apply_compound_support(interaction: Interaction, claims: list[Claim]) -> int:
    supported = 0
    for claim in claims:
        if claim.status not in {VerificationStatus.UNKNOWN, VerificationStatus.SUPPORTED}:
            continue
        sources = cited_numeric_range_support(claim, interaction)
        if sources is None:
            if interaction.grounding_evidence and _COMPARISON.search(claim.verification_text):
                claim.status = VerificationStatus.UNDECIDABLE
                claim.verifier_used = "comparison_evidence_incomplete"
                claim.verification_metadata["support_guard_reason"] = "Both observation and current limit must be verified for a comparison."
            elif _GLOBAL_ABSENCE.search(claim.verification_text) and not (
                interaction.grounding_evidence and all(
                    item.metadata.get("evidence_set_complete") is True for item in interaction.grounding_evidence
                )
            ):
                claim.status = VerificationStatus.UNDECIDABLE
                claim.verifier_used = "global_absence_requires_complete_evidence"
                claim.verification_metadata["support_guard_reason"] = "Retrieved top-k evidence cannot establish a global absence of evidence."
            continue
        claim.status = VerificationStatus.SUPPORTED
        claim.verification_metadata.pop("support_guard_reason", None)
        claim.confidence = 0.98
        claim.verifier_used = "deterministic_cited_numeric_range"
        claim.evidence_ids = [item.evidence_id for item in sources]
        claim.evidence = [item.text for item in sources]
        claim.verification_metadata["support_score_basis"] = "deterministic_two_source_range_check"
        supported += 1
    return supported
