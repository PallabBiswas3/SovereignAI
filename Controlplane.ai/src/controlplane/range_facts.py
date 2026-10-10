"""Exact cited normal-range facts, independent of an NLI model's arithmetic."""
import re
from decimal import Decimal

from adaptivefact.data.schema import VerificationStatus
from adaptivefact.verification.text import split_sentences

_CITE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9:_-]{2,127})\]")
_FACT = re.compile(
    r"(?P<asset>[A-Za-z]+-\d+)\s+(?:current\s+)?normal\s+"
    r"(?P<metric>discharge\s+pressure|pressure|vibration|temperature|speed|load)\s+"
    r"(?:is|was)\s+(?P<low>\d+(?:\.\d+)?)\s*(?:to|[-–])\s*"
    r"(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>bar|mm/s(?:\s+RMS)?|rpm|percent|%|°?C)\.?",
    re.IGNORECASE,
)


def apply_cited_range_facts(interaction, claims):
    """Only a complete normal-range assertion can obtain an exact proof.

    Require one explicit citation to a current controlled document and the
    same asset, metric, endpoints and units in a complete evidence sentence.
    Any incompatible current controlled range remains CONFLICTING. Additional
    safety/authorization/diagnostic conclusions are outside this grammar.
    """
    by_id = {e.evidence_id:e for e in interaction.grounding_evidence}
    def trusted(e):
        kind = str(e.metadata.get('document_type') or e.source_name or '').casefold()
        return (str(e.metadata.get('document_status','')).casefold() == 'current'
                and str(e.metadata.get('trust_level','')).casefold() != 'untrusted'
                and kind in {'equipment_manual','sop','operating_limits','inspection_procedure'})
    def key(m):
        return (m['asset'].casefold(), ' '.join(m['metric'].casefold().split()),
                Decimal(m['low']),Decimal(m['high']),re.sub(r'\s+','',m['unit'].casefold()))
    count=0
    for claim in claims:
        ids=_CITE.findall(claim.verification_text)
        text=_CITE.sub('',claim.verification_text).strip().rstrip('.').strip()
        asserted=_FACT.fullmatch(text)
        if asserted is None or len(ids)!=1 or ids[0] not in by_id:
            continue
        witness=by_id[ids[0]]
        if not trusted(witness):
            continue
        wanted=key(asserted)
        requested_assets={a.casefold() for a in re.findall(r"(?<![\w-])[A-Za-z]+-\d+\b",interaction.prompt)}
        if requested_assets and requested_assets!={wanted[0]}:
            continue
        if wanted[2]>wanted[3]:
            continue
        facts=[]
        ambiguous_competitor=False
        for e in interaction.grounding_evidence:
            if not trusted(e):
                continue
            for sentence in split_sentences(e.text):
                matched=_FACT.fullmatch(sentence.strip())
                if matched is None:
                    # A manual can name the asset once, then state its range in
                    # the next sentence. Inherit only an unambiguous single
                    # explicit asset, never one selected from a mixed-asset text.
                    assets=set(re.findall(r"(?<![\w-])[A-Za-z]+-\d+\b",e.text))
                    if len(assets)==1:
                        matched=_FACT.fullmatch(next(iter(assets))+' '+sentence.strip())
                if matched:
                    facts.append((e,key(matched)))
                else:
                    # Do not override an NLI contradiction when a relevant
                    # current range is negated, conditional or otherwise
                    # outside this exact grammar. Scalar observations and
                    # unrelated operating-permission caveats are not ranges.
                    named={a.casefold() for a in re.findall(r"(?<![\w-])[A-Za-z]+-\d+\b",e.text)}
                    relevant=(not named or wanted[0] in named)
                    has_metric=bool(re.search(rf"\b{re.escape(wanted[1])}\b",sentence,re.IGNORECASE))
                    has_band=bool(re.search(r"\d+(?:\.\d+)?\s*(?:to|[-–])\s*\d+(?:\.\d+)?",sentence))
                    if relevant and has_metric and has_band:
                        ambiguous_competitor=True
        if not any(e.evidence_id==witness.evidence_id and fact==wanted for e,fact in facts):
            continue
        if ambiguous_competitor or any(fact[:2]==wanted[:2] and fact[4]!=wanted[4] for _,fact in facts):
            continue
        conflicts=[e for e,fact in facts if fact[:2]==wanted[:2] and fact[4]==wanted[4] and fact[2:4]!=wanted[2:4]]
        claim.status=VerificationStatus.CONFLICTING if conflicts else VerificationStatus.SUPPORTED
        claim.confidence=1.0
        claim.verifier_used='cited_normal_range_conflict' if conflicts else 'cited_normal_range_exact'
        sources=[witness,*conflicts]
        claim.evidence_ids=[e.evidence_id for e in sources]
        claim.evidence=[e.text for e in sources]
        claim.verification_metadata['range_fact_basis']='complete_cited_current_numeric_assertion'
        count+=1
    return count
