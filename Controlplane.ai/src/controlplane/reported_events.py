"""Exact cited review-request events; not operational authorization."""
import re

from adaptivefact.data.schema import VerificationStatus
from adaptivefact.verification.text import split_sentences

_CITE = re.compile(r'\[([A-Za-z0-9][A-Za-z0-9:_-]{2,127})\]')
_ASSET = re.compile(r'(?<![\w-])[A-Za-z]+-\d+\b')
_EVENT = re.compile(
    r'(?:the\s+)?(?P<actor>operator|technician|inspector|reviewer)\s+'
    r'(?P<verb>requested|recorded|reported)\s+'
    r'(?P<event>maintenance review|inspection review|measurement recheck|engineering review)\.?',
    re.IGNORECASE,
)
_TIME = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z')


def apply_cited_review_events(interaction, claims):
    """Prove only a complete, verbatim event in a cited current observation.

    Never proves work permission, a diagnosis, global absence or imperative
    document text. Reports the record's event, not physical attestation.
    """
    assets={a.casefold() for a in _ASSET.findall(interaction.prompt)}
    if len(assets)!=1:
        return 0
    asset=next(iter(assets))
    timestamp=_TIME.search(interaction.prompt)
    def eligible(source):
        metadata=source.metadata
        return (metadata.get('document_status')=='current'
                and metadata.get('trust_level')!='untrusted'
                and metadata.get('document_type') in {'operations_log','inspection_record','inspection_report','maintenance_history'}
                and {a.casefold() for a in _ASSET.findall(source.text)}=={asset}
                and (timestamp is None or set(_TIME.findall(source.text))=={timestamp.group()})
                and not re.search(r'\b(?:hypothetical|example)\b',source.text,re.IGNORECASE))
    records=[s for s in interaction.grounding_evidence if eligible(s)]
    by_id={s.evidence_id:s for s in records}
    count=0
    for claim in claims:
        citations=_CITE.findall(claim.verification_text)
        clean=_CITE.sub('',claim.verification_text).strip().rstrip('.').strip()
        event=_EVENT.fullmatch(clean)
        if event is None or len(citations)!=1 or citations[0] not in by_id:
            continue
        source=by_id[citations[0]]
        if not any(sentence.strip().rstrip('.').casefold()==clean.casefold()
                   for sentence in split_sentences(source.text)):
            continue
        negative=re.compile(rf"\b{event['actor']}\b[^.!?]{{0,60}}\b(?:not|never|no)\b[^.!?]{{0,40}}\b{event['event']}\b",re.IGNORECASE)
        conflicts=[s for s in records if negative.search(s.text)]
        claim.status=VerificationStatus.CONFLICTING if conflicts else VerificationStatus.SUPPORTED
        claim.confidence=1.0
        claim.verifier_used='cited_review_event_conflict' if conflicts else 'cited_review_event_exact'
        claim.evidence_ids=[s.evidence_id for s in [source,*conflicts]]
        claim.evidence=[s.text for s in [source,*conflicts]]
        claim.verification_metadata['review_event_basis']='complete_cited_current_record_statement_not_authorization'
        count+=1
    return count
