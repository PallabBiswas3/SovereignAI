"""Opt-in source-typed assessment contract; never replaces factual verification."""
import re

from .reporting import render_reported_facts

FIELDS = ('observations', 'operating_limits', 'diagnostic_findings', 'review_disposition')


def assessment_requested(query: str) -> bool:
    return bool(re.search(r'\bstructured assessment\b',query,re.IGNORECASE))


def assessment_sources(graph_evidence: list[dict], diagnostic_evidence: list[dict]) -> dict:
    groups={key:{} for key in FIELDS}
    for item in graph_evidence:
        metadata=item.get('metadata') or {}
        if metadata.get('document_status')!='current' or metadata.get('trust_level')=='untrusted':
            continue
        kind=metadata.get('document_type')
        evidence_id=item['evidence_id']
        if kind in {'operations_log','inspection_report','inspection_record','maintenance_history'}:
            groups['observations'][evidence_id]=item['text']
            groups['review_disposition'][evidence_id]=item['text']
        if kind in {'equipment_manual','sop','operating_limits'}:
            groups['operating_limits'][evidence_id]=item['text']
            groups['review_disposition'][evidence_id]=item['text']
    groups['diagnostic_findings']={item['evidence_id']:item['text'] for item in diagnostic_evidence}
    return groups


def assessment_statements(groups: dict, query: str) -> dict:
    """Bound the opt-in assessment to mechanically extractable source facts.

    No numerical inference, unit conversion or repair advice is performed.
    Documents must name one unambiguous requested asset. Range and scalar
    relations stay separate, and a requested timestamp must actually occur.
    """
    asset_pattern=r'(?<![\w-])[A-Za-z]+-\d+\b'
    assets=set(re.findall(asset_pattern,query))
    result={key:{} for key in FIELDS}
    result['diagnostic_findings']={k:[v] for k,v in groups['diagnostic_findings'].items()}
    if len(assets)!=1:
        return result
    asset=next(iter(assets))
    metric=r'discharge\s+pressure|pressure|vibration|temperature|speed|load'
    unit=r'bar|mm/s(?:\s+RMS)?|rpm|percent|%|°?C'
    wanted_time=re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',query)
    for key in ('observations','operating_limits','review_disposition'):
        for evidence_id,text in groups[key].items():
            named={a.casefold() for a in re.findall(asset_pattern,text)}
            if named!={asset.casefold()}:
                continue
            candidates=[]
            if key=='observations':
                # A shift-log line can bind several semicolon-delimited readings
                # to its explicit asset and timestamp. Never cross sentence boundaries.
                for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',text):
                    if not re.search(rf'\b{re.escape(asset)}\b',sentence,re.IGNORECASE):
                        continue
                    if re.search(r'\b(?:if|unless|could|would|should|might|hypothetical|example|not)\b',sentence,re.IGNORECASE):
                        continue
                    timestamp=re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',sentence)
                    if wanted_time and (timestamp is None or timestamp.group()!=wanted_time.group()):
                        continue
                    for clause in sentence.split(';'):
                        if re.search(r'\b(?:normal|range|above|below|hypothetical|example|not|target)\b',clause,re.IGNORECASE):
                            continue
                        for match in re.finditer(rf'\b(?P<metric>{metric})\s+(?:(?:was|is|of|recorded\s+as)\s+)?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{unit})(?=\W|$)',clause,re.IGNORECASE):
                            if 'pressure' in query.casefold() and 'pressure' not in match['metric'].casefold():
                                continue
                            fact=f"{asset} {match['metric'].casefold()} was {match['value']} {match['unit']}"
                            if timestamp:
                                fact+=' at '+timestamp.group()
                            candidates.append(fact+'.')
            elif key=='operating_limits':
                for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',text):
                    match=re.fullmatch(rf'(?:{re.escape(asset)}\s+)?normal\s+(?P<metric>{metric})\s+is\s+(?P<low>\d+(?:\.\d+)?)\s+to\s+(?P<high>\d+(?:\.\d+)?)\s*(?P<unit>{unit})\.',sentence.strip(),re.IGNORECASE)
                    if match:
                        candidates.append(f"{asset} normal {match['metric'].casefold()} is {match['low']} to {match['high']} {match['unit']}.")
            else:
                # Quote only the existing review request, never invent a required
                # intervention or make a recommendation from a threshold alone.
                for match in re.finditer(r'\bOperator requested maintenance review\.',text,re.IGNORECASE):
                    candidates.append(match.group())
            if candidates:
                result[key][evidence_id]=list(dict.fromkeys(candidates))
    return result


def assessment_schema(groups: dict, statements: dict | None = None) -> dict:
    properties={}
    for key in FIELDS:
        ids=list(groups[key])
        statement={'type':'string','minLength':1,'maxLength':350}
        if key=='diagnostic_findings' and ids:
            statement['enum']=list(groups[key].values())
        elif statements is not None:
            allowed=[s for values in statements[key].values() for s in values]
            if not allowed:
                raise ValueError('no extractable facts for assessment group: '+key)
            statement['enum']=allowed
        citations={'type':'array','minItems':1,'maxItems':1 if statements is not None else 2,'uniqueItems':True,
                   'items':{'type':'string',**({'enum':ids} if ids else {})}}
        properties[key]={'type':'array','minItems':1 if ids else 0,'maxItems':1 if ids else 0,
                         'items':{'type':'object','additionalProperties':False,
                                  'required':['statement','citations'],
                                  'properties':{'statement':statement,'citations':citations}}}
    return {'type':'object','additionalProperties':False,'required':list(FIELDS),'properties':properties}


def render_assessment(data, groups: dict, statements: dict | None = None) -> str:
    if not isinstance(data,dict) or set(data)!=set(FIELDS):
        raise ValueError('assessment must contain the four declared sections')
    lines=[]
    for key in FIELDS:
        items=data[key]
        if not isinstance(items,list) or len(items)!=(1 if groups[key] else 0):
            raise ValueError('assessment coverage does not match available source groups')
        if not items:
            raise ValueError('requested assessment evidence group is unavailable')
        if statements is not None:
            item=items[0]
            if (not isinstance(item,dict) or set(item)!={'statement','citations'}
                or not isinstance(item['citations'],list) or len(item['citations'])!=1
                or item['citations'][0] not in statements[key]
                or item['statement'] not in statements[key][item['citations'][0]]):
                raise ValueError('assessment statement does not match its cited extracted fact')
        if key=='diagnostic_findings':
            item=items[0]
            if (not isinstance(item,dict) or set(item)!={'statement','citations'}
                or not isinstance(item['citations'],list) or len(item['citations'])!=1
                or item['citations'][0] not in groups[key]
                or item['statement']!=groups[key][item['citations'][0]]):
                raise ValueError('diagnostic finding must quote its identified tool evidence')
            # Preserve the complete machine statement, including its clauses;
            # claim extraction/verification must still check every clause.
            lines.append(item['statement'].rstrip('. ')+' ['+item['citations'][0]+'].')
        else:
            lines.append(render_reported_facts({'facts':items},1,set(groups[key])))
    return '\n'.join(lines)
