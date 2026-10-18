"""All evidence pairs survive optimization; only identical computation is shared."""
import torch
from adaptivefact.verification.nli import TransformersNLIScorer


class Tokenizer:
    def __init__(self):
        self.inputs = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.inputs.extend(zip(premises, hypotheses))
        if kwargs.get('return_tensors') == 'pt':
            return {'input_ids': torch.tensor([[len(p), len(h)] for p,h in zip(premises,hypotheses)])}
        return {'input_ids': [[len(p), len(h)] + [1] * len(p) for p,h in zip(premises,hypotheses)]}

    def pad(self, encoded, **kwargs):
        return {'input_ids': torch.tensor([row[:2] for row in encoded['input_ids']])}


class Model:
    config = type('Config', (), {'_commit_hash': 'synthetic-test-revision'})()
    def __init__(self):
        self.seen = []
        self.batches = []

    def __call__(self, input_ids):
        self.seen.extend(input_ids.tolist())
        self.batches.append(input_ids.tolist())
        logits = torch.stack((input_ids[:,0], input_ids[:,1], torch.zeros(len(input_ids))), dim=1).float()
        return type('Output', (), {'logits': logits})()


def scorer():
    obj = TransformersNLIScorer.__new__(TransformersNLIScorer)
    obj.torch = torch
    obj.tokenizer = Tokenizer()
    obj.model = Model()
    obj.device = torch.device('cpu')
    obj.batch_size = 2
    obj.max_length = 512
    obj.label_map = {0:'contradiction',1:'entailment',2:'neutral'}
    obj.profile_model = 'synthetic-test-model'
    return obj


def test_identical_pairs_are_inferred_once_but_results_keep_original_positions():
    nli = scorer()
    p = ['short', 'a longer source', 'short', 'short']
    h = ['claim', 'x', 'claim', 'different']
    results = nli.score(p,h)
    assert len(nli.model.seen) == 3
    assert len(results) == 4
    assert results[0] == results[2]
    assert results[0] != results[3]
    assert results[1].contradiction > results[0].contradiction


def test_changed_evidence_is_never_reused_across_calls():
    nli = scorer()
    first = nli.score(['healthy'],['claim'])
    second = nli.score(['faulty source'],['claim'])
    assert first != second
    assert len(nli.model.seen) == 2


def test_small_mixed_length_batch_avoids_padding_short_pairs_to_long_manuals():
    nli=scorer()
    nli.batch_size=32
    results=nli.score(['s','t','x'*100,'y'*110],['a','b','c','d'])
    assert len(results)==len(nli.model.seen)==4
    assert [len(batch) for batch in nli.model.batches]==[2,2]


def test_empty_input_does_not_infer_and_invalid_alignment_is_rejected():
    import pytest
    nli = scorer()
    assert nli.score([],[]) == []
    with pytest.raises(ValueError):
        nli.score(['a'],[])
    assert not nli.model.seen


def test_request_memo_reuses_exact_pairs_and_does_not_cross_requests():
    from controlplane.verification import LazyTransformersNLIScorer
    lazy = LazyTransformersNLIScorer('fake')
    lazy._delegate = scorer()
    with lazy.request_scope() as runtime:
        a = lazy.score(['source'],['claim'])
        b = lazy.score(['source','new source'],['claim','claim'])
        assert a[0] == b[0]
        a[0].entailment = 0.123
        assert lazy.score(['source'],['claim'])[0] != a[0]
        assert runtime['requested_pairs'] == 4
        assert runtime['inferred_pairs'] == 2
    with lazy.request_scope() as next_runtime:
        lazy.score(['source'],['claim'])
        assert next_runtime['inferred_pairs'] == 1
    assert len(lazy._delegate.model.seen) == 3


def test_request_memo_resets_on_exception_and_is_context_isolated():
    from contextvars import copy_context
    from controlplane.verification import LazyTransformersNLIScorer
    import pytest
    lazy = LazyTransformersNLIScorer('fake')
    lazy._delegate = scorer()
    outer = copy_context()
    with pytest.raises(RuntimeError):
        with lazy.request_scope():
            lazy.score(['source'],['claim'])
            assert outer.run(lazy._request_scores.get) is None
            raise RuntimeError('test')
    assert lazy._request_scores.get() is None


def test_uncited_obsolete_limits_cannot_challenge_current_facts_but_current_conflicts_can():
    from adaptivefact.verification.phase6 import Phase6Pipeline
    pipeline = Phase6Pipeline(scorer())
    items = [
        {'evidence_id':'current-manual','text':'Pump-731 normal discharge pressure is 3.2 to 4.1 bar.', 'metadata':{'document_status':'current'}},
        {'evidence_id':'old-manual','text':'Pump-731 normal discharge pressure is 1.0 to 2.0 bar.', 'metadata':{'document_status':'superseded'}},
        {'evidence_id':'other-current','text':'Pump-731 normal discharge pressure is 2.2 to 2.9 bar.', 'metadata':{'document_status':'current'}},
    ]
    texts,_ = pipeline._retrieve_structured(items,'Pump-731 current normal discharge pressure is 3.2 to 4.1 bar [current-manual].')
    assert items[1]['text'] not in texts
    assert items[0]['text'] in texts and items[2]['text'] in texts
    historical,_ = pipeline._retrieve_structured(items,'Pump-731 superseded manual listed pressure 1.0 to 2.0 bar [old-manual].')
    assert items[1]['text'] in historical
    improperly_cited,_ = pipeline._retrieve_structured(items,'Pump-731 current pressure is 1.0 to 2.0 bar [old-manual].')
    assert items[1]['text'] in improperly_cited  # Authority guard must reject the deliberate misuse.


def test_measurement_and_operating_range_are_not_competing_factual_relations():
    from adaptivefact.verification.phase6 import Phase6Pipeline
    from adaptivefact.verification.nli import NLIScores
    from adaptivefact.data.schema import VerificationStatus
    pipeline = Phase6Pipeline(scorer())
    support = NLIScores(.99,.005,.005)
    contradiction = NLIScores(.005,.99,.005)
    log = 'Pump-731 discharge pressure was 3.7 bar.'
    manual = 'Pump-731 normal discharge pressure is 3.2 to 4.1 bar.'
    state,*_ = pipeline._decide([support,contradiction],[log,manual],[1.,1.],'Pump-731 discharge pressure',log)
    assert state == VerificationStatus.SUPPORTED
    state,*_ = pipeline._decide([support,contradiction],[manual,log],[1.,1.],'Pump-731 discharge pressure',manual)
    assert state == VerificationStatus.SUPPORTED
    wrong_log = 'Pump-731 discharge pressure was 8.9 bar.'
    state,*_ = pipeline._decide([contradiction],[wrong_log],[1.],'Pump-731 discharge pressure',log)
    assert state == VerificationStatus.CONTRADICTED
    other_manual = 'Pump-731 normal discharge pressure is 2.2 to 2.9 bar.'
    state,*_ = pipeline._decide([support,contradiction],[manual,other_manual],[1.,1.],'Pump-731 discharge pressure',manual)
    assert state == VerificationStatus.CONFLICTING
