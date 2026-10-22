"""Literal ranges retain every potentially conflicting current range."""
import pytest
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import NLIScorer
from adaptivefact.verification.phase6 import Phase6Pipeline


class UnusedNLI(NLIScorer):
    def score(self, premises, hypotheses):
        raise AssertionError("This test exercises evidence selection only")

QUERY = "Pump-731 normal discharge pressure is 3.2 to 4.1 bar [manual-1]."


def item(identifier, text, status="current"):
    return {"evidence_id": identifier, "text": text,
            "metadata": {"document_status": status, "document_type": "equipment_manual"}}


def retrieve(extra, query=QUERY):
    pipeline = Phase6Pipeline(UnusedNLI(), retrieval_config=EvidenceRetrieverConfig(top_k=2))
    manual = item("manual-1", "Pump-731 normal discharge pressure is 3.2 to 4.1 bar.")
    return pipeline._retrieve_structured([manual, *extra], query, include_coverage=True)


def test_scalar_readings_and_other_metrics_do_not_consume_range_coverage():
    extras = [item("log", "Pump-731 discharge pressure was 3.7 bar."),
              item("temperature", "Pump-731 normal temperature is 70 to 80 C."),
              item("draft", "Pump-731 normal discharge pressure is 2.2 to 2.9 bar.", "draft")]
    texts, _, omitted = retrieve(extras)
    assert len(texts) == 1 and omitted == 0


@pytest.mark.parametrize("text", [
    "Pump-731 normal discharge pressure is 2.2 to 2.9 bar.",
    "Pump-731 normal discharge pressure is not 3.2 to 4.1 bar.",
    "Pump-731 normal discharge pressure is 3.2 to 4.1 rpm.",
    "If Pump-731 is at startup, normal discharge pressure is 2.2 to 2.9 bar.",
])
def test_current_competitors_remain_in_scope_and_omissions_hold(text):
    texts, _, omitted = retrieve([item("competing-1", text), item("competing-2", text.replace("Pump-731", "Pump-731 at load"))])
    assert len(texts) == 2 and omitted == 1


def test_deliberately_cited_draft_is_not_filtered():
    text = "Pump-731 normal discharge pressure is 2.2 to 2.9 bar."
    texts, _, _ = retrieve([item("draft-1", text, "draft")], QUERY.replace("manual-1", "draft-1"))
    assert text in texts


def test_extended_safety_claim_keeps_broad_evidence_obligation():
    text = "Pump-731 discharge pressure was 3.7 bar."
    texts, _, _ = retrieve([item("log", text)], QUERY.replace(" bar [", " bar and operation is safe ["))
    assert text in texts
