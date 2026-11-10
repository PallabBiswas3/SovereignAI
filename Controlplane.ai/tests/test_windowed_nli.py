"""Sentence-window NLI premises (10 November 2026).

The DeBERTa NLI cross-encoder was trained on one-to-two-sentence premises; whole
retrieved chunks dilute or invert its scores and are truncated at 512 tokens.
Phase 6 now scores structured evidence as short windows, aggregated back to the
source chunk. The stub below reproduces the real model's failure modes (a
multi-sentence premise or a leftover citation gives neutral) so these tests
check the mechanism, not the model.
"""
import pytest

from adaptivefact.verification.nli import NLIScorer, NLIScores
from controlplane.schema import GroundingEvidence, Interaction, VerificationDepth
from controlplane.verification import AdaptiveFactVerificationService

ENTAIL, CONTRADICT, NEUTRAL = NLIScores(0.95, 0.01, 0.04), NLIScores(0.01, 0.95, 0.04), NLIScores(0.02, 0.03, 0.95)
FILLER = " ".join(f"Step {i}: operators shall record ambient conditions and confirm guard integrity." for i in range(1, 9))


class ShortPremiseNLI(NLIScorer):
    """Decides only single-sentence premises; anything longer is 'neutral'."""

    def __init__(self):
        self.hypotheses = []

    def score(self, premises, hypotheses):
        self.hypotheses.extend(hypotheses)
        output = []
        for premise, hypothesis in zip(premises, hypotheses):
            if premise.count(". ") >= 1 or "[" in hypothesis:
                output.append(NEUTRAL)
                continue
            plan = "53B" if "53B" in premise else "11" if "plan 11" in premise else None
            if plan is None:
                output.append(NEUTRAL)
            elif ("53B" in hypothesis) == (plan == "53B"):
                output.append(ENTAIL)
            else:
                output.append(CONTRADICT)
        return output


def _claim(text: str, response: str, *, cite: str | None = "DS-220", windowed: bool = True, nli=None):
    nli = nli or ShortPremiseNLI()
    service = AdaptiveFactVerificationService(nli=nli)
    service.phase6.nli_config.window_premises = windowed
    evidence = [GroundingEvidence(evidence_id="DS-220", document_id="DS-220", chunk_id="DS-220-1", retrieval_score=0.8,
                                  text=text, metadata={"document_status": "current", "document_type": "equipment_manual"})]
    result = service.verify(Interaction(prompt="Which seal flush plan does Pump-102 use?", response=response,
                                        grounding_evidence=evidence, consequential=True), VerificationDepth.STANDARD)
    return result.metadata["claim_results"][0]


LONG = "Pump-102 was manufactured by KSB. " + FILLER + " Pump-102 mechanical seal uses flush plan 53B."


def test_fact_inside_long_chunk_is_supported_only_with_windows():
    response = "Pump-102 mechanical seal uses flush plan 53B [DS-220]."
    assert _claim(LONG, response, windowed=False)["status"] != "supported"
    claim = _claim(LONG, response)
    assert claim["status"] == "supported"


def test_wrong_fact_inside_long_chunk_is_contradicted_with_windows():
    assert _claim(LONG, "Pump-102 mechanical seal uses flush plan 11 [DS-220].")["status"] == "contradicted"


def test_window_about_another_asset_cannot_support():
    text = "Pump-103 mechanical seal uses flush plan 53B. Pump-102 mechanical seal uses flush plan 11."
    claim = _claim(text, "Pump-102 mechanical seal uses flush plan 53B [DS-220].")
    assert claim["status"] != "supported"


def test_window_about_another_asset_cannot_contradict():
    text = "Pump-103 mechanical seal uses flush plan 11. Pump-102 mechanical seal uses flush plan 53B."
    assert _claim(text, "Pump-102 mechanical seal uses flush plan 53B [DS-220].")["status"] == "supported"


def test_sentence_naming_asset_once_is_inherited_by_later_sentences():
    text = "Pump-102 datasheet. " + FILLER + " Mechanical seal uses flush plan 53B."
    assert _claim(text, "Pump-102 mechanical seal uses flush plan 53B [DS-220].")["status"] == "supported"


def test_windows_of_one_uncited_chunk_do_not_corroborate_each_other():
    # Two contradicting sentences in ONE uncited chunk are one source; the
    # corroboration rule needs two sources (or a deliberate citation).
    text = "Pump-102 mechanical seal uses flush plan 11. Pump-102 seal flush plan is plan 11 per the datasheet."
    claim = _claim(text, "Pump-102 mechanical seal uses flush plan 53B.", cite=None)
    assert claim["status"] != "contradicted"


@pytest.mark.parametrize("citation", ["[D1]", "[S1]", "[1]", "[DS-220]", "[dae6f0f7-8ca9-57e3-b820-562bd7788367]"])
def test_citations_of_any_length_are_removed_from_the_nli_hypothesis(citation):
    nli = ShortPremiseNLI()
    _claim(LONG, f"Pump-102 mechanical seal uses flush plan 53B {citation}.", nli=nli)
    assert nli.hypotheses and all("[" not in hypothesis and not hypothesis.endswith(" .") for hypothesis in nli.hypotheses)
