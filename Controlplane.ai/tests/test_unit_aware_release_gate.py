"""Unit-aware release-gate regressions (10 November 2026).

Found by the paraphrase probe in
local_testing/Controlplane.ai/scripts/probe_paraphrase_robustness.py: wrong
claims were released when worded outside small vocabulary lists or stated in a
different unit, and a correct claim in a converted unit (600 cm vs 6 m) was held.
Each fix makes the gate stricter or unit-correct; no policy threshold changed.
"""
from pathlib import Path

import pytest

from adaptivefact.extraction.units import (
    align_units_to_premise,
    extract_quantities,
    scalar_outside_source_ranges,
    unwitnessed_quantities,
)
from adaptivefact.verification.nli import NLIScorer, NLIScores
from controlplane.checker import ControlPlane
from controlplane.policy import PolicyRepository
from controlplane.schema import EnforcementAction, GroundingEvidence, Interaction
from controlplane.verification import AdaptiveFactVerificationService

POLICIES = PolicyRepository(Path(__file__).resolve().parents[1] / "configs/policies")
RELEASED = {EnforcementAction.ALLOW, EnforcementAction.WARN, EnforcementAction.REDACT}


# --- Unit parsing and comparison --------------------------------------------

@pytest.mark.parametrize("claim, source", [
    ("600 cm", "6 m"), ("6000 mm", "6 m"), ("58 psi", "4 bar"), ("400 kPa to 600 kPa", "4.0 to 6.0 bar"),
    ("40 °C", "104 °F"), ("176 °F", "80 °C"), ("3.2 mm/s", "3.2 mm/s RMS"), ("1800 rpm", "30 Hz"),
    ("upper limit 6 bar", "4.0 to 6.0 bar"), ("between 4 and 6 bar", "4.0-6.0 bar"),
])
def test_equal_quantities_in_other_units_are_witnessed(claim, source):
    assert unwitnessed_quantities(claim, [source]) == []


@pytest.mark.parametrize("claim, source", [
    ("6 ft", "6 m"), ("60 cm", "6 m"), ("4.0 to 6.0 psi", "4.0 to 6.0 bar"), ("4.0 psi", "4 bar"),
    ("4 to 7 bar", "4.0 to 6.0 bar"), ("80 °F", "80 °C"), ("3.2 mm/s peak", "3.2 mm/s RMS"),
    ("4 bara", "4 bar"), ("6 bar", "6 mm"),
])
def test_different_quantities_are_not_witnessed(claim, source):
    assert unwitnessed_quantities(claim, [source]) != []


def test_parser_does_not_invent_quantities():
    assert extract_quantities("Pump-102 inspection 2026-10-02 Rev 3") == []
    assert extract_quantities("rose 4 in the last hour; Bay 3 a") == []
    separate = extract_quantities("pressure 5.1 bar and temperature 80 °C")
    assert [(q.dimension, q.is_range) for q in separate] == [("pressure", False), ("temperature", False)]


def test_alignment_rewrites_only_equal_quantities():
    premise = "Pump-102 minimum suction head is 6 m."
    assert align_units_to_premise("Pump-102 minimum suction head is 600 cm", premise).endswith("6 m")
    assert align_units_to_premise("Pump-102 minimum suction head is 60 cm", premise).endswith("60 cm")


def test_scalar_outside_range_uses_converted_values():
    assert scalar_outside_source_ranges("normal at 6.8 bar", "normal range 4.0 to 6.0 bar")
    assert not scalar_outside_source_ranges("recorded 510 kPa", "normal range 4.0 to 6.0 bar")


# --- Release-gate behaviour --------------------------------------------------

class ScriptedNLI(NLIScorer):
    """Entail by default; contradict premises containing a marker; neutral on others."""

    def __init__(self, contradict_premise=None, entail_only_if_hypothesis_has=None):
        self.contradict_premise = contradict_premise
        self.entail_only_if = entail_only_if_hypothesis_has

    def score(self, premises, hypotheses):
        output = []
        for premise, hypothesis in zip(premises, hypotheses):
            if self.contradict_premise and self.contradict_premise in premise:
                output.append(NLIScores(0.01, 0.80, 0.19))
            elif self.entail_only_if is not None and self.entail_only_if not in hypothesis:
                output.append(NLIScores(0.01, 0.04, 0.95))
            else:
                output.append(NLIScores(0.95, 0.01, 0.04))
        return output


def _evidence(*items):
    return [GroundingEvidence(evidence_id=eid, document_id=eid, chunk_id=f"{eid}-1", text=text, retrieval_score=0.9,
                              metadata={"document_status": "current", "document_type": kind, "trust_level": "controlled"})
            for eid, kind, text in items]


LIMIT = ("OPL-3", "operating_limits", "Pump-102 current normal discharge pressure is 4.0 to 6.0 bar.")


def _action(response, evidence, nli, prompt="Pump-102 discharge pressure?"):
    profile = POLICIES.load("internal_assistant")
    profile.checks["hallucination"].settings["use_trained_risk"] = False
    gate = ControlPlane(audit_enabled=False, verification_service=AdaptiveFactVerificationService(nli=nli))
    report = gate.check(Interaction(prompt=prompt, response=response, grounding_evidence=evidence,
                                    consequential=True), profile)
    return report.decision.action


def test_value_match_cannot_outvote_a_single_strong_contradiction():
    evidence = _evidence(LIMIT, ("INS-7", "inspection_record",
                                 "Pump-102 inspection on 2026-10-02: recorded discharge pressure of 6.8 bar."))
    # Mirrors the real DeBERTa scores: contradiction from the limit, neutral elsewhere.
    action = _action("Pump-102 discharge pressure was normal at 6.8 bar during the inspection [INS-7] [OPL-3].",
                     evidence, ScriptedNLI(contradict_premise="4.0 to 6.0 bar", entail_only_if_hypothesis_has="\0"))
    assert action not in RELEASED


def test_in_range_reading_is_still_released_despite_range_contradiction():
    evidence = _evidence(LIMIT, ("INS-8", "inspection_record",
                                 "Pump-102 inspection on 2026-10-09: recorded discharge pressure of 5.1 bar."))
    action = _action("Pump-102 discharge pressure was recorded at 5.1 bar during the inspection [INS-8].",
                     evidence, ScriptedNLI(contradict_premise="4.0 to 6.0 bar"))
    assert action in RELEASED


def test_reading_outside_its_own_stated_range_is_held():
    evidence = _evidence(LIMIT, ("INS-7", "inspection_record",
                                 "Pump-102 inspection on 2026-10-02: recorded discharge pressure of 6.8 bar."))
    action = _action("Pump-102's recorded discharge pressure of 6.8 bar is inside the current normal range of "
                     "4.0 to 6.0 bar [INS-7] [OPL-3].", evidence, ScriptedNLI())
    assert action not in RELEASED


def test_wrong_unit_is_held_even_when_nli_entails():
    action = _action("Pump-102 normal discharge pressure is 4.0 to 6.0 psi [OPL-3].", _evidence(LIMIT), ScriptedNLI())
    assert action not in RELEASED


@pytest.mark.parametrize("claim, released", [
    ("Pump-102 minimum suction head is 600 cm [DS-1].", True),
    ("Pump-102 minimum suction head is 6 m [DS-1].", True),
    ("Pump-102 minimum suction head is 6 ft [DS-1].", False),
])
def test_converted_unit_is_released_only_when_equal(claim, released):
    evidence = _evidence(("DS-1", "equipment_manual", "Pump-102 minimum suction head is 6 m."))
    # NLI entails only when it sees the source's own "6 m" (it cannot convert units).
    action = _action(claim, evidence, ScriptedNLI(entail_only_if_hypothesis_has="6 m"),
                     prompt="What is the Pump-102 minimum suction head?")
    assert (action in RELEASED) is released
