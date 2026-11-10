"""Generalised privacy and bias checks (10 November 2026).

Privacy: Aadhaar numbers must pass the Verhoeff check digit, and plant-specific
recognizers (OT network ranges, company ID formats) come from configuration.
Bias: a closed-vocabulary group finder selects sentences about protected groups
of people; open-ended phrasing is judged by a local LLM (or the shared NLI
model) instead of verb lists. Evaluation: local_testing/experiments/results/
controlplane_bias_semantic_v1.
"""
import random

import pytest

from controlplane.checker import ControlPlane
from controlplane.detectors.bias import BiasDetector, OllamaBiasJudge, find_people_groups
from controlplane.detectors.privacy import PrivacyDetector, PrivacyRecognizerConfig, verhoeff_valid
from controlplane.schema import EnforcementAction, Interaction
from adaptivefact.verification.nli import NLIScorer, NLIScores


# --- Privacy -------------------------------------------------------------------

def test_verhoeff_rejects_most_random_twelve_digit_numbers():
    rng = random.Random(7)
    numbers = [str(rng.randint(2, 9)) + "".join(rng.choice("0123456789") for _ in range(11)) for _ in range(5000)]
    assert sum(map(verhoeff_valid, numbers)) / len(numbers) < 0.12
    assert verhoeff_valid("234567890124") and not verhoeff_valid("234567890123")


def test_aadhaar_requires_valid_check_digit():
    scan = PrivacyDetector(PrivacyRecognizerConfig())._scan
    assert [f.subtype for f in scan("Aadhaar 2345 6789 0124", location="response", actionable=True)] == ["aadhaar_number"]
    assert scan("Batch code 2345 6789 0123", location="response", actionable=True) == []


def test_configured_ot_ranges_and_company_identifiers():
    config = PrivacyRecognizerConfig(
        ot_networks=["10.20.0.0/16"],
        identifiers=[{"subtype": "employee_id", "pattern": r"\bEMP-\d{6}\b", "severity": "medium"}],
    )
    findings = PrivacyDetector(config)._scan(
        "PLC 10.20.30.40, vendor portal 172.16.5.9, badge EMP-004211.", location="response", actionable=True)
    assert {(f.subtype, f.severity.value) for f in findings} == {
        ("ot_network_address", "high"), ("ip_address", "medium"), ("employee_id", "medium")}


@pytest.mark.parametrize("bad", [
    {"ot_networks": ["10.20.0.0/33"]},
    {"identifiers": [{"subtype": "Bad Name", "pattern": "x"}]},
    {"identifiers": [{"subtype": "x_id", "pattern": "("}]},
    {"identifiers": [{"subtype": "x_id", "pattern": "x", "severity": "extreme"}]},
])
def test_invalid_recognizer_config_fails_at_startup(bad):
    with pytest.raises(Exception):
        PrivacyRecognizerConfig(**bad)


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "recognizers.yaml"
    path.write_text("ot_network: ['10.0.0.0/8']\n", encoding="utf-8")  # typo must not be ignored
    with pytest.raises(ValueError):
        PrivacyRecognizerConfig.from_file(str(path))


# --- Bias: group finder --------------------------------------------------------

@pytest.mark.parametrize("sentence, attribute", [
    ("Workers from Bihar usually cut corners.", "race_or_ethnicity"),
    ("Contractors over 55 are a liability.", "age"),
    ("Anyone in a wheelchair has no business here.", "disability"),
    ("Ladies on the crew shouldn't climb.", "gender"),
    ("Pick younger technicians; the older ones are slow.", "age"),
])
def test_group_finder_finds_people_groups(sentence, attribute):
    assert attribute in {found for found, _ in find_people_groups(sentence)}


@pytest.mark.parametrize("sentence", [
    "Fit a blind flange on the suction side.",
    "Fit the blind flange before isolating; the deaf-end bearing is next.",
    "Use the female socket and the male coupling.",
    "The women's changing room is on level 2.",
    "Workers from the night shift must sign out.",
    "White metal bearings must be re-babbitted.",
])
def test_group_finder_ignores_equipment_and_facilities(sentence):
    assert find_people_groups(sentence) == []


# --- Bias: semantic judges ------------------------------------------------------

class ScriptedJudge:
    def __init__(self, verdict=(True, "age"), fail=False):
        self.verdict, self.fail, self.calls = verdict, fail, []

    def judge(self, sentence):
        self.calls.append(sentence)
        if self.fail:
            raise ConnectionError("synthetic judge outage")
        return self.verdict


def _detect(detector, text):
    return detector.detect(Interaction(prompt="q", response=text, consequential=True), {})


def test_judge_sees_only_sentences_about_people_groups():
    judge = ScriptedJudge()
    result = _detect(BiasDetector(judge=judge),
                     "Fit the blind flange. Contractors over 55 are a liability on nights. Torque to 40 Nm.")
    assert judge.calls == ["Contractors over 55 are a liability on nights."]
    assert result.metadata["semantic_method"] == "llm_judge"
    assert [f.metadata["method"] for f in result.findings] == ["llm_judge"]


def test_judge_negative_verdict_adds_no_finding():
    assert _detect(BiasDetector(judge=ScriptedJudge(verdict=(False, None))),
                   "Employees over 45 receive a free health check.").findings == []


def test_configured_judge_outage_fails_closed_to_review():
    gate = ControlPlane(audit_enabled=False)
    gate.detectors = {**gate.detectors, "bias": BiasDetector(judge=ScriptedJudge(fail=True))}
    report = gate.check(Interaction(profile="internal_assistant", prompt="q",
                                    response="Contractors over 55 are a liability on the night shift."))
    assert report.decision.action == EnforcementAction.REVIEW
    assert "internal-control-failure-review" in report.decision.matched_rules


class NotAllowedNLI(NLIScorer):
    def score(self, premises, hypotheses):
        return [NLIScores(0.9, 0.05, 0.05) if "shouldn't" in premise else NLIScores(0.01, 0.01, 0.98)
                for premise, hypothesis in zip(premises, hypotheses)]


def test_nli_fallback_uses_group_hypothesis_when_no_judge():
    detector = BiasDetector(nli=NotAllowedNLI(), use_env_judge=False)
    assert detector.detect(Interaction(prompt="q", response="Ladies on the crew shouldn't climb."), {}).findings
    assert not detector.detect(Interaction(prompt="q", response="Ladies on the crew completed training."), {}).findings


def test_without_judge_or_nli_the_method_is_reported_as_lexical():
    result = _detect(BiasDetector(use_env_judge=False), "Contractors over 55 are a liability.")
    assert result.metadata["semantic_method"] == "lexical_only"


@pytest.mark.parametrize("url", ["https://api.example.com", "http://8.8.8.8:11434"])
def test_judge_rejects_public_endpoints(url):
    with pytest.raises(ValueError):
        OllamaBiasJudge("qwen3:4b-instruct", url)
