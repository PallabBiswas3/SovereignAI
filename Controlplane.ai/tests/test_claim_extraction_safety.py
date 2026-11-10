"""Claim extraction must not change the meaning of safety statements (10 Nov 2026).

Before: "...until the seal is replaced and the permit is closed" produced the
stand-alone fact "the permit is closed"; the right half of "Pump-102 vibration
is 7.4 mm/s and the bearing temperature is 92 °C" lost its asset; and "Do not
restart Pump-102. It has a seal leak." became "Do has a seal leak."
"""
import pytest

from adaptivefact.data.schema import ResponseRecord
from adaptivefact.extraction.claim_extractor import ClaimExtractor


def claims(text: str) -> list[str]:
    record = ResponseRecord(id="r", dataset="t", query="q", generated_response=text)
    return [claim.verification_text for claim in ClaimExtractor().extract(record)]


@pytest.mark.parametrize("text", [
    "Pump-102 must not be restarted until the seal is replaced and the permit is closed.",
    "If vibration exceeds 7.1 mm/s, Pump-102 must be shut down and the bearing must be inspected.",
    "Restart Pump-102 only after the seal is replaced and the permit is closed.",
    "When vibration is high, reduce load and the operator must notify maintenance.",
    "Pump-102 may run unless the seal leaks and the bearing temperature is above 85 °C.",
])
def test_conditions_stay_attached_to_the_whole_requirement(text):
    assert claims(text) == [text]


def test_split_conjunct_keeps_its_asset():
    assert claims("Pump-102 vibration is 7.4 mm/s and the bearing temperature is 92 °C.") == [
        "Pump-102 vibration is 7.4 mm/s", "Pump-102 bearing temperature is 92 °C."]


def test_each_conjunct_keeps_its_own_asset():
    assert claims("Pump-102 vibration is 7.4 mm/s and Pump-103 vibration is 2.1 mm/s.") == [
        "Pump-102 vibration is 7.4 mm/s", "Pump-103 vibration is 2.1 mm/s."]


def test_temporal_preposition_is_not_a_condition():
    assert claims("Pump-102 was inspected after the outage and the seal was replaced.") == [
        "Pump-102 was inspected after the outage", "Pump-102 seal was replaced."]


@pytest.mark.parametrize("text, expected", [
    ("Do not restart Pump-102. It has a seal leak.", "Pump-102 has a seal leak."),
    ("Pump-102 vibration is 7.4 mm/s. It is above the alarm limit.", "Pump-102 vibration is above the alarm limit."),
    # No reliable referent: leave the pronoun rather than guess "Check".
    ("Check the coupling. It is worn.", "It is worn."),
])
def test_pronoun_resolves_to_a_real_subject_or_is_left_alone(text, expected):
    assert claims(text)[-1] == expected
