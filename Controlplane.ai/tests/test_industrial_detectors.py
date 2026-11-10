"""Privacy and bias detectors on industrial text (10 November 2026).

Before: "blind flange", "male NPT fitting" and "white metal bearing" were bias
findings (held for review when consequential), a real age-based staffing rule
was missed, a timestamp was redacted as a phone number, and passwords, PAN
numbers and PLC IP addresses were not detected.
"""
import pytest

from controlplane.actions.response_transformer import redact_response
from controlplane.detectors.bias import BiasDetector
from controlplane.detectors.privacy import PrivacyDetector, mask_privacy_values
from controlplane.schema import Interaction


def _privacy(text):
    return PrivacyDetector()._scan(text, location="response", actionable=True)


def _bias(text):
    return BiasDetector().detect(Interaction(prompt="q", response=text, consequential=True), {}).findings


# --- Privacy -------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Inspection logged 2026-10-02 14:30:55 on Pump-102.",
    "Reading taken 02/10/2026 at 14:30 by shift B.",
    "Ratio 1:1500 and flow 1200.5 m3/h at 2026-11-01T08:15:00Z.",
    "Vibration trend 4.12 4.35 4.40 4.52 4.61 mm/s over five days.",
    "Replace seal, part no. 4521 7788 9012, batch 2026-118.",
    "Motor serial 9876543210 was rewound.",
    "Upgrade HMI firmware 2.1.0.3 before restart.",
    "The password is ******** and must be rotated.",
    "Password: [REDACTED] (rotate quarterly).",
    "A password is required for the historian.",
])
def test_industrial_numbers_and_masked_values_are_not_personal_data(text):
    assert _privacy(text) == []


@pytest.mark.parametrize("text, subtype", [
    ("Contact plant engineer R. Sharma, mobile 98765 43210.", "phone_number"),
    ("Call +91 98765 43210 for the permit.", "phone_number"),
    ("Engineer PAN ABCPE1234F is on file.", "pan_number"),
    ("Aadhaar 2345 6789 0124 was submitted.", "aadhaar_number"),  # Verhoeff-valid synthetic
    ("HMI login password: Plant@2026", "password"),
    ("The pwd=Xy7!secret was shared.", "password"),
    ("PLC address 10.20.30.40 controls Pump-102.", "ip_address"),
    ("Card 4111 1111 1111 1111 was charged.", "payment_card"),
])
def test_personal_and_secret_values_are_detected(text, subtype):
    assert [finding.subtype for finding in _privacy(text)] == [subtype]


def test_timestamp_survives_redaction_and_masking():
    text = "Inspection logged 2026-10-02 14:30:55; call 98765 43210."
    findings = _privacy(text)
    assert redact_response(text, findings) == "Inspection logged 2026-10-02 14:30:55; call [REDACTED PHONE]."
    assert "2026-10-02 14:30:55" in mask_privacy_values(text)


def test_password_is_a_critical_secret_and_card_redaction_keeps_spacing():
    assert _privacy("password: Plant@2026")[0].severity.value == "critical"
    text = "Card 4111 1111 1111 1111 was charged."
    assert redact_response(text, _privacy(text)) == "Card [REDACTED CARD] was charged."


# --- Bias ----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Install the blind flange on the suction line and avoid restarting until the work permit is closed.",
    "Use a male NPT fitting; female adapters are not suitable for this service.",
    "Replace the white metal bearing; black oxide bolts should not be used near fire water lines.",
    "Senior engineers should not be assigned to routine greasing; use the maintenance crew.",
    "The old impeller must not be reused. Young concrete must not be loaded for 7 days.",
    "Per Indian Standard IS 2825, the vessel must not operate above 10 bar.",
    "A deaf-end blind spot exists on the drive side; operators should avoid it.",
    "The man-lift must not be operated in winds above 40 km/h.",
    "Fire the boiler only after purge; check the white noise level on the transmitter.",
    "Operators with a valid permit may enter; others must not enter the confined space.",
    "Keep the crew off the platform during the lift.",
])
def test_engineering_vocabulary_is_not_a_bias_finding(text):
    assert _bias(text) == []


@pytest.mark.parametrize("text, attribute", [
    ("Seniors on the night shift should not be assigned confined-space entry alone.", "age"),
    ("Women should not be assigned to confined-space entry.", "gender"),
    ("Older workers must not operate the overhead crane.", "age"),
    ("Female operators are less capable on night shifts.", "gender"),
    ("Do not hire candidates because they are Muslims.", "religion"),
    ("Migrant workers tend to be careless around rotating equipment.", "race_or_ethnicity"),
    ("Keep disabled employees off the shop floor.", "disability"),
    ("Only male technicians should handle the high-voltage panel.", "gender"),
    ("Bihari labourers are unreliable and should be kept off night shifts.", "race_or_ethnicity"),
])
def test_people_directed_bias_is_detected(text, attribute):
    assert attribute in {finding.metadata.get("protected_attribute") for finding in _bias(text)}
