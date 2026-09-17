from tsdiag.integrations import shape_diagnostic_output, verify_summary_claims


def _result():
    return {
        "domain": "turbofan",
        "task": "remaining_useful_life",
        "decision": "diagnose",
        "detection": {"abnormal": True},
        "localization": {"components": ["fan"], "channels": ["temperature"], "scores": {"fan": 0.8}},
        "hypotheses": [{"label": "fan_degradation", "score": 0.82, "evidence_ids": ["ev-1"]}],
        "confidence": 0.82,
        "uncertainty_estimate": {"value": 0.1, "method": "test"},
        "prognosis": {"remaining_useful_life": 42.0},
        "evidence": [{"evidence_id": "ev-1", "statement": "Fan temperature trend is degrading."}],
        "verification": [],
        "recommended_actions": ["Inspect the fan."],
        "provenance": {"run_id": "demo-1"},
        "tool_trace": [{"tool": "health_index", "status": "ok"}],
    }


def test_claim_grounding_releases_only_exact_result_references():
    summary = "The fan is the localized component. Confidence is 82%. RUL is 42 cycles."
    annotations = [
        {
            "sentence_index": 0,
            "sentence": "The fan is the localized component.",
            "claims": [{
                "claim_text": "fan",
                "source_ref": "/localization/components/0",
                "claimed_value": "fan",
                "evidence_ids": ["ev-1"],
            }],
        },
        {
            "sentence_index": 1,
            "sentence": "Confidence is 82%.",
            "claims": [{
                "claim_text": "82%",
                "source_ref": "/confidence",
                "claimed_value": 0.82,
                "evidence_ids": ["ev-1"],
            }],
        },
        {
            "sentence_index": 2,
            "sentence": "RUL is 42 cycles.",
            "claims": [{
                "claim_text": "42",
                "source_ref": "/prognosis/remaining_useful_life",
                "claimed_value": 42.0,
                "evidence_ids": ["ev-1"],
            }],
        },
    ]
    report = verify_summary_claims(summary, annotations, _result())
    assert report.release_allowed is True
    assert report.claim_count == 3


def test_claim_grounding_fails_closed_on_unannotated_sentence_or_number():
    result = _result()
    missing = verify_summary_claims(
        "The fan is localized. RUL is 99 cycles.",
        [{
            "sentence_index": 0,
            "sentence": "The fan is localized.",
            "claims": [{
                "claim_text": "fan",
                "source_ref": "/localization/components/0",
                "claimed_value": "fan",
            }],
        }],
        result,
    )
    assert missing.release_allowed is False
    assert any("unannotated" in finding.reason for finding in missing.findings)

    unsupported = verify_summary_claims(
        "RUL is 99 cycles.",
        [{
            "sentence_index": 0,
            "sentence": "RUL is 99 cycles.",
            "claims": [{
                "claim_text": "99",
                "source_ref": "/prognosis/remaining_useful_life",
                "claimed_value": 99,
            }],
        }],
        result,
    )
    assert unsupported.release_allowed is False
    assert any("does not match" in finding.reason for finding in unsupported.findings)


def test_presentation_policies_shape_operator_and_engineer_views():
    result = _result()
    operator = shape_diagnostic_output(result, policy_ref="operator-v1")
    engineer = shape_diagnostic_output(result, policy_ref="engineer-v1")

    assert operator["top_hypothesis"] == "fan_degradation"
    assert "confidence" not in operator
    assert "evidence" not in operator
    assert engineer["confidence"] == 0.82
    assert engineer["evidence"][0]["evidence_id"] == "ev-1"
    assert engineer["provenance"]["run_id"] == "demo-1"
