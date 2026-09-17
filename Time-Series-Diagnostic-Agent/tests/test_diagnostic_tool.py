import json

import pytest

from tsdiag.integrations import (
    DIAGNOSE_TOOL_MANIFEST,
    diagnose_tool,
    shape_diagnostic_output,
    verify_summary_claims,
)


def test_diagnostic_tool_executes_json_request_with_provenance():
    cycles = list(range(1, 13))
    payload = json.dumps({
        "domain": "turbofan",
        "task": "remaining_useful_life",
        "inputs": {
            "signal_matrix": [[0.2 * cycle, 0.1 - 0.01 * cycle] for cycle in cycles],
            "channel_names": ["temperature", "noise"],
            "cycle_index": cycles,
        },
        "policy_ref": "turbofan-policy-v2",
        "run_context": {"run_id": "tool-test", "source": "local-llm"},
    })

    result = diagnose_tool(payload)

    assert result["domain"] == "turbofan"
    assert result["metadata"]["pipeline_version"] == "1.1.0"
    assert result["metadata"]["policy_version"] == "turbofan-policy-v2"
    assert result["provenance"]["run_id"] == "tool-test"
    assert result["provenance"]["source"] == "local-llm"
    assert DIAGNOSE_TOOL_MANIFEST["input_schema"]["additionalProperties"] is False


def test_diagnostic_tool_rejects_unknown_envelope_fields():
    with pytest.raises(ValueError, match="unsupported diagnostic tool fields"):
        diagnose_tool({"domain": "bearing", "inputs": {}, "execute_python": "bad"})


def test_turbofan_tool_result_passes_grounding_and_engineer_presentation():
    cycles = list(range(1, 30))
    result = diagnose_tool({
        "domain": "turbofan",
        "task": "remaining_useful_life",
        "inputs": {
            "signal_matrix": [[0.03 * cycle, 1.0 - 0.01 * cycle] for cycle in cycles],
            "channel_names": ["temperature", "pressure"],
            "cycle_index": cycles,
        },
        "policy_ref": "turbofan-policy-v2",
        "run_context": {"run_id": "grounded-demo", "source": "integration-test"},
    })
    confidence = result["confidence"]
    summary = f"Decision is {result['decision']}. Confidence is {confidence}."
    report = verify_summary_claims(
        summary,
        [
            {
                "sentence_index": 0,
                "sentence": f"Decision is {result['decision']}.",
                "claims": [{
                    "claim_text": result["decision"],
                    "source_ref": "/decision",
                    "claimed_value": result["decision"],
                }],
            },
            {
                "sentence_index": 1,
                "sentence": f"Confidence is {confidence}.",
                "claims": [{
                    "claim_text": str(confidence),
                    "source_ref": "/confidence",
                    "claimed_value": confidence,
                }],
            },
        ],
        result,
    )
    engineer = shape_diagnostic_output(result, policy_ref="engineer-v1")

    assert report.release_allowed is True
    assert engineer["provenance"]["run_id"] == "grounded-demo"
    assert engineer["presentation_policy_ref"] == "engineer-v1"
