from fastapi.testclient import TestClient

import tsdiag.api as diagnostic_api


def test_health_reports_service_contract():
    response = TestClient(diagnostic_api.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "time-series-diagnostic-agent",
        "version": "1.1.0",
    }


def test_diagnose_uses_strict_host_neutral_tool(monkeypatch):
    captured = {}

    def fake_diagnose(payload):
        captured.update(payload)
        return {"domain": payload["domain"], "decision": "abstain"}

    monkeypatch.setattr(diagnostic_api, "diagnose_tool", fake_diagnose)
    response = TestClient(diagnostic_api.app).post(
        "/v1/diagnose",
        json={"domain": "bearing", "inputs": {"signal": [0.0, 1.0]}},
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "abstain"
    assert captured["domain"] == "bearing"


def test_diagnose_rejects_unknown_envelope_fields():
    response = TestClient(diagnostic_api.app).post(
        "/v1/diagnose",
        json={"domain": "bearing", "inputs": {}, "unexpected": True},
    )

    assert response.status_code == 422
