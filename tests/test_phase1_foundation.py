from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_persists_and_reports_missing_local_model() -> None:
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Summarize the attached SOP"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"]
    assert payload["model"]
    assert payload["provider"] in {"ollama", "deterministic-unavailable"}

