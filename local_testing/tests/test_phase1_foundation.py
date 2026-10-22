from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_holds_when_controlplane_is_unavailable(monkeypatch) -> None:
    from app.api import chat
    from app.integrations.clients import IntegrationServiceError

    class UnavailableControlPlane:
        async def precheck(self, payload):
            raise IntegrationServiceError("controlplane", "Synthetic unavailable service")

    monkeypatch.setattr(chat, "ControlPlaneClient", lambda *args, **kwargs: UnavailableControlPlane())
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Summarize the attached SOP"})
    assert response.status_code == 503
    assert response.json() == {"detail": "ControlPlane unavailable; response held"}
