from fastapi.testclient import TestClient

from app.main import app


def test_simple_agent_run_has_explicit_verified_plan() -> None:
    with TestClient(app) as client:
        response = client.post("/api/tasks", json={"request": "Explain pump cavitation briefly"})
        assert response.status_code == 200
        run = response.json()
        fetched = client.get(f"/api/tasks/{run['id']}")

    assert run["status"] == "completed"
    assert [step["action"] for step in run["plan"]["steps"]] == [
        "understand_task",
        "generate_response",
        "verify_response",
    ]
    assert all(step["status"] == "completed" for step in run["plan"]["steps"])
    assert fetched.status_code == 200
    assert fetched.json()["id"] == run["id"]

