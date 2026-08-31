from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def test_approved_safe_registered_action_executes_exact_arguments() -> None:
    relative = f"approval-tests/{uuid4()}.md"
    with TestClient(app) as client:
        proposed = client.post("/api/tools/propose", json={
            "tool": "write_file", "args": {"path": relative, "content": "approved exact content"},
        })
        assert proposed.status_code == 200
        proposal = proposed.json()
        assert proposal["decision"] == "REQUIRE_HUMAN_APPROVAL"
        target = Path("workspace") / relative
        assert not target.exists()

        approved = client.post(f"/api/approvals/{proposal['approval_id']}", json={"approve": True, "decided_by": "test-engineer"})
        fetched = client.get(f"/api/approvals/{proposal['approval_id']}")

    assert approved.status_code == 200
    assert approved.json()["status"] == "executed"
    assert target.read_text(encoding="utf-8") == "approved exact content"
    assert fetched.json()["arguments"] == {"path": relative, "content": "approved exact content"}


def test_approval_does_not_override_disabled_destructive_policy() -> None:
    with TestClient(app) as client:
        proposal = client.post("/api/tools/propose", json={"tool": "delete_file", "args": {"path": "uploads/Pump_Inspection_Report.md"}}).json()
        response = client.post(f"/api/approvals/{proposal['approval_id']}", json={"approve": True, "decided_by": "test-engineer"})
    assert response.json()["status"] == "blocked"
    assert Path("workspace/uploads/Pump_Inspection_Report.md").exists()
