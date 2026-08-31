from fastapi.testclient import TestClient
from docx import Document

from app.main import app


def test_scanned_inspection_to_downloadable_approval_note() -> None:
    with TestClient(app) as client:
        response = client.post("/api/tasks", json={
            "request": "Analyze the uploaded Pump-102 inspection report against the maintenance SOP and prepare an approval note",
            "attachments": ["uploads/Pump_Inspection_Report.pdf"],
            "use_case": "engineering",
        })
        assert response.status_code == 200
        run = response.json()
        artifact_response = client.get(run["artifacts"][0]["url"])

    assert run["status"] == "completed"
    assert len(run["plan"]["steps"]) == 13
    assert all(step["status"] == "completed" for step in run["plan"]["steps"])
    assert {source["section"] for source in run["sources"]} == {"7.4", "7.5", "7.6"}
    assert run["governance"]["grounding_score"] == 1.0
    assert artifact_response.status_code == 200
    assert artifact_response.content[:2] == b"PK"
