from __future__ import annotations

import io
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app.main import app


def test_pump_task_persists_workcell_and_exports_verifiable_capsule() -> None:
    with TestClient(app) as client:
        response = client.post("/api/tasks", json={
            "request": "Analyze the uploaded Pump-102 inspection report against the maintenance SOP and prepare an approval note",
            "attachments": ["uploads/Pump_Inspection_Report.pdf"],
            "use_case": "engineering",
            "workcell_id": "pump-inspection",
        })
        assert response.status_code == 200
        run = response.json()
        assert run["workcell_id"] == "pump-inspection"
        assert run["workcell_version"] == "1.0.0"
        assert len(run["workcell_hash"]) == 64
        assert run["workflow_version"] == "1.0.0"
        assert run["workcell_state"]["completed_steps"] == ["validate_inputs", "execute_inspection"]

        created = client.post(f"/api/tasks/{run['id']}/capsule", json={"inputs": []})
        assert created.status_code == 200
        capsule = created.json()
        assert capsule["state"] == "COMPLETE"
        assert capsule["manifest"]["workcell"]["id"] == "pump-inspection"
        assert capsule["manifest"]["inputs"]
        assert all(len(item["sha256"]) == 64 for item in capsule["manifest"]["inputs"])

        verification = client.post(capsule["verify_url"])
        assert verification.status_code == 200
        checked = verification.json()
        assert checked["status"] == "VALID"
        assert checked["signature_status"] == "UNSIGNED"
        assert checked["artifact_valid_count"] == checked["artifact_count"] == 1

        download = client.get(capsule["download_url"])
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        with ZipFile(io.BytesIO(download.content)) as archive:
            names = archive.namelist()
            assert any(name.endswith("capsule_manifest.json") for name in names)
            assert any(name.endswith("evidence/claims.json") for name in names)
            assert any(name.endswith("artifacts/Approval_Note_" + run["id"][:8] + ".docx") for name in names)
