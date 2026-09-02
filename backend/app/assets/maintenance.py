from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.assets.models import MaintenanceActionDraft, MaintenanceRecord
from app.assets.repository import AssetRepository
from app.core.database import HumanApprovalRecord, MaintenanceDraftRecord
from app.identity.content import ContentIdentityService
from app.identity.models import Principal


class CMMSConnector(ABC):
    """Governed maintenance seam. Batch 5 implementations create local drafts only."""

    @abstractmethod
    def list_asset_work_orders(self, principal: Principal, asset_id: str, limit: int = 50) -> list[MaintenanceRecord]:
        raise NotImplementedError

    @abstractmethod
    def read_work_order(self, principal: Principal, draft_id: str) -> MaintenanceActionDraft | None:
        raise NotImplementedError

    @abstractmethod
    def create_work_order_draft(
        self, principal: Principal, *, asset_id: str, action_type: str, priority: str,
        title: str, description: str, reason_claim_ids: list[str],
    ) -> MaintenanceActionDraft:
        raise NotImplementedError


class APELLocalCMMSConnector(CMMSConnector):
    provider_name = "apel-local-cmms-draft-only"

    def __init__(self, session: Session) -> None:
        self.session = session
        self.assets = AssetRepository(session)

    def list_asset_work_orders(self, principal: Principal, asset_id: str, limit: int = 50) -> list[MaintenanceRecord]:
        return self.assets.get_maintenance_history(principal, asset_id, limit)

    def read_work_order(self, principal: Principal, draft_id: str) -> MaintenanceActionDraft | None:
        row = self.session.get(MaintenanceDraftRecord, draft_id)
        if not row or not self.assets.get_asset(principal, row.asset_id):
            return None
        return self.assets._draft(row)

    def create_work_order_draft(
        self, principal: Principal, *, asset_id: str, action_type: str, priority: str,
        title: str, description: str, reason_claim_ids: list[str],
    ) -> MaintenanceActionDraft:
        asset = self.assets.get_asset(principal, asset_id)
        if not asset:
            raise PermissionError("ASSET_ACCESS_DENIED")
        if not principal.has_permission("maintenance.draft.create"):
            raise PermissionError("ACCESS_DENIED")
        if not reason_claim_ids:
            raise ValueError("MAINTENANCE_DRAFT_FAILED: at least one supporting claim is required")
        draft_id = str(uuid4())
        arguments = {
            "draft_id": draft_id, "asset_id": asset_id, "action_type": action_type.upper(),
            "priority": priority.upper(), "title": title, "description": description,
            "reason_claim_ids": sorted(set(reason_claim_ids)),
        }
        approval = HumanApprovalRecord(
            id=str(uuid4()), run_id=None, tool="approve_maintenance_draft",
            args_json=json.dumps(arguments, ensure_ascii=False, sort_keys=True), risk="MEDIUM",
            status="pending", requester_id=principal.user_id,
            organization_id=asset.organization_id, workspace_id=asset.workspace_id,
            action_hash=ContentIdentityService().hash_json({"tool": "approve_maintenance_draft", "arguments": arguments}),
        )
        row = MaintenanceDraftRecord(
            id=draft_id, asset_id=asset_id, action_type=action_type.upper(),
            priority=priority.upper(), title=title.strip(), description=description.strip(),
            reason_claim_ids_json=json.dumps(arguments["reason_claim_ids"]), status="DRAFT",
            created_by=principal.user_id, organization_id=asset.organization_id,
            workspace_id=asset.workspace_id, department_id=asset.department_id,
            classification=asset.classification.name.upper(), approval_id=approval.id,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )
        self.session.add_all([approval, row])
        self.session.commit()
        return self.assets._draft(row)
