from __future__ import annotations

from app.assets.models import AssetFailureCode, AssetResolution
from app.assets.repository import AssetRepository
from app.identity.models import Principal


class AssetResolver:
    """Exact canonical/alias resolution only; deliberately no fuzzy matching."""

    def __init__(self, repository: AssetRepository) -> None:
        self.repository = repository

    def resolve(self, principal: Principal, reference: str) -> AssetResolution:
        candidates = self.repository.resolution_rows(reference)
        if not candidates:
            return AssetResolution(status=AssetFailureCode.asset_not_found.value, reference=reference)
        authorized = [row for row in candidates if self.repository.is_authorized(principal, row)]
        if not authorized:
            return AssetResolution(status=AssetFailureCode.asset_access_denied.value, reference=reference)
        if len(authorized) > 1:
            return AssetResolution(status=AssetFailureCode.ambiguous_asset.value, reference=reference, candidate_ids=[row.id for row in authorized])
        return AssetResolution(status="RESOLVED", reference=reference, asset=self.repository._passport(authorized[0]))
