from __future__ import annotations

from pathlib import Path

import yaml

from controlplane.policy.models import PolicyProfile


class PolicyRepository:
    def __init__(self, root: str | Path = "configs/policies") -> None:
        self.root = Path(root)

    def load(self, profile_id: str) -> PolicyProfile:
        path = self.root / f"{profile_id}.yaml"
        if not path.exists():
            available = ", ".join(self.available()) or "none"
            raise FileNotFoundError(f"Unknown policy profile '{profile_id}'. Available: {available}")
        return PolicyProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def available(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.yaml"))
