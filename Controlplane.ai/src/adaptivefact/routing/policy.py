from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThresholdPolicy:
    tau1: float
    tau2: float

    @classmethod
    def from_json(cls, path: str | Path) -> "ThresholdPolicy":
        raw = json.loads(Path(path).read_text())
        if "thresholds" in raw:
            raw = raw["thresholds"]
        return cls(tau1=float(raw["tau1"]), tau2=float(raw["tau2"]))
