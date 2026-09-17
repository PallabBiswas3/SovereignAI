from __future__ import annotations

from dataclasses import dataclass
from typing import Any


XMEAS_DESCRIPTIONS: dict[str, str] = {
    "XMEAS(1)": "A feed flow",
    "XMEAS(2)": "D feed flow",
    "XMEAS(3)": "E feed flow",
    "XMEAS(4)": "A and C feed flow",
    "XMEAS(5)": "Recycle flow",
    "XMEAS(6)": "Reactor feed rate",
    "XMEAS(7)": "Reactor pressure",
    "XMEAS(8)": "Reactor level",
    "XMEAS(9)": "Reactor temperature",
    "XMEAS(10)": "Purge rate",
    "XMEAS(11)": "Product separator temperature",
    "XMEAS(12)": "Product separator level",
    "XMEAS(13)": "Product separator pressure",
    "XMEAS(14)": "Product separator underflow",
    "XMEAS(15)": "Stripper level",
    "XMEAS(16)": "Stripper pressure",
    "XMEAS(17)": "Stripper underflow",
    "XMEAS(18)": "Stripper temperature",
    "XMEAS(19)": "Stripper steam flow",
    "XMEAS(20)": "Compressor work",
    "XMEAS(21)": "Reactor cooling-water outlet temperature",
    "XMEAS(22)": "Separator cooling-water outlet temperature",
    "XMEAS(23)": "A feed composition",
    "XMEAS(24)": "B feed composition",
    "XMEAS(25)": "C feed composition",
    "XMEAS(26)": "D feed composition",
    "XMEAS(27)": "E feed composition",
    "XMEAS(28)": "F feed composition",
    "XMEAS(29)": "Reactor feed composition A",
    "XMEAS(30)": "Reactor feed composition B",
    "XMEAS(31)": "Reactor feed composition C",
    "XMEAS(32)": "Reactor feed composition D",
    "XMEAS(33)": "Reactor feed composition E",
    "XMEAS(34)": "Reactor feed composition F",
    "XMEAS(35)": "Purge gas composition G",
    "XMEAS(36)": "Purge gas composition H",
    "XMEAS(37)": "Product composition D",
    "XMEAS(38)": "Product composition E",
    "XMEAS(39)": "Product composition F",
    "XMEAS(40)": "Product composition G",
    "XMEAS(41)": "Product composition H",
}

XMV_DESCRIPTIONS: dict[str, str] = {
    "XMV(1)": "D feed flow valve",
    "XMV(2)": "E feed flow valve",
    "XMV(3)": "A feed flow valve",
    "XMV(4)": "A and C feed flow valve",
    "XMV(5)": "Compressor recycle valve",
    "XMV(6)": "Purge valve",
    "XMV(7)": "Separator pot liquid-flow valve",
    "XMV(8)": "Stripper liquid-product flow valve",
    "XMV(9)": "Stripper steam valve",
    "XMV(10)": "Reactor cooling-water flow valve",
    "XMV(11)": "Condenser cooling-water flow valve",
    "XMV(12)": "Agitator speed",
}

TEP_VARIABLE_DESCRIPTIONS = {**XMEAS_DESCRIPTIONS, **XMV_DESCRIPTIONS}
TEP_DEFAULT_CHANNEL_NAMES = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]


@dataclass(frozen=True)
class TEPFaultSpec:
    fault_id: int
    label: str
    description: str
    fault_type: str
    expected_roots: tuple[str, ...]
    affected_variables: tuple[str, ...]
    subsystem: str
    confidence: str = "engineering_prior"

    def to_catalog_record(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_label": self.label,
            "label": self.label,
            "description": self.description,
            "type": self.fault_type,
            "expected_roots": list(self.expected_roots),
            "root_cause": self.expected_roots[0] if self.expected_roots else None,
            "affected_variables": list(self.affected_variables),
            "subsystem": self.subsystem,
            "confidence": self.confidence,
        }


TEP_FAULT_SPECS: dict[int, TEPFaultSpec] = {
    1: TEPFaultSpec(1, "IDV(1): A/C feed ratio step", "A/C feed ratio changed with B composition constant", "step", ("XMEAS(4)", "XMV(4)"), ("XMEAS(4)", "XMEAS(6)", "XMEAS(9)", "XMEAS(29)", "XMEAS(31)", "XMV(4)"), "feed/reaction"),
    2: TEPFaultSpec(2, "IDV(2): B composition step", "B composition changed with A/C ratio constant", "step", ("XMEAS(24)",), ("XMEAS(24)", "XMEAS(30)", "XMEAS(35)", "XMEAS(36)", "XMEAS(40)", "XMEAS(41)"), "feed/composition"),
    3: TEPFaultSpec(3, "IDV(3): D feed temperature step", "D feed temperature changed", "step", ("XMEAS(2)", "XMV(1)"), ("XMEAS(2)", "XMEAS(6)", "XMEAS(9)", "XMEAS(11)", "XMV(1)"), "feed/reaction"),
    4: TEPFaultSpec(4, "IDV(4): Reactor cooling-water inlet temperature step", "Reactor cooling-water inlet temperature changed", "step", ("XMV(10)", "XMEAS(21)", "XMEAS(9)"), ("XMV(10)", "XMEAS(21)", "XMEAS(9)", "XMEAS(7)", "XMEAS(20)"), "reactor cooling"),
    5: TEPFaultSpec(5, "IDV(5): Condenser cooling-water inlet temperature step", "Condenser cooling-water inlet temperature changed", "step", ("XMV(11)", "XMEAS(22)", "XMEAS(11)"), ("XMV(11)", "XMEAS(22)", "XMEAS(11)", "XMEAS(13)", "XMEAS(16)"), "separator/condenser cooling"),
    6: TEPFaultSpec(6, "IDV(6): A feed loss", "A feed loss", "step", ("XMEAS(1)", "XMV(3)"), ("XMEAS(1)", "XMV(3)", "XMEAS(4)", "XMEAS(6)", "XMEAS(9)", "XMEAS(29)"), "feed"),
    7: TEPFaultSpec(7, "IDV(7): C header pressure loss", "C header pressure loss / reduced availability", "step", ("XMEAS(4)", "XMV(4)"), ("XMEAS(4)", "XMV(4)", "XMEAS(6)", "XMEAS(7)", "XMEAS(9)", "XMEAS(31)"), "feed"),
    8: TEPFaultSpec(8, "IDV(8): A/B/C feed composition random variation", "A, B, and C feed composition changed", "random variation", ("XMEAS(23)", "XMEAS(24)", "XMEAS(25)"), ("XMEAS(23)", "XMEAS(24)", "XMEAS(25)", "XMEAS(29)", "XMEAS(30)", "XMEAS(31)", "XMEAS(35)", "XMEAS(36)"), "feed/composition"),
    9: TEPFaultSpec(9, "IDV(9): D feed temperature random variation", "D feed temperature random variation", "random variation", ("XMEAS(2)", "XMV(1)"), ("XMEAS(2)", "XMV(1)", "XMEAS(6)", "XMEAS(9)", "XMEAS(11)"), "feed/reaction"),
    10: TEPFaultSpec(10, "IDV(10): C feed temperature random variation", "C feed temperature random variation", "random variation", ("XMEAS(4)", "XMV(4)"), ("XMEAS(4)", "XMV(4)", "XMEAS(6)", "XMEAS(7)", "XMEAS(9)", "XMEAS(31)"), "feed/reaction"),
    11: TEPFaultSpec(11, "IDV(11): Reactor cooling-water temperature random variation", "Reactor cooling-water inlet temperature random variation", "random variation", ("XMV(10)", "XMEAS(21)", "XMEAS(9)"), ("XMV(10)", "XMEAS(21)", "XMEAS(9)", "XMEAS(7)", "XMEAS(20)"), "reactor cooling"),
    12: TEPFaultSpec(12, "IDV(12): Condenser cooling-water temperature random variation", "Condenser cooling-water inlet temperature random variation", "random variation", ("XMV(11)", "XMEAS(22)", "XMEAS(11)"), ("XMV(11)", "XMEAS(22)", "XMEAS(11)", "XMEAS(13)", "XMEAS(16)"), "separator/condenser cooling"),
    13: TEPFaultSpec(13, "IDV(13): Reaction kinetics slow drift", "Reaction kinetics changed", "slow drift", ("XMEAS(9)", "XMEAS(7)"), ("XMEAS(9)", "XMEAS(7)", "XMEAS(21)", "XMEAS(35)", "XMEAS(36)", "XMEAS(40)", "XMEAS(41)"), "reactor"),
    14: TEPFaultSpec(14, "IDV(14): Reactor cooling-water valve sticking", "Reactor cooling-water valve sticking", "valve sticking", ("XMV(10)",), ("XMV(10)", "XMEAS(21)", "XMEAS(9)", "XMEAS(7)"), "reactor cooling"),
    15: TEPFaultSpec(15, "IDV(15): Condenser cooling-water valve sticking", "Condenser cooling-water valve sticking", "valve sticking", ("XMV(11)",), ("XMV(11)", "XMEAS(22)", "XMEAS(11)", "XMEAS(13)"), "separator/condenser cooling"),
    16: TEPFaultSpec(16, "IDV(16): Unknown fault", "Unknown process fault", "unknown", tuple(), tuple(), "unknown", "benchmark_unknown"),
    17: TEPFaultSpec(17, "IDV(17): Unknown fault", "Unknown process fault", "unknown", tuple(), tuple(), "unknown", "benchmark_unknown"),
    18: TEPFaultSpec(18, "IDV(18): Unknown fault", "Unknown process fault", "unknown", tuple(), tuple(), "unknown", "benchmark_unknown"),
    19: TEPFaultSpec(19, "IDV(19): Unknown fault", "Unknown process fault", "unknown", tuple(), tuple(), "unknown", "benchmark_unknown"),
    20: TEPFaultSpec(20, "IDV(20): Unknown fault", "Unknown process fault", "unknown", tuple(), tuple(), "unknown", "benchmark_unknown"),
    21: TEPFaultSpec(21, "IDV(21): Valve fixed at steady-state position", "A valve was fixed at steady-state position", "constant position", ("XMV(1)", "XMV(3)", "XMV(4)", "XMV(10)", "XMV(11)"), ("XMV(1)", "XMV(3)", "XMV(4)", "XMV(10)", "XMV(11)", "XMEAS(6)", "XMEAS(9)", "XMEAS(21)", "XMEAS(22)"), "control valve"),
}


def tep_fault_catalog() -> dict[str, Any]:
    return {
        "name": "Tennessee Eastman fault catalog",
        "faults": [spec.to_catalog_record() for spec in TEP_FAULT_SPECS.values()],
        "variable_descriptions": dict(TEP_VARIABLE_DESCRIPTIONS),
        "notes": "Expected roots are engineering priors for ranking, not hard labels for all runs.",
    }


def tep_fault_spec(fault_id: int) -> TEPFaultSpec | None:
    return TEP_FAULT_SPECS.get(int(fault_id))


def tep_fault_record(fault_id: int) -> dict[str, Any] | None:
    spec = tep_fault_spec(fault_id)
    return spec.to_catalog_record() if spec else None


TEP_TOPOLOGY_EDGES: tuple[tuple[str, str], ...] = (
    ("XMV(1)", "XMEAS(2)"), ("XMV(2)", "XMEAS(3)"), ("XMV(3)", "XMEAS(1)"), ("XMV(4)", "XMEAS(4)"),
    ("XMEAS(1)", "XMEAS(4)"), ("XMEAS(2)", "XMEAS(6)"), ("XMEAS(3)", "XMEAS(6)"), ("XMEAS(4)", "XMEAS(6)"),
    ("XMEAS(23)", "XMEAS(29)"), ("XMEAS(24)", "XMEAS(30)"), ("XMEAS(25)", "XMEAS(31)"),
    ("XMEAS(6)", "XMEAS(7)"), ("XMEAS(6)", "XMEAS(8)"), ("XMEAS(6)", "XMEAS(9)"),
    ("XMEAS(29)", "XMEAS(9)"), ("XMEAS(30)", "XMEAS(9)"), ("XMEAS(31)", "XMEAS(9)"),
    ("XMV(10)", "XMEAS(21)"), ("XMEAS(21)", "XMEAS(9)"), ("XMEAS(9)", "XMEAS(7)"),
    ("XMEAS(7)", "XMEAS(5)"), ("XMEAS(8)", "XMEAS(5)"), ("XMEAS(9)", "XMEAS(5)"),
    ("XMEAS(5)", "XMEAS(10)"), ("XMV(6)", "XMEAS(10)"),
    ("XMEAS(5)", "XMEAS(11)"), ("XMEAS(5)", "XMEAS(12)"), ("XMEAS(5)", "XMEAS(13)"),
    ("XMV(11)", "XMEAS(22)"), ("XMEAS(22)", "XMEAS(11)"), ("XMEAS(11)", "XMEAS(13)"),
    ("XMV(7)", "XMEAS(14)"), ("XMEAS(14)", "XMEAS(15)"), ("XMEAS(14)", "XMEAS(16)"), ("XMEAS(14)", "XMEAS(17)"),
    ("XMV(9)", "XMEAS(19)"), ("XMEAS(19)", "XMEAS(18)"), ("XMEAS(18)", "XMEAS(17)"),
    ("XMV(8)", "XMEAS(17)"), ("XMEAS(17)", "XMEAS(37)"), ("XMEAS(17)", "XMEAS(38)"), ("XMEAS(17)", "XMEAS(39)"), ("XMEAS(17)", "XMEAS(40)"), ("XMEAS(17)", "XMEAS(41)"),
    ("XMV(5)", "XMEAS(20)"), ("XMEAS(20)", "XMEAS(5)"),
)


def tep_topology() -> dict[str, Any]:
    return {
        "name": "Sparse Tennessee Eastman topology prior",
        "edges": [{"cause": a, "effect": b, "source": "tep_engineering_prior"} for a, b in TEP_TOPOLOGY_EDGES],
    }
