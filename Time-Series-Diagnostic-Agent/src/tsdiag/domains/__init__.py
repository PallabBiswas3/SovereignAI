from .bearing import BEARING_PACK
from .process import PROCESS_PACK
from .process_runner import ProcessDiagnosticPipeline, ProcessDiagnosticResult
from .wind_scada import WIND_SCADA_PACK
from .battery import BATTERY_PACK
from .turbofan import TURBOFAN_PACK
from .transformer import TRANSFORMER_PACK
from .bearing_runner import BearingDiagnosticPipeline
from .runners import (
    BatteryDiagnosticPipeline,
    TransformerDiagnosticPipeline,
)
from .turbofan_plugin import TurbofanDiagnosticPipeline
from .transformer_ad_tfm import ADTFMTransformerDiagnosticPipeline
from .process_cva import CVATEPDiagnosticPipeline, CVATEPDiagnosticResult
from .wind_scada_runner import (
    WindScadaDiagnosticPipeline,
    WindScadaDiagnosticResult as WindScadaBenchmarkResult,
)
from .wind_physics import WindPhysicsConfig, physics_consistency_check

# Backward-compatible name for the former benchmark-specific export. Both names
# now resolve to the same canonical implementation.
WindScadaBenchmarkPipeline = WindScadaDiagnosticPipeline
from .domain_steps import default_domain_tool_registry

DOMAIN_PACKS = {
    pack.key: pack
    for pack in (
        BEARING_PACK,
        PROCESS_PACK,
        WIND_SCADA_PACK,
        BATTERY_PACK,
        TURBOFAN_PACK,
        TRANSFORMER_PACK,
    )
}


def get_domain_pack(key: str):
    try:
        return DOMAIN_PACKS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown domain pack {key!r}. Available: {sorted(DOMAIN_PACKS)}") from exc


__all__ = [
    "BEARING_PACK",
    "PROCESS_PACK",
    "ProcessDiagnosticPipeline",
    "ProcessDiagnosticResult",
    "WIND_SCADA_PACK",
    "BATTERY_PACK",
    "TURBOFAN_PACK",
    "TRANSFORMER_PACK",
    "DOMAIN_PACKS",
    "get_domain_pack",
    "BearingDiagnosticPipeline",
    "WindScadaDiagnosticPipeline",
    "WindScadaBenchmarkPipeline",
    "WindScadaBenchmarkResult",
    "WindPhysicsConfig",
    "physics_consistency_check",
    "BatteryDiagnosticPipeline",
    "TurbofanDiagnosticPipeline",
    "TransformerDiagnosticPipeline",
    "ADTFMTransformerDiagnosticPipeline",
    "CVATEPDiagnosticPipeline",
    "CVATEPDiagnosticResult",
    "default_domain_tool_registry",
]
