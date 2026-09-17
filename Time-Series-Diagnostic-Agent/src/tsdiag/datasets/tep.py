from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np

TEP_SOURCE_REPOSITORY = "https://github.com/camaramm/tennessee-eastman-profBraatz"
TEP_RAW_BASE = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master"
TEP_N_VARIABLES = 52
TEP_TRAIN_FAULT_START = 20  # zero-based in d01.dat...d21.dat training simulations
TEP_TEST_FAULT_START = 160  # zero-based: samples 0..159 normal, 160.. faulty
TEP_SAMPLE_PERIOD_MIN = 3.0

TEP_CHANNEL_NAMES = tuple(
    [f"XMEAS({i})" for i in range(1, 42)]
    + [f"XMV({i})" for i in range(1, 12)]
)

TEP_FAULTS = {
    0: {"description": "Normal operation", "type": "normal"},
    1: {"description": "A/C feed ratio, B composition constant (Stream 4)", "type": "step"},
    2: {"description": "B composition, A/C ratio constant (Stream 4)", "type": "step"},
    3: {"description": "D feed temperature (Stream 2)", "type": "step"},
    4: {"description": "Reactor cooling water inlet temperature", "type": "step"},
    5: {"description": "Condenser cooling water inlet temperature", "type": "step"},
    6: {"description": "A feed loss (Stream 1)", "type": "step"},
    7: {"description": "C header pressure loss / reduced availability (Stream 4)", "type": "step"},
    8: {"description": "A, B, C feed composition (Stream 4)", "type": "random_variation"},
    9: {"description": "D feed temperature (Stream 2)", "type": "random_variation"},
    10: {"description": "C feed temperature (Stream 4)", "type": "random_variation"},
    11: {"description": "Reactor cooling water inlet temperature", "type": "random_variation"},
    12: {"description": "Condenser cooling water inlet temperature", "type": "random_variation"},
    13: {"description": "Reaction kinetics", "type": "slow_drift"},
    14: {"description": "Reactor cooling water valve", "type": "sticking"},
    15: {"description": "Condenser cooling water valve", "type": "sticking"},
    16: {"description": "Unknown", "type": "unknown"},
    17: {"description": "Unknown", "type": "unknown"},
    18: {"description": "Unknown", "type": "unknown"},
    19: {"description": "Unknown", "type": "unknown"},
    20: {"description": "Unknown", "type": "unknown"},
    21: {"description": "Stream 4 valve fixed at steady-state position", "type": "constant_position"},
}

# Conservative engineering proxy targets used only for localization scoring.
# These are not claimed to be canonical ground-truth root-cause labels.
TEP_LOCALIZATION_PROXIES = {
    4: ("XMEAS(21)", "XMV(10)"),
    5: ("XMEAS(22)", "XMV(11)"),
    6: ("XMEAS(1)", "XMV(3)"),
    14: ("XMV(10)", "XMEAS(21)"),
    15: ("XMV(11)", "XMEAS(22)"),
    21: ("XMV(4)", "XMEAS(4)"),
}


def _filename(fault_id: int, split: str) -> str:
    if fault_id not in TEP_FAULTS:
        raise ValueError(f"fault_id must be in 0..21, got {fault_id}")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    suffix = "_te" if split == "test" else ""
    return f"d{fault_id:02d}{suffix}.dat"


def _orient_tep_matrix(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=float)
    if x.ndim == 1:
        if x.size % TEP_N_VARIABLES != 0:
            raise ValueError("flat TEP data length is not divisible by 52")
        x = x.reshape(-1, TEP_N_VARIABLES)
    if x.ndim != 2:
        raise ValueError("TEP data must be a 2-D matrix")
    if x.shape[1] == TEP_N_VARIABLES:
        return x
    if x.shape[0] == TEP_N_VARIABLES:
        return x.T
    raise ValueError(f"expected one TEP dimension to equal 52, got {x.shape}")


def load_tep_dat(path: str | Path) -> np.ndarray:
    """Load a Braatz TEP .dat file and return [samples, 52] regardless of source orientation."""
    data = np.loadtxt(Path(path), dtype=float)
    x = _orient_tep_matrix(data)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path} contains NaN/Inf")
    return x


def download_braatz_tep(
    destination: str | Path,
    *,
    fault_ids: Iterable[int] = range(0, 22),
    splits: Iterable[str] = ("train", "test"),
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download the canonical public Braatz TEP files used by this benchmark.

    Dataset bytes are intentionally not vendored into this repository. This keeps the
    benchmark reproducible while preserving upstream provenance and licensing notices.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for fault_id in fault_ids:
        for split in splits:
            name = _filename(int(fault_id), split)
            target = root / name
            if overwrite or not target.exists():
                urlretrieve(f"{TEP_RAW_BASE}/{name}", target)
            downloaded[name] = target
    return downloaded


def load_tep_reference(data_dir: str | Path, *, use_test_normal: bool = False) -> np.ndarray:
    name = "d00_te.dat" if use_test_normal else "d00.dat"
    return load_tep_dat(Path(data_dir) / name)


def tep_fault_mask(n_samples: int, fault_id: int, *, fault_start: int = TEP_TEST_FAULT_START) -> np.ndarray:
    mask = np.zeros(int(n_samples), dtype=bool)
    if int(fault_id) != 0:
        mask[min(max(0, int(fault_start)), n_samples):] = True
    return mask
