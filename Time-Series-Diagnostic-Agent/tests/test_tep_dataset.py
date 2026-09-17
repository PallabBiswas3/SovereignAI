from pathlib import Path

import numpy as np

from tsdiag.datasets.tep import (
    TEP_CHANNEL_NAMES,
    TEP_TEST_FAULT_START,
    load_tep_dat,
    tep_fault_mask,
)


def test_tep_loader_handles_samples_by_variables(tmp_path: Path):
    x = np.arange(20 * 52, dtype=float).reshape(20, 52)
    path = tmp_path / "sample.dat"
    np.savetxt(path, x)
    loaded = load_tep_dat(path)
    assert loaded.shape == (20, 52)
    assert np.allclose(loaded, x)


def test_tep_loader_handles_braatz_transposed_orientation(tmp_path: Path):
    x = np.arange(20 * 52, dtype=float).reshape(20, 52)
    path = tmp_path / "sample.dat"
    np.savetxt(path, x.T)
    loaded = load_tep_dat(path)
    assert loaded.shape == (20, 52)
    assert np.allclose(loaded, x)


def test_tep_fault_mask_uses_standard_test_onset():
    mask = tep_fault_mask(960, 1)
    assert not mask[:TEP_TEST_FAULT_START].any()
    assert mask[TEP_TEST_FAULT_START:].all()
    assert len(TEP_CHANNEL_NAMES) == 52


def test_normal_tep_mask_is_always_false():
    assert not tep_fault_mask(960, 0).any()
