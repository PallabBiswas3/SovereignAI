import numpy as np
import pytest


torch = pytest.importorskip("torch")

from tsdiag.learned.ad_tfm_ablation import (  # noqa: E402
    PAPER_ABLATION_VARIANTS,
    TFMVariantModel,
    get_variant_spec,
)
from tsdiag.learned.ad_tfm_at import ADTFMConfig  # noqa: E402


def test_four_paper_ablation_variants_are_defined():
    assert [spec.name for spec in PAPER_ABLATION_VARIANTS] == [
        "TFM",
        "AD-TFM",
        "TFM-AT",
        "AD-TFM-AT",
    ]
    assert get_variant_spec("tfm").adaptive_wavelet is False
    assert get_variant_spec("ad-tfm").adaptive_wavelet is True
    assert get_variant_spec("tfm-at").attention is True


def test_ablation_variants_share_output_shape_and_attention_semantics():
    cfg = ADTFMConfig(input_size=6, hidden_size=8, time_dimensions=2, frequency_dimensions=2, attention_size=4, num_classes=5)
    x = torch.randn(3, 12, 6)
    for spec in PAPER_ABLATION_VARIANTS:
        model = TFMVariantModel(cfg, spec)
        logits, attention = model(x, return_attention=True)
        assert tuple(logits.shape) == (3, 5)
        assert tuple(attention.shape) == (3, 12)
        np.testing.assert_allclose(attention.detach().cpu().numpy().sum(axis=1), 1.0, atol=1e-5)
        if not spec.attention:
            expected = np.zeros((3, 12), dtype=np.float32)
            expected[:, -1] = 1.0
            np.testing.assert_allclose(attention.detach().cpu().numpy(), expected, atol=1e-6)


def test_fixed_wavelet_variants_do_not_train_adaptive_parameter_generators():
    cfg = ADTFMConfig(input_size=6, hidden_size=8, time_dimensions=2, frequency_dimensions=2, attention_size=4, num_classes=5)
    fixed = TFMVariantModel(cfg, get_variant_spec("TFM"))
    adaptive = TFMVariantModel(cfg, get_variant_spec("AD-TFM"))
    assert not any(p.requires_grad for p in fixed.cell.wavelet_scale.parameters())
    assert not any(p.requires_grad for p in fixed.cell.wavelet_shift.parameters())
    assert all(p.requires_grad for p in adaptive.cell.wavelet_scale.parameters())
    assert all(p.requires_grad for p in adaptive.cell.wavelet_shift.parameters())
