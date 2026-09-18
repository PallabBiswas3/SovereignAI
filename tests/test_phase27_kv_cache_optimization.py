from app.resources.kv_cache_optimization import KvCacheOptimizationProfile, KvCacheOptimizationStore
from backend.scripts.benchmark_kv_cache import select_result


def test_kv_cache_profile_store_round_trip(tmp_path) -> None:
    store = KvCacheOptimizationStore(tmp_path / "kv_cache_optimization.json")
    store.save(KvCacheOptimizationProfile(
        model="qwen3:4b-instruct",
        cache_type="q8_0",
        context_length=2048,
        quality_score=1.0,
        median_tokens_per_second=10.2,
        resident_size_mb=2500.0,
        system_ram_delta_mb=1700.0,
        available_ram_after_mb=1100.0,
    ))
    profile = store.get("QWEN3:4B-INSTRUCT")
    assert profile is not None
    assert profile.cache_type == "q8_0"
    assert profile.context_length == 2048
    assert profile.flash_attention is True


def test_select_result_prefers_ram_saving_within_guardrails() -> None:
    results = [
        {"cache_type": "f16", "quality_score": 1.0, "median_tokens_per_second": 10.0, "system_ram_delta_mb": 2000.0},
        {"cache_type": "q8_0", "quality_score": 1.0, "median_tokens_per_second": 9.8, "system_ram_delta_mb": 1600.0},
        {"cache_type": "q4_0", "quality_score": 0.8, "median_tokens_per_second": 9.9, "system_ram_delta_mb": 1300.0},
    ]
    selected, policy = select_result(
        results,
        min_quality=0.95,
        max_tps_regression=0.05,
        min_ram_saving_mb=128.0,
    )
    assert selected["cache_type"] == "q8_0"
    assert results[1]["eligible"] is True
    assert results[2]["eligible"] is False
    assert policy["selection_priority"].startswith("maximize RAM saving")


def test_select_result_keeps_f16_when_compression_saves_too_little_or_is_slow() -> None:
    results = [
        {"cache_type": "f16", "quality_score": 1.0, "median_tokens_per_second": 10.0, "system_ram_delta_mb": 2000.0},
        {"cache_type": "q8_0", "quality_score": 1.0, "median_tokens_per_second": 9.2, "system_ram_delta_mb": 1550.0},
        {"cache_type": "q4_0", "quality_score": 1.0, "median_tokens_per_second": 9.9, "system_ram_delta_mb": 1930.0},
    ]
    selected, _ = select_result(
        results,
        min_quality=0.95,
        max_tps_regression=0.05,
        min_ram_saving_mb=128.0,
    )
    assert selected["cache_type"] == "f16"
