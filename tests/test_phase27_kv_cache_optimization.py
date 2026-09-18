from app.resources.kv_cache_optimization import (
    KvCacheConfirmationStore,
    KvCacheContextConfirmation,
    KvCacheOptimizationProfile,
    KvCacheOptimizationStore,
    evaluate_context_confirmation,
    select_kv_cache_result,
)


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


def test_confirmation_store_round_trip(tmp_path) -> None:
    store = KvCacheConfirmationStore(tmp_path / "kv_cache_confirmations.json")
    store.save(KvCacheContextConfirmation(
        model="qwen3:4b-instruct",
        context_length=4096,
        confirmed=True,
        paired_speed_ratio=1.08,
        candidate_win_rate=1.0,
        resident_saving_mb=270.0,
        baseline_quality_score=1.0,
        candidate_quality_score=1.0,
        long_context_passed=True,
    ))
    result = store.get("QWEN3:4B-INSTRUCT", 4096)
    assert result is not None
    assert result.confirmed is True
    assert result.candidate_cache == "q8_0"
    assert result.resident_saving_mb == 270.0


def test_select_result_prefers_ram_saving_within_guardrails() -> None:
    results = [
        {"cache_type": "f16", "quality_score": 1.0, "median_tokens_per_second": 10.0, "system_ram_delta_mb": 2000.0},
        {"cache_type": "q8_0", "quality_score": 1.0, "median_tokens_per_second": 9.8, "system_ram_delta_mb": 1600.0},
        {"cache_type": "q4_0", "quality_score": 0.8, "median_tokens_per_second": 9.9, "system_ram_delta_mb": 1300.0},
    ]
    selected, policy = select_kv_cache_result(
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
    selected, _ = select_kv_cache_result(
        results,
        min_quality=0.95,
        max_tps_regression=0.05,
        min_ram_saving_mb=128.0,
    )
    assert selected["cache_type"] == "f16"


def test_controlled_confirmation_accepts_repeatable_q8_win() -> None:
    result = evaluate_context_confirmation(
        model="qwen3:4b-instruct",
        context_length=4096,
        paired_speed_ratios=[1.08, 1.06, 1.10, 1.07],
        baseline_resident_sizes_mb=[2973.5, 2973.5, 2973.5, 2973.5],
        candidate_resident_sizes_mb=[2703.6, 2703.6, 2703.6, 2703.6],
        baseline_quality_score=1.0,
        candidate_quality_score=1.0,
        baseline_long_context_passed=True,
        candidate_long_context_passed=True,
    )
    assert result.confirmed is True
    assert result.paired_speed_ratio == 1.075
    assert result.candidate_win_rate == 1.0
    assert result.resident_saving_mb == 269.9
    assert result.reasons == []


def test_controlled_confirmation_rejects_unstable_or_unvalidated_q8() -> None:
    result = evaluate_context_confirmation(
        model="qwen3:4b-instruct",
        context_length=4096,
        paired_speed_ratios=[1.10, 0.96, 1.01, 0.98],
        baseline_resident_sizes_mb=[2973.5, 2973.5],
        candidate_resident_sizes_mb=[2703.6, 2703.6],
        baseline_quality_score=1.0,
        candidate_quality_score=1.0,
        baseline_long_context_passed=True,
        candidate_long_context_passed=False,
    )
    assert result.confirmed is False
    assert result.candidate_win_rate == 0.5
    assert any("win rate" in reason for reason in result.reasons)
    assert any("long-context" in reason for reason in result.reasons)
