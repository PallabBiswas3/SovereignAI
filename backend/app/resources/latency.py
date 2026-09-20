from __future__ import annotations

from typing import Any


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def build_latency_breakdown(
    provider_metrics: dict[str, Any],
    *,
    generation_wall_seconds: float,
    event_callback_seconds: float,
    event_callback_count: int,
    first_ui_frame_seconds: float | None,
) -> dict[str, Any]:
    """Separate provider latency from SovereignAI application overhead.

    Ollama's total duration represents model-runtime work, while scheduler queue time and
    SovereignAI's streaming/event persistence happen outside that duration. Keeping these
    buckets separate makes direct CLI-vs-workbench comparisons actionable.
    """

    wall = max(0.0, float(generation_wall_seconds))
    queue = max(0.0, _metric(provider_metrics, "queue_wait_seconds"))
    model_total = max(0.0, _metric(provider_metrics, "total_duration_seconds"))
    callback = max(0.0, float(event_callback_seconds))
    application_overhead = max(0.0, wall - queue - model_total)
    overhead_percent = (application_overhead / wall * 100.0) if wall > 0 else 0.0
    callback_percent = (callback / wall * 100.0) if wall > 0 else 0.0

    return {
        "generation_wall_seconds": round(wall, 6),
        "queue_wait_seconds": round(queue, 6),
        "model_reported_total_seconds": round(model_total, 6),
        "load_duration_seconds": provider_metrics.get("load_duration_seconds"),
        "time_to_first_token_seconds": provider_metrics.get("time_to_first_token_seconds"),
        "decode_tokens_per_second": provider_metrics.get("tokens_per_second"),
        "prompt_token_count": provider_metrics.get("prompt_token_count"),
        "generated_token_count": provider_metrics.get("token_count"),
        "warm_status": provider_metrics.get("warm_status", "unknown"),
        "first_ui_frame_seconds": (
            round(float(first_ui_frame_seconds), 6) if first_ui_frame_seconds is not None else None
        ),
        "event_callback_seconds": round(callback, 6),
        "event_callback_count": max(0, int(event_callback_count)),
        "event_callback_percent": round(callback_percent, 2),
        "application_overhead_seconds": round(application_overhead, 6),
        "application_overhead_percent": round(overhead_percent, 2),
    }
