from __future__ import annotations

from app.assets.models import HistoricalSeries, TrendAnalysis, TrendDirection, TrendWindow


class InsufficientHistoryError(ValueError):
    code = "INSUFFICIENT_HISTORY"


class TrendAnalyzer:
    """Deterministic condition-series arithmetic; this is not failure prediction."""

    def __init__(self, slope_tolerance_per_day: float = 1e-6, rolling_window: int = 3) -> None:
        self.tolerance = abs(slope_tolerance_per_day)
        self.rolling_window = max(1, rolling_window)

    def analyze(self, series: HistoricalSeries, threshold: float | None = None) -> TrendAnalysis:
        values = [item for item in series.measurements if item.quality.value in {"GOOD", "UNCERTAIN"}]
        if len(values) < 2:
            raise InsufficientHistoryError("INSUFFICIENT_HISTORY: at least two usable measurements are required")
        values.sort(key=lambda item: item.timestamp)
        origin = values[0].timestamp
        x = [(item.timestamp - origin).total_seconds() / 86400.0 for item in values]
        y = [item.value for item in values]
        mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
        denominator = sum((item - mean_x) ** 2 for item in x)
        slope = sum((left - mean_x) * (right - mean_y) for left, right in zip(x, y)) / denominator if denominator else 0.0
        direction = TrendDirection.flat
        if slope > self.tolerance:
            direction = TrendDirection.increasing
        elif slope < -self.tolerance:
            direction = TrendDirection.decreasing
        abnormal = sum(1 for item in y if threshold is not None and item > threshold)
        crossings = 0
        above = [item > threshold for item in y] if threshold is not None else []
        for previous, current in zip(above, above[1:]):
            if previous != current:
                crossings += 1
        time_above = sum((values[index + 1].timestamp - values[index].timestamp).total_seconds() for index in range(len(values) - 1) if threshold is not None and y[index] > threshold)
        change = y[-1] - y[0]
        return TrendAnalysis(
            asset_id=series.asset_id, metric=series.metric, unit=series.unit,
            window=TrendWindow(start=values[0].timestamp, end=values[-1].timestamp),
            sample_count=len(y), latest=y[-1], mean=round(mean_y, 6), minimum=min(y), maximum=max(y),
            change=round(change, 6), percentage_change=round(change / y[0] * 100, 6) if y[0] else None,
            slope_per_day=round(slope, 9), rolling_mean=round(sum(y[-self.rolling_window:]) / min(len(y), self.rolling_window), 6),
            threshold_crossings=crossings, time_above_threshold_seconds=time_above,
            abnormal_readings=abnormal, trend=direction,
        )
