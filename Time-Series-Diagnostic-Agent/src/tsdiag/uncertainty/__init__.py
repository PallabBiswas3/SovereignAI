"""Model-independent conformal prediction and selective reporting."""

from .conformal import (
    ConformalClassifier, ConformalizedQuantileRegressor, SelectiveDecision,
    SplitConformalRegressor, conformal_quantile, interval_coverage,
    mean_interval_width, set_coverage,
)
from .selective import (
    accuracy_at_coverage, aurc, expected_calibration_error, fit_temperature,
    risk_coverage_curve, selective_summary, softmax,
)

__all__ = [
    "ConformalClassifier", "ConformalizedQuantileRegressor", "SelectiveDecision",
    "SplitConformalRegressor", "conformal_quantile", "interval_coverage",
    "mean_interval_width", "set_coverage", "accuracy_at_coverage", "aurc",
    "expected_calibration_error", "fit_temperature", "risk_coverage_curve",
    "selective_summary", "softmax",
]
