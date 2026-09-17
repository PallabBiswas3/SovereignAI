"""
Metrics for the benchmark harness: classification quality, calibration,
latency percentiles, and cost. Kept as plain functions (not classes) so
they're trivial to call from ad-hoc notebooks during experimentation,
not just from the runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float] | None = None,
) -> dict[str, float]:
    """Binary classification metrics for hallucination detection.

    y_true / y_pred: 1 = hallucinated, 0 = supported/clean.
    y_score: predicted probability of the positive (hallucinated) class,
        used for AUROC if provided.
    """
    if len(y_true) == 0:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "auroc": float("nan"), "n": 0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "n": len(y_true),
    }

    if y_score is not None and len(set(y_true)) > 1:
        result["auroc"] = float(roc_auc_score(y_true, y_score))
        result["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")

    return result


def selective_triage_metrics(
    y_true: list[int],
    decisions: list[str],
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate a four-way decision without turning abstentions into errors.

    ``hallucinated`` and ``supported`` are confirmed model decisions. ``mixed``
    and ``unknown`` are abstentions that require review. The returned metrics
    deliberately distinguish factuality detection from the operational safety
    action, because treating every abstention as a confirmed hallucination can
    make a safe system appear to have a 100% false-positive rate.
    """
    if len(y_true) != len(decisions):
        raise ValueError("y_true and decisions must have the same length")
    if y_score is not None and len(y_score) != len(y_true):
        raise ValueError("y_score and y_true must have the same length")

    normalized = [getattr(decision, "value", decision) for decision in decisions]
    valid = {"supported", "hallucinated", "mixed", "unknown"}
    invalid = sorted({str(decision) for decision in normalized if decision not in valid})
    if invalid:
        raise ValueError(f"unsupported decision label(s): {', '.join(invalid)}")

    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "coverage": float("nan"),
            "abstention_rate": float("nan"),
            "intervention_rate": float("nan"),
            "human_review_candidate_rate": float("nan"),
        }

    confirmed_indices = [
        index for index, decision in enumerate(normalized)
        if decision in {"supported", "hallucinated"}
    ]
    abstained_indices = [
        index for index, decision in enumerate(normalized)
        if decision in {"mixed", "unknown"}
    ]
    hallucination_indices = [
        index for index, decision in enumerate(normalized)
        if decision == "hallucinated"
    ]
    release_indices = [
        index for index, decision in enumerate(normalized)
        if decision == "supported"
    ]
    review_indices = [
        index for index, decision in enumerate(normalized)
        if decision != "supported"
    ]

    total_positive = sum(int(label == 1) for label in y_true)
    total_negative = n - total_positive
    confirmed_tp = sum(y_true[index] == 1 for index in hallucination_indices)
    confirmed_fp = sum(y_true[index] == 0 for index in hallucination_indices)
    released_fn = sum(y_true[index] == 1 for index in release_indices)
    released_tn = sum(y_true[index] == 0 for index in release_indices)
    abstained_positive = sum(y_true[index] == 1 for index in abstained_indices)
    abstained_negative = sum(y_true[index] == 0 for index in abstained_indices)
    review_positive = sum(y_true[index] == 1 for index in review_indices)

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    confirmed_true = [y_true[index] for index in confirmed_indices]
    confirmed_pred = [
        int(normalized[index] == "hallucinated") for index in confirmed_indices
    ]
    confirmed_score = (
        [y_score[index] for index in confirmed_indices]
        if y_score is not None
        else None
    )

    ranking: dict[str, float | None] = {"auroc": None, "auprc": None}
    if y_score is not None and len(set(y_true)) > 1:
        ranking = {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }

    return {
        "n": n,
        "ground_truth": {
            "hallucinated": total_positive,
            "supported": total_negative,
        },
        "decision_counts": {
            label: normalized.count(label)
            for label in ("supported", "hallucinated", "mixed", "unknown")
        },
        "coverage": float(len(confirmed_indices) / n),
        "abstention_rate": float(len(abstained_indices) / n),
        "intervention_rate": float(len(review_indices) / n),
        "human_review_candidate_rate": float(len(abstained_indices) / n),
        "confirmed_detection": {
            "precision": safe_ratio(confirmed_tp, confirmed_tp + confirmed_fp),
            "end_to_end_recall": safe_ratio(confirmed_tp, total_positive),
            "decided_subset_metrics": classification_metrics(
                confirmed_true,
                confirmed_pred,
                confirmed_score,
            ),
        },
        "safety": {
            "unsafe_release_count": released_fn,
            "unsafe_release_rate": safe_ratio(released_fn, total_positive),
            "supported_release_precision": safe_ratio(
                released_tn,
                released_tn + released_fn,
            ),
            "intervention_precision": safe_ratio(review_positive, len(review_indices)),
            "intervention_recall": safe_ratio(review_positive, total_positive),
        },
        "abstentions": {
            "count": len(abstained_indices),
            "hallucinated": abstained_positive,
            "supported": abstained_negative,
        },
        "ranking": ranking,
    }


def remediation_metrics(y_true: list[int], actions: list[str]) -> dict[str, Any]:
    """Measure automatic handling without equating refusal with usefulness.

    A response is automated when its action is anything except human review.
    Useful answers are unchanged or extractively corrected releases. Separate
    retention and containment metrics expose systems that achieve automation
    merely by blocking or abstaining from everything.
    """
    if len(y_true) != len(actions):
        raise ValueError("y_true and actions must have the same length")
    normalized = [getattr(action, "value", action) for action in actions]
    valid = {
        "release",
        "corrected_release",
        "block",
        "safe_abstain",
        "human_review",
    }
    invalid = sorted({str(action) for action in normalized if action not in valid})
    if invalid:
        raise ValueError(f"unsupported remediation action(s): {', '.join(invalid)}")

    n = len(y_true)
    if n == 0:
        return {"n": 0, "automation_rate": float("nan")}

    counts = {action: normalized.count(action) for action in sorted(valid)}
    automated = [index for index, action in enumerate(normalized) if action != "human_review"]
    useful = [
        index for index, action in enumerate(normalized)
        if action in {"release", "corrected_release"}
    ]
    original_releases = [
        index for index, action in enumerate(normalized) if action == "release"
    ]
    total_positive = sum(label == 1 for label in y_true)
    total_negative = n - total_positive
    unsafe_original_releases = sum(y_true[index] == 1 for index in original_releases)
    contained_hallucinations = sum(
        y_true[index] == 1 and normalized[index] != "release"
        for index in range(n)
    )
    retained_supported = sum(y_true[index] == 0 for index in useful)
    correct_safe_handling = contained_hallucinations + retained_supported

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    return {
        "n": n,
        "action_counts": counts,
        "automation_rate": float(len(automated) / n),
        "human_review_rate": float(counts["human_review"] / n),
        "useful_answer_rate": float(len(useful) / n),
        "unchanged_release_rate": float(counts["release"] / n),
        "corrected_release_rate": float(counts["corrected_release"] / n),
        "safe_abstention_rate": float(counts["safe_abstain"] / n),
        "block_rate": float(counts["block"] / n),
        "unsafe_original_release_count": unsafe_original_releases,
        "unsafe_original_release_rate": safe_ratio(
            unsafe_original_releases,
            total_positive,
        ),
        "hallucination_containment_rate": safe_ratio(
            contained_hallucinations,
            total_positive,
        ),
        "supported_answer_retention_rate": safe_ratio(
            retained_supported,
            total_negative,
        ),
        "safe_handling_rate": float(correct_safe_handling / n),
        "warning": (
            "Automation includes blocking and safe abstention. Use useful_answer_rate "
            "and supported_answer_retention_rate to detect refusal-based metric gaming."
        ),
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: list[int], y_prob: list[float], n_bins: int = 10
) -> float:
    """Standard binned ECE: weighted average |confidence - accuracy| across
    equal-width probability bins. Matters here specifically because the
    router thresholds (tau_1, tau_2) act directly on these probabilities
    (§18) -- a miscalibrated-but-accurate classifier can still produce a
    badly miscalibrated router."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_arr)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_confidence = y_prob_arr[mask].mean()
        bin_accuracy = y_true_arr[mask].mean()
        ece += (bin_count / n) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def calibration_metrics(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> dict[str, float]:
    if len(y_true) == 0:
        return {"ece": float("nan"), "brier_score": float("nan")}
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """P50/P95/P99 wall-clock latency, per Objective 2 of the master
    prompt. Also reports mean since percentiles alone can hide a heavy
    but rare tail that mean would catch, and vice versa."""
    if len(latencies_ms) == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan")}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Cost metrics
# ---------------------------------------------------------------------------


def cost_summary(costs: list[float]) -> dict[str, float]:
    """Aggregate cost. Values are whatever unit the caller uses
    consistently (e.g. USD, or a normalized "compute unit") — this
    function doesn't assume a currency, it just aggregates."""
    if len(costs) == 0:
        return {"total": 0.0, "mean": 0.0, "per_1k_requests": 0.0}
    arr = np.asarray(costs, dtype=float)
    n = len(arr)
    return {
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "per_1k_requests": float(arr.mean() * 1000),
    }
