from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from ..tools import (
    build_root_feature_rows,
    causal_graph_filter,
    contribution_analysis,
    fault_onset_timing,
    granger_causality,
    knowledge_guided_root_cause_decision,
    pca_monitoring,
    pre_post_shift_evidence,
    process_diagnosis,
    root_cause_rank_enhanced,
    standardize_against_normal,
    stationarity_analysis,
    temporal_fault_type_evidence,
)


@dataclass
class TEPAblationPrediction:
    """One root-cause prediction from one evidence configuration."""

    fault_id: int
    variant: str
    predicted_root: str | None
    top3_roots: list[str]
    predicted_fault_id: int | None
    predicted_fault_label: str | None
    confidence: float
    decision_source: str
    abstain_reason: str | None
    expected_roots: list[str]
    root_hit: bool | None
    top3_root_hit: bool | None
    fault_id_hit: bool | None
    false_confident: bool | None


TEP_ABLATION_VARIANTS: tuple[str, ...] = (
    "generic",
    "topology_only",
    "catalog_only",
    "topology_catalog",
)


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains NaN or Inf")
    return arr


def _causal_input(x: np.ndarray, stationarity: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    recommendations = stationarity.get("differencing_recommendations", [])
    difference_channels = []
    for rec in recommendations:
        if rec.get("difference"):
            try:
                difference_channels.append(int(rec["channel"]))
            except Exception:
                continue
    if not difference_channels:
        return x, []

    transformed = x.copy()
    for j in difference_channels:
        if 0 <= j < transformed.shape[1] and transformed.shape[0] > 1:
            transformed[1:, j] = np.diff(x[:, j])
            transformed[0, j] = transformed[1, j]
    return transformed, difference_channels


def _score_map_from_contributions(contributions: dict[str, Any], names: list[str]) -> dict[str, float]:
    arr = np.asarray(contributions.get("variable_contributions", []), dtype=float).ravel()
    if arr.size != len(names):
        return {}
    return {names[i]: float(arr[i]) for i in range(len(names))}


def _top_roots(ranking: dict[str, Any], k: int = 3) -> list[str]:
    rows = ranking.get("root_cause_ranking", [])
    return [str(row["variable"]) for row in rows[:k] if "variable" in row]


def _generic_prediction(ranking: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    diagnosis = process_diagnosis(
        ranking.get("root_cause_ranking", []),
        ranking.get("propagation_paths", []),
        fault_catalog=None,
        confidence_threshold=threshold,
    )
    best = ranking.get("root_cause_ranking", [{}])[0] if ranking.get("root_cause_ranking") else {}
    return {
        "predicted_root": diagnosis.get("root_cause"),
        "top3_roots": _top_roots(ranking),
        "predicted_fault_id": None,
        "predicted_fault_label": diagnosis.get("fault_label"),
        "confidence": float(diagnosis.get("confidence", best.get("score", 0.0)) or 0.0),
        "decision_source": "generic_contribution_onset_granger_shift",
        "abstain_reason": diagnosis.get("abstain_reason"),
    }


def _catalog_prediction(
    ranking: dict[str, Any],
    *,
    fault_catalog: Any,
    shift_scores: dict[str, float],
    contribution_scores: dict[str, float],
    onset_order: Iterable[str],
    fault_type_scores: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    kg = knowledge_guided_root_cause_decision(
        ranking.get("root_cause_ranking", []),
        fault_catalog,
        shift_scores=shift_scores,
        contribution_scores=contribution_scores,
        onset_order=onset_order,
        fault_type_scores=fault_type_scores,
        threshold=threshold,
    )
    confidence = float(kg.get("confidence", 0.0) or 0.0)
    abstain_reason = kg.get("abstain_reason")
    if not abstain_reason and confidence < threshold:
        abstain_reason = "catalog_evidence_below_threshold"
    return {
        "predicted_root": kg.get("root_cause"),
        "top3_roots": [str(x) for x in kg.get("ranked_roots", kg.get("ranked_variables", []))[:3]],
        "predicted_fault_id": kg.get("fault_id") if abstain_reason is None else None,
        "predicted_fault_label": kg.get("fault_label") if abstain_reason is None else None,
        "confidence": confidence,
        "decision_source": kg.get("decision_source", "tep_fault_catalog"),
        "abstain_reason": abstain_reason,
        "catalog_ranking": kg.get("catalog_ranking", []),
        "catalog_margin": kg.get("catalog_margin"),
    }


def _with_scores(
    raw: dict[str, Any],
    *,
    fault_id: int,
    variant: str,
    expected_roots: list[str],
    confidence_threshold: float,
) -> TEPAblationPrediction:
    root = raw.get("predicted_root")
    top3 = [str(x) for x in raw.get("top3_roots", [])]
    pred_fault = raw.get("predicted_fault_id")
    pred_fault_int = None if pred_fault is None else int(pred_fault)
    root_hit = None if not expected_roots else root in expected_roots
    top3_hit = None if not expected_roots else bool(set(top3) & set(expected_roots))
    fault_id_hit = None if pred_fault_int is None else pred_fault_int == int(fault_id)
    confidence = float(raw.get("confidence", 0.0) or 0.0)
    false_confident = None
    if expected_roots and root is not None:
        false_confident = bool((root not in expected_roots) and confidence >= confidence_threshold)

    return TEPAblationPrediction(
        fault_id=int(fault_id),
        variant=variant,
        predicted_root=root,
        top3_roots=top3,
        predicted_fault_id=pred_fault_int,
        predicted_fault_label=raw.get("predicted_fault_label"),
        confidence=confidence,
        decision_source=str(raw.get("decision_source", variant)),
        abstain_reason=raw.get("abstain_reason"),
        expected_roots=expected_roots,
        root_hit=root_hit,
        top3_root_hit=top3_hit,
        fault_id_hit=fault_id_hit,
        false_confident=false_confident,
    )


def run_tep_root_cause_ablation(
    signal_matrix,
    normal_reference,
    channel_names,
    alarm_mask,
    *,
    fault_id: int,
    expected_roots: Iterable[str] = (),
    process_topology: Any = None,
    fault_catalog: Any = None,
    variance_target: float = 0.95,
    control_alpha: float = 0.99,
    maxlag: int = 1,
    granger_alpha: float = 0.05,
    onset_z_threshold: float = 3.5,
    onset_persistence: int = 3,
    confidence_threshold: float = 0.18,
) -> dict[str, Any]:
    """Compare generic/knowledge variants and emit leakage-free calibration features.

    True fault ID and expected roots are used only for scoring after predictions and
    as calibration targets. Candidate features are computed without the true label.
    """
    x = _as_2d(signal_matrix)
    ref = _as_2d(normal_reference)
    names = [str(n) for n in channel_names]
    if x.shape[1] != ref.shape[1] or len(names) != x.shape[1]:
        raise ValueError("current/reference/channel dimensions must match")

    standardized = standardize_against_normal(x, ref)
    z = standardized["standardized_signal"]
    z_ref = standardized["standardized_reference"]

    pca = pca_monitoring(z, z_ref, variance_target=variance_target, alpha=control_alpha)
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    if alarm.shape[0] != x.shape[0]:
        raise ValueError("alarm_mask length mismatch")
    if not np.any(alarm):
        alarm = np.asarray(pca["alarm_mask"], dtype=bool)

    contributions = contribution_analysis(z, pca["pca_state"], alarm)
    shift = pre_post_shift_evidence(z, names, alarm_mask=alarm)
    type_evidence = temporal_fault_type_evidence(z, names, alarm_mask=alarm)
    stationarity = stationarity_analysis(z)
    causal_matrix, differenced_channels = _causal_input(z, stationarity)
    granger = granger_causality(causal_matrix, names, maxlag=maxlag, alpha=granger_alpha)
    onset = fault_onset_timing(
        z,
        alarm,
        names,
        z_threshold=onset_z_threshold,
        persistence=onset_persistence,
    )

    filtered_generic = causal_graph_filter(
        granger["directed_edges"],
        process_topology=None,
        p_value_threshold=granger_alpha,
    )
    filtered_topology = causal_graph_filter(
        granger["directed_edges"],
        process_topology=process_topology,
        p_value_threshold=granger_alpha,
    )

    contribution_scores = _score_map_from_contributions(contributions, names)
    common_rank_kwargs = dict(
        variable_contributions=contributions["variable_contributions"],
        shift_scores=shift["shift_scores"],
        channel_names=names,
    )
    ranking_generic = root_cause_rank_enhanced(
        contributions["suspect_variables"],
        filtered_generic["filtered_causal_graph"],
        onset["onset_order"],
        **common_rank_kwargs,
    )
    ranking_topology = root_cause_rank_enhanced(
        contributions["suspect_variables"],
        filtered_topology["filtered_causal_graph"],
        onset["onset_order"],
        **common_rank_kwargs,
    )

    expected = [str(v) for v in expected_roots]
    raw_predictions = {
        "generic": _generic_prediction(ranking_generic, threshold=confidence_threshold),
        "topology_only": _generic_prediction(ranking_topology, threshold=confidence_threshold),
    }
    if fault_catalog:
        raw_predictions["catalog_only"] = _catalog_prediction(
            ranking_generic,
            fault_catalog=fault_catalog,
            shift_scores=shift["shift_scores"],
            contribution_scores=contribution_scores,
            onset_order=onset["onset_order"],
            fault_type_scores=type_evidence["fault_type_scores"],
            threshold=confidence_threshold,
        )
        raw_predictions["topology_catalog"] = _catalog_prediction(
            ranking_topology,
            fault_catalog=fault_catalog,
            shift_scores=shift["shift_scores"],
            contribution_scores=contribution_scores,
            onset_order=onset["onset_order"],
            fault_type_scores=type_evidence["fault_type_scores"],
            threshold=confidence_threshold,
        )

    predictions = [
        _with_scores(
            raw_predictions[variant],
            fault_id=fault_id,
            variant=variant,
            expected_roots=expected,
            confidence_threshold=confidence_threshold,
        )
        for variant in TEP_ABLATION_VARIANTS
        if variant in raw_predictions
    ]

    feature_rows = build_root_feature_rows(
        names,
        contribution_scores=contribution_scores,
        shift_scores=shift["shift_scores"],
        onset_order=onset["onset_order"],
        process_topology=process_topology,
        fault_catalog=fault_catalog,
        fault_type_scores=type_evidence["fault_type_scores"],
    )
    calibration_record = {
        "fault_id": int(fault_id),
        "expected_roots": expected,
        "candidates": feature_rows,
        "fault_type_scores": type_evidence["fault_type_scores"],
    }

    return {
        "predictions": predictions,
        "rows": [asdict(p) for p in predictions],
        "calibration_record": calibration_record,
        "artifacts": {
            "selected_channels": names,
            "contribution_ranking": [names[int(i)] for i in np.argsort(contributions["variable_contributions"])[::-1][:5]],
            "shift_ranking": shift["ranked_variables"][:5],
            "fault_type_scores": type_evidence["fault_type_scores"],
            "fault_type_metrics": type_evidence["metrics"],
            "onset_order": onset["onset_order"][:5],
            "generic_top3": _top_roots(ranking_generic),
            "topology_top3": _top_roots(ranking_topology),
            "generic_edge_count": len(filtered_generic["filtered_causal_graph"]["edges"]),
            "topology_edge_count": len(filtered_topology["filtered_causal_graph"]["edges"]),
            "differenced_channels": differenced_channels,
        },
    }


def summarize_tep_ablation(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    variants = sorted({str(r["variant"]) for r in rows})
    summary: dict[str, Any] = {}
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        root_cases = [r for r in group if r.get("root_hit") is not None]
        top3_cases = [r for r in group if r.get("top3_root_hit") is not None]
        fault_cases = [r for r in group if r.get("fault_id_hit") is not None]
        false_cases = [r for r in group if r.get("false_confident") is not None]
        abstained = [r for r in group if r.get("abstain_reason")]
        summary[variant] = {
            "cases": len(group),
            "root_accuracy": float(np.mean([bool(r["root_hit"]) for r in root_cases])) if root_cases else None,
            "root_cases": len(root_cases),
            "top3_root_accuracy": float(np.mean([bool(r["top3_root_hit"]) for r in top3_cases])) if top3_cases else None,
            "top3_root_cases": len(top3_cases),
            "fault_id_accuracy": float(np.mean([bool(r["fault_id_hit"]) for r in fault_cases])) if fault_cases else None,
            "fault_id_cases": len(fault_cases),
            "abstention_rate": float(len(abstained) / len(group)) if group else None,
            "false_confident_rate": float(np.mean([bool(r["false_confident"]) for r in false_cases])) if false_cases else None,
            "mean_confidence": float(np.mean([float(r.get("confidence", 0.0) or 0.0) for r in group])) if group else None,
        }
    return summary
