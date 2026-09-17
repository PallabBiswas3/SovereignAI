from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np


FEATURE_NAMES: tuple[str, ...] = (
    "contribution_score",
    "pre_post_shift_score",
    "onset_earliness_score",
    "topology_upstreamness_score",
    "fault_type_agreement_score",
    "catalog_root_prior_score",
)


@dataclass(frozen=True)
class RootRankerConfig:
    weights: dict[str, float]
    margin_threshold: float = 0.0
    max_abstention_rate: float = 0.35
    max_false_confident_rate: float = 0.50


@dataclass(frozen=True)
class RankedRoot:
    variable: str
    score: float
    features: dict[str, float]


def _normalize_map(values: dict[str, float]) -> dict[str, float]:
    clean = {str(k): max(0.0, float(v)) for k, v in values.items()}
    scale = max(clean.values()) if clean else 0.0
    if scale <= 1e-12:
        return {k: 0.0 for k in clean}
    return {k: float(v / scale) for k, v in clean.items()}


def _topology_edges(process_topology: Any) -> list[tuple[str, str]]:
    if not process_topology:
        return []
    raw = process_topology.get("edges", []) if isinstance(process_topology, dict) else process_topology
    out: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        for cause, effects in raw.items():
            if isinstance(effects, str):
                effects = [effects]
            for effect in effects:
                out.append((str(cause), str(effect)))
        return out
    for edge in raw:
        if isinstance(edge, dict):
            cause = edge.get("cause") or edge.get("source") or edge.get("from")
            effect = edge.get("effect") or edge.get("target") or edge.get("to")
            if cause is not None and effect is not None:
                out.append((str(cause), str(effect)))
        elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
            out.append((str(edge[0]), str(edge[1])))
    return out


def topology_upstreamness_scores(
    candidate_names: Iterable[str],
    process_topology: Any,
) -> dict[str, float]:
    """Compute a bounded structural upstreamness prior from a directed topology.

    Upstream variables are rewarded for outgoing influence and penalized for being
    primarily downstream receivers. This uses only process structure, never the
    true fault label.
    """
    names = [str(x) for x in candidate_names]
    edges = _topology_edges(process_topology)
    incoming = {name: 0.0 for name in names}
    outgoing = {name: 0.0 for name in names}
    for cause, effect in edges:
        if cause in outgoing:
            outgoing[cause] += 1.0
        if effect in incoming:
            incoming[effect] += 1.0
    max_out = max(outgoing.values()) if outgoing else 1.0
    max_in = max(incoming.values()) if incoming else 1.0
    max_out = max(max_out, 1.0)
    max_in = max(max_in, 1.0)
    return {
        name: float(
            np.clip(
                0.70 * (outgoing[name] / max_out)
                + 0.30 * (1.0 - incoming[name] / max_in),
                0.0,
                1.0,
            )
        )
        for name in names
    }


def _fault_type_key(raw: str | None) -> str:
    value = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "random" in value:
        return "random_variation"
    if "drift" in value:
        return "slow_drift"
    if "sticking" in value or "stiction" in value:
        return "valve_sticking"
    if "constant" in value or "fixed" in value:
        return "constant_position"
    if "step" in value:
        return "step"
    if "unknown" in value:
        return "unknown"
    return value or "unknown"


def _catalog_records(fault_catalog: Any) -> list[dict[str, Any]]:
    if isinstance(fault_catalog, dict):
        raw = fault_catalog.get("faults", [])
    elif isinstance(fault_catalog, list):
        raw = fault_catalog
    else:
        raw = []
    return [row for row in raw if isinstance(row, dict)]


def _record_roots(record: dict[str, Any]) -> list[str]:
    roots = record.get("expected_roots")
    if roots:
        return [str(x) for x in roots]
    root = record.get("root_cause") or record.get("root_variable")
    return [str(root)] if root else []


def catalog_feature_scores(
    candidate_names: Iterable[str],
    fault_catalog: Any,
    *,
    ranked_variables: Iterable[str],
    variable_scores: dict[str, float] | None = None,
    fault_type_scores: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-root catalog prior and temporal-type agreement scores.

    Catalog priors are computed against the full fault catalog. The true IDV label
    is never supplied, preventing label leakage during ranking or calibration.
    """
    from .tep_reasoning import rank_tep_fault_catalog

    names = [str(x) for x in candidate_names]
    ranked = rank_tep_fault_catalog(
        ranked_variables,
        fault_catalog,
        variable_scores=variable_scores,
        fault_type_scores=fault_type_scores,
    )["fault_catalog_ranking"]

    prior = {name: 0.0 for name in names}
    type_agreement = {name: 0.0 for name in names}
    records = {row.get("fault_id"): row for row in _catalog_records(fault_catalog)}

    for row in ranked:
        score = max(0.0, float(row.get("score", 0.0) or 0.0))
        roots = [str(x) for x in row.get("expected_roots", [])]
        for root in roots:
            if root in prior:
                prior[root] = max(prior[root], score)
        record = records.get(row.get("fault_id"))
        if record:
            key = _fault_type_key(str(record.get("type") or record.get("fault_type") or ""))
            type_score = max(0.0, float((fault_type_scores or {}).get(key, 0.0)))
            for root in _record_roots(record):
                if root in type_agreement:
                    type_agreement[root] = max(type_agreement[root], type_score)

    return _normalize_map(prior), type_agreement


def build_root_feature_rows(
    candidate_names: Iterable[str],
    *,
    contribution_scores: dict[str, float],
    shift_scores: dict[str, float],
    onset_order: Iterable[str],
    process_topology: Any,
    fault_catalog: Any,
    fault_type_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Construct the six leakage-free root-ranking features requested by PR #10."""
    names = [str(x) for x in candidate_names]
    contrib = _normalize_map({name: contribution_scores.get(name, 0.0) for name in names})
    shift = _normalize_map({name: shift_scores.get(name, 0.0) for name in names})

    ordered = [str(x) for x in onset_order]
    onset_pos = {name: i for i, name in enumerate(ordered)}
    denom = max(1, len(ordered) - 1)
    onset = {
        name: float(1.0 - onset_pos[name] / denom) if name in onset_pos else 0.0
        for name in names
    }
    topology = topology_upstreamness_scores(names, process_topology)

    evidence_mix = {
        name: 0.55 * shift.get(name, 0.0)
        + 0.30 * contrib.get(name, 0.0)
        + 0.15 * onset.get(name, 0.0)
        for name in names
    }
    ranked_variables = [name for name, _ in sorted(evidence_mix.items(), key=lambda kv: kv[1], reverse=True)]
    catalog_prior, type_agreement = catalog_feature_scores(
        names,
        fault_catalog,
        ranked_variables=ranked_variables,
        variable_scores=evidence_mix,
        fault_type_scores=fault_type_scores,
    )

    rows = []
    for name in names:
        rows.append(
            {
                "variable": name,
                "contribution_score": float(contrib.get(name, 0.0)),
                "pre_post_shift_score": float(shift.get(name, 0.0)),
                "onset_earliness_score": float(onset.get(name, 0.0)),
                "topology_upstreamness_score": float(topology.get(name, 0.0)),
                "fault_type_agreement_score": float(type_agreement.get(name, 0.0)),
                "catalog_root_prior_score": float(catalog_prior.get(name, 0.0)),
            }
        )
    return rows


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {name: max(0.0, float(weights.get(name, 0.0))) for name in FEATURE_NAMES}
    total = sum(clean.values())
    if total <= 1e-12:
        return {name: 1.0 / len(FEATURE_NAMES) for name in FEATURE_NAMES}
    return {name: value / total for name, value in clean.items()}


def score_root_candidates(
    candidates: Iterable[dict[str, Any]],
    weights: dict[str, float],
) -> list[RankedRoot]:
    w = _normalized_weights(weights)
    ranked: list[RankedRoot] = []
    for row in candidates:
        features = {name: float(np.clip(row.get(name, 0.0), 0.0, 1.0)) for name in FEATURE_NAMES}
        score = float(sum(w[name] * features[name] for name in FEATURE_NAMES))
        ranked.append(RankedRoot(variable=str(row["variable"]), score=score, features=features))
    ranked.sort(key=lambda r: (-r.score, r.variable))
    return ranked


def predict_root(
    record: dict[str, Any],
    config: RootRankerConfig,
) -> dict[str, Any]:
    ranked = score_root_candidates(record.get("candidates", []), config.weights)
    if not ranked:
        return {
            "predicted_root": None,
            "top3_roots": [],
            "confidence": 0.0,
            "score_margin": 0.0,
            "abstained": True,
            "abstain_reason": "no_root_candidates",
            "ranking": [],
        }
    second = ranked[1].score if len(ranked) > 1 else 0.0
    margin = float(ranked[0].score - second)
    abstained = margin < float(config.margin_threshold)
    return {
        "predicted_root": None if abstained else ranked[0].variable,
        "raw_top_root": ranked[0].variable,
        "top3_roots": [row.variable for row in ranked[:3]],
        "confidence": float(ranked[0].score),
        "score_margin": margin,
        "abstained": abstained,
        "abstain_reason": "calibrated_margin_below_threshold" if abstained else None,
        "ranking": [asdict(row) for row in ranked],
    }


def evaluate_root_ranker(
    records: Iterable[dict[str, Any]],
    config: RootRankerConfig,
) -> dict[str, Any]:
    rows = []
    for record in records:
        expected = {str(x) for x in record.get("expected_roots", [])}
        if not expected:
            continue
        prediction = predict_root(record, config)
        raw_top = prediction.get("raw_top_root")
        accepted = prediction.get("predicted_root")
        top3 = set(prediction.get("top3_roots", []))
        root_hit = bool(accepted in expected) if accepted is not None else False
        raw_hit = bool(raw_top in expected) if raw_top is not None else False
        top3_hit = bool(top3 & expected)
        false_confident = bool(accepted is not None and accepted not in expected)
        rows.append(
            {
                "fault_id": int(record["fault_id"]),
                "expected_roots": sorted(expected),
                **prediction,
                "root_hit": root_hit,
                "raw_top1_hit": raw_hit,
                "top3_root_hit": top3_hit,
                "false_confident": false_confident,
            }
        )

    n = len(rows)
    if not n:
        return {
            "top1_root_accuracy": None,
            "raw_top1_root_accuracy": None,
            "top3_root_accuracy": None,
            "abstention_rate": None,
            "false_confident_rate": None,
            "cases": 0,
            "rows": [],
        }
    return {
        "top1_root_accuracy": float(np.mean([row["root_hit"] for row in rows])),
        "raw_top1_root_accuracy": float(np.mean([row["raw_top1_hit"] for row in rows])),
        "top3_root_accuracy": float(np.mean([row["top3_root_hit"] for row in rows])),
        "abstention_rate": float(np.mean([row["abstained"] for row in rows])),
        "false_confident_rate": float(np.mean([row["false_confident"] for row in rows])),
        "cases": n,
        "rows": rows,
    }


def _integer_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in _integer_compositions(total - head, parts - 1):
            yield (head,) + tail


def weight_grid(step: float = 0.20) -> list[dict[str, float]]:
    units = int(round(1.0 / float(step)))
    if units <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("step must evenly divide 1.0")
    return [
        {name: value / units for name, value in zip(FEATURE_NAMES, composition)}
        for composition in _integer_compositions(units, len(FEATURE_NAMES))
    ]


def optimize_root_ranker(
    records: Iterable[dict[str, Any]],
    *,
    max_abstention_rate: float = 0.35,
    max_false_confident_rate: float = 0.50,
    weight_step: float = 0.20,
    margin_grid: Iterable[float] = (0.0, 0.02, 0.04, 0.06, 0.08),
) -> dict[str, Any]:
    records = list(records)
    trials = []
    for weights, margin in product(weight_grid(weight_step), margin_grid):
        config = RootRankerConfig(
            weights=weights,
            margin_threshold=float(margin),
            max_abstention_rate=float(max_abstention_rate),
            max_false_confident_rate=float(max_false_confident_rate),
        )
        metrics = evaluate_root_ranker(records, config)
        abstention = metrics["abstention_rate"] if metrics["abstention_rate"] is not None else 1.0
        false_confident = metrics["false_confident_rate"] if metrics["false_confident_rate"] is not None else 1.0
        feasible = bool(abstention <= max_abstention_rate and false_confident <= max_false_confident_rate)
        trials.append((config, metrics, feasible))

    def objective(item):
        config, metrics, feasible = item
        top1 = metrics["top1_root_accuracy"] or 0.0
        raw_top1 = metrics["raw_top1_root_accuracy"] or 0.0
        top3 = metrics["top3_root_accuracy"] or 0.0
        abstention = metrics["abstention_rate"] or 0.0
        false_confident = metrics["false_confident_rate"] or 0.0
        violation = max(0.0, abstention - max_abstention_rate) + max(0.0, false_confident - max_false_confident_rate)
        return (
            1 if feasible else 0,
            top1,
            raw_top1,
            top3,
            -false_confident,
            -abstention,
            -violation,
            -config.margin_threshold,
        )

    best_config, best_metrics, feasible = max(trials, key=objective)
    return {
        "config": best_config,
        "metrics": best_metrics,
        "feasible": feasible,
        "search_space_size": len(trials),
    }


def _fault_level_folds(records: list[dict[str, Any]], n_splits: int) -> list[list[int]]:
    ids = sorted({int(row["fault_id"]) for row in records if row.get("expected_roots")})
    n_splits = min(max(2, int(n_splits)), max(2, len(ids)))
    folds = [[] for _ in range(n_splits)]
    for i, fault_id in enumerate(ids):
        folds[i % n_splits].append(fault_id)
    return folds


def cross_validate_root_ranker(
    records: Iterable[dict[str, Any]],
    *,
    n_splits: int = 4,
    max_abstention_rate: float = 0.35,
    max_false_confident_rate: float = 0.50,
    weight_step: float = 0.20,
    margin_grid: Iterable[float] = (0.0, 0.02, 0.04, 0.06, 0.08),
) -> dict[str, Any]:
    """Fault-level cross-validation for leakage-resistant weight calibration."""
    records = [row for row in records if row.get("expected_roots")]
    folds = _fault_level_folds(records, n_splits)
    held_out_rows = []
    fold_reports = []

    for fold_index, validation_ids in enumerate(folds):
        validation_set = set(validation_ids)
        train = [row for row in records if int(row["fault_id"]) not in validation_set]
        valid = [row for row in records if int(row["fault_id"]) in validation_set]
        fitted = optimize_root_ranker(
            train,
            max_abstention_rate=max_abstention_rate,
            max_false_confident_rate=max_false_confident_rate,
            weight_step=weight_step,
            margin_grid=margin_grid,
        )
        validation_metrics = evaluate_root_ranker(valid, fitted["config"])
        held_out_rows.extend(validation_metrics["rows"])
        fold_reports.append(
            {
                "fold": fold_index,
                "validation_fault_ids": validation_ids,
                "train_fault_ids": sorted({int(row["fault_id"]) for row in train}),
                "config": asdict(fitted["config"]),
                "train_metrics": {k: v for k, v in fitted["metrics"].items() if k != "rows"},
                "validation_metrics": {k: v for k, v in validation_metrics.items() if k != "rows"},
            }
        )

    n = len(held_out_rows)
    cv_metrics = {
        "top1_root_accuracy": float(np.mean([row["root_hit"] for row in held_out_rows])) if n else None,
        "raw_top1_root_accuracy": float(np.mean([row["raw_top1_hit"] for row in held_out_rows])) if n else None,
        "top3_root_accuracy": float(np.mean([row["top3_root_hit"] for row in held_out_rows])) if n else None,
        "abstention_rate": float(np.mean([row["abstained"] for row in held_out_rows])) if n else None,
        "false_confident_rate": float(np.mean([row["false_confident"] for row in held_out_rows])) if n else None,
        "cases": n,
    }

    final_fit = optimize_root_ranker(
        records,
        max_abstention_rate=max_abstention_rate,
        max_false_confident_rate=max_false_confident_rate,
        weight_step=weight_step,
        margin_grid=margin_grid,
    )
    return {
        "method": f"{len(folds)}-fold fault-level cross-validation",
        "feature_names": list(FEATURE_NAMES),
        "folds": fold_reports,
        "held_out_rows": held_out_rows,
        "cv_metrics": cv_metrics,
        "final_config": asdict(final_fit["config"]),
        "final_training_metrics": {k: v for k, v in final_fit["metrics"].items() if k != "rows"},
        "search_space_size_per_fold": final_fit["search_space_size"],
    }
