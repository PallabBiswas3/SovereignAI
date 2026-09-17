from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

import numpy as np


def _normalize_edge(edge: Any) -> dict[str, Any]:
    if isinstance(edge, dict):
        cause = edge.get("cause") or edge.get("source") or edge.get("from")
        effect = edge.get("effect") or edge.get("target") or edge.get("to")
        if cause is None or effect is None:
            raise ValueError("edge dictionaries must define cause/source/from and effect/target/to")
        out = dict(edge)
        out["cause"] = str(cause)
        out["effect"] = str(effect)
        return out
    if isinstance(edge, (tuple, list)) and len(edge) >= 2:
        return {"cause": str(edge[0]), "effect": str(edge[1])}
    raise TypeError("edges must be mappings or (cause, effect) pairs")


def _topology_edge_set(process_topology: Any) -> set[tuple[str, str]] | None:
    """Normalize common process-topology representations to directed edges.

    Accepted forms:
    - iterable of (cause, effect) pairs
    - iterable of edge dictionaries
    - mapping {cause: [effect, ...]}
    - mapping {"edges": [...]}
    """
    if process_topology is None:
        return None

    if isinstance(process_topology, dict) and "edges" in process_topology:
        process_topology = process_topology["edges"]

    edges: set[tuple[str, str]] = set()
    if isinstance(process_topology, dict):
        for cause, effects in process_topology.items():
            if isinstance(effects, str):
                effects = [effects]
            for effect in effects:
                edges.add((str(cause), str(effect)))
        return edges

    for edge in process_topology:
        normalized = _normalize_edge(edge)
        edges.add((normalized["cause"], normalized["effect"]))
    return edges


def causal_graph_filter(
    directed_edges: Iterable[Any],
    process_topology: Any = None,
    *,
    p_value_threshold: float | None = None,
    allow_unknown_when_no_topology: bool = True,
) -> dict[str, Any]:
    """Filter data-driven causal links with significance and known process topology.

    Granger links are predictive relationships, not physical-causality proof. When a
    process topology is supplied, only topology-consistent directions are retained.
    If no topology is supplied, significant data-driven links are kept and marked as
    unverified by physics/topology.
    """
    normalized_edges = [_normalize_edge(edge) for edge in directed_edges]
    topology = _topology_edge_set(process_topology)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for edge in normalized_edges:
        reason = None
        p_value = edge.get("p_value")
        if p_value_threshold is not None and p_value is not None:
            if float(p_value) > p_value_threshold:
                reason = "above_p_value_threshold"

        if reason is None and topology is not None:
            pair = (edge["cause"], edge["effect"])
            if pair not in topology:
                reason = "not_allowed_by_process_topology"

        if reason is None:
            enriched = dict(edge)
            enriched["topology_verified"] = topology is not None
            kept.append(enriched)
        else:
            rejected = dict(edge)
            rejected["removal_reason"] = reason
            removed.append(rejected)

    if topology is None and not allow_unknown_when_no_topology:
        removed.extend(
            {**edge, "removal_reason": "topology_unavailable"} for edge in kept
        )
        kept = []

    nodes = sorted({e["cause"] for e in kept} | {e["effect"] for e in kept})
    return {
        "filtered_causal_graph": {
            "nodes": nodes,
            "edges": kept,
            "topology_available": topology is not None,
        },
        "removed_edges": removed,
    }


def _first_persistent_true(mask: np.ndarray, persistence: int) -> int | None:
    if persistence <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if idx.size else None
    run = 0
    for i, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= persistence:
            return i - persistence + 1
    return None


def fault_onset_timing(
    signal_matrix,
    alarm_mask,
    channel_names=None,
    *,
    timestamps=None,
    baseline_fraction: float = 0.2,
    z_threshold: float = 3.5,
    persistence: int = 3,
) -> dict[str, Any]:
    """Estimate per-channel fault onset using robust baseline deviations.

    The global alarm mask is used only to constrain where onset may be searched. A
    robust median/MAD baseline is learned from the pre-alarm region when possible,
    otherwise from the first baseline_fraction of the record. Each channel onset is
    the first persistent threshold exceedance.
    """
    x = np.asarray(signal_matrix, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("signal_matrix must be [samples, channels]")

    n, d = x.shape
    names = channel_names or [f"ch{i}" for i in range(d)]
    if len(names) != d:
        raise ValueError("channel_names length mismatch")

    alarm = np.asarray(alarm_mask, dtype=bool)
    if alarm.ndim == 2:
        if alarm.shape != x.shape:
            raise ValueError("2-D alarm_mask must match signal_matrix")
        global_alarm = np.any(alarm, axis=1)
    elif alarm.ndim == 1 and alarm.shape[0] == n:
        global_alarm = alarm
    else:
        raise ValueError("alarm_mask must have length n_samples or match signal_matrix")

    first_global = int(np.flatnonzero(global_alarm)[0]) if np.any(global_alarm) else None
    default_baseline_end = max(10, int(round(n * baseline_fraction)))
    baseline_end = first_global if first_global is not None and first_global >= 10 else default_baseline_end
    baseline_end = min(max(10, baseline_end), n)

    baseline = x[:baseline_end]
    center = np.nanmedian(baseline, axis=0)
    mad = np.nanmedian(np.abs(baseline - center), axis=0)
    robust_scale = 1.4826 * mad
    fallback = np.nanstd(baseline, axis=0)
    scale = np.where(robust_scale > 1e-12, robust_scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)

    z = np.abs((x - center) / scale)
    channel_alarm = z >= float(z_threshold)
    if first_global is not None:
        search_gate = np.arange(n) >= max(0, first_global - persistence + 1)
        channel_alarm &= search_gate[:, None]

    onset_records = []
    for j, name in enumerate(names):
        idx = _first_persistent_true(channel_alarm[:, j], persistence)
        onset_value = None
        if idx is not None and timestamps is not None:
            ts = np.asarray(timestamps)
            if ts.shape[0] != n:
                raise ValueError("timestamps length mismatch")
            onset_value = ts[idx].item() if hasattr(ts[idx], "item") else ts[idx]
        onset_records.append(
            {
                "channel": str(name),
                "index": idx,
                "timestamp": onset_value,
                "max_abs_z": float(np.nanmax(z[:, j])),
            }
        )

    ordered = sorted(
        [r for r in onset_records if r["index"] is not None],
        key=lambda r: (r["index"], -r["max_abs_z"]),
    )

    return {
        "onset_times": {r["channel"]: r["index"] for r in onset_records},
        "onset_timestamps": {r["channel"]: r["timestamp"] for r in onset_records},
        "onset_order": [r["channel"] for r in ordered],
        "onset_details": onset_records,
        "baseline": {"end_index": baseline_end, "center": center, "scale": scale},
        "channel_alarm_mask": channel_alarm,
        "z_scores": z,
    }


def _extract_graph_edges(filtered_causal_graph: Any) -> list[dict[str, Any]]:
    if filtered_causal_graph is None:
        return []
    if isinstance(filtered_causal_graph, dict) and "edges" in filtered_causal_graph:
        raw = filtered_causal_graph["edges"]
    else:
        raw = filtered_causal_graph
    return [_normalize_edge(edge) for edge in raw]


def _propagation_paths_from(root: str, edges: list[dict[str, Any]], max_depth: int = 5):
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["cause"]].append(edge["effect"])

    paths = []
    queue = deque([(root, [root])])
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for nxt in adjacency.get(node, []):
            if nxt in path:
                continue
            new_path = path + [nxt]
            paths.append(new_path)
            queue.append((nxt, new_path))
    return paths


def root_cause_rank(
    suspect_variables,
    filtered_causal_graph,
    onset_order,
    *,
    variable_contributions=None,
    channel_names=None,
    max_path_depth: int = 5,
) -> dict[str, Any]:
    """Rank process root causes from anomaly contribution, causality and onset timing.

    The score combines four transparent signals:
    - suspect/contribution evidence
    - early abnormal onset
    - outgoing causal influence
    - low incoming causal dependence
    """
    edges = _extract_graph_edges(filtered_causal_graph)
    nodes = sorted({e["cause"] for e in edges} | {e["effect"] for e in edges})

    names = list(channel_names) if channel_names is not None else None
    suspects: set[str] = set()
    for item in suspect_variables or []:
        if isinstance(item, (int, np.integer)) and names is not None and 0 <= int(item) < len(names):
            suspects.add(str(names[int(item)]))
        else:
            suspects.add(str(item))

    nodes = sorted(set(nodes) | suspects | {str(x) for x in (onset_order or [])})
    if not nodes:
        return {"root_cause_ranking": [], "propagation_paths": [], "confidence": 0.0}

    outgoing = defaultdict(int)
    incoming = defaultdict(int)
    for edge in edges:
        outgoing[edge["cause"]] += 1
        incoming[edge["effect"]] += 1

    onset_position = {str(name): i for i, name in enumerate(onset_order or [])}
    denom_onset = max(1, len(onset_position) - 1)
    max_out = max([outgoing[n] for n in nodes] + [1])
    max_in = max([incoming[n] for n in nodes] + [1])

    contribution_map: dict[str, float] = {}
    if variable_contributions is not None:
        arr = np.asarray(variable_contributions, dtype=float).ravel()
        if names is not None and arr.size == len(names):
            contribution_map = {str(names[i]): float(arr[i]) for i in range(len(names))}
        elif arr.size == len(nodes):
            contribution_map = {nodes[i]: float(arr[i]) for i in range(len(nodes))}

    rows = []
    for node in nodes:
        suspect_score = 1.0 if node in suspects else 0.0
        contribution_score = max(0.0, contribution_map.get(node, 0.0))
        if contribution_map:
            max_contrib = max(contribution_map.values()) or 1.0
            contribution_score /= max_contrib
        evidence_score = max(suspect_score, contribution_score)

        if node in onset_position:
            onset_score = 1.0 - onset_position[node] / denom_onset
        else:
            onset_score = 0.0
        outgoing_score = outgoing[node] / max_out
        upstream_score = 1.0 - incoming[node] / max_in

        score = (
            0.35 * evidence_score
            + 0.30 * onset_score
            + 0.25 * outgoing_score
            + 0.10 * upstream_score
        )
        rows.append(
            {
                "variable": node,
                "score": float(score),
                "suspect_score": float(suspect_score),
                "contribution_score": float(contribution_score),
                "onset_score": float(onset_score),
                "outgoing_score": float(outgoing_score),
                "upstream_score": float(upstream_score),
                "outgoing_edges": int(outgoing[node]),
                "incoming_edges": int(incoming[node]),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    second_score = rows[1]["score"] if len(rows) > 1 else 0.0
    separation = max(0.0, best["score"] - second_score)
    absolute = min(1.0, best["score"])
    confidence = float(np.clip(0.65 * absolute + 0.35 * separation, 0.0, 1.0))

    propagation_paths = _propagation_paths_from(best["variable"], edges, max_depth=max_path_depth)
    return {
        "root_cause_ranking": rows,
        "propagation_paths": propagation_paths,
        "confidence": confidence,
    }


def _catalog_match(root: str, propagation_paths, fault_catalog):
    if not fault_catalog:
        return None

    if isinstance(fault_catalog, dict):
        direct = fault_catalog.get(root)
        if isinstance(direct, str):
            return {"fault_label": direct, "match_type": "root_variable"}
        if isinstance(direct, dict):
            return {"fault_label": direct.get("label", root), "match_type": "root_variable", **direct}
        records = fault_catalog.get("faults") if "faults" in fault_catalog else None
        if records is None:
            return None
    else:
        records = fault_catalog

    path_nodes = set(node for path in propagation_paths for node in path)
    best = None
    best_score = -1.0
    for record in records or []:
        if not isinstance(record, dict):
            continue
        expected_root = record.get("root_cause") or record.get("root_variable")
        affected = set(str(x) for x in record.get("affected_variables", []))
        score = 0.0
        if expected_root is not None and str(expected_root) == root:
            score += 1.0
        if affected:
            score += len(affected & path_nodes) / len(affected)
        if score > best_score:
            best_score = score
            best = {
                "fault_label": record.get("label") or record.get("fault_label"),
                "catalog_score": float(score),
                "match_type": "catalog_pattern",
                "catalog_record": record,
            }
    return best if best_score > 0 else None


def process_diagnosis(
    root_cause_ranking,
    propagation_paths,
    fault_catalog=None,
    *,
    confidence_threshold: float = 0.35,
) -> dict[str, Any]:
    """Convert root-cause evidence into a process fault diagnosis or abstention."""
    ranking = list(root_cause_ranking or [])
    if not ranking:
        return {
            "fault_label": None,
            "root_cause": None,
            "confidence": 0.0,
            "abstain_reason": "no_root_cause_candidates",
            "affected_variables": [],
            "propagation_paths": list(propagation_paths or []),
        }

    best = ranking[0]
    root = str(best["variable"])
    base_confidence = float(np.clip(best.get("score", 0.0), 0.0, 1.0))
    paths = list(propagation_paths or [])
    affected = sorted({node for path in paths for node in path if node != root})

    catalog = _catalog_match(root, paths, fault_catalog)
    label = catalog.get("fault_label") if catalog else f"Process anomaly rooted at {root}"
    if catalog and catalog.get("catalog_score") is not None:
        catalog_strength = min(1.0, float(catalog["catalog_score"]) / 2.0)
        confidence = float(np.clip(0.75 * base_confidence + 0.25 * catalog_strength, 0.0, 1.0))
    else:
        confidence = base_confidence

    abstain_reason = None
    if confidence < confidence_threshold:
        label = None
        abstain_reason = "root_cause_evidence_below_threshold"

    return {
        "fault_label": label,
        "root_cause": root if label is not None else None,
        "confidence": confidence,
        "abstain_reason": abstain_reason,
        "affected_variables": affected,
        "propagation_paths": paths,
        "catalog_match": catalog,
    }
