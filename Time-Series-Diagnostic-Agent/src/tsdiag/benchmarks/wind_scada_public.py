from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

import numpy as np

from ..contracts import DiagnosticRequest, RunContext
from ..datasets.wind_care import (
    CARE_DOI,
    CARE_SAMPLE_PERIOD_MINUTES,
    care_normal_mask,
    load_care_event,
    load_care_event_info,
    validate_care_layout,
)
from ..detectors.residual_changepoint import (
    DEFAULT_CUSUM_DRIFT,
    DEFAULT_CUSUM_HOLD_SAMPLES,
    DEFAULT_CUSUM_THRESHOLD,
)
from ..evaluation.wind_scada import evaluate_wind_event, summarize_wind_events
from ..evaluation.wind_calibration import (
    DEFAULT_WIND_CALIBRATION_PATH,
    load_wind_calibrated_config,
)
from ..pipeline import diagnose
from .wind_scada import WindBenchmarkEventResult, _aligned_sensor_matrices, _ids, _write_summary_markdown


def run_care_benchmark(
    data_dir: str | Path,
    *,
    event_ids: Iterable[int] | None = None,
    wind_farm: str | None = None,
    n_regimes: int = 4,
    residual_threshold: float | None = None,
    persistence: int | None = None,
    cusum_drift: float | None = None,
    cusum_threshold: float | None = None,
    cusum_hold_samples: int = DEFAULT_CUSUM_HOLD_SAMPLES,
    criticality_threshold: int = 72,
    event_minimum_corroborated_run: int = 6,
    event_minimum_residual_fraction: float | None = None,
    event_minimum_drift_fraction: float = 0.02,
    event_ood_abstain_fraction: float = 0.10,
    artifact_checksums: Mapping[str, str] | None = None,
    calibration_path: str | Path | None = DEFAULT_WIND_CALIBRATION_PATH,
    output_dir: str | Path | None = None,
) -> dict:
    """CARE benchmark through the exact public structured-diagnosis boundary.

    Event labels are loaded for scoring only after ``diagnose`` returns. The
    inference request contains measurements, healthy/reference masks and fixed
    policy configuration, but never ``event.is_anomaly`` or event ground truth.
    """
    root = Path(data_dir)
    layout = validate_care_layout(root)
    if not all(layout.values()) and wind_farm is None:
        raise FileNotFoundError(f"incomplete CARE layout: {layout}")
    if wind_farm is not None and not layout.get(str(wind_farm).upper(), False):
        raise FileNotFoundError(f"CARE Wind Farm {wind_farm} is not available")

    info = load_care_event_info(root, wind_farm=wind_farm)
    selected_ids = _ids(event_ids, info["event_id"].astype(int).tolist())
    calibration = (
        load_wind_calibrated_config(calibration_path)
        if calibration_path is not None
        else None
    )
    calibrated_config = {} if calibration is None else dict(calibration["config"])
    residual_threshold = float(
        residual_threshold if residual_threshold is not None
        else calibrated_config.get("residual_threshold", 3.5)
    )
    persistence = int(
        persistence if persistence is not None
        else calibrated_config.get("persistence", 3)
    )
    cusum_drift = float(
        cusum_drift if cusum_drift is not None
        else calibrated_config.get("cusum_drift", DEFAULT_CUSUM_DRIFT)
    )
    cusum_threshold = float(
        cusum_threshold if cusum_threshold is not None
        else calibrated_config.get("cusum_threshold", DEFAULT_CUSUM_THRESHOLD)
    )
    event_minimum_residual_fraction = float(
        event_minimum_residual_fraction if event_minimum_residual_fraction is not None
        else calibrated_config.get("event_minimum_residual_fraction", 0.02)
    )
    if calibration is not None:
        overlap = sorted(set(selected_ids) & {int(value) for value in calibration["event_ids"]})
        if overlap:
            raise ValueError(
                f"calibration events overlap this benchmark evaluation: {overlap}"
            )
    resolved_checksums = dict(artifact_checksums or {})
    if calibration_path is not None and Path(calibration_path).is_file():
        digest = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
        resolved_checksums["wind_calibration"] = digest
    event_rows: list[WindBenchmarkEventResult] = []
    evaluation_rows = []
    provenance_rows: list[dict] = []
    failures = []
    total_started = perf_counter()

    for event_id in selected_ids:
        try:
            event_data = load_care_event(root, event_id, statistics=("avg",))
            if event_data.event.is_anomaly is None:
                raise ValueError("event_label is unavailable")

            train, prediction, channel_names, train_ts, pred_ts = _aligned_sensor_matrices(event_data)
            healthy_train = care_normal_mask(event_data.train)
            pred_evaluable = (
                np.ones(len(prediction), dtype=bool)
                if event_data.event.wind_farm == "A"
                else care_normal_mask(event_data.prediction)
            )

            request = DiagnosticRequest(
                domain="wind_scada",
                task="condition_monitoring",
                inputs={
                    "train_matrix": train,
                    "prediction_matrix": prediction,
                    "channel_names": channel_names,
                    "train_timestamps": train_ts,
                    "prediction_timestamps": pred_ts,
                    "healthy_train_mask": healthy_train,
                    "evaluable_mask": pred_evaluable,
                    "n_regimes": n_regimes,
                    "residual_threshold": residual_threshold,
                    "persistence": persistence,
                    "cusum_drift": cusum_drift,
                    "cusum_threshold": cusum_threshold,
                    "cusum_hold_samples": cusum_hold_samples,
                    "event_minimum_corroborated_run": event_minimum_corroborated_run,
                    "event_minimum_residual_fraction": event_minimum_residual_fraction,
                    "event_minimum_drift_fraction": event_minimum_drift_fraction,
                    "event_ood_abstain_fraction": event_ood_abstain_fraction,
                },
                policy_ref="wind-care-event-policy:1.0",
                model_refs={"normal_behavior": "wind-nbm-regime-v1"},
                run_context=RunContext(
                    run_id=f"care-{event_id}",
                    source="CARE-v6-benchmark",
                    dataset_id="care-to-compare-v6:zenodo-15846963",
                    protocol_id="care-v6-event-policy-v1",
                    artifact_checksums=resolved_checksums,
                ),
            )

            started = perf_counter()
            result = diagnose(request)
            runtime = perf_counter() - started
            if result.metadata.get("pipeline_version") is None:
                raise RuntimeError("public diagnosis did not record pipeline_version")
            if result.provenance is None:
                raise RuntimeError("public diagnosis did not record run provenance")
            artifacts = result.metadata.get("wind_artifacts", {})
            cp = artifacts.get("residual_changepoint", {})
            physics = artifacts.get("physics_consistency", {})
            fusion = artifacts.get("fusion", {})
            anomaly = artifacts.get("anomaly_detection", {})
            regime_assignment = artifacts.get("regime_assignment", {})

            alarm_mask = np.asarray(result.metadata.get("alarm_mask", []), dtype=bool)
            if alarm_mask.shape[0] != len(prediction):
                raise RuntimeError("public Wind result did not preserve sample-level alarm_mask")
            residual_alarm_mask = np.asarray(anomaly.get("alarm_mask", alarm_mask), dtype=bool)
            drift_alarm_mask = np.asarray(cp.get("alarm_mask", alarm_mask), dtype=bool)
            ood_mask = np.asarray(
                regime_assignment.get("out_of_distribution_mask", np.zeros(len(prediction), dtype=bool)),
                dtype=bool,
            )
            physics_count = len(physics.get("verification_findings", []))

            # Ground truth enters only here, after inference has completed.
            evaluation = evaluate_wind_event(
                event_id=event_id,
                is_anomaly_event=bool(event_data.event.is_anomaly),
                alarm_mask=alarm_mask,
                residual_alarm_mask=residual_alarm_mask,
                drift_alarm_mask=drift_alarm_mask,
                out_of_distribution_mask=ood_mask,
                physics_finding_count=physics_count,
                timestamps=pred_ts,
                event_start=event_data.event.event_start,
                event_end=event_data.event.event_end,
                normal_mask=pred_evaluable,
                criticality_threshold=criticality_threshold,
                sample_period_minutes=CARE_SAMPLE_PERIOD_MINUTES,
                minimum_corroborated_run=event_minimum_corroborated_run,
                minimum_residual_fraction=event_minimum_residual_fraction,
                minimum_drift_fraction=event_minimum_drift_fraction,
                ood_abstain_fraction=event_ood_abstain_fraction,
            )

            public_to_eval = {"diagnose": "fault", "monitor": "monitor", "abstain": "abstain"}
            if public_to_eval[result.decision] != evaluation.event_decision:
                raise RuntimeError(
                    f"public/evaluator decision mismatch: {result.decision} vs {evaluation.event_decision}"
                )
            evaluation_rows.append(evaluation)

            event_rows.append(WindBenchmarkEventResult(
                event_id=event_id,
                wind_farm=event_data.event.wind_farm,
                asset_id=event_data.event.asset_id,
                true_label="anomaly" if event_data.event.is_anomaly else "normal",
                event_detected=evaluation.event_detected,
                event_decision=evaluation.event_decision,
                abstained=evaluation.abstained,
                event_evidence_score=evaluation.evidence_score,
                decision_confidence=evaluation.decision_confidence,
                decision_reason=evaluation.decision_reason,
                legacy_criticality_detected=evaluation.legacy_criticality_detected,
                max_criticality=evaluation.max_criticality,
                alarm_fraction=evaluation.alarm_fraction,
                residual_alarm_fraction=evaluation.residual_alarm_fraction,
                drift_alarm_fraction=evaluation.drift_alarm_fraction,
                corroborated_alarm_fraction=evaluation.corroborated_alarm_fraction,
                longest_corroborated_run=evaluation.longest_corroborated_run,
                out_of_distribution_fraction=evaluation.out_of_distribution_fraction,
                fused_alarm_fraction=float(fusion.get("fused_alarm_fraction", 0.0)),
                event_window_recall=evaluation.event_window_recall,
                first_detection_index=evaluation.first_detection_index,
                time_to_event_end_minutes=evaluation.time_to_event_end_minutes,
                early_warning_lead_time_minutes=evaluation.early_warning_lead_time_minutes,
                affected_channels=list(result.metadata.get("affected_channels", [])),
                change_point_count=len(cp.get("change_points", [])),
                physics_finding_count=physics_count,
                confidence=float(result.confidence),
                tool_call_count=len(result.tool_trace),
                runtime_seconds=float(runtime),
                n_train=int(train.shape[0]),
                n_prediction=int(prediction.shape[0]),
                n_channels=int(train.shape[1]),
            ))
            provenance_rows.append(asdict(result.provenance))
        except Exception as exc:
            failures.append({"event_id": int(event_id), "error": f"{type(exc).__name__}: {exc}"})

    total_runtime = perf_counter() - total_started
    summary = summarize_wind_events(evaluation_rows)
    summary.update({
        "successful_events": len(event_rows),
        "failed_events": len(failures),
        "mean_runtime_seconds": float(np.mean([r.runtime_seconds for r in event_rows])) if event_rows else None,
        "total_runtime_seconds": float(total_runtime),
        "mean_tool_calls": float(np.mean([r.tool_call_count for r in event_rows])) if event_rows else None,
        "total_tool_calls": int(sum(r.tool_call_count for r in event_rows)),
        "mean_residual_alarm_fraction": float(np.mean([r.residual_alarm_fraction for r in event_rows])) if event_rows else None,
        "mean_drift_alarm_fraction": float(np.mean([r.drift_alarm_fraction for r in event_rows])) if event_rows else None,
        "mean_fused_alarm_fraction": float(np.mean([r.fused_alarm_fraction for r in event_rows])) if event_rows else None,
    })

    payload = {
        "source": f"CARE to Compare v6 ({CARE_DOI})",
        "data_dir": str(root),
        "execution_path": "DiagnosticRequest -> diagnose() -> WindScadaPlugin -> shared WorkflowExecutor",
        "configuration": {
            "statistics": ["avg"],
            "n_regimes": int(n_regimes),
            "residual_threshold": float(residual_threshold),
            "persistence": int(persistence),
            "cusum_drift": float(cusum_drift),
            "cusum_threshold": float(cusum_threshold),
            "cusum_hold_samples": int(cusum_hold_samples),
            "criticality_threshold": int(criticality_threshold),
            "event_minimum_corroborated_run": int(event_minimum_corroborated_run),
            "event_minimum_residual_fraction": float(event_minimum_residual_fraction),
            "event_minimum_drift_fraction": float(event_minimum_drift_fraction),
            "event_ood_abstain_fraction": float(event_ood_abstain_fraction),
            "calibration": None if calibration is None else {
                "path": str(calibration_path),
                "development_pool_id": calibration["development_pool_id"],
                "target_event_far": calibration["target_event_far"],
                "achieved_event_far": calibration["achieved_event_far"],
                "achieved_event_far_ci": calibration["achieved_event_far_ci"],
                "n_events_used": calibration["n_events_used"],
            },
        },
        "summary": summary,
        "events": [asdict(row) for row in event_rows],
        "run_provenance": provenance_rows,
        "source_artifact_checksums": resolved_checksums,
        "failures": failures,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "wind_scada_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary_markdown(payload, out / "care_v6_summary.md")
        rows = [asdict(row) for row in event_rows]
        if rows:
            with (out / "wind_scada_event_table.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    row = dict(row)
                    row["affected_channels"] = "|".join(row["affected_channels"])
                    writer.writerow(row)

    return payload
