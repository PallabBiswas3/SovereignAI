#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from adaptivefact.benchmark.metrics import calibration_metrics
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.risk.calibration import ProbabilityCalibrator, fit_candidate_calibrators
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.model import RiskModelBundle
from adaptivefact.risk.threshold_optimizer import optimize_routing_thresholds, routing_statistics
from adaptivefact.risk.training import group_validation_split, labels_from_records


def _read_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _split_phase2_validation(records, *, random_state: int, candidates: int):
    first = group_validation_split(
        records,
        validation_size=0.50,
        random_state=random_state + 101,
        candidates=candidates,
    )
    calibration_fit = [records[i] for i in first.train_indices]
    remaining = [records[i] for i in first.validation_indices]

    second = group_validation_split(
        remaining,
        validation_size=0.50,
        random_state=random_state + 202,
        candidates=candidates,
    )
    calibration_select = [remaining[i] for i in second.train_indices]
    routing_tune = [remaining[i] for i in second.validation_indices]
    return calibration_fit, calibration_select, routing_tune


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: calibrate the selected risk model and optimize routing thresholds")
    parser.add_argument("--config", default="configs/phase3_ragtruth.yaml")
    args = parser.parse_args()

    cfg = _read_yaml(args.config)
    phase2_cfg = _read_yaml(cfg["phase2"]["config"])
    phase2_manifest = json.loads(Path(cfg["phase2"]["manifest"]).read_text())

    ds_cfg = phase2_cfg["dataset"]
    exp2 = phase2_cfg["experiment"]
    exp3 = cfg["experiment"]

    loader = RAGTruthLoader(ds_cfg["root"], include_low_quality=ds_cfg.get("include_low_quality", False))
    train_records = loader.load_list(split="train")
    test_records = loader.load_list(split="test")

    if exp2.get("max_train_records"):
        train_records = train_records[: exp2["max_train_records"]]
    if exp3.get("max_test_records"):
        test_records = test_records[: exp3["max_test_records"]]

    phase2_split = group_validation_split(
        train_records,
        validation_size=exp2.get("validation_size", 0.2),
        random_state=exp2.get("random_state", 42),
        candidates=exp2.get("split_candidates", 20),
    )
    phase2_validation = [train_records[i] for i in phase2_split.validation_indices]

    calibration_fit, calibration_select, routing_tune = _split_phase2_validation(
        phase2_validation,
        random_state=exp3.get("random_state", 42),
        candidates=exp3.get("split_candidates", 30),
    )

    feature_config = FeatureExtractionConfig(**phase2_manifest["feature_config"])
    extractor = RiskFeatureExtractor(feature_config)
    model = RiskModelBundle.load(cfg["phase2"]["selected_model"])

    def raw_probabilities(records):
        return model.predict_proba(extractor.transform(records))

    p_cal_fit = raw_probabilities(calibration_fit)
    p_cal_select = raw_probabilities(calibration_select)
    p_route = raw_probabilities(routing_tune)
    p_test = raw_probabilities(test_records)

    y_cal_fit = labels_from_records(calibration_fit)
    y_cal_select = labels_from_records(calibration_select)
    y_route = labels_from_records(routing_tune)
    y_test = labels_from_records(test_records)

    calibrators = fit_candidate_calibrators(
        p_cal_fit,
        y_cal_fit,
        random_state=exp3.get("random_state", 42),
    )

    selection_results = {}
    for name, calibrator in calibrators.items():
        calibrated = calibrator.predict(p_cal_select)
        selection_results[name] = calibration_metrics(
            y_cal_select.tolist(),
            calibrated.tolist(),
            n_bins=exp3.get("ece_bins", 10),
        )

    selected_name = min(
        selection_results,
        key=lambda name: (
            selection_results[name]["brier_score"],
            selection_results[name]["ece"],
        ),
    )
    selected_calibrator = calibrators[selected_name]

    calibrated_route = selected_calibrator.predict(p_route)
    threshold_cfg = cfg["thresholds"]
    thresholds = optimize_routing_thresholds(
        y_route,
        calibrated_route,
        max_fast_hallucination_rate=threshold_cfg.get("max_fast_hallucination_rate", 0.03),
        min_hallucination_recall_outside_fast=threshold_cfg.get("min_hallucination_recall_outside_fast", 0.98),
        min_fast_count=threshold_cfg.get("min_fast_count", 25),
        min_agentic_hallucination_rate=threshold_cfg.get("min_agentic_hallucination_rate", 0.75),
        min_agentic_count=threshold_cfg.get("min_agentic_count", 25),
        grid_points=threshold_cfg.get("grid_points", 300),
    )

    calibrated_test = selected_calibrator.predict(p_test)
    test_stats = routing_statistics(y_test, calibrated_test, thresholds.tau1, thresholds.tau2)

    output_dir = Path(exp3.get("output_dir", "models/phase3"))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_calibrator.save(output_dir / "selected_calibrator.joblib")

    threshold_payload = {
        "tau1": thresholds.tau1,
        "tau2": thresholds.tau2,
        "constraints": threshold_cfg,
        "routing_tune": thresholds.to_dict(),
    }
    (output_dir / "thresholds.json").write_text(json.dumps(threshold_payload, indent=2))

    manifest = {
        "selected_risk_model": phase2_manifest["selected_model"],
        "selected_calibrator": selected_name,
        "calibrator_selection_rule": "lowest Brier score on calibration-selection split; ECE breaks ties",
        "subset_sizes": {
            "phase2_validation": len(phase2_validation),
            "calibration_fit": len(calibration_fit),
            "calibration_select": len(calibration_select),
            "routing_tune": len(routing_tune),
            "official_test": len(test_records),
        },
        "hallucination_rates": {
            "calibration_fit": float(y_cal_fit.mean()),
            "calibration_select": float(y_cal_select.mean()),
            "routing_tune": float(y_route.mean()),
            "official_test": float(y_test.mean()),
        },
        "calibrator_selection_metrics": selection_results,
        "thresholds": threshold_payload,
        "official_test": {
            "raw_calibration": calibration_metrics(y_test.tolist(), p_test.tolist(), n_bins=exp3.get("ece_bins", 10)),
            "calibrated": calibration_metrics(y_test.tolist(), calibrated_test.tolist(), n_bins=exp3.get("ece_bins", 10)),
            "routing": test_stats,
        },
        "note": "tau1/tau2 are Phase-3 provisional routing thresholds. Re-optimize later when real lightweight/agentic cost and accuracy are available.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"\nSaved Phase 3 artifacts to {output_dir}")


if __name__ == "__main__":
    main()
