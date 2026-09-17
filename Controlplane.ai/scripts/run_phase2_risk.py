#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
import yaml

from adaptivefact.benchmark.metrics import latency_percentiles
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.training import (
    evaluate_probabilities,
    group_validation_split,
    labels_from_records,
    train_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Phase 2 Risk Estimator V1")
    parser.add_argument("--config", default="configs/phase2_ragtruth.yaml")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    ds_cfg = raw["dataset"]
    exp_cfg = raw["experiment"]
    feat_cfg = FeatureExtractionConfig(**raw.get("features", {}))

    loader = RAGTruthLoader(ds_cfg["root"], include_low_quality=ds_cfg.get("include_low_quality", False))
    train_records = loader.load_list(split="train")
    test_records = loader.load_list(split="test")

    max_train = exp_cfg.get("max_train_records")
    max_test = exp_cfg.get("max_test_records")
    if max_train:
        train_records = train_records[:max_train]
    if max_test:
        test_records = test_records[:max_test]

    split = group_validation_split(
        train_records,
        validation_size=exp_cfg.get("validation_size", 0.2),
        random_state=exp_cfg.get("random_state", 42),
        candidates=exp_cfg.get("split_candidates", 20),
    )
    fit_records = [train_records[i] for i in split.train_indices]
    val_records = [train_records[i] for i in split.validation_indices]

    extractor = RiskFeatureExtractor(feat_cfg)
    print(f"Extracting {len(extractor.feature_names)} features...")
    X_fit = extractor.transform(fit_records)
    X_val = extractor.transform(val_records)
    X_test = extractor.transform(test_records)
    y_fit = labels_from_records(fit_records)
    y_val = labels_from_records(val_records)
    y_test = labels_from_records(test_records)

    models = train_models(
        X_fit,
        y_fit,
        extractor.feature_names,
        random_state=exp_cfg.get("random_state", 42),
    )

    output_dir = Path(exp_cfg.get("model_dir", "models/phase2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_sample_size = int(exp_cfg.get("latency_sample_size", 500))
    latency_records = test_records[:latency_sample_size]

    results = {}
    for name, bundle in models.items():
        val_prob = bundle.predict_proba(X_val)
        test_prob = bundle.predict_proba(X_test)

        end_to_end_latencies = []
        model_only_latencies = []
        for record in latency_records:
            start = perf_counter()
            row = extractor.transform([record])
            model_start = perf_counter()
            bundle.predict_proba(row)
            model_only_latencies.append((perf_counter() - model_start) * 1000.0)
            end_to_end_latencies.append((perf_counter() - start) * 1000.0)

        results[name] = {
            "validation": evaluate_probabilities(y_val, val_prob),
            "test": evaluate_probabilities(y_test, test_prob),
            "latency_ms": {
                "end_to_end_risk": latency_percentiles(end_to_end_latencies),
                "model_only": latency_percentiles(model_only_latencies),
                "n": len(latency_records),
            },
        }
        bundle.save(output_dir / f"{name}.joblib")

    logreg_f1 = results["logreg"]["validation"]["f1"]
    xgb_f1 = results["xgboost"]["validation"]["f1"]
    min_gain = float(exp_cfg.get("min_f1_gain_for_xgboost", 0.03))
    selected = "xgboost" if xgb_f1 - logreg_f1 >= min_gain else "logreg"
    models[selected].save(output_dir / "selected_model.joblib")

    logreg_classifier = models["logreg"].estimator.named_steps["classifier"]
    feature_importance = {
        "logreg_coefficients": {
            name: float(value)
            for name, value in sorted(
                zip(extractor.feature_names, logreg_classifier.coef_[0]),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )
        },
        "xgboost_importance": {
            name: float(value)
            for name, value in sorted(
                zip(extractor.feature_names, models["xgboost"].estimator.feature_importances_),
                key=lambda pair: pair[1],
                reverse=True,
            )
        },
    }

    manifest = {
        "artifact_provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "config_path": str(Path(args.config)),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "response_data_sha256": hashlib.sha256((Path(ds_cfg["root"]) / "response.jsonl").read_bytes()).hexdigest(),
            "source_data_sha256": hashlib.sha256((Path(ds_cfg["root"]) / "source_info.jsonl").read_bytes()).hexdigest(),
        },
        "selected_model": selected,
        "selection_rule": f"xgboost only if validation F1 gain >= {min_gain:.3f}",
        "feature_names": extractor.feature_names,
        "feature_config": feat_cfg.to_dict(),
        "feature_importance": feature_importance,
        "n_fit": len(fit_records),
        "n_validation": len(val_records),
        "n_test": len(test_records),
        "fit_hallucination_rate": float(y_fit.mean()),
        "validation_hallucination_rate": float(y_val.mean()),
        "test_hallucination_rate": float(y_test.mean()),
        "results": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"\nSaved models and manifest to {output_dir}")


if __name__ == "__main__":
    main()
