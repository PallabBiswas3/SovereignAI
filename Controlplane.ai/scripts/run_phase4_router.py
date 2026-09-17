#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from adaptivefact.data.loaders.halueval import HaluEvalLoader
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.risk.calibration import ProbabilityCalibrator
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.model import RiskModelBundle
from adaptivefact.routing.evaluation import evaluate_router
from adaptivefact.routing.policy import ThresholdPolicy
from adaptivefact.routing.router import AdaptiveRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: evaluate the adaptive router on RAGTruth and HaluEval")
    parser.add_argument("--config", default="configs/phase4_router.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    phase2_manifest = json.loads(Path(cfg["artifacts"]["phase2_manifest"]).read_text())

    model = RiskModelBundle.load(cfg["artifacts"]["risk_model"])
    calibrator = ProbabilityCalibrator.load(cfg["artifacts"]["calibrator"])
    policy = ThresholdPolicy.from_json(cfg["artifacts"]["thresholds"])
    extractor = RiskFeatureExtractor(FeatureExtractionConfig(**phase2_manifest["feature_config"]))

    router = AdaptiveRouter(
        model,
        extractor,
        calibrator,
        tau1=policy.tau1,
        tau2=policy.tau2,
    )

    rag_cfg = cfg["ragtruth"]
    rag_loader = RAGTruthLoader(rag_cfg["root"], include_low_quality=rag_cfg.get("include_low_quality", False))
    rag_records = rag_loader.load_list(split="test")
    if rag_cfg.get("max_records"):
        rag_records = rag_records[: rag_cfg["max_records"]]

    results = {"ragtruth_test": evaluate_router(router, rag_records)}

    halu_cfg = cfg.get("halueval", {})
    if halu_cfg.get("enabled", True):
        halu_loader = HaluEvalLoader(halu_cfg["root"], task=halu_cfg.get("task", "qa"))
        halu_records = halu_loader.load_list(max_source_records=halu_cfg.get("max_source_records"))
        results[f"halueval_{halu_cfg.get('task', 'qa')}"] = evaluate_router(router, halu_records)

    output_dir = Path(cfg["experiment"].get("output_dir", "results/phase4"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "router_evaluation.json"
    output_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nSaved Phase 4 evaluation to {output_path}")


if __name__ == "__main__":
    main()
