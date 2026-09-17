#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.extraction.claim_extractor import ClaimExtractionConfig
from adaptivefact.verification.deterministic import DeterministicVerifierConfig
from adaptivefact.verification.phase5 import Phase5Pipeline, summarize_phase5


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: claim extraction + deterministic verification")
    parser.add_argument("--config", default="configs/phase5_ragtruth.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})

    loader = RAGTruthLoader(
        data_cfg.get("root", "data/raw/ragtruth/dataset"),
        include_low_quality=bool(data_cfg.get("include_low_quality", False)),
    )
    records = list(loader.load(split=exp_cfg.get("split", "test")))
    max_records = exp_cfg.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]

    extraction_cfg = ClaimExtractionConfig(**cfg.get("extraction", {}))
    verifier_cfg = DeterministicVerifierConfig(**cfg.get("verification", {}))
    pipeline = Phase5Pipeline(extraction_cfg, verifier_cfg)

    processed = []
    timings = []
    for idx, record in enumerate(records, start=1):
        enriched, timing = pipeline.process(record)
        processed.append(enriched)
        timings.append(timing)
        if idx % 250 == 0:
            print(f"Processed {idx}/{len(records)} responses...")

    summary = summarize_phase5(processed, timings)
    summary["experiment"] = exp_cfg.get("name", "phase5")
    summary["config"] = {
        "extraction": cfg.get("extraction", {}),
        "verification": cfg.get("verification", {}),
    }

    output_cfg = cfg.get("output", {})
    out_dir = Path(output_cfg.get("directory", "results/phase5"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / output_cfg.get("summary_file", "phase5_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sample_limit = int(exp_cfg.get("save_enriched_max_records", 100))
    enriched_path = out_dir / output_cfg.get("enriched_file", "enriched_records_sample.jsonl")
    with enriched_path.open("w", encoding="utf-8") as f:
        for record in processed[:sample_limit]:
            f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved Phase 5 summary to {summary_path}")
    print(f"Saved enriched sample to {enriched_path}")


if __name__ == "__main__":
    main()
