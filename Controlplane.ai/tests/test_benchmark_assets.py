import json
from pathlib import Path

import yaml


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_candidate_benchmark_has_requested_split_sizes_and_blinded_test():
    development = load_jsonl("data/controlplane/benchmark_v1/development_candidates.jsonl")
    validation = load_jsonl("data/controlplane/benchmark_v1/validation_candidates.jsonl")
    test_inputs = load_jsonl("data/controlplane/benchmark_v1/test_inputs_blinded.jsonl")

    assert len(development) == 200
    assert len(validation) == 100
    assert len(test_inputs) == 400
    assert all(item["annotation_status"] == "candidate_unreviewed" for item in development)
    assert all(item["annotation_status"] == "candidate_unreviewed" for item in validation)
    assert all("expected_action" not in item for item in test_inputs)
    assert all(not any(key.startswith("gold_") for key in item) for item in test_inputs)


def test_model_comparison_matrix_contains_all_requested_routes():
    config = yaml.safe_load(Path("configs/model_comparison.yaml").read_text(encoding="utf-8"))
    ids = {item["id"] for item in config["models"]}
    assert {
        "deberta_small",
        "deberta_large",
        "terra_evidence_judge",
        "sol_evidence_judge",
        "hybrid_deberta_terra",
    } <= ids
