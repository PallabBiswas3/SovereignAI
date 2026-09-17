import json

import numpy as np

from adaptivefact.data.loaders.halueval import HaluEvalLoader
from adaptivefact.data.schema import ResponseLabel, ResponseRecord, VerificationRoute
from adaptivefact.risk.calibration import ProbabilityCalibrator
from adaptivefact.routing.router import AdaptiveRouter


class FakeExtractor:
    feature_names = ["risk"]

    def transform(self, records):
        return np.asarray([[float(r.metadata["risk"])] for r in records])


class FakeModel:
    feature_names = ["risk"]

    def predict_proba(self, X):
        return X[:, 0]


def test_router_three_routes():
    router = AdaptiveRouter(
        FakeModel(),
        FakeExtractor(),
        ProbabilityCalibrator("identity"),
        tau1=0.2,
        tau2=0.7,
    )
    def record(risk):
        return ResponseRecord(
            id=str(risk), dataset="x", query="q", generated_response="a",
            ground_truth_label=ResponseLabel.SUPPORTED, metadata={"risk": risk}
        )

    assert router.route(record(0.1)).route == VerificationRoute.FAST_ACCEPT
    assert router.route(record(0.4)).route == VerificationRoute.LIGHTWEIGHT
    assert router.route(record(0.9)).route == VerificationRoute.AGENTIC


def test_halueval_qa_loader_creates_supported_and_hallucinated_pair(tmp_path):
    rows = [{
        "knowledge": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "right_answer": "Paris.",
        "hallucinated_answer": "Berlin."
    }]
    (tmp_path / "qa_data.json").write_text(json.dumps(rows), encoding="utf-8")
    records = HaluEvalLoader(tmp_path, task="qa").load_list()
    assert len(records) == 2
    assert records[0].ground_truth_label == ResponseLabel.SUPPORTED
    assert records[1].ground_truth_label == ResponseLabel.HALLUCINATED
    assert records[0].context == rows[0]["knowledge"]
