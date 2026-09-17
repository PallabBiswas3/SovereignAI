from __future__ import annotations

import pytest

from adaptivefact.benchmark.metrics import remediation_metrics, selective_triage_metrics
from adaptivefact.benchmark.sampling import sample_records, sampling_summary
from adaptivefact.data.schema import ResponseLabel, ResponseRecord


def test_selective_metrics_keep_abstentions_separate_from_confirmed_errors():
    result = selective_triage_metrics(
        [1, 0, 1, 0],
        ["hallucinated", "mixed", "supported", "supported"],
        [0.9, 0.7, 0.2, 0.1],
    )

    assert result["coverage"] == 0.75
    assert result["abstention_rate"] == 0.25
    assert result["confirmed_detection"]["precision"] == 1.0
    assert result["confirmed_detection"]["end_to_end_recall"] == 0.5
    assert result["safety"]["unsafe_release_count"] == 1
    assert result["safety"]["unsafe_release_rate"] == 0.5
    assert result["abstentions"] == {"count": 1, "hallucinated": 0, "supported": 1}


def test_selective_metrics_validate_lengths_and_labels():
    with pytest.raises(ValueError):
        selective_triage_metrics([1], [])
    with pytest.raises(ValueError):
        selective_triage_metrics([1], ["blocked"])


def _records(supported: int, hallucinated: int) -> list[ResponseRecord]:
    records = []
    for index in range(supported):
        records.append(
            ResponseRecord(
                id=f"supported-{index}",
                dataset="synthetic",
                source_id=f"source-s-{index}",
                query="q",
                generated_response="r",
                ground_truth_label=ResponseLabel.SUPPORTED,
            )
        )
    for index in range(hallucinated):
        records.append(
            ResponseRecord(
                id=f"hallucinated-{index}",
                dataset="synthetic",
                source_id=f"source-h-{index}",
                query="q",
                generated_response="r",
                ground_truth_label=ResponseLabel.HALLUCINATED,
            )
        )
    return records


def test_stratified_sampling_is_reproducible_and_preserves_prevalence():
    records = _records(supported=8, hallucinated=2)
    first = sample_records(records, 5, method="stratified_random", seed=7)
    second = sample_records(records, 5, method="stratified_random", seed=7)

    assert [record.id for record in first] == [record.id for record in second]
    assert sampling_summary(first)["label_counts"] == {
        "supported": 4,
        "hallucinated": 1,
    }
    assert sampling_summary(first)["unique_source_ids"] == 5


def test_sampling_rejects_unknown_method():
    with pytest.raises(ValueError):
        sample_records(_records(2, 1), 2, method="not-a-method")


def test_remediation_metrics_separate_automation_from_useful_answers():
    result = remediation_metrics(
        [0, 0, 1, 1, 0],
        ["release", "corrected_release", "block", "safe_abstain", "human_review"],
    )

    assert result["automation_rate"] == 0.8
    assert result["human_review_rate"] == 0.2
    assert result["useful_answer_rate"] == 0.4
    assert result["hallucination_containment_rate"] == 1.0
    assert result["supported_answer_retention_rate"] == pytest.approx(2 / 3)


def test_remediation_metrics_reject_unknown_actions():
    with pytest.raises(ValueError):
        remediation_metrics([0], ["pretend_supported"])
