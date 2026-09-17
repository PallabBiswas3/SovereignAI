import numpy as np

from adaptivefact.data.schema import ResponseLabel, ResponseRecord
from adaptivefact.risk.features import FeatureExtractionConfig, RiskFeatureExtractor
from adaptivefact.risk.training import group_validation_split, labels_from_records, train_models


def make_record(i, answer, label, source_id=None):
    return ResponseRecord(
        id=f"r{i}",
        dataset="synthetic",
        source_id=source_id or f"s{i}",
        query="What was the company's 2025 revenue?",
        context="In 2025, Acme Corp reported revenue of 18 crore.",
        generated_response=answer,
        ground_truth_label=label,
    )


def test_feature_support_ratios_detect_number_mismatch():
    extractor = RiskFeatureExtractor(FeatureExtractionConfig())
    supported = extractor.extract(
        make_record(1, "Acme Corp reported revenue of 18 crore in 2025.", ResponseLabel.SUPPORTED)
    )
    bad = extractor.extract(
        make_record(2, "Acme Corp reported revenue of 28 crore in 2025.", ResponseLabel.HALLUCINATED)
    )
    assert supported["number_support_ratio"] > bad["number_support_ratio"]
    assert supported["date_support_ratio"] == 1.0


def test_feature_matrix_is_finite():
    extractor = RiskFeatureExtractor(FeatureExtractionConfig())
    records = [
        make_record(1, "Acme Corp reported revenue of 18 crore in 2025.", ResponseLabel.SUPPORTED),
        make_record(2, "Acme Corp reported revenue of 28 crore in 2025.", ResponseLabel.HALLUCINATED),
    ]
    X = extractor.transform(records)
    assert X.shape == (2, len(extractor.feature_names))
    assert np.isfinite(X).all()


def test_group_validation_split_has_no_source_overlap():
    records = []
    for source in range(12):
        for j in range(2):
            label = ResponseLabel.HALLUCINATED if (source + j) % 2 else ResponseLabel.SUPPORTED
            records.append(make_record(source * 2 + j, "Some answer 18 crore.", label, source_id=f"s{source}"))

    split = group_validation_split(records, validation_size=0.25, candidates=10)
    train_sources = {records[i].source_id for i in split.train_indices}
    val_sources = {records[i].source_id for i in split.validation_indices}
    assert train_sources.isdisjoint(val_sources)


def test_models_train_and_predict_probabilities():
    extractor = RiskFeatureExtractor(FeatureExtractionConfig())
    records = []
    for i in range(24):
        if i % 2:
            answer = f"Acme Corp reported revenue of {30 + i} crore in 2025."
            label = ResponseLabel.HALLUCINATED
        else:
            answer = "Acme Corp reported revenue of 18 crore in 2025."
            label = ResponseLabel.SUPPORTED
        records.append(make_record(i, answer, label))

    X = extractor.transform(records)
    y = labels_from_records(records)
    models = train_models(X, y, extractor.feature_names)
    for bundle in models.values():
        prob = bundle.predict_proba(X)
        assert prob.shape == (len(records),)
        assert ((prob >= 0) & (prob <= 1)).all()
