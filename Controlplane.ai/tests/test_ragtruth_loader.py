"""
Phase 0 smoke tests. These run against the REAL downloaded RAGTruth
data (not mocks) — run scripts/download_ragtruth.py first, or these
will be skipped with a clear message.

What this proves:
  1. The loader parses the actual on-disk format correctly.
  2. Every yielded record satisfies the ResponseRecord schema (pydantic
     validation happens automatically on construction, so a malformed
     record would raise before it ever reached these assertions).
  3. The label_type -> VerificationStatus mapping is doing something
     sane (conflicts vs baseless info aren't collapsed together).
  4. The benchmark harness can run trivial components over the loaded
     data end-to-end and produce metrics + a comparison table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivefact.benchmark.experiment import ExperimentConfig, run_experiment
from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.loaders.ragtruth import RAGTruthLoader
from adaptivefact.data.schema import ResponseLabel, VerificationStatus

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "ragtruth"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "response.jsonl").exists(),
    reason="RAGTruth not downloaded — run scripts/download_ragtruth.py first",
)


@pytest.fixture(scope="module")
def loader() -> RAGTruthLoader:
    return RAGTruthLoader(DATA_ROOT)


@pytest.fixture(scope="module")
def test_split_records(loader: RAGTruthLoader) -> list:
    return loader.load_list(split="test")


def test_loader_yields_records(test_split_records):
    assert len(test_split_records) > 0
    # RAGTruth test split is documented as 2,700 responses; after
    # filtering to quality == "good" it should be close to but at most that.
    assert len(test_split_records) <= 2700


def test_records_are_valid_schema(test_split_records):
    for record in test_split_records[:50]:
        assert record.dataset == "ragtruth"
        assert record.query  # prompt should be non-empty
        assert record.generated_response
        assert record.ground_truth_label in (ResponseLabel.SUPPORTED, ResponseLabel.HALLUCINATED)


def test_quality_filter_excludes_bad_quality(loader: RAGTruthLoader):
    strict = loader.load_list(split="test")
    loose = RAGTruthLoader(DATA_ROOT, include_low_quality=True).load_list(split="test")
    assert len(loose) >= len(strict)


def test_hallucinated_records_have_spans(test_split_records):
    hallucinated = [r for r in test_split_records if r.ground_truth_label == ResponseLabel.HALLUCINATED]
    assert len(hallucinated) > 0
    for record in hallucinated[:20]:
        assert len(record.ground_truth_spans) > 0
        for span in record.ground_truth_spans:
            assert span.end > span.start
            assert span.text in record.generated_response


def test_label_type_mapping_distinguishes_conflict_from_baseless(test_split_records):
    """Conflict labels and Baseless-Info labels must map to DIFFERENT
    VerificationStatus values (CONTRADICTED vs UNKNOWN) per §6.3 of the
    master prompt — an unsupported claim is not automatically false."""
    seen_statuses = set()
    for record in test_split_records:
        for span in record.ground_truth_spans:
            if span.normalized_type is not None:
                seen_statuses.add(span.normalized_type)
    assert VerificationStatus.CONTRADICTED in seen_statuses
    assert VerificationStatus.UNKNOWN in seen_statuses


def test_clean_records_have_no_spans(test_split_records):
    clean = [r for r in test_split_records if r.ground_truth_label == ResponseLabel.SUPPORTED]
    assert len(clean) > 0
    assert all(len(r.ground_truth_spans) == 0 for r in clean[:50])


# ---------------------------------------------------------------------------
# Benchmark harness smoke test — proves the runner works end to end using
# two trivial baseline components. Neither is a real system component;
# they exist purely to validate the harness plumbing before Phase 2+
# components exist.
# ---------------------------------------------------------------------------


class AlwaysSupportedBaseline(Component):
    """Predicts 'not hallucinated' for everything. Establishes the floor:
    any real component should beat this on recall."""

    label = "Always-supported baseline"

    def run(self, record) -> ComponentResult:
        return ComponentResult(prediction=0, confidence=0.0, latency_ms=0.01)


class SpanCountBaseline(Component):
    """Cheats by reading ground-truth span count directly — this is NOT a
    valid system component (it uses the label!) and exists solely to
    prove the harness can distinguish a perfect predictor from a trivial
    one in the comparison table."""

    label = "Oracle (uses ground truth — sanity check only)"

    def run(self, record) -> ComponentResult:
        has_spans = len(record.ground_truth_spans) > 0
        return ComponentResult(prediction=int(has_spans), confidence=float(has_spans), latency_ms=0.01)


def test_benchmark_harness_end_to_end(loader: RAGTruthLoader, tmp_path):
    config = ExperimentConfig(
        name="phase0_smoke_test",
        split="test",
        max_records=200,  # keep the test fast
        output_dir=str(tmp_path),
    )
    components = {
        "always_supported": AlwaysSupportedBaseline(),
        "oracle": SpanCountBaseline(),
    }

    result = run_experiment(config, loader, components)

    # The oracle must get perfect recall (it's reading the label directly).
    assert result["components"]["oracle"]["metrics"]["recall"] == pytest.approx(1.0)
    # The always-supported baseline must get zero recall (it never predicts positive).
    assert result["components"]["always_supported"]["metrics"]["recall"] == pytest.approx(0.0)

    # Result file was actually written.
    written_files = list(tmp_path.glob("phase0_smoke_test_*.json"))
    assert len(written_files) == 1
