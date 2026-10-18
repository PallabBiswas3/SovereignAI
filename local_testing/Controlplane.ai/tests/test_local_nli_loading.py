"""ControlPlane must not contact a model registry during an industrial check."""

from adaptivefact.verification.nli import NLIScores, TransformersNLIScorer
from controlplane.verification import LazyTransformersNLIScorer, build_adaptive_verification_from_env


def test_controlplane_nli_is_local_only_by_default(monkeypatch) -> None:
    monkeypatch.setenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", "1")
    monkeypatch.delenv("CONTROLPLANE_NLI_LOCAL_ONLY", raising=False)

    service = build_adaptive_verification_from_env()

    assert service is not None
    assert service.phase6 is not None
    assert service.phase6.nli.local_files_only is True


def test_controlplane_nli_download_requires_explicit_opt_out(monkeypatch) -> None:
    monkeypatch.setenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", "1")
    monkeypatch.setenv("CONTROLPLANE_NLI_LOCAL_ONLY", "0")

    service = build_adaptive_verification_from_env()

    assert service is not None
    assert service.phase6 is not None
    assert service.phase6.nli.local_files_only is False


def test_lazy_nli_forwards_local_only_to_transformers_loader(monkeypatch) -> None:
    called: dict[str, object] = {}

    class FakeNLI:
        def __init__(self, model_name: str, *, device=None, local_files_only=False, batch_size=16) -> None:
            called.update(model_name=model_name, device=device, local_files_only=local_files_only, batch_size=batch_size)

        def score(self, premises, hypotheses):
            return [NLIScores(entailment=0.9, contradiction=0.05, neutral=0.05)]

        def score_profiled(self, premises, hypotheses):
            return self.score(premises, hypotheses), {}

    monkeypatch.setattr("controlplane.verification.TransformersNLIScorer", FakeNLI)
    scorer = LazyTransformersNLIScorer("cached-model", local_files_only=True)

    assert scorer.score(["evidence"], ["claim"])[0].entailment == 0.9
    assert called == {"model_name": "cached-model", "device": None, "local_files_only": True, "batch_size": 16}


def test_nli_batch_size_is_bounded_and_configurable(monkeypatch) -> None:
    monkeypatch.setenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", "1")
    monkeypatch.setenv("CONTROLPLANE_NLI_BATCH_SIZE", "32")
    service = build_adaptive_verification_from_env()
    assert service is not None and service.phase6 is not None
    assert service.phase6.nli.batch_size == 32

    monkeypatch.setenv("CONTROLPLANE_NLI_BATCH_SIZE", "65")
    import pytest
    with pytest.raises(ValueError, match="between 1 and 64"):
        build_adaptive_verification_from_env()


def test_transformers_loader_applies_local_only_to_tokenizer_and_model(monkeypatch) -> None:
    import transformers

    called: list[tuple[str, bool]] = []

    class FakeModel:
        config = type("Config", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}, "num_labels": 3})()

        def eval(self):
            return self

        def to(self, device):
            return self

    def tokenizer_load(model_name, *, local_files_only):
        called.append(("tokenizer", local_files_only))
        return object()

    def model_load(model_name, *, local_files_only):
        called.append(("model", local_files_only))
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model_load)

    TransformersNLIScorer("cached-model", device="cpu", local_files_only=True)

    assert called == [("tokenizer", True), ("model", True)]
