from __future__ import annotations

import abc
from dataclasses import dataclass


class SupportScorer(abc.ABC):
    """Return P(claim is supported by evidence) for each pair.

    These scorers are intentionally support-only. Low support must not be
    interpreted as contradiction without a separate contradiction-capable
    verifier such as NLI.
    """

    name: str

    @abc.abstractmethod
    def score(self, documents: list[str], claims: list[str]) -> list[float]:
        raise NotImplementedError


@dataclass
class MiniCheckSupportScorer(SupportScorer):
    model_name: str = "flan-t5-large"
    cache_dir: str = "./ckpts/minicheck"

    name: str = "minicheck"

    def __post_init__(self) -> None:
        self._delegate = None

    def _load(self):
        if self._delegate is None:
            try:
                from minicheck.minicheck import MiniCheck
            except ImportError as exc:
                raise ImportError(
                    "MiniCheck backend is optional. Install with: "
                    "pip install 'minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main'"
                ) from exc
            self._delegate = MiniCheck(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._delegate

    def score(self, documents: list[str], claims: list[str]) -> list[float]:
        if len(documents) != len(claims):
            raise ValueError("documents and claims must have the same length")
        if not documents:
            return []
        scorer = self._load()
        _, raw_prob, _, _ = scorer.score(docs=documents, claims=claims)
        return [float(value) for value in raw_prob]


@dataclass
class AlignScoreSupportScorer(SupportScorer):
    checkpoint_path: str
    model: str = "roberta-base"
    device: str = "cpu"
    batch_size: int = 16
    evaluation_mode: str = "nli_sp"

    name: str = "alignscore"

    def __post_init__(self) -> None:
        self._delegate = None

    def _load(self):
        if self._delegate is None:
            try:
                from alignscore import AlignScore
            except ImportError as exc:
                raise ImportError(
                    "AlignScore backend is optional. Install the official AlignScore package first."
                ) from exc
            self._delegate = AlignScore(
                model=self.model,
                batch_size=self.batch_size,
                device=self.device,
                ckpt_path=self.checkpoint_path,
                evaluation_mode=self.evaluation_mode,
            )
        return self._delegate

    def score(self, documents: list[str], claims: list[str]) -> list[float]:
        if len(documents) != len(claims):
            raise ValueError("documents and claims must have the same length")
        if not documents:
            return []
        values = self._load().score(contexts=documents, claims=claims)
        return [float(value) for value in values]
