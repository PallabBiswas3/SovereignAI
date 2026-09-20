from __future__ import annotations

import abc
from dataclasses import dataclass


class SupportScorer(abc.ABC):
    """Return P(claim is supported by evidence) for each document/claim pair.

    These scorers are support/alignment models only. A low score is not treated
    as contradiction; contradiction remains the responsibility of the NLI path.
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
                    "MiniCheck backend is optional. Install the official package with: "
                    "pip install 'minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main'"
                ) from exc
            self._delegate = MiniCheck(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._delegate

    def score(self, documents: list[str], claims: list[str]) -> list[float]:
        if len(documents) != len(claims):
            raise ValueError("documents and claims must have the same length")
        if not documents:
            return []
        _, raw_prob, _, _ = self._load().score(docs=documents, claims=claims)
        return [float(value) for value in raw_prob]


@dataclass
class AlignScoreSupportScorer(SupportScorer):
    """Adapter for the current official ``yuh-zha/Align`` implementation.

    The class name is retained because the experiment compares an AlignScore-
    style text-alignment verifier with MiniCheck and DeBERTa. The maintained
    official API imports ``Align`` from the ``align`` package and calls the
    instance directly with ``contexts`` and ``claims``.
    """

    checkpoint_path: str
    model: str = "roberta-base"
    device: str = "cpu"
    batch_size: int = 16

    name: str = "alignscore"

    def __post_init__(self) -> None:
        self._delegate = None

    def _load(self):
        if self._delegate is None:
            try:
                from align import Align
            except ImportError as exc:
                raise ImportError(
                    "Align backend is optional. Clone/install the official "
                    "https://github.com/yuh-zha/Align package before enabling it."
                ) from exc
            self._delegate = Align(
                model=self.model,
                batch_size=self.batch_size,
                device=self.device,
                ckpt_path=self.checkpoint_path,
            )
        return self._delegate

    def score(self, documents: list[str], claims: list[str]) -> list[float]:
        if len(documents) != len(claims):
            raise ValueError("documents and claims must have the same length")
        if not documents:
            return []
        values = self._load()(contexts=documents, claims=claims)
        return [float(value) for value in values]
