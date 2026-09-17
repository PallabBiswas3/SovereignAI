from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


class NLIScorer(abc.ABC):
    @abc.abstractmethod
    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        raise NotImplementedError


class TransformersNLIScorer(NLIScorer):
    """Sequence-classification NLI scorer with robust label-name handling."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        fallback_label_order: tuple[str, str, str] = ("contradiction", "entailment", "neutral"),
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformersNLIScorer requires transformers and torch. "
                "Install the Phase 1 dependencies from requirements.txt."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        normalized = {}
        for idx, label in id2label.items():
            if "entail" in label:
                normalized[idx] = "entailment"
            elif "contrad" in label:
                normalized[idx] = "contradiction"
            elif "neutral" in label:
                normalized[idx] = "neutral"

        if len(normalized) != 3 and self.model.config.num_labels == 3:
            normalized = {i: name for i, name in enumerate(fallback_label_order)}
        self.label_map = normalized

    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        if not premises:
            return []

        outputs: list[NLIScores] = []
        for start in range(0, len(premises), self.batch_size):
            p_batch = premises[start : start + self.batch_size]
            h_batch = hypotheses[start : start + self.batch_size]
            tokens = self.tokenizer(
                p_batch,
                h_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with self.torch.inference_mode():
                logits = self.model(**tokens).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                values = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                for idx, prob in enumerate(row):
                    label = self.label_map.get(idx)
                    if label is not None:
                        values[label] = float(prob)
                outputs.append(NLIScores(**values))

        return outputs
