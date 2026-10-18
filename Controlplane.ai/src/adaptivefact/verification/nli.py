from __future__ import annotations

import abc
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np


def profile_model_identity(model_name: str) -> str:
    """Keep local model paths (which may contain personal data) out of profiles."""
    if Path(model_name).is_absolute() or "\\" in model_name or Path(model_name).exists():
        return f"local-model-sha256:{sha256(model_name.encode()).hexdigest()[:16]}"
    return model_name


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
        local_files_only: bool = False,
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
        self.profile_model = profile_model_identity(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)
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
        scores, _ = self.score_profiled(premises, hypotheses)
        return scores

    def score_profiled(self, premises: list[str], hypotheses: list[str]) -> tuple[list[NLIScores], dict]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        if not premises:
            return [], {"requested_pairs": 0, "unique_pairs": 0, "forward_passes": 0,
                        "batch_sizes": [], "padded_lengths": [], "tokens_processed": 0,
                        "tokenization_ms": 0.0, "padding_ms": 0.0, "forward_ms": 0.0, "total_ms": 0.0}

        total_started = perf_counter()
        # Share only byte-identical pair computations within this call. Source
        # identity/authority and each original evidence position remain separate
        # in Phase6; no scores or source text are cached across requests.
        unique = list(dict.fromkeys(zip(premises, hypotheses)))
        positions = {pair: index for index, pair in enumerate(unique)}
        tokenize_started = perf_counter()
        encoded = self.tokenizer(
            [pair[0] for pair in unique], [pair[1] for pair in unique],
            padding=False, truncation=True, max_length=self.max_length,
        )
        tokenization_ms = (perf_counter() - tokenize_started) * 1000.0
        order = sorted(range(len(unique)), key=lambda index: len(encoded["input_ids"][index]))
        outputs: dict[int, NLIScores] = {}
        padding_ms = 0.0
        forward_ms = 0.0
        batch_sizes: list[int] = []
        padded_lengths: list[int] = []
        start = 0
        while start < len(order):
            end = start + 1
            shortest = len(encoded['input_ids'][order[start]])
            # A small response can still mix short tool statements and long
            # manuals. Split before padding would double the shortest pair;
            # this changes batch shape only, never tokens or evidence coverage.
            while (end < len(order) and end - start < self.batch_size
                   and len(encoded['input_ids'][order[end]]) <= 2 * shortest):
                end += 1
            indices = order[start:end]
            start = end
            pad_started = perf_counter()
            tokens = self.tokenizer.pad(
                {key: [value[index] for index in indices] for key, value in encoded.items()},
                padding=True, return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            padding_ms += (perf_counter() - pad_started) * 1000.0
            batch_sizes.append(len(indices))
            padded_lengths.append(int(tokens["input_ids"].shape[1]))
            forward_started = perf_counter()
            with self.torch.inference_mode():
                logits = self.model(**tokens).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
            forward_ms += (perf_counter() - forward_started) * 1000.0

            for index, row in zip(indices, probs):
                values = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                for idx, prob in enumerate(row):
                    label = self.label_map.get(idx)
                    if label is not None:
                        values[label] = float(prob)
                outputs[index] = NLIScores(**values)

        # Return independent score objects in the caller's original order.
        scores = [NLIScores(**vars(outputs[positions[pair]])) for pair in zip(premises, hypotheses)]
        return scores, {
            "requested_pairs": len(premises), "unique_pairs": len(unique),
            "forward_passes": len(batch_sizes), "batch_sizes": batch_sizes,
            "padded_lengths": padded_lengths,
            "tokens_processed": sum(size * length for size, length in zip(batch_sizes, padded_lengths)),
            "tokenization_ms": tokenization_ms, "padding_ms": padding_ms,
            "forward_ms": forward_ms, "total_ms": (perf_counter() - total_started) * 1000.0,
            "device": str(self.device), "cpu_threads": self.torch.get_num_threads(),
            "model": self.profile_model,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "runtime": f"torch {self.torch.__version__}",
        }
