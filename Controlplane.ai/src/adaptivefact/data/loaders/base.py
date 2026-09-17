"""
Abstract dataset loader interface.

Per §49: "the framework should work with several datasets rather than
one hard-coded benchmark." Every loader (RAGTruth now; HaluEval,
FActScore, custom sets later per the dataset roadmap) implements this
same interface so the benchmark harness and every pipeline stage never
need to know which dataset a record came from.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path

from adaptivefact.data.schema import ResponseRecord


class BaseDatasetLoader(abc.ABC):
    """Subclass this for every new dataset."""

    #: short machine-readable name, e.g. "ragtruth". Stamped onto every
    #: ResponseRecord.dataset field produced by this loader.
    name: str

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root {self.root} does not exist. "
                f"Did you run the corresponding scripts/download_*.py?"
            )

    @abc.abstractmethod
    def load(self, split: str | None = None) -> Iterator[ResponseRecord]:
        """Yield ResponseRecord instances.

        Args:
            split: if given, only yield records from that split
                (e.g. "train", "test"). If None, yield everything.
        """
        raise NotImplementedError

    def load_list(self, split: str | None = None) -> list[ResponseRecord]:
        """Convenience wrapper — materializes the generator into a list.
        Fine for RAGTruth-scale data (~18k records); avoid for anything
        that doesn't comfortably fit in memory."""
        return list(self.load(split=split))
