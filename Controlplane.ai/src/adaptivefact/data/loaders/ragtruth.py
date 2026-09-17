"""
Loader for RAGTruth (Wu et al., 2023) — arXiv:2401.00396,
github.com/ParticleMedia/RAGTruth.

Chosen as the Phase 0 dataset per §50 of the master project prompt:
context is already known, so claims can be checked directly against the
provided context rather than requiring open-domain search, which gives
cleaner ground truth for validating the core adaptive-verification idea
before extending to web-based verification (Phase 9).

--------------------------------------------------------------------------
ACTUAL FILE SCHEMA (confirmed by cloning the repo directly — do not trust
descriptions from secondary sources, they drift):

dataset/response.jsonl   (17,790 lines, one generated response per line)
    id             str   — response id, unique within this file
    source_id      str   — foreign key into source_info.jsonl
    model          str   — generator model, e.g. "gpt-4-0613", "mistral-7B-instruct"
    temperature    float
    labels         list  — possibly EMPTY list; each element is:
                              start, end    int   (char offsets into `response`)
                              text          str   (the hallucinated span itself)
                              label_type    str   one of:
                                  "Evident Conflict"      -> contradicts context
                                  "Subtle Conflict"       -> contradicts context
                                  "Evident Baseless Info" -> not supported by context
                                  "Subtle Baseless Info"  -> not supported by context
                              meta          str   (human-readable explanation)
                              implicit_true bool
                              due_to_null   bool
    split          str   — "train" (15,090) or "test" (2,700)
    quality        str   — "good" (17,617) / "incorrect_refusal" (144) / "truncated" (29)
    response       str   — the generated text itself

dataset/source_info.jsonl  (2,965 lines, one source document per line)
    source_id      str        — join key
    task_type      str        — "Summary" / "QA" / "Data2txt"
    source         str        — origin corpus, e.g. "CNN/DM"
    source_info    str | dict — the retrieved/source content (i.e. our `context`).
                                 IMPORTANT: this field's *type* depends on task_type
                                 (confirmed by inspecting all 2,965 records, not
                                 assumed):
                                   Summary   (943/2965)  -> str, plain article text
                                   QA        (989/2965)  -> dict {"question": str, "passages": str}
                                   Data2txt (1033/2965)  -> dict, a fixed 9-key structured
                                             business record (name/address/city/state/
                                             categories/hours/attributes/business_stars/
                                             review_info)
                                 A loader that assumes `source_info` is always a
                                 plain string will crash (or silently mishandle)
                                 two of the three task types. See _normalize_context().
    prompt         str        — the full instruction given to the generator (i.e. our `query`)

Label-type mapping to our VerificationStatus (see §6.3 of the master
prompt: an unsupported claim must be classified UNKNOWN, never
auto-FALSE) is deliberately NOT a blanket "everything -> CONTRADICTED".
"Conflict" labels are genuine contradictions; "Baseless Info" labels
mean the context simply doesn't address the claim, which is closer to
UNKNOWN/unsupported than to a confirmed factual conflict.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from adaptivefact.data.loaders.base import BaseDatasetLoader
from adaptivefact.data.schema import (
    GenerationMetadata,
    HallucinationSpan,
    ResponseLabel,
    ResponseRecord,
    VerificationStatus,
)

_LABEL_TYPE_TO_STATUS = {
    "Evident Conflict": VerificationStatus.CONTRADICTED,
    "Subtle Conflict": VerificationStatus.CONTRADICTED,
    "Evident Baseless Info": VerificationStatus.UNKNOWN,
    "Subtle Baseless Info": VerificationStatus.UNKNOWN,
}


def _normalize_context(task_type: str | None, source_info: str | dict) -> tuple[str, dict | None]:
    """Returns (context_string, structured_source_or_None).

    See the type-by-task_type breakdown in the module docstring. The
    structured form is preserved (not discarded) for Data2txt records
    specifically because §14/§15 of the master prompt call for explicit
    number/date/entity matching against context — having the original
    field-value pairs (e.g. `hours`, `business_stars`) available
    unflattened will make that far more precise later than re-parsing
    them back out of a JSON-dumped string.
    """
    if isinstance(source_info, str):
        return source_info, None

    if task_type == "QA":
        # source_info == {"question": ..., "passages": ...}; `question`
        # duplicates what's already in the prompt, `passages` is the
        # actual retrieved evidence.
        return source_info.get("passages", ""), None

    # Data2txt (or any other future dict-shaped task_type): no single
    # natural string field, so serialize the whole record but keep the
    # structured original too.
    return json.dumps(source_info, indent=2, ensure_ascii=False), source_info


class RAGTruthLoader(BaseDatasetLoader):
    """Expects `root` to point at the RAGTruth repo's `dataset/` directory
    (i.e. the directory directly containing response.jsonl and
    source_info.jsonl). Use scripts/download_ragtruth.py to fetch it."""

    name = "ragtruth"

    def __init__(self, root: str | Path, include_low_quality: bool = False):
        super().__init__(root)
        self.response_path = self.root / "response.jsonl"
        self.source_info_path = self.root / "source_info.jsonl"
        for p in (self.response_path, self.source_info_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected {p} — is `root` pointing at the RAGTruth "
                    f"`dataset/` directory? (run scripts/download_ragtruth.py)"
                )
        self.include_low_quality = include_low_quality
        self._source_info_cache: dict[str, dict] | None = None

    def _load_source_info(self) -> dict[str, dict]:
        if self._source_info_cache is None:
            cache: dict[str, dict] = {}
            with open(self.source_info_path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    cache[obj["source_id"]] = obj
            self._source_info_cache = cache
        return self._source_info_cache

    def load(self, split: str | None = None) -> Iterator[ResponseRecord]:
        source_info = self._load_source_info()

        with open(self.response_path, encoding="utf-8") as f:
            for line in f:
                resp = json.loads(line)

                if split is not None and resp.get("split") != split:
                    continue
                if not self.include_low_quality and resp.get("quality") != "good":
                    continue

                src = source_info.get(resp["source_id"])
                if src is None:
                    # Shouldn't happen in a clean download, but don't let
                    # one bad join kill the whole load.
                    continue

                spans = [
                    HallucinationSpan(
                        start=lab["start"],
                        end=lab["end"],
                        text=lab["text"],
                        label_type=lab["label_type"],
                        normalized_type=_LABEL_TYPE_TO_STATUS.get(lab["label_type"]),
                        meta=lab.get("meta"),
                    )
                    for lab in resp.get("labels", [])
                ]

                ground_truth_label = (
                    ResponseLabel.HALLUCINATED if spans else ResponseLabel.SUPPORTED
                )

                context_str, structured_source = _normalize_context(
                    src.get("task_type"), src.get("source_info", "")
                )

                metadata: dict = {
                    "source_corpus": src.get("source"),
                    "num_labels": len(spans),
                }
                if structured_source is not None:
                    metadata["structured_source"] = structured_source

                yield ResponseRecord(
                    id=f"ragtruth_{resp['id']}",
                    dataset=self.name,
                    source_id=resp["source_id"],
                    split=resp.get("split"),
                    query=src.get("prompt", ""),
                    context=context_str,
                    generated_response=resp["response"],
                    ground_truth_label=ground_truth_label,
                    ground_truth_spans=spans,
                    generation_metadata=GenerationMetadata(
                        model=resp.get("model"),
                        temperature=resp.get("temperature"),
                        task_type=src.get("task_type"),
                        quality=resp.get("quality"),
                        logits_available=False,  # RAGTruth ships text only, no logits
                    ),
                    metadata=metadata,
                )
