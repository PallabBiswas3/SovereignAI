from __future__ import annotations

import json
from pathlib import Path

from adaptivefact.data.schema import (
    GenerationMetadata,
    ResponseLabel,
    ResponseRecord,
)


class HaluEvalLoader:
    """Load HaluEval into the project's ResponseRecord schema."""

    def __init__(self, root: str | Path, *, task: str = "qa") -> None:
        self.root = Path(root)
        self.task = task

    def load_list(
        self,
        *,
        max_source_records: int | None = None,
    ) -> list[ResponseRecord]:

        if self.task == "qa":
            return self._load_qa(
                max_source_records=max_source_records
            )

        if self.task == "general":
            return self._load_general(
                max_source_records=max_source_records
            )

        raise ValueError(
            "Phase 4 HaluEval loader currently supports "
            "task='qa' or task='general'"
        )

    def _load_json(self, filename: str) -> list[dict]:
        path = self.root / filename

        if not path.exists():
            raise FileNotFoundError(
                f"HaluEval file not found: {path}"
            )

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return []

        # First try regular JSON.
        try:
            data = json.loads(text)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                return [data]

            raise ValueError(
                f"Unexpected JSON format in {path}"
            )

        # HaluEval files may be JSONL:
        # one JSON object per line.
        except json.JSONDecodeError:
            rows = []

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(
                        json.loads(line)
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line "
                        f"{line_number} of {path}"
                    ) from exc

            return rows

    def _load_qa(
        self,
        *,
        max_source_records: int | None,
    ) -> list[ResponseRecord]:

        rows = self._load_json(
            "qa_data.json"
        )

        if max_source_records is not None:
            rows = rows[:max_source_records]

        records: list[ResponseRecord] = []

        for idx, row in enumerate(rows):
            source_id = f"halueval-qa-{idx}"

            common = {
                "dataset": "halueval_qa",
                "source_id": source_id,
                "split": "external_test",
                "query": str(
                    row.get("question", "")
                ),
                "context": str(
                    row.get("knowledge", "")
                ),
                "generation_metadata":
                    GenerationMetadata(
                        task_type="QA"
                    ),
                "metadata": {
                    "halueval_source_index": idx
                },
            }

            right_answer = str(
                row.get(
                    "right_answer",
                    ""
                )
            )

            hallucinated_answer = str(
                row.get(
                    "hallucinated_answer",
                    ""
                )
            )

            records.append(
                ResponseRecord(
                    id=f"{source_id}-supported",
                    generated_response=right_answer,
                    ground_truth_label=
                        ResponseLabel.SUPPORTED,
                    **common,
                )
            )

            records.append(
                ResponseRecord(
                    id=f"{source_id}-hallucinated",
                    generated_response=
                        hallucinated_answer,
                    ground_truth_label=
                        ResponseLabel.HALLUCINATED,
                    **common,
                )
            )

        return records

    def _load_general(
        self,
        *,
        max_source_records: int | None,
    ) -> list[ResponseRecord]:

        rows = self._load_json(
            "general_data.json"
        )

        if max_source_records is not None:
            rows = rows[:max_source_records]

        records: list[ResponseRecord] = []

        for idx, row in enumerate(rows):
            label = str(
                row.get(
                    "hallucination_label",
                    ""
                )
            ).strip().lower()

            if label == "yes":
                ground_truth = (
                    ResponseLabel.HALLUCINATED
                )
            else:
                ground_truth = (
                    ResponseLabel.SUPPORTED
                )

            records.append(
                ResponseRecord(
                    id=f"halueval-general-{idx}",
                    dataset="halueval_general",
                    source_id=
                        f"halueval-general-{idx}",
                    split="external_test",
                    query=str(
                        row.get(
                            "user_query",
                            ""
                        )
                    ),
                    context=None,
                    generated_response=str(
                        row.get(
                            "chatgpt_response",
                            ""
                        )
                    ),
                    ground_truth_label=
                        ground_truth,
                    generation_metadata=
                        GenerationMetadata(
                            model="ChatGPT",
                            task_type="General",
                        ),
                    metadata={
                        "halueval_source_index": idx
                    },
                )
            )

        return records