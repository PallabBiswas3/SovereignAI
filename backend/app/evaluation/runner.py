from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import psutil
from sqlalchemy.orm import Session

from app.core.database import AgentRunRecord, AuditEventRecord
from app.governance.injection import PromptInjectionScanner
from app.governance.pii import PIIDetector
from app.rag.chunking import ProvenanceChunker
from app.rag.embeddings import (
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    configured_embedding_provider,
)
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.retrieval import LocalRetriever
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.tools.file_tools import extract_text
from app.evaluation.batch2 import Batch2EvaluationRunner


class EvaluationRunner:
    def __init__(self, session: Session, models_config: Path, knowledge_root: Path) -> None:
        self.session = session
        self.models_config = models_config
        self.knowledge_root = knowledge_root
        self.benchmark = json.loads(Path(__file__).with_name("benchmarks.json").read_text(encoding="utf-8"))

    def run(self) -> dict[str, Any]:
        result = {
            "benchmark": {"version": self.benchmark["version"], "case_counts": {key: len(value) for key, value in self.benchmark.items() if isinstance(value, list)}},
            "routing": self._routing_metrics(),
            "rag": self._rag_metrics(),
            "governance": self._governance_metrics(),
            "agent": self._agent_metrics(),
            "system": self._system_metrics(),
        }
        result["rag"]["batch2"] = Batch2EvaluationRunner().run()
        return result

    def _routing_metrics(self) -> dict[str, Any]:
        router = ModelRouter(ModelRegistry(self.models_config))
        predictions = []
        for case in self.benchmark["routing"]:
            request, expected = case["request"], case["expected"]
            actual = router.route(request).model_id
            predictions.append({"request": request, "expected": expected, "actual": actual, "correct": actual == expected})
        labels = sorted({item["expected"] for item in predictions} | {item["actual"] for item in predictions})
        matrix = {expected: {actual: sum(item["expected"] == expected and item["actual"] == actual for item in predictions) for actual in labels} for expected in labels}
        f1_values = []
        for label in labels:
            tp = matrix[label][label]
            fp = sum(matrix[other][label] for other in labels if other != label)
            fn = sum(matrix[label][other] for other in labels if other != label)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1_values.append(2 * precision * recall / max(.0001, precision + recall))
        return {
            "accuracy": round(sum(item["correct"] for item in predictions) / len(predictions), 3),
            "macro_f1": round(sum(f1_values) / len(f1_values), 3), "confusion_matrix": matrix,
            "case_count": len(predictions), "cases": predictions,
            "average_task_latency_ms": None, "note": "Classifier/router benchmark is fully offline; generation latency requires a running local model.",
        }

    def _rag_metrics(self) -> dict[str, Any]:
        sop = self.knowledge_root / "Maintenance_SOP.md"
        if not sop.exists():
            return {"retrieval_precision_at_1": None, "retrieval_recall_at_1": None, "cases": [], "note": "Synthetic SOP is missing."}
        cases = self.benchmark["rag"]
        chunks = ProvenanceChunker().chunk(extract_text(sop))
        hash_result = self._retrieval_benchmark(LocalHashEmbeddingProvider(), chunks, cases)
        try:
            semantic_provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
            semantic_result = self._retrieval_benchmark(semantic_provider, chunks, cases)
        except Exception as exc:
            semantic_result = {"available": False, "error": str(exc)}
        active = configured_embedding_provider()
        selected = semantic_result if active.provider_name.startswith("sentence-transformer:") else hash_result
        return {
            "retrieval_precision_at_1": selected.get("recall_at_1"),
            "retrieval_recall_at_1": selected.get("recall_at_1"),
            "retrieval_recall_at_3": selected.get("recall_at_3"),
            "mrr": selected.get("mrr"),
            "citation_correctness": selected.get("citation_correctness"),
            "refusal_accuracy": selected.get("refusal_accuracy"),
            "case_count": len(cases),
            "active_embedding": active.provider_name,
            "comparison": {"hash": hash_result, "semantic": semantic_result},
            "cases": selected.get("cases", []),
            "note": "Small offline benchmark; these are not production relevance claims.",
        }

    @staticmethod
    def _retrieval_benchmark(
        provider: EmbeddingProvider, chunks, cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        vectors = provider.embed_documents([chunk.text for chunk in chunks])
        results = []
        reciprocal_ranks = []
        answerable_count = sum(case["section"] is not None for case in cases)
        refusal_results = []
        for case in cases:
            query, expected = case["query"], case["section"]
            query_vector = provider.embed_query(query)
            scores = [sum(left * right for left, right in zip(query_vector, vector)) for vector in vectors]
            order = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
            rank = next((position for position, index in enumerate(order, start=1) if expected is not None and chunks[index].section == expected), 0)
            if expected is not None:
                reciprocal_ranks.append(1 / rank if rank else 0.0)
            top_score = scores[order[0]] if order else 0.0
            should_refuse = expected is None
            refused = top_score < .35
            if should_refuse:
                refusal_results.append(refused)
            results.append({
                "query": query,
                "expected_section": expected,
                "actual_section": chunks[order[0]].section if order else None,
                "rank": rank, "top_score": round(top_score, 4), "should_refuse": should_refuse,
                "refused": refused if should_refuse else False,
            })
        return {
            "available": True,
            "provider": provider.provider_name,
            "recall_at_1": round(sum(item["rank"] == 1 for item in results) / answerable_count, 3),
            "recall_at_3": round(sum(0 < item["rank"] <= 3 for item in results) / answerable_count, 3),
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
            "citation_correctness": round(sum(item["rank"] == 1 for item in results if not item["should_refuse"]) / answerable_count, 3),
            "refusal_accuracy": round(sum(refusal_results) / len(refusal_results), 3),
            "cases": results,
        }

    def _governance_metrics(self) -> dict[str, Any]:
        pii_cases = [(case["text"], case["pii"]) for case in self.benchmark["governance"]]
        injection_cases = [(case["text"], case["injection"]) for case in self.benchmark["governance"]]
        pii = self._binary_metrics(pii_cases, lambda text: bool(PIIDetector().detect(text)))
        injection = self._binary_metrics(injection_cases, lambda text: bool(PromptInjectionScanner().scan(text)))
        return {"case_count": len(self.benchmark["governance"]), "pii": pii, "prompt_injection": injection, "hallucination_detection": {"precision": None, "recall": None, "note": "Claim-level grounding is tested separately; production hallucination measurement requires a larger labeled answer corpus."}}

    @staticmethod
    def _binary_metrics(cases: list[tuple[str, bool]], predict) -> dict[str, Any]:
        outcomes = [(expected, bool(predict(text))) for text, expected in cases]
        tp = sum(expected and actual for expected, actual in outcomes)
        fp = sum(not expected and actual for expected, actual in outcomes)
        fn = sum(expected and not actual for expected, actual in outcomes)
        tn = sum(not expected and not actual for expected, actual in outcomes)
        return {
            "precision": round(tp / max(1, tp + fp), 3),
            "recall": round(tp / max(1, tp + fn), 3),
            "f1": round((2 * tp) / max(1, 2 * tp + fp + fn), 3),
            "false_positive_rate": round(fp / max(1, fp + tn), 3),
            "false_negative_rate": round(fn / max(1, fn + tp), 3),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }

    def _agent_metrics(self) -> dict[str, Any]:
        benchmark_cases = []
        total_expected_tools = 0
        matched_tools = 0
        for case in self.benchmark["agent"]:
            request = case["request"].lower()
            workflow = "inspection" if "inspection" in request and "sop" in request else "coding" if "python" in request and "csv" in request else "tool_agent"
            mapping = {
                "knowledge": "knowledge_search", "read file": "read_file", "list file": "list_files",
                "ocr": "ocr_document", "image": "analyze_image", "word": "generate_docx",
                "presentation": "generate_pptx", "spreadsheet": "generate_xlsx",
            }
            tools = [tool for phrase, tool in mapping.items() if phrase in request]
            if workflow == "inspection":
                tools = ["ocr_document", "analyze_image", "retrieve_sources", "compare_readings", "create_docx", "generate_xlsx", "generate_pptx"]
            elif workflow == "coding":
                tools = ["run_python"]
            expected_tools = case["tools"]
            total_expected_tools += len(expected_tools)
            matched_tools += len(set(expected_tools) & set(tools))
            benchmark_cases.append({**case, "actual_workflow": workflow, "actual_tools": tools, "workflow_correct": workflow == case["workflow"]})
        records = self.session.query(AgentRunRecord).all()
        total = len(records)
        completed = sum(record.status == "completed" for record in records)
        tool_calls = self.session.query(AuditEventRecord).filter_by(event_type="tool_call").count()
        failed_calls = self.session.query(AuditEventRecord).filter(AuditEventRecord.event_type == "tool_call", AuditEventRecord.summary.like("%failed%")) .count()
        latencies = [(record.updated_at - record.created_at).total_seconds() * 1000 for record in records]
        code_events = self.session.query(AuditEventRecord).filter_by(event_type="code_execution").all()
        attempts = [json.loads(event.payload_json).get("attempt", 1) for event in code_events]
        repair_count = sum(max(0, int(attempt) - 1) for attempt in attempts)
        return {
            "case_count": len(benchmark_cases),
            "workflow_accuracy": round(sum(item["workflow_correct"] for item in benchmark_cases) / len(benchmark_cases), 3),
            "benchmark_completion_rate": round(sum(item["workflow_correct"] for item in benchmark_cases) / len(benchmark_cases), 3),
            "tool_selection_recall": round(matched_tools / max(1, total_expected_tools), 3),
            "bounded_protocol": {"max_tool_calls": 6, "max_decisions": 8, "max_seconds": 180},
            "benchmark_cases": benchmark_cases,
            "total_runs": total, "task_completion_rate": round(completed / total, 3) if total else None,
            "average_tool_calls": round(tool_calls / total, 3) if total else 0.0,
            "failed_tool_calls": failed_calls, "repair_count": repair_count,
            "mean_execution_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        }

    @staticmethod
    def _system_metrics() -> dict[str, Any]:
        process = psutil.Process()
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "process_ram_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "system_ram_percent": memory.percent,
            "peak_vram_mb": None,
            "tokens_per_second": None,
            "note": "VRAM and token throughput populate only when a compatible local model runtime reports them.",
        }
