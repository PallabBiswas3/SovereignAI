from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import Base
from app.evidence.context import ContextCompiler
from app.evidence.models import Calculation, Claim, EvidenceBundle, Measurement, Rule, RuleSource, SupportStatus
from app.evidence.verification import VerificationEngine
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.reranking import LocalCrossEncoderReranker
from app.rag.retrieval import LocalRetriever, RetrievedChunk
from app.resources.scheduler import ResourceScheduler


class Batch2EvaluationRunner:
    def __init__(self) -> None:
        benchmark_path = Path(__file__).with_name("batch2_benchmarks.json")
        self.benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    def run(self) -> dict[str, object]:
        with TemporaryDirectory() as directory:
            engine = create_engine("sqlite:///:memory:")
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                embeddings = LocalHashEmbeddingProvider(128)
                self._ingest_fixture(session, embeddings, Path(directory))
                settings = Settings(
                    embedding_provider="hash",
                    hybrid_dense_top_k=20,
                    hybrid_sparse_top_k=20,
                    hybrid_fusion_candidate_limit=20,
                    hybrid_final_context_k=5,
                    reranker_enabled=True,
                )
                dense = LocalRetriever(session, embeddings, acl_scope="internal")
                hybrid = HybridRetriever(session, embeddings, settings=settings)
                reranker = LocalCrossEncoderReranker(
                    settings.reranker_model,
                    local_files_only=True,
                    scheduler=ResourceScheduler(),
                )
                hybrid_rerank = HybridRetriever(session, embeddings, reranker=reranker, settings=settings)
                dense_result = self._measure(lambda query: dense.search(query, 5), "dense")
                hybrid_result = self._measure(lambda query: hybrid.search(query, 5), "hybrid", hybrid)
                rerank_result = self._measure(
                    lambda query: hybrid_rerank.search(query, 5), "hybrid_rerank", hybrid_rerank
                )
                context = self._context_evaluation(hybrid.search("PU-102 vibration limit", 20))
                verification = self._verification_evaluation()
        reranker_available = rerank_result.get("reranker_available") is True
        improvement = (
            round(float(rerank_result["mrr"]) - float(hybrid_result["mrr"]), 4)
            if reranker_available else None
        )
        return {
            "benchmark_version": self.benchmark["version"],
            "case_count": len(self.benchmark["retrieval"]),
            "comparison": {
                "dense": dense_result,
                "hybrid": hybrid_result,
                "hybrid_rerank": rerank_result,
            },
            "reranker_improvement_mrr": improvement,
            "context_compiler": context,
            "claim_verification": verification,
            "note": "Small deterministic offline benchmark; values are measurements for this fixture, not production quality claims.",
        }

    @staticmethod
    def _ingest_fixture(session: Session, embeddings, root: Path) -> None:
        fixtures = [
            ("SOP-MNT-42.md", "[PAGE 37]\n## Section 7.4\nSOP-MNT-42 applies to Pump-102 / PU-102. Maximum normal vibration shall not exceed 5.5 mm/s RMS.\n[PAGE 39]\n## Section 7.6\nPump-102 pressure range is 4.8 to 5.5 bar.", {"classification": "internal", "revision": "Rev 3"}),
            ("SOP-MNT-42-Rev2.md", "[PAGE 37]\n## Section 7.4\nSOP-MNT-42 Rev 2 says Pump-102 vibration shall not exceed 6.0 mm/s RMS.", {"classification": "internal", "revision": "Rev 2"}),
            ("history.md", "[PAGE 1]\n## Section H1\nPump-102 historical readings include vibration and bearing temperature.", {"classification": "internal"}),
            ("unrelated.md", "[PAGE 1]\n## Section L1\nXV-101 lubrication is monthly.", {"classification": "internal"}),
        ]
        for filename, content, metadata in fixtures:
            path = root / filename
            path.write_text(content, encoding="utf-8")
            KnowledgeIngestionService(session, embeddings).ingest(path, metadata)

    def _measure(
        self,
        search: Callable[[str], list[RetrievedChunk]],
        name: str,
        hybrid: HybridRetriever | None = None,
    ) -> dict[str, object]:
        cases: list[dict[str, object]] = []
        latencies: list[float] = []
        reciprocal: list[float] = []
        for case in self.benchmark["retrieval"]:
            started = monotonic()
            results = search(case["query"])
            latencies.append((monotonic() - started) * 1000)
            expected = case["expected_section"]
            rank = next((index for index, item in enumerate(results, start=1)
                         if expected is not None and item.source.get("section") == expected), 0)
            if case["answerable"]:
                reciprocal.append(1 / rank if rank else 0.0)
            refused = not self._question_supported(case["query"], results)
            cases.append({
                "id": case["id"], "category": case["category"], "rank": rank,
                "top_section": results[0].source.get("section") if results else None,
                "refused": refused, "expected_refusal": not case["answerable"],
            })
        answerable = [item for item in cases if not item["expected_refusal"]]
        unanswerable = [item for item in cases if item["expected_refusal"]]
        telemetry = hybrid.last_telemetry.to_dict() if hybrid else {}
        return {
            "strategy": name,
            "recall_at_1": round(sum(item["rank"] == 1 for item in answerable) / len(answerable), 4),
            "recall_at_3": round(sum(0 < item["rank"] <= 3 for item in answerable) / len(answerable), 4),
            "recall_at_5": round(sum(0 < item["rank"] <= 5 for item in answerable) / len(answerable), 4),
            "mrr": round(sum(reciprocal) / len(reciprocal), 4),
            "citation_precision": round(sum(item["rank"] == 1 for item in answerable) / len(answerable), 4),
            "unsupported_question_refusal_rate": round(sum(item["refused"] for item in unanswerable) / max(1, len(unanswerable)), 4),
            "mean_total_latency_ms": round(sum(latencies) / len(latencies), 6),
            "dense_latency_ms": telemetry.get("dense_duration_ms"),
            "sparse_latency_ms": telemetry.get("sparse_duration_ms"),
            "fusion_latency_ms": telemetry.get("fusion_duration_ms"),
            "reranker_latency_ms": telemetry.get("reranker_duration_ms"),
            "reranker_available": telemetry.get("reranker_available"),
            "warning": telemetry.get("warning"),
            "cases": cases,
        }

    @staticmethod
    def _question_supported(query: str, results: list[RetrievedChunk]) -> bool:
        if not results:
            return False
        stop = {"what", "which", "does", "this", "that", "current", "pump", "section", "require"}
        query_terms = {item for item in re.findall(r"[a-z]{4,}", query.lower()) if item not in stop}
        evidence_terms = set(re.findall(r"[a-z]{4,}", results[0].text.lower()))
        dense = float(results[0].scores.get("dense") or results[0].score)
        return dense >= 0.45 or len(query_terms & evidence_terms) / max(1, len(query_terms)) >= 0.34

    @staticmethod
    def _context_evaluation(candidates: list[RetrievedChunk]) -> dict[str, object]:
        expanded = (candidates * 5)[:20]
        compiled = ContextCompiler(Settings(context_max_evidence_chunks=5)).compile(
            task="PU-102 vibration limit", evidence=expanded,
            selected_model="general", context_window=8192, execution_mode="STANDARD",
        )
        required_retained = any(re.search(r"5\.5|6\.0", source.text) for source in compiled.evidence)
        return {
            **compiled.metrics.model_dump(mode="json"),
            "required_fact_preserved": required_retained,
        }

    @staticmethod
    def _verification_evaluation() -> dict[str, object]:
        measurement = Measurement(id="M1", metric="pressure", original_value=500,
                                  original_unit="kPa", source_id="E1")
        rule = Rule(id="R1", metric="pressure", operator="between", lower_bound=4.8,
                    upper_bound=5.5, unit="bar", rule_type="normal_range",
                    source=RuleSource(source_id="E2"))
        cases = []
        for actual, expected in ((True, "SUPPORTED"), (False, "UNSUPPORTED")):
            calculation = Calculation(id="CALC1", expression="4.8 <= 5 <= 5.5",
                                      inputs=["M1", "R1"], result=actual)
            claim = Claim(id="CL1", text="Pump pressure is within range.",
                          claim_type="engineering_finding", evidence_ids=["M1", "R1"],
                          calculation_ids=["CALC1"], support_status=SupportStatus.supported)
            result = VerificationEngine().verify(EvidenceBundle(
                measurements=[measurement.model_copy(deep=True)], rules=[rule.model_copy(deep=True)],
                calculations=[calculation], claims=[claim],
            ))
            observed = result.claims[0].support_status.value
            cases.append({"expected": expected, "observed": observed, "correct": observed == expected})
        return {"case_count": len(cases), "deterministic_correctness": sum(item["correct"] for item in cases) / len(cases), "cases": cases}
