from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.evaluation.runner import EvaluationRunner


ROOT = Path(__file__).resolve().parents[1]


def test_offline_evaluation_produces_measurable_metrics() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        metrics = EvaluationRunner(session, ROOT / "config" / "models.yaml", ROOT / "knowledge_base").run()
    assert metrics["routing"]["accuracy"] >= 2 / 3
    assert metrics["benchmark"]["case_counts"] == {"routing": 20, "rag": 20, "governance": 20, "agent": 10}
    assert metrics["routing"]["case_count"] == 20
    assert metrics["routing"]["macro_f1"] > 0
    assert metrics["routing"]["confusion_matrix"]
    assert metrics["rag"]["retrieval_precision_at_1"] is not None
    assert metrics["rag"]["retrieval_recall_at_3"] is not None
    assert metrics["rag"]["mrr"] is not None
    assert metrics["rag"]["citation_correctness"] is not None
    assert metrics["rag"]["refusal_accuracy"] is not None
    assert set(metrics["rag"]["comparison"]) == {"hash", "semantic"}
    assert metrics["governance"]["pii"]["precision"] == 1.0
    assert metrics["governance"]["prompt_injection"]["recall"] == 1.0
    assert metrics["governance"]["pii"]["confusion_matrix"]
    assert metrics["agent"]["case_count"] == 10
    assert metrics["agent"]["workflow_accuracy"] == 1.0
    assert metrics["agent"]["tool_selection_recall"] == 1.0
    assert metrics["system"]["process_ram_mb"] > 0
