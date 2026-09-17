import json

from tsdiag.evaluation.cross_domain import build_cross_domain_report, write_cross_domain_report
from tsdiag.integrations import diagnose_tool


def _battery_result(run_id: str):
    voltage = [[3.70, 3.69] for _ in range(12)]
    temperature = [[30.0, 30.1] for _ in range(12)]
    return diagnose_tool({
        "domain": "battery",
        "task": "anomaly_localization",
        "inputs": {
            "cell_voltage": voltage,
            "cell_temperature": temperature,
            "cell_ids": ["a", "b"],
            "timestamps": list(range(12)),
        },
        "run_context": {
            "run_id": run_id,
            "source": "unit-test",
            "dataset_id": "fixture-battery",
            "protocol_id": "fixture-v1",
        },
    })


def test_cross_domain_report_preserves_versioned_provenance(tmp_path):
    report = build_cross_domain_report([_battery_result("run-1")], report_id="fixture")

    assert report["report_version"] == "1.0"
    assert report["summary"]["run_count"] == 1
    assert report["summary"]["domains_present"] == ["battery"]
    assert report["summary"]["complete_six_domain_coverage"] is False
    run = report["runs"][0]
    assert run["pipeline_version"] == "1.1.0"
    assert run["dataset_id"] == "fixture-battery"
    assert run["protocol_id"] == "fixture-v1"
    assert len(run["input_hash"]) == 64
    assert len(run["result_hash"]) == 64

    paths = write_cross_domain_report(report, tmp_path)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["report_id"] == "fixture"
    assert "| battery |" in paths["markdown"].read_text(encoding="utf-8")
