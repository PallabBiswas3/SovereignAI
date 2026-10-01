from pathlib import Path

from demo import generate_demo_data


def test_fresh_demo_generation_supplies_documented_inspection_fixture(
    tmp_path: Path, monkeypatch,
) -> None:
    uploads = tmp_path / "uploads"
    knowledge = tmp_path / "knowledge"
    monkeypatch.setattr(generate_demo_data, "UPLOADS", uploads)
    monkeypatch.setattr(generate_demo_data, "KNOWLEDGE", knowledge)

    generate_demo_data.generate_demo_data()

    report = uploads / "Pump_Inspection_Report.md"
    assert report.is_file()
    content = report.read_text(encoding="utf-8")
    assert "Pump-102" in content
    assert "8.2 mm/s RMS" in content
    assert "86 C" in content
    assert "4.4 bar" in content
    assert (uploads / "Pump_Inspection_Report.pdf").is_file()
    assert (uploads / "pump_sensor_readings.csv").is_file()
    assert (knowledge / "Maintenance_SOP.pdf").is_file()

    generate_demo_data.generate_demo_data()
    assert report.read_text(encoding="utf-8") == content
