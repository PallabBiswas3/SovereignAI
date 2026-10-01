"""Generate synthetic, non-confidential demo assets locally."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "workspace" / "uploads"
KNOWLEDGE = ROOT / "knowledge_base"


def font(size: int):
    for candidate in [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def scanned_inspection_pdf() -> None:
    lines = [
        "PUMP-102 INSPECTION REPORT", "Inspection date: 15 August 2026", "Department: Mechanical Maintenance",
        "", "OBSERVED READINGS", "Overall vibration: 8.2 mm/s RMS", "Bearing temperature: 86 C",
        "Discharge pressure: 4.4 bar", "", "OBSERVATIONS", "Audible bearing noise was present.",
        "No casing crack or irreparable damage was observed.", "First abnormal vibration finding after overhaul.",
    ]
    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    y = 140
    for index, line in enumerate(lines):
        draw.text((145, y), line, fill="#111111", font=font(54 if index == 0 else 38))
        y += 92 if index == 0 else 72
    image.save(UPLOADS / "Pump_Inspection_Report.pdf", "PDF", resolution=150.0)


def inspection_markdown() -> None:
    """Provide the text fixture used by the artifact and approval demos."""
    (UPLOADS / "Pump_Inspection_Report.md").write_text(
        "# Pump-102 Inspection Report\n\n"
        "Inspection date: 2026-08-15\n"
        "Department: Mechanical Maintenance\n"
        "Equipment: Pump-102\n\n"
        "Observed readings:\n\n"
        "- Overall vibration at drive-end bearing: 8.2 mm/s RMS\n"
        "- Bearing temperature: 86 C\n"
        "- Discharge pressure: 4.4 bar\n\n"
        "Observations: Audible bearing noise was present. No casing crack or "
        "irreparable casing damage was observed. This is the first abnormal "
        "vibration finding after the last overhaul.\n\n"
        "Inspector recommendation: Perform alignment and bearing checks. "
        "Engineering disposition is pending comparison with the current "
        "maintenance SOP.\n",
        encoding="utf-8",
    )


def sop_pdf() -> None:
    pages = [
        ("7.4 Vibration Limits", "Overall vibration shall not exceed 6.0 mm/s RMS. Above 9.0 mm/s requires removal from service."),
        ("7.5 Bearing Temperature", "Normal bearing temperature is up to 80 C. Above 90 C requires shutdown and inspection."),
        ("7.6 Discharge Pressure", "Pump-102 normal discharge pressure is 4.8 to 5.5 bar."),
        ("8.2 Replacement Decision", "Replace only for irreparable casing damage, repeated critical vibration, or repair cost above 60 percent."),
    ]
    output = canvas.Canvas(str(KNOWLEDGE / "Maintenance_SOP.pdf"), pagesize=A4)
    for page, (heading, body) in enumerate(pages, start=1):
        output.setFont("Helvetica-Bold", 16); output.drawString(65, 780, "Maintenance SOP 12 - Rotating Equipment")
        output.setFont("Helvetica-Bold", 13); output.drawString(65, 730, f"Section {heading}")
        output.setFont("Helvetica", 11); output.drawString(65, 695, body)
        output.drawString(65, 50, f"Page {page}"); output.showPage()
    output.save()


def sensor_csv() -> None:
    rows = [
        ("2026-08-15T08:00:00", 72.0, 3.2), ("2026-08-15T09:00:00", 73.1, 3.4),
        ("2026-08-15T10:00:00", 74.2, 3.5), ("2026-08-15T11:00:00", 86.0, 8.2),
        ("2026-08-15T12:00:00", 75.0, 3.7), ("2026-08-15T13:00:00", 91.5, 9.6),
        ("2026-08-15T14:00:00", 74.8, 3.6), ("2026-08-15T15:00:00", 75.2, 3.8),
    ]
    with (UPLOADS / "pump_sensor_readings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["timestamp", "temperature_c", "vibration_mm_s"]); writer.writerows(rows)


def generate_demo_data() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    inspection_markdown()
    scanned_inspection_pdf()
    sop_pdf()
    sensor_csv()


if __name__ == "__main__":
    generate_demo_data()
    print("Generated synthetic inspection Markdown/PDF, SOP PDF, and sensor CSV.")
