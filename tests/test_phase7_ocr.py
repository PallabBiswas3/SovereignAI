from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfWriter

from app.multimodal.ocr import LocalOCRService


def test_scanned_pdf_detection(tmp_path: Path) -> None:
    path = tmp_path / "blank_scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    assert LocalOCRService.is_scanned_pdf(path)


def test_local_ocr_preserves_confidence_and_boxes(tmp_path: Path) -> None:
    image_path = tmp_path / "inspection.png"
    image = Image.new("RGB", (1000, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 40), "Pump-102 Inspection", fill="black", font_size=42)
    draw.text((35, 125), "Vibration: 8.2 mm/s", fill="black", font_size=38)
    image.save(image_path)
    result = LocalOCRService(confidence_threshold=0.1).extract(image_path)
    assert result.available
    assert result.pages[0].page == 1
    assert result.pages[0].words
    assert all(len(word.bbox) == 4 for word in result.pages[0].words)
    assert 0 <= result.mean_confidence <= 1

