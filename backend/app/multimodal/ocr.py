from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel, Field
from pypdf import PdfReader


class OCRWord(BaseModel):
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class OCRPage(BaseModel):
    page: int
    text: str
    confidence: float
    words: list[OCRWord] = Field(default_factory=list)


class OCRResult(BaseModel):
    text: str
    pages: list[OCRPage]
    mean_confidence: float
    low_confidence: bool
    engine: str = "tesseract"
    available: bool = True
    warning: str | None = None


class LocalOCRService:
    def __init__(self, language: str = "eng", confidence_threshold: float = 0.65) -> None:
        self.language = language
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def is_scanned_pdf(path: Path, min_chars_per_page: int = 30) -> bool:
        reader = PdfReader(str(path))
        if not reader.pages:
            return False
        counts = [len((page.extract_text() or "").strip()) for page in reader.pages]
        return sum(counts) / len(counts) < min_chars_per_page

    def extract(self, path: Path) -> OCRResult:
        if not shutil.which("tesseract"):
            return OCRResult(text="", pages=[], mean_confidence=0.0, low_confidence=True, available=False, warning="Tesseract is not installed; OCR was not performed.")
        images = self._load_images(path)
        pages = [self._ocr_image(image, index + 1) for index, image in enumerate(images)]
        mean = sum(page.confidence for page in pages) / len(pages) if pages else 0.0
        low = mean < self.confidence_threshold
        return OCRResult(
            text="\n\n".join(f"[PAGE {page.page}]\n{page.text}" for page in pages),
            pages=pages,
            mean_confidence=round(mean, 4),
            low_confidence=low,
            warning=(f"OCR confidence {mean:.0%} is below the {self.confidence_threshold:.0%} threshold; verify extracted facts." if low else None),
        )

    def _load_images(self, path: Path) -> list[Image.Image]:
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(str(path))
            try:
                return [page.render(scale=2.2).to_pil() for page in document]
            finally:
                document.close()
        image = Image.open(path)
        frames: list[Image.Image] = []
        frame = 0
        while True:
            try:
                image.seek(frame)
                frames.append(image.convert("RGB").copy())
                frame += 1
            except EOFError:
                break
        return frames

    def _ocr_image(self, image: Image.Image, page_number: int) -> OCRPage:
        import pytesseract
        from pytesseract import Output

        prepared = ImageOps.autocontrast(ImageOps.grayscale(image)).filter(ImageFilter.SHARPEN)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.25)
        data = pytesseract.image_to_data(prepared, lang=self.language, output_type=Output.DICT, config="--psm 6")
        words: list[OCRWord] = []
        lines: dict[tuple[int, int, int], list[str]] = {}
        for index, raw in enumerate(data["text"]):
            token = raw.strip()
            try:
                confidence = max(0.0, float(data["conf"][index]) / 100.0)
            except (ValueError, TypeError):
                confidence = 0.0
            if not token:
                continue
            word = OCRWord(
                text=token,
                confidence=round(confidence, 4),
                bbox=(int(data["left"][index]), int(data["top"][index]), int(data["width"][index]), int(data["height"][index])),
            )
            words.append(word)
            key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
            lines.setdefault(key, []).append(token)
        text = "\n".join(" ".join(tokens) for tokens in lines.values())
        mean = sum(word.confidence for word in words) / len(words) if words else 0.0
        return OCRPage(page=page_number, text=text, confidence=round(mean, 4), words=words)


class DocumentTextExtractor:
    def __init__(self, ocr: LocalOCRService | None = None) -> None:
        self.ocr = ocr or LocalOCRService()

    def extract(self, path: Path) -> tuple[str, OCRResult | None]:
        from app.tools.file_tools import extract_text

        if path.suffix.lower() == ".pdf" and self.ocr.is_scanned_pdf(path):
            result = self.ocr.extract(path)
            if not result.available:
                raise RuntimeError(result.warning or "OCR unavailable")
            return result.text, result
        return extract_text(path), None

