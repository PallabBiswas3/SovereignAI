import * as pdfjs from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

/* =========================
   TYPES
========================= */

export interface PDFChunk {
  content: string;
  chunkIndex: number;
  pageStart: number;
  pageEnd: number;
  metadata: {
    fileName: string;
    totalPages: number;
    charCount: number;
  };
}

export interface PDFExtractionResult {
  fullText: string;
  chunks: PDFChunk[];
  totalPages: number;
  fileName: string;
  sourceDocId: string;
}

/* =========================
   CONSTANTS
========================= */

const CHUNK_SIZE = 600;       // words per chunk
const CHUNK_OVERLAP = 100;    // overlapping words between chunks for context continuity
const MIN_CHUNK_WORDS = 30;   // discard tiny chunks

/* =========================
   CORE TEXT EXTRACTION
========================= */

export const extractTextFromPDF = async (file: File): Promise<string> => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;

    let fullText = "";
    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map((item: any) => item.str)
        .filter(Boolean)
        .join(" ");
      fullText += pageText + "\n";
    }

    return fullText.trim();
  } catch (error) {
    console.error("Error extracting text from PDF:", error);
    throw new Error("Failed to extract text from PDF");
  }
};

/* =========================
   PER-PAGE TEXT EXTRACTION
========================= */

const extractPageTexts = async (
  file: File
): Promise<{ text: string; page: number }[]> => {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
  const pages: { text: string; page: number }[] = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();
    const pageText = textContent.items
      .map((item: any) => item.str)
      .filter(Boolean)
      .join(" ");
    pages.push({ text: pageText, page: pageNum });
  }

  return pages;
};

/* =========================
   SLIDING WINDOW CHUNKER
========================= */

export const extractChunksFromPDF = async (
  file: File
): Promise<PDFExtractionResult> => {
  const sourceDocId = `pdf_${Date.now()}_${file.name.replace(/\s+/g, "_")}`;
  const pageTexts = await extractPageTexts(file);
  const totalPages = pageTexts.length;

  // Build word-level index with page tracking
  interface WordEntry {
    word: string;
    page: number;
  }

  const wordIndex: WordEntry[] = [];
  for (const { text, page } of pageTexts) {
    const words = text.split(/\s+/).filter(Boolean);
    for (const word of words) {
      wordIndex.push({ word, page });
    }
  }

  const fullText = wordIndex.map((w) => w.word).join(" ");
  const chunks: PDFChunk[] = [];
  let chunkIndex = 0;
  let start = 0;

  while (start < wordIndex.length) {
    const end = Math.min(start + CHUNK_SIZE, wordIndex.length);
    const slice = wordIndex.slice(start, end);

    if (slice.length < MIN_CHUNK_WORDS) break;

    const content = slice.map((w) => w.word).join(" ");
    const pageStart = slice[0].page;
    const pageEnd = slice[slice.length - 1].page;

    chunks.push({
      content,
      chunkIndex,
      pageStart,
      pageEnd,
      metadata: {
        fileName: file.name,
        totalPages,
        charCount: content.length,
      },
    });

    chunkIndex++;
    // Slide forward by (CHUNK_SIZE - CHUNK_OVERLAP) for overlapping context
    start += CHUNK_SIZE - CHUNK_OVERLAP;
  }

  return {
    fullText,
    chunks,
    totalPages,
    fileName: file.name,
    sourceDocId,
  };
};

/* =========================
   VALIDATION
========================= */

export const validatePDFFile = (file: File): boolean => {
  return file.type === "application/pdf" && file.size <= 10 * 1024 * 1024;
};

export const getPDFPageCount = async (file: File): Promise<number> => {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
  return pdf.numPages;
};