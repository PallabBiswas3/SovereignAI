from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.artifacts.docx_generator import DocxGenerator
from app.artifacts.pptx_generator import PptxGenerator
from app.artifacts.service import ArtifactService
from app.artifacts.xlsx_generator import XlsxGenerator
from app.multimodal.ocr import LocalOCRService
from app.multimodal.vision import OllamaVisionProvider
from app.rag.embeddings import configured_embedding_provider
from app.rag.factory import configured_hybrid_retriever
from app.tools.base import Tool, ToolResult, ToolRisk
from app.tools.file_tools import SafeWorkspace
from app.resources.cache import get_cache_backend
from app.core.config import get_settings
from app.identity.models import Principal, ResourceScope


def _filename(value: str, extension: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(value).name).strip() or f"artifact{extension}"
    return name if name.lower().endswith(extension) else f"{name}{extension}"


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"
    description = "Search the indexed internal knowledge base and return cited source chunks"
    risk = ToolRisk.low
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}

    def __init__(self, session: Session, principal: Principal | None = None) -> None:
        self.session = session
        self.principal = principal

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, error="A query is required")
        limit = max(1, min(int(args.get("limit", 5)), 10))
        settings = get_settings()
        results = configured_hybrid_retriever(
            self.session, cache=get_cache_backend(), settings=settings,
            principal=self.principal if settings.auth_mode.lower() == "local" else None,
        ).search(query, limit)
        return ToolResult(success=True, output={"results": [
            item.to_dict()
            for item in results
        ]})


class OCRDocumentTool(Tool):
    name = "ocr_document"
    description = "Run local Tesseract OCR on a scanned PDF or image"
    risk = ToolRisk.low
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = self.workspace.resolve(str(args.get("path", "")), must_exist=True)
            result = LocalOCRService(cache=get_cache_backend()).extract(path)
            return ToolResult(success=result.available, output=result.model_dump(), error=result.warning if not result.available else None)
        except (ValueError, FileNotFoundError, OSError) as exc:
            return ToolResult(success=False, error=str(exc))


class AnalyzeImageTool(Tool):
    name = "analyze_image"
    description = "Analyze an image using the configured local vision model"
    risk = ToolRisk.low
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["path"]}

    def __init__(self, workspace: SafeWorkspace, endpoint: str, model_tag: str) -> None:
        self.workspace, self.endpoint, self.model_tag = workspace, endpoint, model_tag

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            path = self.workspace.resolve(str(args.get("path", "")), must_exist=True)
            result = await OllamaVisionProvider(
                self.endpoint, self.model_tag, cache=get_cache_backend()
            ).analyze_image(
                path, str(args.get("prompt", "Describe relevant industrial evidence and uncertainty."))
            )
            return ToolResult(success=result.available, output=result.model_dump(), error=result.warning if not result.available else None)
        except (ValueError, FileNotFoundError, OSError) as exc:
            return ToolResult(success=False, error=str(exc))


class _ArtifactTool(Tool):
    risk = ToolRisk.low

    def __init__(self, artifacts: ArtifactService, scope: ResourceScope | None = None) -> None:
        self.artifacts = artifacts
        self.scope = scope

    def register(self, path: Path, run_id: str | None) -> ToolResult:
        record = self.artifacts.register(path, run_id, scope=self.scope)
        return ToolResult(success=True, output={
            "artifact": {"id": record.id, "name": record.name, "url": f"/api/artifacts/{record.id}"}
        }, generated_files=[record.path])


class GenerateDocxTool(_ArtifactTool):
    name = "generate_docx"
    description = "Generate and register a Word report from titled sections"
    input_schema = {"type": "object", "properties": {"filename": {"type": "string"}, "title": {"type": "string"}, "sections": {"type": "array"}, "sources": {"type": "array"}, "run_id": {"type": "string"}}, "required": ["title", "sections"]}

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        filename = _filename(str(args.get("filename", "report.docx")), ".docx")
        path = self.artifacts.root / filename
        DocxGenerator().generate_report(path, str(args["title"]), list(args["sections"]), list(args.get("sources", [])))
        return self.register(path, str(args.get("run_id")) if args.get("run_id") else None)


class GenerateXlsxTool(_ArtifactTool):
    name = "generate_xlsx"
    description = "Generate and register an Excel workbook from structured rows"
    input_schema = {"type": "object", "properties": {"filename": {"type": "string"}, "title": {"type": "string"}, "rows": {"type": "array"}, "run_id": {"type": "string"}}, "required": ["title", "rows"]}

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        filename = _filename(str(args.get("filename", "analysis.xlsx")), ".xlsx")
        path = self.artifacts.root / filename
        XlsxGenerator().generate(path, str(args["title"]), list(args["rows"]))
        return self.register(path, str(args.get("run_id")) if args.get("run_id") else None)


class GeneratePptxTool(_ArtifactTool):
    name = "generate_pptx"
    description = "Generate and register a PowerPoint briefing from title and bullet slides"
    input_schema = {"type": "object", "properties": {"filename": {"type": "string"}, "title": {"type": "string"}, "slides": {"type": "array"}, "run_id": {"type": "string"}}, "required": ["title", "slides"]}

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        filename = _filename(str(args.get("filename", "briefing.pptx")), ".pptx")
        path = self.artifacts.root / filename
        PptxGenerator().generate(path, str(args["title"]), list(args["slides"]))
        return self.register(path, str(args.get("run_id")) if args.get("run_id") else None)
