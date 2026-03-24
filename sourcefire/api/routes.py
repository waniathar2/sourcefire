"""FastAPI router for the Sourcefire API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from sourcefire.api.models import QueryRequest, SourceResponse, StatusResponse
from sourcefire.config import CODEBASE_PATH, GEMINI_API_KEY

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Module-level dependency state — set once at startup via init_dependencies()
# ---------------------------------------------------------------------------

_pool: Any = None
_graph: Any = None
_profile: Any = None
_index_status: dict[str, Any] = {
    "files_indexed": 0,
    "last_indexed": "never",
    "index_status": "not_ready",
    "language": "generic",
}


def init_dependencies(pool: Any, graph: Any, index_status: dict[str, Any], profile: Any = None) -> None:
    """Inject shared dependencies from the application lifespan."""
    global _pool, _graph, _index_status, _profile
    _pool = pool
    _graph = graph
    _index_status = index_status
    _profile = profile


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".dart": "dart",
    ".py": "python",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".toml": "toml",
    ".xml": "xml",
    ".sql": "sql",
    ".graphql": "graphql",
    ".proto": "protobuf",
    ".tf": "hcl",
    ".dockerfile": "dockerfile",
}


def _detect_language(file_path: Path) -> str:
    name = file_path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return _EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "plaintext")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    """Stream a RAG response for the given query via Server-Sent Events."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Set it in your .env file.",
        )

    from sourcefire.chain.rag_chain import stream_rag_response

    async def _event_generator() -> AsyncGenerator[dict[str, str], None]:
        async for chunk in stream_rag_response(
            pool=_pool,
            graph=_graph,
            query=request.query,
            mode=request.mode,
            model=request.model,
            history=request.history,
            profile=_profile,
        ):
            yield {"data": json.dumps(chunk)}

    return EventSourceResponse(_event_generator())


@router.get("/sources", response_model=SourceResponse)
async def sources(path: str = Query(..., description="Relative path within the codebase")) -> SourceResponse:
    """Return the content and detected language of a source file."""
    codebase_resolved = CODEBASE_PATH.resolve()
    full_path = (CODEBASE_PATH / path).resolve()

    if not str(full_path).startswith(str(codebase_resolved)):
        raise HTTPException(status_code=400, detail="Path traversal detected.")

    if not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}") from exc

    return SourceResponse(
        content=content,
        language=_detect_language(full_path),
    )


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Return current index status."""
    return StatusResponse(
        files_indexed=_index_status.get("files_indexed", 0),
        last_indexed=str(_index_status.get("last_indexed", "never")),
        index_status=str(_index_status.get("index_status", "not_ready")),
        language=str(_index_status.get("language", "generic")),
    )
