"""Tests for API routes and models."""

from pathlib import Path


def test_source_path_traversal_blocked():
    from src.config import CODEBASE_PATH

    full = (CODEBASE_PATH / "../../etc/passwd").resolve()
    assert not str(full).startswith(str(CODEBASE_PATH.resolve()))


def test_query_request_defaults():
    from src.api.models import QueryRequest

    req = QueryRequest(query="test error")
    assert req.mode == "debug"
    assert req.model == "gemini-2.5-flash"
    assert req.history == []


def test_status_response_model():
    from src.api.models import StatusResponse

    s = StatusResponse(files_indexed=172, last_indexed="2026-03-23", index_status="ready")
    assert s.files_indexed == 172
