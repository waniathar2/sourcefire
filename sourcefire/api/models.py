"""Pydantic request/response models for the Sourcefire API."""

from typing import Literal

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    mode: Literal["debug", "feature", "explain"] = "debug"
    model: Literal["gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"] = "gemini-3.1-flash-lite-preview"
    history: list[dict] = []


class StatusResponse(BaseModel):
    files_indexed: int
    last_indexed: str
    index_status: str
    language: str = "generic"


class SourceResponse(BaseModel):
    content: str
    language: str
