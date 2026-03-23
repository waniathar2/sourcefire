"""Pydantic request/response models for the Cravv Observatory API."""

from typing import Literal

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    mode: Literal["debug", "feature", "explain"] = "debug"
    model: Literal["gemini-2.5-flash", "gemini-2.5-pro"] = "gemini-2.5-flash"
    history: list[dict] = []


class StatusResponse(BaseModel):
    files_indexed: int
    last_indexed: str
    index_status: str


class SourceResponse(BaseModel):
    content: str
    language: str
