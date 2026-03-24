"""Configuration for Sourcefire — loaded from .sourcefire/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w


@dataclass
class SourcefireConfig:
    """All Sourcefire configuration for a project."""

    # Resolved at runtime, not stored in TOML
    project_dir: Path = field(default_factory=Path.cwd)
    sourcefire_dir: Path = field(default_factory=lambda: Path.cwd() / ".sourcefire")

    # [project]
    project_name: str = ""
    language: str = "auto"

    # [indexer]
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    chunk_size: int = 1000
    chunk_overlap: int = 300

    # [llm]
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"

    # [server]
    host: str = "127.0.0.1"
    port: int = 8000

    # [retrieval]
    top_k: int = 8
    relevance_threshold: float = 0.3

    # Versioning
    config_version: int = 1

    @property
    def gemini_api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    @property
    def chroma_dir(self) -> Path:
        return self.sourcefire_dir / "chroma"

    @property
    def graph_path(self) -> Path:
        return self.sourcefire_dir / "graph.json"

    @property
    def config_path(self) -> Path:
        return self.sourcefire_dir / "config.toml"

    @property
    def lock_path(self) -> Path:
        return self.sourcefire_dir / ".lock"


# Constants used by other modules
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKEN_BUDGET: dict[str, int] = {
    "gemini-2.5-flash": 100_000,
    "gemini-2.5-pro": 200_000,
}
MAX_HISTORY_PAIRS: int = 5
RESPONSE_HEADROOM: int = 8_000


def default_config(project_dir: Path) -> SourcefireConfig:
    """Return a SourcefireConfig with sensible defaults for the given project."""
    return SourcefireConfig(
        project_dir=project_dir,
        sourcefire_dir=project_dir / ".sourcefire",
        project_name=project_dir.name,
    )


def load_config(project_dir: Path, sourcefire_dir: Path) -> SourcefireConfig:
    """Load config from .sourcefire/config.toml."""
    config_path = sourcefire_dir / "config.toml"
    raw = config_path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)

    project = data.get("project", {})
    indexer = data.get("indexer", {})
    llm = data.get("llm", {})
    server = data.get("server", {})
    retrieval = data.get("retrieval", {})

    return SourcefireConfig(
        project_dir=project_dir,
        sourcefire_dir=sourcefire_dir,
        config_version=data.get("config_version", 1),
        project_name=project.get("name", project_dir.name),
        language=project.get("language", "auto"),
        include=indexer.get("include", []),
        exclude=indexer.get("exclude", []),
        chunk_size=indexer.get("chunk_size", 1000),
        chunk_overlap=indexer.get("chunk_overlap", 300),
        provider=llm.get("provider", "gemini"),
        model=llm.get("model", "gemini-2.5-flash"),
        api_key_env=llm.get("api_key_env", "GEMINI_API_KEY"),
        host=server.get("host", "127.0.0.1"),
        port=server.get("port", 8000),
        top_k=retrieval.get("top_k", 8),
        relevance_threshold=retrieval.get("relevance_threshold", 0.3),
    )


def save_config(config: SourcefireConfig) -> None:
    """Write config to .sourcefire/config.toml."""
    data = {
        "config_version": config.config_version,
        "project": {
            "name": config.project_name,
            "language": config.language,
        },
        "indexer": {
            "include": config.include,
            "exclude": config.exclude,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        "llm": {
            "provider": config.provider,
            "model": config.model,
            "api_key_env": config.api_key_env,
        },
        "server": {
            "host": config.host,
            "port": config.port,
        },
        "retrieval": {
            "top_k": config.top_k,
            "relevance_threshold": config.relevance_threshold,
        },
    }
    config.config_path.parent.mkdir(parents=True, exist_ok=True)
    config.config_path.write_text(tomli_w.dumps(data), encoding="utf-8")
