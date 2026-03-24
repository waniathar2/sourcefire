# Sourcefire CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Sourcefire from a project-local FastAPI app with PostgreSQL into a globally-installable CLI tool with per-project `.sourcefire/` directories, ChromaDB for embedded vector storage, a file watcher for live re-indexing, and LLM-powered auto-initialization.

**Architecture:** Single `sourcefire` command that auto-detects or initializes a project, manages config via `.sourcefire/config.toml`, stores vectors in ChromaDB (`.sourcefire/chroma/`), and runs a FastAPI web UI with a background file watcher. The package is renamed from `src` to `sourcefire` for global installability.

**Tech Stack:** Python 3.11+, ChromaDB, FastAPI, LangChain, sentence-transformers, tree-sitter, watchfiles, tomli-w

**Spec:** `docs/superpowers/specs/2026-03-24-sourcefire-cli-design.md`

---

## File Structure

### New Files
- `sourcefire/cli.py` — CLI entry point (arg parsing, project discovery, init, server launch)
- `sourcefire/init.py` — Auto-init logic (LLM config generation, `.sourcefire/` setup)
- `sourcefire/watcher.py` — File watcher for live incremental re-indexing
- `sourcefire/db.py` — ChromaDB wrapper (async-safe operations via `run_in_executor`)

### Renamed Files (src/ -> sourcefire/)
- `src/__init__.py` -> `sourcefire/__init__.py`
- `src/config.py` -> `sourcefire/config.py`
- `src/api/__init__.py` -> `sourcefire/api/__init__.py`
- `src/api/models.py` -> `sourcefire/api/models.py`
- `src/api/routes.py` -> `sourcefire/api/routes.py`
- `src/chain/__init__.py` -> `sourcefire/chain/__init__.py`
- `src/chain/prompts.py` -> `sourcefire/chain/prompts.py`
- `src/chain/rag_chain.py` -> `sourcefire/chain/rag_chain.py`
- `src/indexer/__init__.py` -> `sourcefire/indexer/__init__.py`
- `src/indexer/embeddings.py` -> `sourcefire/indexer/embeddings.py`
- `src/indexer/metadata.py` -> `sourcefire/indexer/metadata.py`
- `src/indexer/language_profiles.py` -> `sourcefire/indexer/language_profiles.py`
- `src/indexer/pipeline.py` -> `sourcefire/indexer/pipeline.py`
- `src/retriever/__init__.py` -> `sourcefire/retriever/__init__.py`
- `src/retriever/graph.py` -> `sourcefire/retriever/graph.py`
- `src/retriever/search.py` -> `sourcefire/retriever/search.py`
- `static/` -> `sourcefire/static/`
- `prompts/` -> `sourcefire/prompts/`

### Modified Files
- `pyproject.toml` — New name, deps, scripts entry, package-data
- `sourcefire/config.py` — TOML-based config instead of env vars
- `sourcefire/indexer/pipeline.py` — ChromaDB instead of psycopg
- `sourcefire/retriever/search.py` — ChromaDB query API instead of SQL
- `sourcefire/retriever/graph.py` — JSON file persistence + `remove_file()` method
- `sourcefire/chain/rag_chain.py` — Replace `pool` with `collection`, `CODEBASE_PATH` with `project_dir`
- `sourcefire/chain/prompts.py` — `importlib.resources` for system.md path
- `sourcefire/api/routes.py` — Replace `pool` with `collection` + `project_dir`
- `sourcefire/api/models.py` — Update model name literals
- `sourcefire/indexer/embeddings.py` — Update import path
- `.env.example` — Remove DB URL and CODEBASE_PATH

### Deleted Files
- `main.py` — Replaced by `sourcefire/cli.py`
- `src/` — Entire directory replaced by `sourcefire/`
- `src/tara_rag.egg-info/` — Old build artifact

---

## Task 1: Package Rename and pyproject.toml

Rename `src/` to `sourcefire/`, update `pyproject.toml`, and fix all internal imports. This is the foundation — everything else builds on this.

**Files:**
- Rename: `src/` -> `sourcefire/`
- Move: `static/` -> `sourcefire/static/`
- Move: `prompts/` -> `sourcefire/prompts/`
- Modify: `pyproject.toml`
- Delete: `main.py`
- Delete: `src/tara_rag.egg-info/`

- [ ] **Step 1: Rename src/ to sourcefire/**

```bash
git mv src sourcefire
```

- [ ] **Step 2: Move static/ and prompts/ into the package**

```bash
git mv static sourcefire/static
git mv prompts sourcefire/prompts
```

- [ ] **Step 3: Update all internal imports**

Replace every `from src.` with `from sourcefire.` and every `import src.` with `import sourcefire.` across all `.py` files in the `sourcefire/` directory. Files that need updating:

- `sourcefire/config.py` — no `src.` imports (standalone)
- `sourcefire/indexer/pipeline.py` — `from src.config` -> `from sourcefire.config`, `from src.indexer.` -> `from sourcefire.indexer.`, `from src.retriever.` -> `from sourcefire.retriever.`
- `sourcefire/retriever/search.py` — `from src.config` -> `from sourcefire.config`, `from src.indexer.` -> `from sourcefire.indexer.`
- `sourcefire/retriever/graph.py` — no `src.` imports (standalone)
- `sourcefire/chain/prompts.py` — `from src.config` -> `from sourcefire.config`
- `sourcefire/chain/rag_chain.py` — `from src.config` -> `from sourcefire.config`, `from src.indexer.` -> `from sourcefire.indexer.`, `from src.retriever.` -> `from sourcefire.retriever.`, `from src.chain.` -> `from sourcefire.chain.`
- `sourcefire/api/routes.py` — `from src.api.` -> `from sourcefire.api.`, `from src.config` -> `from sourcefire.config`
- `sourcefire/indexer/embeddings.py` — `from src.config` -> `from sourcefire.config`
- `sourcefire/indexer/metadata.py` — `from src.indexer.language_profiles` -> `from sourcefire.indexer.language_profiles`
- `sourcefire/api/routes.py` — ALSO fix the lazy import inside `query()` function body: `from src.chain.rag_chain` -> `from sourcefire.chain.rag_chain`

**Important:** ALL import fixes must happen in this step. Do not defer any `from src.` fixes to later tasks — they will cause `ImportError` at module load time and block subsequent tasks.

- [ ] **Step 4: Update pyproject.toml**

```toml
[project]
name = "sourcefire"
version = "0.2.0"
description = "Sourcefire — AI-powered codebase RAG from your terminal"
requires-python = ">=3.11"
dependencies = [
    "chromadb",
    "langchain",
    "langchain-google-genai",
    "fastapi",
    "uvicorn[standard]",
    "sse-starlette",
    "sentence-transformers",
    "tree-sitter",
    "tree-sitter-languages",
    "python-dotenv",
    "watchfiles",
    "tomli-w",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx"]

[project.scripts]
sourcefire = "sourcefire.cli:main"

[tool.setuptools.package-data]
sourcefire = ["static/**/*", "prompts/**/*"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 5: Delete old main.py and egg-info**

```bash
git rm main.py
rm -rf src/tara_rag.egg-info  # may not be tracked
```

- [ ] **Step 6: Update .env.example**

```
GEMINI_API_KEY=AIza...
```

Remove `COCOINDEX_DATABASE_URL` and `CODEBASE_PATH` lines.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rename src/ to sourcefire/ package, update deps for CLI tool"
```

---

## Task 2: Config Module — TOML-Based Configuration

Replace the environment-variable-based config with a TOML-file-based config that reads from `.sourcefire/config.toml`.

**Files:**
- Modify: `sourcefire/config.py`

- [ ] **Step 1: Rewrite sourcefire/config.py**

The new config module loads from a TOML file and provides a `SourcefireConfig` dataclass. Environment variables (`GEMINI_API_KEY`) still work as overrides. The module also provides `load_config()` and `default_config()` functions.

```python
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

    # Derived
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


# Kept for backwards compat during migration — modules that import these
# will be updated in later tasks to use config object instead.
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
    config.config_path.write_text(
        tomli_w.dumps(data), encoding="utf-8"
    )
```

- [ ] **Step 2: Verify the module parses correctly**

```bash
python -c "from sourcefire.config import SourcefireConfig, default_config; c = default_config(Path('.')); print(c)"
```

- [ ] **Step 3: Commit**

```bash
git add sourcefire/config.py
git commit -m "refactor: TOML-based config module with SourcefireConfig dataclass"
```

---

## Task 3: ChromaDB Wrapper

Create `sourcefire/db.py` — a thin async-safe wrapper around ChromaDB that all other modules use. This replaces all psycopg database operations.

**Files:**
- Create: `sourcefire/db.py`

- [ ] **Step 1: Write sourcefire/db.py**

```python
"""ChromaDB wrapper for Sourcefire — async-safe via run_in_executor."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

import chromadb

COLLECTION_NAME = "code_chunks"


def create_client(chroma_dir: Path) -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the code_chunks collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Delete and recreate the collection (for full re-index)."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass  # Collection doesn't exist
    return get_collection(client)


# ---------------------------------------------------------------------------
# Sync operations (called directly during indexing, or via executor for async)
# ---------------------------------------------------------------------------


def add_chunks(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, str]],
) -> None:
    """Add chunks to the collection. Batch size handled by caller."""
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_file_chunks(collection: chromadb.Collection, filename: str) -> None:
    """Delete all chunks for a given filename."""
    collection.delete(where={"filename": filename})


def query_similar(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 8,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """Query for similar chunks. Returns list of result dicts."""
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    rows: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return rows

    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else 1.0
        relevance = 1.0 - distance  # cosine distance -> similarity

        rows.append({
            "filename": meta.get("filename", ""),
            "location": meta.get("location", ""),
            "code": results["documents"][0][i] if results["documents"] else "",
            "feature": meta.get("feature", ""),
            "layer": meta.get("layer", ""),
            "file_type": meta.get("file_type", ""),
            "relevance": relevance,
        })

    return rows


def get_chunks_by_files(
    collection: chromadb.Collection,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Retrieve all chunks for the given filenames."""
    if not filenames:
        return []

    # ChromaDB $in filter
    results = collection.get(
        where={"filename": {"$in": filenames}},
        include=["documents", "metadatas"],
    )

    rows: list[dict[str, Any]] = []
    if not results["ids"]:
        return rows

    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] if results["metadatas"] else {}
        rows.append({
            "filename": meta.get("filename", ""),
            "location": meta.get("location", ""),
            "code": results["documents"][i] if results["documents"] else "",
            "feature": meta.get("feature", ""),
            "layer": meta.get("layer", ""),
            "file_type": meta.get("file_type", ""),
        })

    return rows


def get_indexed_files(collection: chromadb.Collection) -> set[str]:
    """Return set of all filenames currently in the collection."""
    results = collection.get(include=["metadatas"])
    files: set[str] = set()
    if results["metadatas"]:
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                files.add(meta["filename"])
    return files


# ---------------------------------------------------------------------------
# Async wrappers (for use in FastAPI routes / RAG chain)
# ---------------------------------------------------------------------------


async def async_query_similar(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 8,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """Async wrapper for query_similar."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(query_similar, collection, query_embedding, n_results, where)
    )


async def async_get_chunks_by_files(
    collection: chromadb.Collection,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Async wrapper for get_chunks_by_files."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(get_chunks_by_files, collection, filenames)
    )


async def async_delete_file_chunks(
    collection: chromadb.Collection,
    filename: str,
) -> None:
    """Async wrapper for delete_file_chunks."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, partial(delete_file_chunks, collection, filename)
    )
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from sourcefire.db import create_client, get_collection, query_similar; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sourcefire/db.py
git commit -m "feat: add ChromaDB wrapper with async-safe operations"
```

---

## Task 4: Rewrite Indexing Pipeline for ChromaDB

Replace all psycopg database operations in the pipeline with ChromaDB operations using `sourcefire/db.py`. Remove PostgreSQL schema DDL. Accept config object instead of reading globals.

**Files:**
- Modify: `sourcefire/indexer/pipeline.py`

- [ ] **Step 1: Rewrite pipeline.py**

Key changes:
- Replace `ConnectionPool` parameter with `chromadb.Collection`
- Remove all SQL DDL (`_SCHEMA_SQL`, `_IVFFLAT_SQL`)
- Remove `_create_schema`, `_try_create_ivfflat_index`, `_upsert_chunks`, `_build_import_graph` DB functions
- Replace with ChromaDB `add_chunks` from `sourcefire/db`
- Accept `SourcefireConfig` for `CODEBASE_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, include/exclude patterns
- Return import graph edges as dict instead of writing to DB
- `_collect_files` uses config include/exclude patterns (authoritative, not merged with profile)

```python
"""Full indexing pipeline for Sourcefire.

Scans a codebase, chunks files (AST-aware when a language profile exists,
simple split otherwise), embeds all chunks, and inserts them into ChromaDB.
Also builds the import graph for graph-augmented retrieval.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import chromadb

from sourcefire.config import SourcefireConfig
from sourcefire.db import add_chunks, reset_collection, delete_file_chunks
from sourcefire.indexer.embeddings import embed_batch
from sourcefire.indexer.language_profiles import LanguageProfile, get_profile, get_profile_for_extension
from sourcefire.indexer.metadata import chunk_source_file, extract_metadata
from sourcefire.retriever.graph import ImportGraph

# ---------------------------------------------------------------------------
# .gitignore parsing
# ---------------------------------------------------------------------------


def _parse_gitignore(codebase_path: Path) -> list[str]:
    """Parse .gitignore and return a list of glob patterns to exclude."""
    gitignore_path = codebase_path / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    try:
        for line in gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("/"):
                line = line[1:]
            if line.endswith("/"):
                line = line + "**"
            patterns.append(line)
    except OSError:
        pass
    return patterns


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------


def _match_patterns(rel_path: str, patterns: list[str]) -> bool:
    """Return True if *rel_path* matches any of *patterns* using fnmatch."""
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Non-AST chunker (simple recursive text split)
# ---------------------------------------------------------------------------


def _chunk_plain_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 300,
) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------


def _collect_files(
    codebase_path: Path,
    config: SourcefireConfig,
    profile: LanguageProfile | None,
) -> list[Path]:
    """Return all files under *codebase_path* that pass include/exclude filters.

    Config patterns are authoritative. If config has include patterns, those
    are used. Otherwise falls back to language profile patterns.
    """
    # Config include/exclude take precedence
    include_patterns: list[str] = list(config.include) if config.include else []
    exclude_patterns: list[str] = list(config.exclude) if config.exclude else []

    # If no config patterns, fall back to profile
    if not include_patterns and profile:
        include_patterns = list(profile.include_patterns)
    if not exclude_patterns and profile:
        exclude_patterns = list(profile.exclude_patterns)

    # Always exclude .gitignore patterns
    exclude_patterns.extend(_parse_gitignore(codebase_path))

    # Broad default if nothing specified
    if not include_patterns:
        include_patterns = ["**/*"]

    matched: list[Path] = []
    for pattern in include_patterns:
        for file_path in codebase_path.glob(pattern):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(codebase_path).as_posix()
            if not _match_patterns(rel, exclude_patterns):
                matched.append(file_path)

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in matched:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


# ---------------------------------------------------------------------------
# Chunk production
# ---------------------------------------------------------------------------


def _chunks_for_file(
    file_path: Path,
    codebase_path: Path,
    profile: LanguageProfile | None,
    chunk_size: int = 1000,
) -> list[dict[str, Any]]:
    """Return a list of chunk dicts for *file_path*."""
    rel = file_path.relative_to(codebase_path).as_posix()
    source = file_path.read_text(encoding="utf-8", errors="replace")

    file_profile = get_profile_for_extension(file_path.suffix) or profile

    if file_profile and file_path.suffix in [e for e in file_profile.file_extensions]:
        raw_chunks = chunk_source_file(source, rel, file_profile, chunk_size=chunk_size)
        chunks_out: list[dict[str, Any]] = []
        for idx, chunk in enumerate(raw_chunks):
            meta = chunk["metadata"]
            chunks_out.append({
                "filename": rel,
                "location": f"{rel}:{idx}",
                "code": chunk["text"],
                "feature": meta.get("feature", ""),
                "layer": meta.get("layer", ""),
                "file_type": meta.get("file_type", ""),
            })
        return chunks_out
    else:
        meta = extract_metadata("", rel, file_profile)
        raw_texts = _chunk_plain_text(source, chunk_size=chunk_size)
        return [
            {
                "filename": rel,
                "location": f"{rel}:{idx}",
                "code": text,
                "feature": meta.get("feature", ""),
                "layer": meta.get("layer", ""),
                "file_type": meta.get("file_type", ""),
            }
            for idx, text in enumerate(raw_texts)
        ]


# ---------------------------------------------------------------------------
# Mtime tracking for incremental indexing
# ---------------------------------------------------------------------------


def _get_stored_mtimes(collection: chromadb.Collection) -> dict[str, float]:
    """Get stored mtimes for all indexed files from ChromaDB metadata."""
    results = collection.get(include=["metadatas"])
    mtimes: dict[str, float] = {}
    if results["metadatas"]:
        for meta in results["metadatas"]:
            if meta and "filename" in meta and "mtime" in meta:
                try:
                    mtimes[meta["filename"]] = float(meta["mtime"])
                except (ValueError, TypeError):
                    pass
    return mtimes


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_indexing(
    collection: chromadb.Collection,
    config: SourcefireConfig,
    client: chromadb.ClientAPI | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """Run the full indexing pipeline.

    Args:
        collection: ChromaDB collection to write to.
        config: Sourcefire configuration.
        client: ChromaDB client (needed for full reset).
        full: If True, reset collection and re-index everything.
              If False, incremental — compare file mtimes and only
              re-index changed/new files, delete removed files.

    Returns:
        A stats dict with keys: ``files``, ``chunks``, ``edges``, ``language``.
        Also includes ``import_edges`` — the raw import map for graph building.
    """
    codebase_path = config.project_dir
    print(f"[pipeline] Scanning codebase at: {codebase_path}")

    # Detect language
    language_override = config.language if config.language != "auto" else None
    profile = get_profile(codebase_path, language_override)
    lang_name = profile.language if profile else "generic"
    print(f"[pipeline] Detected language: {lang_name}")

    # Collect files on disk
    files = _collect_files(codebase_path, config, profile)
    print(f"[pipeline] Found {len(files)} files to index.")

    # Empty project check
    if not files:
        print("[pipeline] Error: No source files found matching the configured patterns.")
        print("Run `sourcefire --reinit` to regenerate patterns, or edit .sourcefire/config.toml manually.")
        return {
            "files": 0, "chunks": 0, "edges": 0,
            "language": lang_name, "import_edges": {},
        }

    # Full reset if requested
    if full and client:
        collection = reset_collection(client)
        print("[pipeline] Collection reset for full re-index.")
    elif not full:
        # Incremental mode: determine which files changed
        from sourcefire.db import get_indexed_files
        indexed_files = get_indexed_files(collection)

        # Build mtime map for current files on disk
        current_files: dict[str, float] = {}
        for f in files:
            rel = f.relative_to(codebase_path).as_posix()
            current_files[rel] = f.stat().st_mtime

        # Determine changed/new/deleted files
        # For files already indexed, compare mtime stored in metadata
        stored_mtimes = _get_stored_mtimes(collection)
        changed: list[Path] = []
        for f in files:
            rel = f.relative_to(codebase_path).as_posix()
            stored_mtime = stored_mtimes.get(rel, 0.0)
            if rel not in indexed_files or f.stat().st_mtime > stored_mtime:
                changed.append(f)

        deleted = indexed_files - set(current_files.keys())

        # Delete removed files
        for rel in deleted:
            delete_file_chunks(collection, rel)

        if not changed and not deleted:
            print("[pipeline] Index is up to date.")
            return {
                "files": len(files), "chunks": collection.count(), "edges": 0,
                "language": lang_name, "import_edges": {},
            }

        # Only process changed files
        print(f"[pipeline] {len(changed)} changed, {len(deleted)} deleted files.")
        files = changed

    # Produce chunks and collect imports
    all_chunks: list[dict[str, Any]] = []
    file_imports: dict[str, list[str]] = {}

    for file_path in files:
        rel = file_path.relative_to(codebase_path).as_posix()
        chunks = _chunks_for_file(file_path, codebase_path, profile, chunk_size=config.chunk_size)
        all_chunks.extend(chunks)

        file_profile = get_profile_for_extension(file_path.suffix) or profile
        if file_profile and file_path.suffix in file_profile.file_extensions:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            meta = extract_metadata(source, rel, file_profile)
            if meta.get("imports"):
                file_imports[rel] = meta["imports"]

    print(f"[pipeline] Produced {len(all_chunks)} chunks.")

    if not all_chunks:
        return {
            "files": len(files), "chunks": 0, "edges": 0,
            "language": lang_name, "import_edges": {},
        }

    # Embed in one batch
    print("[pipeline] Embedding chunks...")
    texts = [c["code"] for c in all_chunks]
    embeddings = embed_batch(texts)
    print("[pipeline] Embeddings done.")

    # Build mtime lookup for embedding in metadata
    file_mtimes: dict[str, str] = {}
    for file_path in files:
        rel = file_path.relative_to(codebase_path).as_posix()
        file_mtimes[rel] = str(file_path.stat().st_mtime)

    # Delete old chunks for changed files (incremental mode)
    if not full:
        for file_path in files:
            rel = file_path.relative_to(codebase_path).as_posix()
            delete_file_chunks(collection, rel)

    # Insert into ChromaDB in batches (ChromaDB has a batch limit of ~5461)
    BATCH_SIZE = 5000
    print("[pipeline] Inserting into ChromaDB...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        batch_emb = embeddings[i:i + BATCH_SIZE]
        add_chunks(
            collection,
            ids=[c["location"] for c in batch],
            documents=[c["code"] for c in batch],
            embeddings=batch_emb,
            metadatas=[
                {
                    "filename": c["filename"],
                    "location": c["location"],
                    "feature": c["feature"],
                    "layer": c["layer"],
                    "file_type": c["file_type"],
                    "mtime": file_mtimes.get(c["filename"], "0"),
                }
                for c in batch
            ],
        )
    print(f"[pipeline] Inserted {len(all_chunks)} chunks.")

    # Build import graph edges (returned, not stored in DB)
    edge_count = sum(len(v) for v in file_imports.values())
    print(f"[pipeline] Import edges: {edge_count}")

    return {
        "files": len(files),
        "chunks": len(all_chunks),
        "edges": edge_count,
        "language": lang_name,
        "import_edges": file_imports,
    }


def index_files(
    collection: chromadb.Collection,
    file_paths: list[Path],
    config: SourcefireConfig,
    profile: LanguageProfile | None,
) -> dict[str, list[str]]:
    """Index specific files (for incremental re-indexing).

    Deletes existing chunks for each file, then re-chunks, embeds, and inserts.
    Returns the import map for updated files.
    """
    codebase_path = config.project_dir
    file_imports: dict[str, list[str]] = {}
    all_chunks: list[dict[str, Any]] = []

    for file_path in file_paths:
        rel = file_path.relative_to(codebase_path).as_posix()

        # Delete old chunks
        delete_file_chunks(collection, rel)

        # Re-chunk
        chunks = _chunks_for_file(file_path, codebase_path, profile, chunk_size=config.chunk_size)
        all_chunks.extend(chunks)

        # Collect imports
        file_profile = get_profile_for_extension(file_path.suffix) or profile
        if file_profile and file_path.suffix in file_profile.file_extensions:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            meta = extract_metadata(source, rel, file_profile)
            if meta.get("imports"):
                file_imports[rel] = meta["imports"]

    if all_chunks:
        texts = [c["code"] for c in all_chunks]
        embeddings = embed_batch(texts)

        BATCH_SIZE = 5000
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            batch_emb = embeddings[i:i + BATCH_SIZE]
            add_chunks(
                collection,
                ids=[c["location"] for c in batch],
                documents=[c["code"] for c in batch],
                embeddings=batch_emb,
                metadatas=[
                    {
                        "filename": c["filename"],
                        "location": c["location"],
                        "feature": c["feature"],
                        "layer": c["layer"],
                        "file_type": c["file_type"],
                    }
                    for c in batch
                ],
            )

    return file_imports
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from sourcefire.indexer.pipeline import run_indexing, index_files; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sourcefire/indexer/pipeline.py
git commit -m "refactor: replace psycopg with ChromaDB in indexing pipeline"
```

---

## Task 5: Update Retriever — Search and Graph

Replace SQL-based search with ChromaDB queries. Add JSON persistence to ImportGraph.

**Files:**
- Modify: `sourcefire/retriever/search.py`
- Modify: `sourcefire/retriever/graph.py`

- [ ] **Step 1: Rewrite search.py**

Replace `semantic_search` and `get_chunks_by_filenames` to use ChromaDB async wrappers.

```python
"""Vector search retriever for Sourcefire.

Provides:
- parse_file_references: extract file references from stack traces and error
  messages, driven by language profile patterns.
- semantic_search: cosine similarity search against ChromaDB.
- get_chunks_by_filenames: retrieve all chunks for a specific set of files.
"""

from __future__ import annotations

import re
from typing import Any

from sourcefire.db import async_query_similar, async_get_chunks_by_files


# ---------------------------------------------------------------------------
# Stack trace / error message parsing
# ---------------------------------------------------------------------------

_COMPILED_FILE_REF_PATTERNS: dict[str, list[re.Pattern]] = {}


def parse_file_references(text: str, file_ref_patterns: list[str] | None = None) -> list[dict[str, Any]]:
    """Extract file references from stack traces and error messages."""
    if not file_ref_patterns:
        file_ref_patterns = [r"\b([\w./\\-]+\.\w+)(?::(\d+))?"]

    cache_key = str(file_ref_patterns)
    if cache_key not in _COMPILED_FILE_REF_PATTERNS:
        _COMPILED_FILE_REF_PATTERNS[cache_key] = [re.compile(p) for p in file_ref_patterns]

    compiled = _COMPILED_FILE_REF_PATTERNS[cache_key]

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for regex in compiled:
        for m in regex.finditer(text):
            raw_path = m.group(1) if m.group(1) else ""
            if not raw_path:
                continue
            line = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else 0
            key = (raw_path, line)
            if key not in seen:
                seen.add(key)
                results.append({"file": raw_path, "line": line})

    return results


# ---------------------------------------------------------------------------
# ChromaDB search functions
# ---------------------------------------------------------------------------


async def semantic_search(
    collection: Any,
    query_vector: list[float],
    top_k: int = 8,
    threshold: float = 0.3,
    feature: str | None = None,
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cosine similarity search against ChromaDB."""
    where: dict | None = None

    if feature and filenames:
        where = {"$and": [{"feature": feature}, {"filename": {"$in": filenames}}]}
    elif feature:
        where = {"feature": feature}
    elif filenames:
        where = {"filename": {"$in": filenames}}

    rows = await async_query_similar(collection, query_vector, n_results=top_k, where=where)

    # Filter by threshold
    return [r for r in rows if r.get("relevance", 0) >= threshold]


async def get_chunks_by_filenames(
    collection: Any,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Retrieve all chunks for the given file paths."""
    return await async_get_chunks_by_files(collection, filenames)
```

- [ ] **Step 2: Add JSON persistence and remove_file to graph.py**

Add `to_json`, `from_json`, `save`, `load`, and `remove_file` methods to `ImportGraph`:

Add these methods to the existing `ImportGraph` class:

```python
    # ------------------------------------------------------------------
    # File removal (for incremental re-index)
    # ------------------------------------------------------------------

    def remove_file(self, file_path: str) -> None:
        """Remove all edges involving *file_path*."""
        # Remove forward edges
        if file_path in self._forward:
            for target in self._forward[file_path]:
                self._reverse[target].discard(file_path)
            del self._forward[file_path]
        # Remove reverse edges
        if file_path in self._reverse:
            for source in self._reverse[file_path]:
                self._forward[source].discard(file_path)
            del self._reverse[file_path]

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON storage."""
        edges = []
        for source, targets in self._forward.items():
            for target in targets:
                edges.append({"source": source, "target": target})
        return {"edges": edges}

    @classmethod
    def from_dict(cls, data: dict, external_prefixes: tuple[str, ...] = ()) -> "ImportGraph":
        """Deserialize from a dict."""
        graph = cls(external_prefixes=external_prefixes)
        for edge in data.get("edges", []):
            graph.add_edge(edge["source"], edge["target"])
        return graph

    def save(self, path: Path) -> None:
        """Save graph to a JSON file."""
        import json
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, external_prefixes: tuple[str, ...] = ()) -> "ImportGraph":
        """Load graph from a JSON file."""
        import json
        if not path.is_file():
            return cls(external_prefixes=external_prefixes)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, external_prefixes=external_prefixes)
```

Also add `from pathlib import Path` to the imports at the top.

- [ ] **Step 3: Verify**

```bash
python -c "from sourcefire.retriever.search import semantic_search, parse_file_references; print('OK')"
python -c "from sourcefire.retriever.graph import ImportGraph; g = ImportGraph(); g.add_edge('a','b'); print(g.to_dict()); g.remove_file('a'); print(g.to_dict())"
```

- [ ] **Step 4: Commit**

```bash
git add sourcefire/retriever/search.py sourcefire/retriever/graph.py
git commit -m "refactor: ChromaDB-backed search, JSON-persisted import graph"
```

---

## Task 6: Update RAG Chain and Prompts

Replace `pool` with `collection` throughout rag_chain.py. Replace `CODEBASE_PATH` with a `project_dir` parameter. Fix prompts.py path resolution.

**Files:**
- Modify: `sourcefire/chain/rag_chain.py`
- Modify: `sourcefire/chain/prompts.py`

- [ ] **Step 1: Fix prompts.py path resolution**

Replace the relative path traversal with `importlib.resources`:

```python
# Old:
_SYSTEM_MD_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "system.md"

# New:
from importlib.resources import files as _resource_files
_SYSTEM_MD_PATH = Path(str(_resource_files("sourcefire") / "prompts" / "system.md"))
```

- [ ] **Step 2: Update rag_chain.py**

The changes are pervasive but mechanical:

1. Replace `from src.config import CODEBASE_PATH` with accepting `project_dir: Path` as a parameter
2. Replace `pool` parameter with `collection` (ChromaDB) in all functions:
   - `_retrieve_debug(collection, ...)`
   - `_retrieve_feature(collection, ...)`
   - `_retrieve_explain(collection, ...)`
   - `retrieve_for_mode(collection, ...)`
   - `stream_rag_response(collection, ...)`
3. In `_load_static_context()`: accept `project_dir: Path` parameter instead of using global `CODEBASE_PATH`
4. In `_get_tools(graph, profile, collection, project_dir)`: replace every `CODEBASE_PATH` reference with `project_dir`, replace `pool` with `collection`
5. In `semantic_code_search` tool: replace `semantic_search(pool, ...)` with the synchronous `query_similar` from `sourcefire.db` (since tools run in threads)
6. In `find_similar_code` tool: same synchronous replacement
7. Update `GEMINI_API_KEY` import — get from config or env

The function signatures become:

```python
async def stream_rag_response(
    collection: Any,
    graph: ImportGraph,
    query: str,
    mode: str,
    model: str,
    history: list[dict[str, str]] | None = None,
    profile: LanguageProfile | None = None,
    project_dir: Path | None = None,
    gemini_api_key: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
```

And every internal call passes `collection` instead of `pool`.

- [ ] **Step 3: Verify imports compile**

```bash
python -c "from sourcefire.chain.rag_chain import stream_rag_response; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add sourcefire/chain/rag_chain.py sourcefire/chain/prompts.py
git commit -m "refactor: update RAG chain for ChromaDB collection and project_dir"
```

---

## Task 7: Update API Routes

Replace `pool` with `collection` and `CODEBASE_PATH` with `project_dir` in the routes module.

**Files:**
- Modify: `sourcefire/api/routes.py`
- Modify: `sourcefire/api/models.py`

- [ ] **Step 1: Update routes.py**

1. Replace `_pool` with `_collection`
2. Add `_project_dir: Path` module-level dependency
3. Update `init_dependencies(collection, graph, index_status, profile, project_dir, gemini_api_key)`
4. In `query()` route: pass `collection` and `project_dir` to `stream_rag_response`
5. In `sources()` route: use `_project_dir` instead of `CODEBASE_PATH`
6. Remove `from sourcefire.config import CODEBASE_PATH, GEMINI_API_KEY`

```python
_collection: Any = None
_graph: Any = None
_profile: Any = None
_project_dir: Path | None = None
_gemini_api_key: str = ""
_index_status: dict[str, Any] = { ... }

def init_dependencies(collection, graph, index_status, profile=None, project_dir=None, gemini_api_key=""):
    global _collection, _graph, _index_status, _profile, _project_dir, _gemini_api_key
    _collection = collection
    _graph = graph
    _index_status = index_status
    _profile = profile
    _project_dir = project_dir
    _gemini_api_key = gemini_api_key
```

Update `query()`:
```python
async for chunk in stream_rag_response(
    collection=_collection,
    graph=_graph,
    query=request.query,
    mode=request.mode,
    model=request.model,
    history=request.history,
    profile=_profile,
    project_dir=_project_dir,
    gemini_api_key=_gemini_api_key,
):
```

Update `sources()`:
```python
codebase_resolved = _project_dir.resolve()
full_path = (_project_dir / path).resolve()
```

- [ ] **Step 2: Update models.py**

Update the model literal type to match new defaults:

```python
model: Literal["gemini-2.5-flash", "gemini-2.5-pro"] = "gemini-2.5-flash"
```

- [ ] **Step 3: Verify**

```bash
python -c "from sourcefire.api.routes import init_dependencies, router; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add sourcefire/api/routes.py sourcefire/api/models.py
git commit -m "refactor: update API routes for ChromaDB and project_dir injection"
```

---

## Task 8: Auto-Init Module

Create `sourcefire/init.py` — handles first-run detection, LLM-powered config generation, and `.sourcefire/` setup.

**Files:**
- Create: `sourcefire/init.py`

- [ ] **Step 1: Write sourcefire/init.py**

```python
"""Auto-initialization for Sourcefire — creates .sourcefire/ with LLM-generated config."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from sourcefire.config import SourcefireConfig, default_config, save_config
from sourcefire.indexer.language_profiles import get_profile


def scan_file_tree(project_dir: Path, max_files: int = 5000) -> str:
    """Scan the project directory and return a text representation of the file tree."""
    skip_dirs = {
        ".git", "node_modules", "__pycache__", "build", "dist", "target",
        ".dart_tool", ".next", "venv", ".venv", ".idea", ".vs", ".sourcefire",
        ".tox", "eggs", "*.egg-info",
    }

    lines: list[str] = []
    file_count = 0

    for root, dirs, files in os.walk(project_dir):
        # Skip hidden/build directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        rel_root = Path(root).relative_to(project_dir).as_posix()
        if rel_root == ".":
            rel_root = ""

        for f in sorted(files):
            if file_count >= max_files:
                break
            rel_path = f"{rel_root}/{f}" if rel_root else f
            lines.append(rel_path)
            file_count += 1

        if file_count >= max_files:
            break

    return "\n".join(lines)


def _generate_patterns_via_llm(file_tree: str, api_key: str) -> dict[str, list[str]] | None:
    """Ask the LLM to generate include/exclude patterns from the file tree.

    Returns {"include": [...], "exclude": [...]} or None on failure.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
        )

        prompt = (
            "Given this project file tree, determine which files are source code worth "
            "indexing for a code RAG system. Respond with ONLY a TOML code block containing "
            "two arrays: `include` (glob patterns for source files, configs, and docs) and "
            "`exclude` (glob patterns for build artifacts, dependencies, generated files, "
            "and non-code assets). Be comprehensive but conservative.\n\n"
            "Always include these in exclude: .git/**, .sourcefire/**\n\n"
            f"```\n{file_tree}\n```"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)

        # Extract TOML block
        match = re.search(r"```(?:toml)?\s*\n(.*?)\n```", text, re.DOTALL)
        if not match:
            return None

        toml_str = match.group(1)
        data = tomllib.loads(toml_str)

        include = data.get("include", [])
        exclude = data.get("exclude", [])

        if isinstance(include, list) and isinstance(exclude, list):
            # Ensure .sourcefire/** is always excluded
            if ".sourcefire/**" not in exclude:
                exclude.append(".sourcefire/**")
            return {"include": include, "exclude": exclude}

        return None

    except Exception as exc:
        print(f"[init] LLM config generation failed: {exc}")
        return None


def _fallback_patterns(project_dir: Path, language_override: str | None = None) -> dict[str, list[str]]:
    """Generate patterns from language profile when LLM is unavailable."""
    profile = get_profile(project_dir, language_override)

    if profile:
        include = list(profile.include_patterns)
        exclude = list(profile.exclude_patterns)
    else:
        include = ["**/*"]
        exclude = []

    # Always add standard excludes
    for pat in [".git/**", ".sourcefire/**", "node_modules/**", "__pycache__/**",
                "*.pyc", ".venv/**", "venv/**", "dist/**", "build/**"]:
        if pat not in exclude:
            exclude.append(pat)

    # Always include docs
    for pat in ["README.md", "CLAUDE.md"]:
        if pat not in include:
            include.append(pat)

    return {"include": include, "exclude": exclude}


def auto_init(
    project_dir: Path,
    sourcefire_dir: Path | None = None,
    api_key: str = "",
    language_override: str | None = None,
) -> SourcefireConfig:
    """Initialize .sourcefire/ directory with LLM-generated config.

    Args:
        project_dir: The project root directory.
        sourcefire_dir: Path to .sourcefire/ (default: project_dir / ".sourcefire").
        api_key: Gemini API key for LLM config generation.
        language_override: Force a specific language.

    Returns:
        The generated SourcefireConfig.
    """
    if sourcefire_dir is None:
        sourcefire_dir = project_dir / ".sourcefire"

    print(f"[init] Initializing Sourcefire for: {project_dir.name}")

    # Create .sourcefire/ directory
    sourcefire_dir.mkdir(parents=True, exist_ok=True)

    # Scan file tree
    print("[init] Scanning project structure...")
    file_tree = scan_file_tree(project_dir)

    # Try LLM-powered pattern generation
    patterns: dict[str, list[str]] | None = None
    if api_key:
        print("[init] Generating config via LLM...")
        patterns = _generate_patterns_via_llm(file_tree, api_key)

    if patterns:
        print(f"[init] LLM generated {len(patterns['include'])} include, {len(patterns['exclude'])} exclude patterns.")
    else:
        print("[init] Using language-profile defaults for patterns.")
        patterns = _fallback_patterns(project_dir, language_override)

    # Detect language
    profile = get_profile(project_dir, language_override)
    language = profile.language if profile else "auto"

    # Build config
    config = default_config(project_dir)
    config.sourcefire_dir = sourcefire_dir
    config.include = patterns["include"]
    config.exclude = patterns["exclude"]
    config.language = language

    # Save
    save_config(config)
    print(f"[init] Config written to: {config.config_path}")

    # Print gitignore suggestion
    print("\nTip: Add to your .gitignore:")
    print("  .sourcefire/chroma/")
    print("  .sourcefire/graph.json")
    print("  .sourcefire/.lock\n")

    return config


def reinit_patterns(
    config: SourcefireConfig,
    api_key: str = "",
) -> SourcefireConfig:
    """Regenerate only the [indexer] include/exclude patterns, preserving other config.

    Used by `sourcefire --reinit`.
    """
    print(f"[init] Regenerating patterns for: {config.project_dir.name}")

    file_tree = scan_file_tree(config.project_dir)

    patterns: dict[str, list[str]] | None = None
    if api_key:
        print("[init] Generating patterns via LLM...")
        patterns = _generate_patterns_via_llm(file_tree, api_key)

    if patterns:
        print(f"[init] LLM generated {len(patterns['include'])} include, {len(patterns['exclude'])} exclude patterns.")
    else:
        print("[init] Using language-profile defaults.")
        language_override = config.language if config.language != "auto" else None
        patterns = _fallback_patterns(config.project_dir, language_override)

    # Only update indexer patterns — preserve all other user-edited config
    config.include = patterns["include"]
    config.exclude = patterns["exclude"]

    save_config(config)
    print(f"[init] Updated patterns in: {config.config_path}")

    return config
```

- [ ] **Step 2: Verify**

```bash
python -c "from sourcefire.init import auto_init, reinit_patterns, scan_file_tree; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sourcefire/init.py
git commit -m "feat: LLM-powered auto-init for .sourcefire/ directory"
```

---

## Task 9: File Watcher Module

Create `sourcefire/watcher.py` — watches for file changes and triggers incremental re-indexing.

**Files:**
- Create: `sourcefire/watcher.py`

- [ ] **Step 1: Write sourcefire/watcher.py**

```python
"""File watcher for live incremental re-indexing."""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import Any

from watchfiles import awatch, Change

from sourcefire.config import SourcefireConfig
from sourcefire.db import delete_file_chunks
from sourcefire.indexer.language_profiles import LanguageProfile, get_profile
from sourcefire.indexer.pipeline import index_files
from sourcefire.retriever.graph import ImportGraph


def _should_watch(rel_path: str, config: SourcefireConfig) -> bool:
    """Return True if the file matches include patterns and not exclude patterns."""
    for pattern in config.exclude:
        if fnmatch.fnmatch(rel_path, pattern):
            return False

    if not config.include:
        return True

    for pattern in config.include:
        if fnmatch.fnmatch(rel_path, pattern):
            return True

    return False


async def watch_and_reindex(
    config: SourcefireConfig,
    collection: Any,
    graph: ImportGraph,
    profile: LanguageProfile | None,
) -> None:
    """Watch project directory and incrementally re-index changed files.

    Runs as a background asyncio task. Batches changes within a 1-second
    debounce window.
    """
    project_dir = config.project_dir

    print(f"[watcher] Watching {project_dir} for changes...")

    async for changes in awatch(
        project_dir,
        debounce=1000,  # 1 second debounce
        recursive=True,
        step=200,  # check every 200ms
    ):
        changed_files: list[Path] = []
        deleted_files: list[str] = []

        for change_type, path_str in changes:
            path = Path(path_str)
            try:
                rel = path.relative_to(project_dir).as_posix()
            except ValueError:
                continue

            if not _should_watch(rel, config):
                continue

            if change_type in (Change.added, Change.modified):
                if path.is_file():
                    changed_files.append(path)
            elif change_type == Change.deleted:
                deleted_files.append(rel)

        # Handle deletions
        for rel in deleted_files:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, delete_file_chunks, collection, rel)
                graph.remove_file(rel)
                print(f"[watcher] Removed: {rel}")
            except Exception as exc:
                print(f"[watcher] Error removing {rel}: {exc}")

        # Handle additions/modifications
        if changed_files:
            try:
                loop = asyncio.get_event_loop()
                file_imports = await loop.run_in_executor(
                    None, index_files, collection, changed_files, config, profile
                )

                # Update graph
                for file_path in changed_files:
                    rel = file_path.relative_to(project_dir).as_posix()
                    graph.remove_file(rel)

                for source_file, imports in file_imports.items():
                    for imp in imports:
                        graph.add_edge(source_file, ImportGraph._resolve_import(source_file, imp))

                rel_names = [p.relative_to(project_dir).as_posix() for p in changed_files]
                print(f"[watcher] Re-indexed: {', '.join(rel_names)}")
            except Exception as exc:
                print(f"[watcher] Error re-indexing: {exc}")
```

- [ ] **Step 2: Verify**

```bash
python -c "from sourcefire.watcher import watch_and_reindex; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sourcefire/watcher.py
git commit -m "feat: file watcher for live incremental re-indexing"
```

---

## Task 10: CLI Entry Point

Create `sourcefire/cli.py` — the main entry point that ties everything together: project discovery, lockfile, init, indexing, server launch, and graceful shutdown.

**Files:**
- Create: `sourcefire/cli.py`

- [ ] **Step 1: Write sourcefire/cli.py**

```python
"""Sourcefire CLI — single command entry point."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import signal
import sys
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sourcefire",
        description="Sourcefire — AI-powered codebase RAG from your terminal",
    )
    parser.add_argument("--port", type=int, default=None, help="Server port (default: from config or 8000)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--reinit", action="store_true", help="Regenerate .sourcefire/config.toml via LLM")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def discover_project() -> tuple[Path, Path]:
    """Walk up from cwd to find .sourcefire/, like git finds .git/.

    Returns (project_dir, sourcefire_dir).
    If not found, returns (cwd, cwd/.sourcefire).
    """
    current = Path.cwd().resolve()
    while True:
        candidate = current / ".sourcefire"
        if candidate.is_dir():
            return current, candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Not found — use cwd
    cwd = Path.cwd().resolve()
    return cwd, cwd / ".sourcefire"


def acquire_lock(lock_path: Path) -> int | None:
    """Acquire an exclusive file lock. Returns fd on success, None on failure."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (OSError, BlockingIOError):
        return None


def release_lock(fd: int, lock_path: Path) -> None:
    """Release the file lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def main() -> None:
    """Sourcefire CLI entry point."""
    args = parse_args()

    project_dir, sourcefire_dir = discover_project()

    # Acquire lock
    lock_fd = acquire_lock(sourcefire_dir / ".lock")
    if lock_fd is None:
        print("Error: Another sourcefire instance is already running for this project.")
        sys.exit(1)

    # Check for API key — prompt interactively if missing
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("No GEMINI_API_KEY found in environment.")
        api_key = input("Enter your Gemini API key: ").strip()
        if not api_key:
            print("Error: A Gemini API key is required to run Sourcefire.")
            release_lock(lock_fd, sourcefire_dir / ".lock")
            sys.exit(1)
        # Persist to .env in project root so it's available next time
        env_path = project_dir / ".env"
        with open(env_path, "a") as f:
            f.write(f"\nGEMINI_API_KEY={api_key}\n")
        os.environ["GEMINI_API_KEY"] = api_key
        print(f"API key saved to {env_path}")

    # Auto-init or reinit
    needs_init = not sourcefire_dir.exists() or not (sourcefire_dir / "config.toml").exists()

    if needs_init:
        from sourcefire.init import auto_init
        config = auto_init(
            project_dir=project_dir,
            sourcefire_dir=sourcefire_dir,
            api_key=api_key,
        )
    elif args.reinit:
        # Reinit: regenerate only [indexer] patterns, preserve other user edits
        from sourcefire.config import load_config
        from sourcefire.init import reinit_patterns
        config = load_config(project_dir, sourcefire_dir)
        config = reinit_patterns(config, api_key=api_key)
    else:
        from sourcefire.config import load_config
        config = load_config(project_dir, sourcefire_dir)

    # Override port from CLI
    if args.port:
        config.port = args.port

    # Store state for lifespan access
    _app_state["config"] = config
    _app_state["project_dir"] = project_dir
    _app_state["sourcefire_dir"] = sourcefire_dir
    _app_state["api_key"] = api_key
    _app_state["args"] = args
    _app_state["lock_fd"] = lock_fd

    # Run server
    try:
        uvicorn.run(
            "sourcefire.cli:app",
            host=config.host,
            port=config.port,
            reload=False,
            log_level="info" if args.verbose else "warning",
        )
    finally:
        release_lock(lock_fd, sourcefire_dir / ".lock")


# ---------------------------------------------------------------------------
# App state (shared between main() and lifespan)
# ---------------------------------------------------------------------------

_app_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: index codebase, build graph, start watcher."""
    config = _app_state["config"]
    project_dir = _app_state["project_dir"]
    sourcefire_dir = _app_state["sourcefire_dir"]
    api_key = _app_state["api_key"]
    args = _app_state["args"]

    from sourcefire.db import create_client, get_collection, reset_collection
    from sourcefire.indexer.language_profiles import get_profile
    from sourcefire.indexer.pipeline import run_indexing
    from sourcefire.retriever.graph import ImportGraph
    from sourcefire.api.routes import init_dependencies
    from sourcefire.watcher import watch_and_reindex

    print(f"[sourcefire] Project: {project_dir.name}")
    print(f"[sourcefire] Config: {sourcefire_dir / 'config.toml'}")

    # Detect language
    language_override = config.language if config.language != "auto" else None
    profile = get_profile(project_dir, language_override)
    lang_name = profile.language if profile else "generic"
    print(f"[sourcefire] Language: {lang_name}")

    # Create ChromaDB client
    client = create_client(config.chroma_dir)
    collection = get_collection(client)

    # Determine if this is a first run (empty collection)
    existing_count = collection.count()
    is_first_run = existing_count == 0

    # Run indexing
    if is_first_run:
        print("[sourcefire] First run — full index...")
        stats = run_indexing(collection, config, client=client, full=True)
    else:
        print("[sourcefire] Re-indexing changed files...")
        stats = run_indexing(collection, config, client=client, full=False)

    print(f"[sourcefire] Indexed: {stats['files']} files, {stats['chunks']} chunks")

    # Build import graph
    external_prefixes = profile.external_import_prefixes if profile else ()
    graph = ImportGraph(external_prefixes=external_prefixes)

    # Load from file or build from index stats
    import_edges = stats.get("import_edges", {})
    if import_edges:
        for source_file, imports in import_edges.items():
            for imp in imports:
                resolved = ImportGraph._resolve_import(source_file, imp)
                graph.add_edge(source_file, resolved)
    elif config.graph_path.is_file():
        graph = ImportGraph.load(config.graph_path, external_prefixes=external_prefixes)

    print(f"[sourcefire] Import graph: {graph.node_count} nodes")

    # Build index status
    index_status = {
        "files_indexed": stats.get("files", 0),
        "last_indexed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_status": "ready",
        "language": lang_name,
    }

    # Inject dependencies into routes
    init_dependencies(collection, graph, index_status, profile, project_dir, api_key)

    # Start file watcher
    watcher_task = asyncio.create_task(
        watch_and_reindex(config, collection, graph, profile)
    )

    # Open browser
    url = f"http://{config.host}:{config.port}"
    print(f"[sourcefire] Ready — {url}")
    if not args.no_open:
        webbrowser.open(url)

    yield

    # Shutdown
    print("[sourcefire] Shutting down...")
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass

    # Save graph
    graph.save(config.graph_path)
    print("[sourcefire] Graph saved.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

from importlib.resources import files as _resource_files

_static_dir = str(Path(_resource_files("sourcefire")) / "static")

app = FastAPI(
    title="Sourcefire",
    description="AI-powered codebase RAG. Created by Athar Wani.",
    version="0.2.0",
    lifespan=lifespan,
)

from sourcefire.api.routes import router  # noqa: E402

app.include_router(router)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(os.path.join(_static_dir, "index.html"))
```

- [ ] **Step 2: Verify it parses**

```bash
python -c "from sourcefire.cli import main, discover_project; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sourcefire/cli.py
git commit -m "feat: sourcefire CLI entry point with auto-init, indexing, watcher, and server"
```

---

## ~~Task 11: Update embeddings.py Import~~ (MERGED INTO TASK 1)

This task is handled as part of Task 1 Step 3 — all `from src.` imports including `embeddings.py` and `metadata.py` are fixed in the same step as the package rename to prevent `ImportError` in subsequent tasks.

---

## Task 12: Clean Up and Integration Test

Remove old files, install the package, and verify the full flow works.

**Files:**
- Delete: `src/` (if any remnants)
- Modify: `.gitignore` (add `.sourcefire/chroma/`, `.sourcefire/graph.json`, `.sourcefire/.lock`)

- [ ] **Step 1: Update .gitignore**

Add these lines:

```
.sourcefire/chroma/
.sourcefire/graph.json
.sourcefire/.lock
```

- [ ] **Step 2: Install the package in dev mode**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 3: Verify the CLI is available**

```bash
which sourcefire
sourcefire --help
```

Expected output: help text with `--port`, `--no-open`, `--reinit`, `--verbose` flags.

- [ ] **Step 4: Test in a sample project**

```bash
cd /tmp
mkdir test-project && cd test-project
echo 'def hello(): return "world"' > main.py
echo 'GEMINI_API_KEY=...' > .env  # or export it
sourcefire --no-open
```

Verify:
- `.sourcefire/` directory is created
- `.sourcefire/config.toml` has LLM-generated patterns
- `.sourcefire/chroma/` has data
- Server starts on port 8000
- Ctrl+C gracefully shuts down (saves graph, releases lock)

- [ ] **Step 5: Test file watcher**

While the server is running, create a new file in the test project:

```bash
echo 'def new_func(): pass' > new_module.py
```

Check the terminal — should see `[watcher] Re-indexed: new_module.py`.

- [ ] **Step 6: Test lockfile**

In another terminal, try running `sourcefire` in the same directory. Should see:

```
Error: Another sourcefire instance is already running for this project.
```

- [ ] **Step 7: Test subdirectory discovery**

```bash
cd /tmp/test-project/some/nested/dir
sourcefire --no-open
```

Should find `.sourcefire/` in `/tmp/test-project/` and use it.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: sourcefire CLI tool — complete integration"
```

---

## Task Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Package rename `src/` -> `sourcefire/`, fix ALL imports, pyproject.toml | — |
| 2 | TOML-based config module | 1 |
| 3 | ChromaDB wrapper (`db.py`) | 1 |
| 4 | Rewrite indexing pipeline for ChromaDB (full + incremental) | 2, 3 |
| 5 | Update retriever (search + graph) | 3 |
| 6 | Update RAG chain and prompts | 4, 5 |
| 7 | Update API routes | 6 |
| 8 | Auto-init module (+ `reinit_patterns` for `--reinit`) | 2 |
| 9 | File watcher module | 3, 4 |
| 10 | CLI entry point (with interactive API key prompt) | All above |
| ~~11~~ | ~~Merged into Task 1~~ | — |
| 12 | Clean up and integration test | All above |
