"""Full indexing pipeline for the Cravv Observatory RAG system.

Scans the Flutter codebase, chunks files (AST-aware for Dart, simple split for
others), embeds all chunks, and upserts them into the pgvector table. Also
populates the import_graph table for graph-augmented retrieval.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from psycopg_pool import ConnectionPool

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CODEBASE_PATH,
    EXCLUDED_PATTERNS,
    INCLUDED_PATTERNS,
)
from src.indexer.embeddings import embed_batch
from src.indexer.metadata import chunk_dart_file, extract_dart_metadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 384
TABLE_NAME: str = "code_embeddings"

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    location TEXT NOT NULL,
    code TEXT NOT NULL,
    feature TEXT DEFAULT '',
    layer TEXT DEFAULT '',
    file_type TEXT DEFAULT '',
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    UNIQUE(filename, location)
);

CREATE INDEX IF NOT EXISTS idx_code_embeddings_filename
    ON {TABLE_NAME} (filename);

CREATE INDEX IF NOT EXISTS idx_code_embeddings_feature
    ON {TABLE_NAME} (feature);

CREATE TABLE IF NOT EXISTS import_graph (
    source_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    PRIMARY KEY (source_path, target_path)
);
"""

# The ivfflat index is created separately because it requires enough rows to be
# present first (at least lists * 10 rows).  We attempt it after upsert and
# swallow errors gracefully.
_IVFFLAT_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_code_embeddings_embedding
    ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);
"""


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
# Non-Dart chunker (simple recursive text split)
# ---------------------------------------------------------------------------


def _chunk_non_dart(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Each successive chunk starts *chunk_overlap* characters before the end of
    the previous chunk, preserving context across boundaries.
    """
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


def _collect_files(codebase_path: Path) -> list[Path]:
    """Return all files under *codebase_path* that pass include/exclude filters."""
    matched: list[Path] = []

    for pattern in INCLUDED_PATTERNS:
        for file_path in codebase_path.glob(pattern):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(codebase_path).as_posix()
            if not _match_patterns(rel, EXCLUDED_PATTERNS):
                matched.append(file_path)

    # Deduplicate while preserving order (multiple patterns may match same file).
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
) -> list[dict[str, Any]]:
    """Return a list of chunk dicts for *file_path*.

    Each dict has:
        filename  : str — relative POSIX path (e.g. "lib/core/router/app_router.dart")
        location  : str — "<filename>:<chunk_index>"
        code      : str — chunk text
        feature   : str
        layer     : str
        file_type : str
    """
    rel = file_path.relative_to(codebase_path).as_posix()
    source = file_path.read_text(encoding="utf-8", errors="replace")

    if file_path.suffix == ".dart":
        raw_chunks = chunk_dart_file(source, rel, chunk_size=CHUNK_SIZE)
        chunks_out: list[dict[str, Any]] = []
        for idx, chunk in enumerate(raw_chunks):
            meta = chunk["metadata"]
            chunks_out.append(
                {
                    "filename": rel,
                    "location": f"{rel}:{idx}",
                    "code": chunk["text"],
                    "feature": meta.get("feature", ""),
                    "layer": meta.get("layer", ""),
                    "file_type": meta.get("file_type", ""),
                }
            )
        return chunks_out
    else:
        # Non-Dart: plain text split; derive metadata from path only.
        meta = extract_dart_metadata("", rel)  # path-based inference; source empty
        raw_texts = _chunk_non_dart(source)
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
# Database helpers
# ---------------------------------------------------------------------------


def _create_schema(pool: ConnectionPool) -> None:
    """Create tables and basic indexes (idempotent)."""
    with pool.connection() as conn:
        conn.execute(_SCHEMA_SQL)
        conn.commit()


def _try_create_ivfflat_index(pool: ConnectionPool) -> None:
    """Attempt to create the ivfflat index; log and continue on failure."""
    try:
        with pool.connection() as conn:
            conn.execute(_IVFFLAT_SQL)
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[pipeline] ivfflat index creation skipped (not enough rows?): {exc}"
        )


def _upsert_chunks(
    pool: ConnectionPool,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    """Upsert *chunks* + *embeddings* into the code_embeddings table."""
    upsert_sql = f"""
        INSERT INTO {TABLE_NAME}
            (filename, location, code, feature, layer, file_type, embedding)
        VALUES
            (%(filename)s, %(location)s, %(code)s,
             %(feature)s, %(layer)s, %(file_type)s,
             %(embedding)s::vector)
        ON CONFLICT (filename, location) DO UPDATE SET
            code      = EXCLUDED.code,
            feature   = EXCLUDED.feature,
            layer     = EXCLUDED.layer,
            file_type = EXCLUDED.file_type,
            embedding = EXCLUDED.embedding;
    """
    rows = [
        {**chunk, "embedding": str(emb)}
        for chunk, emb in zip(chunks, embeddings)
    ]
    with pool.connection() as conn:
        conn.executemany(upsert_sql, rows)
        conn.commit()


def _build_import_graph(
    pool: ConnectionPool,
    file_imports: dict[str, list[str]],
) -> int:
    """Populate the import_graph table from *file_imports*. Returns edge count."""
    from src.retriever.graph import ImportGraph

    graph = ImportGraph.from_import_map(file_imports)

    rows: list[dict[str, str]] = []
    for source, targets in graph._forward.items():  # noqa: SLF001
        for target in targets:
            rows.append({"source_path": source, "target_path": target})

    if not rows:
        return 0

    upsert_sql = """
        INSERT INTO import_graph (source_path, target_path)
        VALUES (%(source_path)s, %(target_path)s)
        ON CONFLICT DO NOTHING;
    """
    with pool.connection() as conn:
        conn.executemany(upsert_sql, rows)
        conn.commit()

    return len(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_indexing(pool: ConnectionPool) -> dict[str, int]:
    """Run the full indexing pipeline.

    Args:
        pool: An open psycopg ConnectionPool pointed at the target database.

    Returns:
        A stats dict with keys: ``files``, ``chunks``, ``edges``.
    """
    print(f"[pipeline] Scanning codebase at: {CODEBASE_PATH}")

    # 1. Ensure schema exists.
    _create_schema(pool)
    print("[pipeline] Schema ready.")

    # 2. Collect files.
    files = _collect_files(CODEBASE_PATH)
    print(f"[pipeline] Found {len(files)} files to index.")

    # 3. Produce chunks and collect import maps in one pass.
    all_chunks: list[dict[str, Any]] = []
    file_imports: dict[str, list[str]] = {}

    for file_path in files:
        rel = file_path.relative_to(CODEBASE_PATH).as_posix()
        chunks = _chunks_for_file(file_path, CODEBASE_PATH)
        all_chunks.extend(chunks)

        # Collect imports from Dart files for the graph table.
        if file_path.suffix == ".dart":
            source = file_path.read_text(encoding="utf-8", errors="replace")
            meta = extract_dart_metadata(source, rel)
            file_imports[rel] = meta.get("imports", [])

    print(f"[pipeline] Produced {len(all_chunks)} chunks.")

    if not all_chunks:
        return {"files": len(files), "chunks": 0, "edges": 0}

    # 4. Embed in one batch.
    print("[pipeline] Embedding chunks...")
    texts = [c["code"] for c in all_chunks]
    embeddings = embed_batch(texts)
    print("[pipeline] Embeddings done.")

    # 5. Upsert into pgvector table.
    print("[pipeline] Upserting into database...")
    _upsert_chunks(pool, all_chunks, embeddings)
    print(f"[pipeline] Upserted {len(all_chunks)} chunks.")

    # 6. Try to create the ivfflat ANN index (requires >= 200 rows).
    _try_create_ivfflat_index(pool)

    # 7. Build import graph table.
    print("[pipeline] Building import graph...")
    edge_count = _build_import_graph(pool, file_imports)
    print(f"[pipeline] Import graph: {edge_count} edges.")

    return {
        "files": len(files),
        "chunks": len(all_chunks),
        "edges": edge_count,
    }
