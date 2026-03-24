"""Vector search retriever for Sourcefire.

Provides:
- parse_file_references: extract file references from stack traces and error
  messages, driven by language profile patterns.
- build_metadata_filter: build a SQL WHERE clause fragment for feature/filename
  filtering.
- semantic_search: cosine similarity search against the code_embeddings table.
- get_chunks_by_filenames: retrieve all chunks for a specific set of files.
"""

from __future__ import annotations

import re
from typing import Any

from sourcefire.config import RELEVANCE_THRESHOLD, TOP_K
from sourcefire.indexer.pipeline import TABLE_NAME

# ---------------------------------------------------------------------------
# Stack trace / error message parsing
# ---------------------------------------------------------------------------

# Compiled regex cache (populated on first call per profile)
_COMPILED_FILE_REF_PATTERNS: dict[str, list[re.Pattern]] = {}


def parse_file_references(text: str, file_ref_patterns: list[str] | None = None) -> list[dict[str, Any]]:
    """Extract file references from stack traces and error messages.

    Args:
        text: The text to parse (stack trace, error message, or query).
        file_ref_patterns: Regex patterns from the language profile. Each pattern
            should have group(1) = file path, group(2) = optional line number.
            If None, a generic fallback is used.

    Returns a list of dicts, each with:
        file : str  — relative file path
        line : int  — line number (0 if not present)
    """
    if not file_ref_patterns:
        # Generic fallback: match any path-like reference with an extension
        file_ref_patterns = [r"\b([\w./\\-]+\.\w+)(?::(\d+))?"]

    # Get or compile patterns
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
# Metadata filter builder
# ---------------------------------------------------------------------------


def build_metadata_filter(
    feature: str | None = None,
    filenames: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE clause fragment for optional metadata filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if feature:
        clauses.append("feature = %s")
        params.append(feature)

    if filenames:
        clauses.append("filename = ANY(%s)")
        params.append(filenames)

    sql = " AND ".join(clauses)
    return sql, params


# ---------------------------------------------------------------------------
# Database search functions
# ---------------------------------------------------------------------------


async def semantic_search(
    pool: Any,
    query_vector: list[float],
    top_k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    feature: str | None = None,
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cosine similarity search against the code_embeddings table."""
    meta_sql, meta_params = build_metadata_filter(feature=feature, filenames=filenames)

    where_clause = "WHERE 1 - (embedding <=> %s::vector) >= %s"
    params: list[Any] = [str(query_vector), threshold]

    if meta_sql:
        where_clause += f" AND {meta_sql}"
        params.extend(meta_params)

    sql = f"""
        SELECT
            filename,
            location,
            code,
            feature,
            layer,
            file_type,
            1 - (embedding <=> %s::vector) AS relevance
        FROM {TABLE_NAME}
        {where_clause}
        ORDER BY relevance DESC
        LIMIT %s
    """
    # The vector is used twice: once in SELECT for the relevance score and
    # once in the WHERE clause.
    params = [str(query_vector), str(query_vector), threshold]
    if meta_sql:
        params.extend(meta_params)
    params.append(top_k)

    rows: list[dict[str, Any]] = []
    with pool.connection() as conn:
        cursor = conn.execute(sql, params)
        for row in cursor.fetchall():
            rows.append(
                {
                    "filename": row[0],
                    "location": row[1],
                    "code": row[2],
                    "feature": row[3],
                    "layer": row[4],
                    "file_type": row[5],
                    "relevance": float(row[6]),
                }
            )

    return rows


async def get_chunks_by_filenames(
    pool: Any,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Retrieve all chunks for the given file paths."""
    if not filenames:
        return []

    sql = f"""
        SELECT filename, location, code, feature, layer, file_type
        FROM {TABLE_NAME}
        WHERE filename = ANY(%s)
        ORDER BY filename, location
    """
    rows: list[dict[str, Any]] = []
    with pool.connection() as conn:
        cursor = conn.execute(sql, [filenames])
        for row in cursor.fetchall():
            rows.append(
                {
                    "filename": row[0],
                    "location": row[1],
                    "code": row[2],
                    "feature": row[3],
                    "layer": row[4],
                    "file_type": row[5],
                }
            )

    return rows
