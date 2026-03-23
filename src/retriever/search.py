"""Vector search retriever for the Cravv Observatory RAG system.

Provides:
- parse_file_references: extract lib/...dart file references from Flutter
  stack traces and error messages.
- build_metadata_filter: build a SQL WHERE clause fragment for feature/filename
  filtering.
- semantic_search: cosine similarity search against the code_embeddings table.
- get_chunks_by_filenames: retrieve all chunks for a specific set of files.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import RELEVANCE_THRESHOLD, TOP_K
from src.indexer.pipeline import TABLE_NAME

# ---------------------------------------------------------------------------
# Stack trace / error message parsing
# ---------------------------------------------------------------------------

# Matches Flutter stack trace lines, e.g.:
#   package:cravv/features/auth/presentation/providers/auth_notifier.dart:44:5
# Capture group 1: path after "package:cravv/" (does NOT include "lib/")
# Capture group 2: line number
_PACKAGE_RE = re.compile(
    r"package:[^/]+/((?:features|core|lib)[^\s:)]+\.dart)"
    r":(\d+)"
)

# Matches direct lib/-prefixed references, e.g.:
#   lib/core/network/dio_client.dart:18
# Capture group 1: full path including "lib/"
# Capture group 2: line number (optional — may not be present)
_LIB_RE = re.compile(
    r"\b(lib/[^\s:)]+\.dart)"
    r"(?::(\d+))?"
)


def parse_file_references(text: str) -> list[dict[str, Any]]:
    """Extract file references from Flutter stack traces and error messages.

    Returns a list of dicts, each with:
        file : str  — relative path starting with "lib/" (e.g. "lib/core/...")
        line : int  — line number (0 if not present in the source text)

    Deduplication is applied: the first occurrence of each (file, line) pair
    is kept; subsequent duplicates are dropped.
    """
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    # 1. Package-URI style (Flutter stack traces)
    for m in _PACKAGE_RE.finditer(text):
        raw_path = m.group(1)
        # If the path already starts with "lib/", keep it as-is.
        # Otherwise prepend "lib/".
        if raw_path.startswith("lib/"):
            file_path = raw_path
        else:
            file_path = "lib/" + raw_path
        line = int(m.group(2))
        key = (file_path, line)
        if key not in seen:
            seen.add(key)
            results.append({"file": file_path, "line": line})

    # 2. Direct lib/ references (error messages, comments, etc.)
    for m in _LIB_RE.finditer(text):
        file_path = m.group(1)
        line = int(m.group(2)) if m.group(2) else 0
        key = (file_path, line)
        if key not in seen:
            seen.add(key)
            results.append({"file": file_path, "line": line})

    return results


# ---------------------------------------------------------------------------
# Metadata filter builder
# ---------------------------------------------------------------------------


def build_metadata_filter(
    feature: str | None = None,
    filenames: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE clause fragment for optional metadata filters.

    Args:
        feature:   If provided, adds ``feature = %s`` to the clause.
        filenames: If provided, adds ``filename = ANY(%s)`` to the clause.

    Returns:
        A 2-tuple of (sql_fragment, params_list).  sql_fragment is an empty
        string when no filters are requested.
    """
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
    """Cosine similarity search against the code_embeddings table.

    Args:
        pool:         A psycopg ConnectionPool.
        query_vector: The embedding vector to search with (list of floats).
        top_k:        Maximum number of results to return.
        threshold:    Minimum cosine similarity (0–1) to include a result.
        feature:      Optional feature name filter.
        filenames:    Optional list of filenames to restrict the search to.

    Returns:
        List of dicts with keys: filename, location, code, feature, layer,
        file_type, relevance.
    """
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
    """Retrieve all chunks for the given file paths.

    Args:
        pool:      A psycopg ConnectionPool.
        filenames: List of relative file paths (e.g. "lib/core/router/app_router.dart").

    Returns:
        List of dicts with keys: filename, location, code, feature, layer, file_type.
    """
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
