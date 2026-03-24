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

    return [r for r in rows if r.get("relevance", 0) >= threshold]


async def get_chunks_by_filenames(
    collection: Any,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Retrieve all chunks for the given file paths."""
    return await async_get_chunks_by_files(collection, filenames)
