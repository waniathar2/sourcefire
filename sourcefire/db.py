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
        pass
    return get_collection(client)


# ---------------------------------------------------------------------------
# Sync operations
# ---------------------------------------------------------------------------


def add_chunks(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, str]],
) -> None:
    """Add chunks to the collection."""
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


def get_indexed_files_and_mtimes(collection: chromadb.Collection) -> tuple[set[str], dict[str, float]]:
    """Return (set of filenames, {filename: mtime}) for all indexed chunks.

    Uses pagination to avoid loading the entire collection into memory at once.
    """
    files: set[str] = set()
    mtimes: dict[str, float] = {}
    batch_size = 10000
    offset = 0

    total = collection.count()
    if total == 0:
        return files, mtimes

    while offset < total:
        results = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        if not results["metadatas"]:
            break
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                fname = meta["filename"]
                files.add(fname)
                if "mtime" in meta:
                    try:
                        stored = float(meta["mtime"])
                        # Keep the max mtime per file (chunks share the same mtime)
                        if fname not in mtimes or stored > mtimes[fname]:
                            mtimes[fname] = stored
                    except (ValueError, TypeError):
                        pass
        offset += batch_size

    return files, mtimes


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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(query_similar, collection, query_embedding, n_results, where)
    )


async def async_get_chunks_by_files(
    collection: chromadb.Collection,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Async wrapper for get_chunks_by_files."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(get_chunks_by_files, collection, filenames)
    )


async def async_delete_file_chunks(
    collection: chromadb.Collection,
    filename: str,
) -> None:
    """Async wrapper for delete_file_chunks."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, partial(delete_file_chunks, collection, filename)
    )
