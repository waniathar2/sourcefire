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


def get_indexed_files(collection: chromadb.Collection) -> set[str]:
    """Return set of all filenames currently in the collection."""
    results = collection.get(include=["metadatas"])
    files: set[str] = set()
    if results["metadatas"]:
        for meta in results["metadatas"]:
            if meta and "filename" in meta:
                files.add(meta["filename"])
    return files


def get_stored_mtimes(collection: chromadb.Collection) -> dict[str, float]:
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
