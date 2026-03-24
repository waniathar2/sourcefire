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
from sourcefire.db import add_chunks, reset_collection, delete_file_chunks, get_indexed_files_and_mtimes
from sourcefire.indexer.embeddings import embed_batch
from sourcefire.indexer.language_profiles import LanguageProfile, get_profile, get_profile_for_extension
from sourcefire.indexer.metadata import chunk_source_file, extract_metadata


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

    # Guard against infinite loop if overlap >= size
    chunk_overlap = min(chunk_overlap, chunk_size - 1)

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
    include_patterns: list[str] = list(config.include) if config.include else []
    exclude_patterns: list[str] = list(config.exclude) if config.exclude else []

    # If no config patterns, fall back to profile
    if not include_patterns and profile:
        include_patterns = list(profile.include_patterns)
    if not exclude_patterns and profile:
        exclude_patterns = list(profile.exclude_patterns)

    # Always exclude .gitignore patterns
    exclude_patterns.extend(_parse_gitignore(codebase_path))

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
# Public entry points
# ---------------------------------------------------------------------------


def run_indexing(
    collection: chromadb.Collection,
    config: SourcefireConfig,
    client: chromadb.ClientAPI | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """Run the indexing pipeline.

    Args:
        collection: ChromaDB collection to write to.
        config: Sourcefire configuration.
        client: ChromaDB client (needed for full reset).
        full: If True, reset collection and re-index everything.
              If False, incremental — compare file mtimes and only
              re-index changed/new files, delete removed files.

    Returns:
        A stats dict with keys: files, chunks, edges, language, import_edges.
    """
    codebase_path = config.project_dir
    print(f"[pipeline] Scanning codebase at: {codebase_path}")

    language_override = config.language if config.language != "auto" else None
    profile = get_profile(codebase_path, language_override)
    lang_name = profile.language if profile else "generic"
    print(f"[pipeline] Detected language: {lang_name}")

    # Collect all files on disk
    all_disk_files = _collect_files(codebase_path, config, profile)
    print(f"[pipeline] Found {len(all_disk_files)} files to index.")

    if not all_disk_files:
        print("[pipeline] Error: No source files found matching the configured patterns.")
        print("Run `sourcefire --reinit` to regenerate patterns, or edit .sourcefire/config.toml manually.")
        return {
            "files": 0, "chunks": 0, "edges": 0,
            "language": lang_name, "import_edges": {},
        }

    # Determine which files to process
    if full and client:
        collection = reset_collection(client)
        print("[pipeline] Collection reset for full re-index.")
        files_to_index = all_disk_files
    elif not full:
        # Incremental: compare mtimes
        indexed_files, stored_mtimes = get_indexed_files_and_mtimes(collection)

        current_files: dict[str, Path] = {}
        for f in all_disk_files:
            rel = f.relative_to(codebase_path).as_posix()
            current_files[rel] = f

        # Find changed/new files
        changed: list[Path] = []
        for rel, f in current_files.items():
            stored_mtime = stored_mtimes.get(rel, 0.0)
            if rel not in indexed_files or f.stat().st_mtime > stored_mtime:
                changed.append(f)

        # Find deleted files
        deleted = indexed_files - set(current_files.keys())
        for rel in deleted:
            delete_file_chunks(collection, rel)

        if not changed and not deleted:
            print("[pipeline] Index is up to date.")
            return {
                "files": len(all_disk_files), "chunks": collection.count(), "edges": 0,
                "language": lang_name, "import_edges": {},
            }

        print(f"[pipeline] {len(changed)} changed, {len(deleted)} deleted files.")
        files_to_index = changed

        # Delete old chunks for changed files before re-inserting
        for f in changed:
            rel = f.relative_to(codebase_path).as_posix()
            delete_file_chunks(collection, rel)
    else:
        files_to_index = all_disk_files

    # Produce chunks and collect imports
    all_chunks: list[dict[str, Any]] = []
    file_imports: dict[str, list[str]] = {}

    for file_path in files_to_index:
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
            "files": len(all_disk_files), "chunks": 0, "edges": 0,
            "language": lang_name, "import_edges": file_imports,
        }

    # Embed
    print("[pipeline] Embedding chunks...")
    texts = [c["code"] for c in all_chunks]
    embeddings = embed_batch(texts)
    print("[pipeline] Embeddings done.")

    # Build mtime lookup
    file_mtimes: dict[str, str] = {}
    for file_path in files_to_index:
        rel = file_path.relative_to(codebase_path).as_posix()
        file_mtimes[rel] = str(file_path.stat().st_mtime)

    # Insert into ChromaDB in batches
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

    edge_count = sum(len(v) for v in file_imports.values())
    print(f"[pipeline] Import edges: {edge_count}")

    return {
        "files": len(all_disk_files),
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
    """Index specific files (for incremental re-indexing by the watcher).

    Deletes existing chunks for each file, then re-chunks, embeds, and inserts.
    Returns the import map for updated files.
    """
    codebase_path = config.project_dir
    file_imports: dict[str, list[str]] = {}
    all_chunks: list[dict[str, Any]] = []

    for file_path in file_paths:
        rel = file_path.relative_to(codebase_path).as_posix()
        delete_file_chunks(collection, rel)

        chunks = _chunks_for_file(file_path, codebase_path, profile, chunk_size=config.chunk_size)
        all_chunks.extend(chunks)

        file_profile = get_profile_for_extension(file_path.suffix) or profile
        if file_profile and file_path.suffix in file_profile.file_extensions:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            meta = extract_metadata(source, rel, file_profile)
            if meta.get("imports"):
                file_imports[rel] = meta["imports"]

    if all_chunks:
        texts = [c["code"] for c in all_chunks]
        embeddings = embed_batch(texts)

        file_mtimes: dict[str, str] = {}
        for file_path in file_paths:
            rel = file_path.relative_to(codebase_path).as_posix()
            file_mtimes[rel] = str(file_path.stat().st_mtime)

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
                        "mtime": file_mtimes.get(c["filename"], "0"),
                    }
                    for c in batch
                ],
            )

    return file_imports
