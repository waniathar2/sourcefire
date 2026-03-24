"""File watcher for live incremental re-indexing."""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import Any

from watchfiles import awatch, Change

from sourcefire.config import SourcefireConfig
from sourcefire.db import delete_file_chunks
from sourcefire.indexer.language_profiles import LanguageProfile
from sourcefire.indexer.pipeline import index_files
from sourcefire.retriever.graph import ImportGraph


# Always skip these regardless of config
_ALWAYS_EXCLUDE = (
    ".sourcefire/",
    ".git/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
)


def _should_watch(rel_path: str, config: SourcefireConfig) -> bool:
    """Return True if the file matches include patterns and not exclude patterns."""
    # Hard excludes — never watch these
    for prefix in _ALWAYS_EXCLUDE:
        if rel_path.startswith(prefix):
            return False

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
        debounce=1000,
        recursive=True,
        step=200,
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
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, delete_file_chunks, collection, rel)
                graph.remove_file(rel)
                print(f"[watcher] Removed: {rel}")
            except Exception as exc:
                print(f"[watcher] Error removing {rel}: {exc}")

        # Handle additions/modifications
        if changed_files:
            try:
                loop = asyncio.get_running_loop()
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
