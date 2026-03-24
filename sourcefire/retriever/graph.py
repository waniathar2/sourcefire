"""Bidirectional import graph for source files."""

from __future__ import annotations

import json
import posixpath
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import ClassVar


class ImportGraph:
    """Directed graph that tracks import relationships between source files.

    Edges are stored in both forward (imports) and reverse (importers)
    directions so either direction can be queried in O(1).
    """

    # Default external schemes — overridden at construction time by the profile
    _external_prefixes: tuple[str, ...] = ()

    def __init__(self, external_prefixes: tuple[str, ...] = ()) -> None:
        self._forward: defaultdict[str, set[str]] = defaultdict(set)
        self._reverse: defaultdict[str, set[str]] = defaultdict(set)
        self._external_prefixes = external_prefixes

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str) -> None:
        """Record that *source* imports *target*."""
        self._forward[source].add(target)
        self._reverse[target].add(source)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_imports(self, file_path: str) -> list[str]:
        """Return files that *file_path* imports (forward edges)."""
        return list(self._forward.get(file_path, []))

    def get_importers(self, file_path: str) -> list[str]:
        """Return files that import *file_path* (reverse edges)."""
        return list(self._reverse.get(file_path, []))

    def get_neighbors(self, file_path: str, hops: int = 1) -> list[str]:
        """Return all files reachable from *file_path* within *hops* steps.

        Traversal goes in *both* directions (imports and importers).
        The seed file itself is excluded from the result.
        """
        visited: set[str] = {file_path}
        queue: deque[tuple[str, int]] = deque([(file_path, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in (*self._forward.get(current, ()), *self._reverse.get(current, ())):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        visited.discard(file_path)
        return list(visited)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of unique nodes (files) in the graph."""
        nodes: set[str] = set(self._forward.keys()) | set(self._reverse.keys())
        return len(nodes)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_import_map(
        cls,
        file_imports: dict[str, list[str]],
        base_dir: str = "",
        external_prefixes: tuple[str, ...] = (),
    ) -> "ImportGraph":
        """Build an ImportGraph from a mapping of ``{file: [import_strings]}``.

        - Imports starting with any prefix in *external_prefixes* are skipped.
        - Relative imports (``../foo``, ``./bar``) are resolved relative to
          the importing file's directory.

        Args:
            file_imports: Mapping from source file path to its raw import list.
            base_dir: Optional prefix (unused, kept for forward-compatibility).
            external_prefixes: Prefixes that indicate external packages.
        """
        graph = cls(external_prefixes=external_prefixes)
        for source_file, imports in file_imports.items():
            for raw_import in imports:
                if external_prefixes and any(raw_import.startswith(scheme) for scheme in external_prefixes):
                    continue
                resolved = cls._resolve_import(source_file, raw_import)
                graph.add_edge(source_file, resolved)
        return graph

    @staticmethod
    def _resolve_import(source_file: str, relative_import: str) -> str:
        """Resolve *relative_import* relative to *source_file*'s directory."""
        source_dir = str(PurePosixPath(source_file).parent)
        joined = posixpath.join(source_dir, relative_import)
        return posixpath.normpath(joined)

    # ------------------------------------------------------------------
    # File removal (for incremental re-index)
    # ------------------------------------------------------------------

    def remove_file(self, file_path: str) -> None:
        """Remove all edges involving *file_path*."""
        if file_path in self._forward:
            for target in self._forward[file_path]:
                self._reverse[target].discard(file_path)
            del self._forward[file_path]
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
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, external_prefixes: tuple[str, ...] = ()) -> "ImportGraph":
        """Load graph from a JSON file."""
        if not path.is_file():
            return cls(external_prefixes=external_prefixes)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, external_prefixes=external_prefixes)
