"""Bidirectional import graph for Dart/Flutter source files."""

from __future__ import annotations

import posixpath
from collections import defaultdict, deque
from pathlib import PurePosixPath
from typing import ClassVar


class ImportGraph:
    """Directed graph that tracks import relationships between source files.

    Edges are stored in both forward (imports) and reverse (importers)
    directions so either direction can be queried in O(1).
    """

    # Import schemes that refer to external packages and should not be
    # resolved as local file paths.
    _EXTERNAL_SCHEMES: ClassVar[tuple[str, ...]] = ("package:", "dart:")

    def __init__(self) -> None:
        self._forward: defaultdict[str, set[str]] = defaultdict(set)
        self._reverse: defaultdict[str, set[str]] = defaultdict(set)

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
    ) -> "ImportGraph":
        """Build an ImportGraph from a mapping of ``{file: [import_strings]}``.

        - ``package:`` and ``dart:`` imports are skipped (external).
        - Relative imports (``../foo.dart``, ``./bar.dart``) are resolved
          relative to the importing file's directory.

        Args:
            file_imports: Mapping from source file path to its raw import list.
            base_dir: Optional prefix that was stripped from file paths when
                the metadata was collected.  Not currently used for path
                arithmetic but kept for forward-compatibility.
        """
        graph = cls()
        for source_file, imports in file_imports.items():
            for raw_import in imports:
                if any(raw_import.startswith(scheme) for scheme in cls._EXTERNAL_SCHEMES):
                    continue
                resolved = cls._resolve_import(source_file, raw_import)
                graph.add_edge(source_file, resolved)
        return graph

    @staticmethod
    def _resolve_import(source_file: str, relative_import: str) -> str:
        """Resolve *relative_import* relative to *source_file*'s directory.

        Uses ``pathlib.PurePosixPath`` so the logic is platform-independent.

        Examples::

            _resolve_import("lib/a.dart", "../b.dart")  # -> "lib/b.dart"
            _resolve_import("lib/sub/a.dart", "./c.dart")  # -> "lib/sub/c.dart"
        """
        source_dir = str(PurePosixPath(source_file).parent)
        joined = posixpath.join(source_dir, relative_import)
        # posixpath.normpath collapses ".." and "." segments correctly.
        return posixpath.normpath(joined)
