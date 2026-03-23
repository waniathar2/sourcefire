"""
Dart AST Metadata Extractor

Extracts structural metadata from Dart source files using tree-sitter (when available)
or regex fallback. The tree-sitter code paths exist for future compatibility but
tree-sitter-languages is not installed in the current environment, so regex is used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional tree-sitter import — guarded so regex fallback is used when
# tree-sitter-languages is not installed.
# ---------------------------------------------------------------------------
try:
    from tree_sitter_languages import get_language, get_parser  # type: ignore

    _DART_LANGUAGE = get_language("dart")
    _DART_PARSER = get_parser("dart")
except Exception:
    _DART_LANGUAGE = None
    _DART_PARSER = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_dart_metadata(source: str, file_path: str) -> dict[str, Any]:
    """Return a metadata dict for a Dart source file.

    Keys:
        imports   : list[str] — imported URIs
        exports   : list[str] — top-level class/enum/mixin/extension names
        layer     : str       — 'presentation' | 'domain' | 'data' | 'core' | 'unknown'
        feature   : str       — feature name, e.g. 'auth', or 'core' / 'unknown'
        file_type : str       — 'notifier' | 'screen' | 'model' | ... | 'unknown'
    """
    if _DART_PARSER is not None:
        imports = _extract_imports_tree_sitter(source)
        exports = _extract_exports_tree_sitter(source)
    else:
        imports = _extract_imports_regex(source)
        exports = _extract_exports_regex(source)

    return {
        "imports": imports,
        "exports": exports,
        "layer": _infer_layer(file_path),
        "feature": _infer_feature(file_path),
        "file_type": _infer_file_type(file_path),
    }


def chunk_dart_file(
    source: str,
    file_path: str,
    chunk_size: int = 1000,
) -> list[dict[str, Any]]:
    """Split a Dart file into chunks and attach metadata to each chunk.

    Splitting strategy:
    1. If tree-sitter is available, split at top-level declaration boundaries.
    2. Otherwise, use regex to split at class/enum/mixin/extension boundaries.
    3. If the source is shorter than chunk_size, return a single chunk.

    Each chunk dict has keys:
        text     : str  — the chunk text
        metadata : dict — output of extract_dart_metadata for the whole file
    """
    metadata = extract_dart_metadata(source, file_path)

    if _DART_PARSER is not None:
        raw_chunks = _chunk_tree_sitter(source, chunk_size)
    else:
        raw_chunks = _chunk_regex(source, chunk_size)

    return [{"text": text, "metadata": metadata} for text in raw_chunks]


# ---------------------------------------------------------------------------
# Tree-sitter implementations (only called when _DART_PARSER is not None)
# ---------------------------------------------------------------------------


def _extract_imports_tree_sitter(source: str) -> list[str]:
    """Extract import URIs using tree-sitter."""
    tree = _DART_PARSER.parse(source.encode())
    imports: list[str] = []
    _walk_for_imports(tree.root_node, imports)
    return imports


def _walk_for_imports(node: Any, imports: list[str]) -> None:
    if node.type == "import_specification":
        for child in node.children:
            if child.type == "string_literal":
                uri = child.text.decode().strip("'\"")
                imports.append(uri)
    for child in node.children:
        _walk_for_imports(child, imports)


def _extract_exports_tree_sitter(source: str) -> list[str]:
    """Extract top-level declaration names using tree-sitter."""
    tree = _DART_PARSER.parse(source.encode())
    exports: list[str] = []
    _walk_for_exports(tree.root_node, exports)
    return exports


def _walk_for_exports(node: Any, exports: list[str]) -> None:
    if node.type in (
        "class_definition",
        "enum_declaration",
        "mixin_declaration",
        "extension_declaration",
    ):
        for child in node.children:
            if child.type == "identifier":
                exports.append(child.text.decode())
                break
    for child in node.children:
        _walk_for_exports(child, exports)


def _chunk_tree_sitter(source: str, chunk_size: int) -> list[str]:
    """Split source at top-level declaration boundaries using tree-sitter."""
    tree = _DART_PARSER.parse(source.encode())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for node in tree.root_node.children:
        if node.type in (
            "class_definition",
            "enum_declaration",
            "mixin_declaration",
            "extension_declaration",
        ):
            text = node.text.decode()
            if current_len + len(text) > chunk_size and current:
                chunks.append("\n".join(current).strip())
                current = []
                current_len = 0
            current.append(text)
            current_len += len(text)

    if current:
        chunks.append("\n".join(current).strip())

    return chunks or [source]


# ---------------------------------------------------------------------------
# Regex implementations (always available, used when tree-sitter is absent)
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(r"""import\s+'([^']+)'""")
_EXPORT_RE = re.compile(
    r"""^(?:abstract\s+)?(?:class|enum|mixin|extension)\s+(\w+)""",
    re.MULTILINE,
)
_BOUNDARY_RE = re.compile(
    r"""^(?:abstract\s+)?(?:class|enum|mixin|extension)\s+\w+""",
    re.MULTILINE,
)


def _extract_imports_regex(source: str) -> list[str]:
    return _IMPORT_RE.findall(source)


def _extract_exports_regex(source: str) -> list[str]:
    return _EXPORT_RE.findall(source)


def _chunk_regex(source: str, chunk_size: int) -> list[str]:
    """Split source at class/enum/mixin/extension boundaries using regex."""
    if len(source) <= chunk_size:
        return [source]

    boundaries = [m.start() for m in _BOUNDARY_RE.finditer(source)]
    if not boundaries:
        # No recognisable boundaries — fall back to size-based splitting.
        return [source[i : i + chunk_size] for i in range(0, len(source), chunk_size)]

    # Always include the preamble (imports, top-level comments) before first boundary.
    segments: list[str] = []
    starts = boundaries + [len(source)]
    if boundaries[0] > 0:
        preamble = source[: boundaries[0]].strip()
        if preamble:
            segments.append(preamble)

    current_text = ""
    for idx, start in enumerate(boundaries):
        end = starts[idx + 1]
        segment = source[start:end].strip()
        if len(current_text) + len(segment) > chunk_size and current_text:
            segments.append(current_text.strip())
            current_text = segment
        else:
            current_text = (current_text + "\n\n" + segment).strip() if current_text else segment

    if current_text:
        segments.append(current_text.strip())

    return segments or [source]


# ---------------------------------------------------------------------------
# Path-based inference helpers
# ---------------------------------------------------------------------------

_LAYER_PARTS = ("presentation", "domain", "data", "core")

_FILE_TYPE_SUFFIXES: list[tuple[str, str]] = [
    ("_remote_datasource", "datasource"),
    ("_datasource", "datasource"),
    ("_repository_impl", "repository"),
    ("_repository", "repository"),
    ("_notifier", "notifier"),
    ("_provider", "provider"),
    ("_interceptor", "interceptor"),
    ("_screen", "screen"),
    ("_widget", "widget"),
    ("_model", "model"),
    ("_entity", "entity"),
]

_WIDGETS_DIR_RE = re.compile(r"/widgets/")
_FEATURE_RE = re.compile(r"features/(\w+)/")


def _infer_layer(file_path: str) -> str:
    for part in _LAYER_PARTS:
        if f"/{part}/" in file_path:
            return part
    return "unknown"


def _infer_feature(file_path: str) -> str:
    match = _FEATURE_RE.search(file_path)
    if match:
        return match.group(1)
    if "/core/" in file_path:
        return "core"
    return "unknown"


def _infer_file_type(file_path: str) -> str:
    stem = Path(file_path).stem.lower()

    for suffix, file_type in _FILE_TYPE_SUFFIXES:
        if stem.endswith(suffix):
            return file_type

    if _WIDGETS_DIR_RE.search(file_path):
        return "widget"

    return "unknown"
