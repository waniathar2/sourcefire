"""Language-agnostic code metadata extractor.

Extracts structural metadata from source files using tree-sitter (when available)
or regex fallback. Language-specific behavior is driven by LanguageProfile.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from sourcefire.indexer.language_profiles import LanguageProfile, get_profile_for_extension

# ---------------------------------------------------------------------------
# Optional tree-sitter import
# ---------------------------------------------------------------------------
try:
    from tree_sitter_languages import get_language, get_parser  # type: ignore

    _HAS_TREE_SITTER = True
except Exception:
    _HAS_TREE_SITTER = False

# Cache loaded parsers by language name
_PARSERS: dict[str, Any] = {}


def _get_parser(language: str) -> Any | None:
    """Get a tree-sitter parser for the given language, or None."""
    if not _HAS_TREE_SITTER:
        return None
    if language not in _PARSERS:
        try:
            _PARSERS[language] = get_parser(language)
        except Exception:
            _PARSERS[language] = None
    return _PARSERS[language]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_metadata(source: str, file_path: str, profile: Optional[LanguageProfile] = None) -> dict[str, Any]:
    """Return a metadata dict for a source file.

    Keys:
        imports   : list[str] — imported URIs/modules
        exports   : list[str] — top-level declaration names
        layer     : str       — architecture layer inferred from path
        feature   : str       — feature name inferred from path
        file_type : str       — file role inferred from path
    """
    if profile is None:
        ext = Path(file_path).suffix
        profile = get_profile_for_extension(ext)

    if profile is None:
        # No profile — return path-only defaults
        return {
            "imports": [],
            "exports": [],
            "layer": "unknown",
            "feature": "unknown",
            "file_type": "unknown",
        }

    parser = _get_parser(profile.tree_sitter_language) if profile.tree_sitter_language else None

    if parser is not None and source:
        imports = _extract_imports_tree_sitter(source, parser, profile)
        exports = _extract_exports_tree_sitter(source, parser, profile)
    elif source:
        imports = _extract_imports_regex(source, profile)
        exports = _extract_exports_regex(source, profile)
    else:
        imports = []
        exports = []

    return {
        "imports": imports,
        "exports": exports,
        "layer": _infer_layer(file_path, profile),
        "feature": _infer_feature(file_path, profile),
        "file_type": _infer_file_type(file_path, profile),
    }


def chunk_source_file(
    source: str,
    file_path: str,
    profile: Optional[LanguageProfile] = None,
    chunk_size: int = 1000,
) -> list[dict[str, Any]]:
    """Split a source file into chunks and attach metadata to each chunk.

    Splitting strategy:
    1. If tree-sitter is available and a profile exists, split at declaration boundaries.
    2. Otherwise, use regex boundary splitting.
    3. If the source is shorter than chunk_size, return a single chunk.

    Each chunk dict has keys:
        text     : str  — the chunk text
        metadata : dict — output of extract_metadata for the whole file
    """
    if profile is None:
        ext = Path(file_path).suffix
        profile = get_profile_for_extension(ext)

    metadata = extract_metadata(source, file_path, profile)

    if profile is None:
        # No profile — return source as a single chunk
        return [{"text": source, "metadata": metadata}]

    parser = _get_parser(profile.tree_sitter_language) if profile.tree_sitter_language else None

    if parser is not None:
        raw_chunks = _chunk_tree_sitter(source, parser, profile, chunk_size)
    else:
        raw_chunks = _chunk_regex(source, profile, chunk_size)

    return [{"text": text, "metadata": metadata} for text in raw_chunks]


# ---------------------------------------------------------------------------
# Tree-sitter implementations
# ---------------------------------------------------------------------------


def _extract_imports_tree_sitter(source: str, parser: Any, profile: LanguageProfile) -> list[str]:
    """Extract import URIs/modules using tree-sitter."""
    tree = parser.parse(source.encode())
    imports: list[str] = []
    _walk_for_imports(tree.root_node, imports, profile)
    return imports


def _walk_for_imports(node: Any, imports: list[str], profile: LanguageProfile) -> None:
    if node.type in profile.import_node_types:
        for child in node.children:
            if child.type == profile.string_literal_type:
                uri = child.text.decode().strip("'\"")
                imports.append(uri)
    for child in node.children:
        _walk_for_imports(child, imports, profile)


def _extract_exports_tree_sitter(source: str, parser: Any, profile: LanguageProfile) -> list[str]:
    """Extract top-level declaration names using tree-sitter."""
    tree = parser.parse(source.encode())
    exports: list[str] = []
    _walk_for_exports(tree.root_node, exports, profile)
    return exports


def _walk_for_exports(node: Any, exports: list[str], profile: LanguageProfile) -> None:
    if node.type in profile.export_node_types:
        for child in node.children:
            if child.type == "identifier":
                exports.append(child.text.decode())
                break
    for child in node.children:
        _walk_for_exports(child, exports, profile)


def _chunk_tree_sitter(source: str, parser: Any, profile: LanguageProfile, chunk_size: int) -> list[str]:
    """Split source at top-level declaration boundaries using tree-sitter."""
    tree = parser.parse(source.encode())
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for node in tree.root_node.children:
        if node.type in profile.boundary_node_types:
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
# Regex implementations
# ---------------------------------------------------------------------------


def _extract_imports_regex(source: str, profile: LanguageProfile) -> list[str]:
    if not profile.import_pattern:
        return []
    regex = re.compile(profile.import_pattern, re.MULTILINE)
    results = []
    for m in regex.finditer(source):
        # Take the first non-None group (different patterns use different groups)
        for g in m.groups():
            if g:
                results.append(g)
                break
    return results


def _extract_exports_regex(source: str, profile: LanguageProfile) -> list[str]:
    if not profile.export_pattern:
        return []
    return re.compile(profile.export_pattern, re.MULTILINE).findall(source)


def _chunk_regex(source: str, profile: LanguageProfile, chunk_size: int) -> list[str]:
    """Split source at declaration boundaries using regex."""
    if len(source) <= chunk_size:
        return [source]

    if not profile.boundary_pattern:
        # No boundary pattern — fall back to size-based splitting
        return [source[i : i + chunk_size] for i in range(0, len(source), chunk_size)]

    boundary_re = re.compile(profile.boundary_pattern, re.MULTILINE)
    boundaries = [m.start() for m in boundary_re.finditer(source)]

    if not boundaries:
        return [source[i : i + chunk_size] for i in range(0, len(source), chunk_size)]

    # Include the preamble (imports, top-level comments) before first boundary.
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


def _infer_layer(file_path: str, profile: LanguageProfile) -> str:
    for part in profile.layer_parts:
        if f"/{part}/" in file_path:
            return part
    return "unknown"


def _infer_feature(file_path: str, profile: LanguageProfile) -> str:
    if profile.feature_regex:
        match = re.search(profile.feature_regex, file_path)
        if match:
            return match.group(1)
    if "/core/" in file_path:
        return "core"
    return "unknown"


def _infer_file_type(file_path: str, profile: LanguageProfile) -> str:
    stem = Path(file_path).stem.lower()

    for suffix, file_type in profile.file_type_suffixes:
        if stem.endswith(suffix.lower()):
            return file_type

    for pattern, file_type in profile.directory_type_patterns.items():
        if pattern in file_path:
            return file_type

    return "unknown"
