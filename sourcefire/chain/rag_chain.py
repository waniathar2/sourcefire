"""LangChain RAG chain with mode-aware retrieval and Gemini API streaming."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from sourcefire.indexer.embeddings import embed_text
from sourcefire.indexer.language_profiles import LanguageProfile
from sourcefire.retriever.search import semantic_search, get_chunks_by_filenames, parse_file_references
from sourcefire.retriever.graph import ImportGraph
from sourcefire.chain.prompts import assemble_prompt
from sourcefire.db import query_similar


# ---------------------------------------------------------------------------
# Static context loader
# ---------------------------------------------------------------------------


def _load_static_context(project_dir: Path) -> tuple[str, str]:
    """Load CLAUDE.md from project_dir.

    Returns:
        A 2-tuple of (claude_md_content, memory_content).
    """
    claude_md = ""
    claude_md_path = project_dir / "CLAUDE.md"
    if claude_md_path.is_file():
        try:
            claude_md = claude_md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

    return claude_md, ""


# ---------------------------------------------------------------------------
# Mode-specific retrievers
# ---------------------------------------------------------------------------


async def _retrieve_debug(
    collection: Any,
    graph: ImportGraph,
    query: str,
    query_vector: list[float],
    top_k: int,
    profile: LanguageProfile | None = None,
) -> list[dict[str, Any]]:
    """Debug mode: parse stack trace -> direct lookup -> graph expansion -> semantic."""
    chunks: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()

    file_ref_patterns = profile.file_ref_patterns if profile else None
    file_refs = parse_file_references(query, file_ref_patterns)
    direct_filenames = [ref["file"] for ref in file_refs]

    if direct_filenames:
        direct_chunks = await get_chunks_by_filenames(collection, direct_filenames)
        for c in direct_chunks:
            c["priority"] = "direct"
            c.setdefault("relevance", 1.0)
            chunks.append(c)
            seen_filenames.add(c["filename"])

        graph_filenames: list[str] = []
        for fname in direct_filenames:
            graph_filenames.extend(graph.get_neighbors(fname, hops=1))

        graph_filenames = [f for f in graph_filenames if f not in seen_filenames]

        if graph_filenames:
            graph_chunks = await get_chunks_by_filenames(collection, graph_filenames)
            for c in graph_chunks:
                c["priority"] = "graph"
                c.setdefault("relevance", 0.6)
                chunks.append(c)
                seen_filenames.add(c["filename"])

    semantic_chunks = await semantic_search(collection, query_vector, top_k=top_k)
    for c in semantic_chunks:
        if c["filename"] not in seen_filenames:
            c["priority"] = "semantic"
            chunks.append(c)
            seen_filenames.add(c["filename"])

    return chunks


async def _retrieve_feature(
    collection: Any,
    graph: ImportGraph,
    query: str,
    query_vector: list[float],
    top_k: int,
    profile: LanguageProfile | None = None,
) -> list[dict[str, Any]]:
    """Feature mode: semantic search -> best feature -> retrieve feature chunks."""
    _FEATURE_CAP = 15

    seed_chunks = await semantic_search(collection, query_vector, top_k=top_k)

    feature_scores: dict[str, list[float]] = {}
    for c in seed_chunks:
        feat = c.get("feature") or "core"
        feature_scores.setdefault(feat, []).append(float(c.get("relevance", 0.0)))

    if not feature_scores:
        for c in seed_chunks:
            c["priority"] = "semantic"
        return seed_chunks

    best_feature = max(feature_scores, key=lambda f: sum(feature_scores[f]) / len(feature_scores[f]))

    feature_chunks = await semantic_search(
        collection,
        query_vector,
        top_k=_FEATURE_CAP,
        feature=best_feature,
    )
    for c in feature_chunks:
        c["priority"] = "semantic"

    return feature_chunks


async def _retrieve_explain(
    collection: Any,
    graph: ImportGraph,
    query: str,
    query_vector: list[float],
    top_k: int,
    profile: LanguageProfile | None = None,
) -> list[dict[str, Any]]:
    """Explain mode: semantic search -> import graph expansion in both directions."""
    chunks: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()

    seed_chunks = await semantic_search(collection, query_vector, top_k=top_k)
    for c in seed_chunks:
        c["priority"] = "semantic"
        chunks.append(c)
        seen_filenames.add(c["filename"])

    neighbor_filenames: list[str] = []
    for c in seed_chunks:
        for neighbor in graph.get_neighbors(c["filename"], hops=1):
            if neighbor not in seen_filenames:
                neighbor_filenames.append(neighbor)
                seen_filenames.add(neighbor)

    if neighbor_filenames:
        neighbor_chunks = await get_chunks_by_filenames(collection, neighbor_filenames)
        for c in neighbor_chunks:
            c["priority"] = "graph"
            c.setdefault("relevance", 0.5)
            chunks.append(c)

    return chunks


# ---------------------------------------------------------------------------
# Public retrieval entry point
# ---------------------------------------------------------------------------


async def retrieve_for_mode(
    collection: Any,
    graph: ImportGraph,
    query: str,
    mode: str,
    top_k: int = 8,
    profile: LanguageProfile | None = None,
) -> list[dict[str, Any]]:
    """Embed *query* and dispatch to the mode-specific retriever."""
    loop = asyncio.get_running_loop()
    query_vector: list[float] = await loop.run_in_executor(None, embed_text, query)

    if mode == "debug":
        return await _retrieve_debug(collection, graph, query, query_vector, top_k, profile)
    elif mode == "feature":
        return await _retrieve_feature(collection, graph, query, query_vector, top_k, profile)
    else:
        return await _retrieve_explain(collection, graph, query, query_vector, top_k, profile)


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

def _get_tools(
    graph: ImportGraph,
    profile: LanguageProfile | None = None,
    collection: Any = None,
    project_dir: Path | None = None,
) -> list[Any]:

    _project_dir = project_dir or Path.cwd()
    searchable_exts = tuple(profile.searchable_extensions) if profile else (".py", ".js", ".ts", ".go", ".rs", ".java", ".dart", ".yaml", ".json", ".md")

    @tool
    def read_local_file(filepath: str) -> str:
        """Reads the complete content of a file from the repository.
        Use when you need to see exact implementation details.
        Provide the relative filepath (e.g. 'src/main.py').
        """
        full_path = (_project_dir / filepath).resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        if not full_path.is_file():
            return f"Error: File '{filepath}' not found in the codebase."
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 30000:
                return content[:30000] + "\n\n... [File truncated because it is too large] ..."
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    @tool
    def list_directory(dir_path: str) -> str:
        """Lists files and folders within a specific directory in the repository.
        Use to understand project structure or find files.
        Provide the relative path (e.g. 'src/components'). Use '.' for root.
        """
        if dir_path == ".":
            full_path = _project_dir.resolve()
        else:
            full_path = (_project_dir / dir_path).resolve()

        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        if not full_path.is_dir():
            return f"Error: Directory '{dir_path}' not found."

        try:
            items = []
            for item in full_path.iterdir():
                suffix = "/" if item.is_dir() else ""
                items.append(item.name + suffix)
            return f"Contents of '{dir_path}':\n" + "\n".join(sorted(items))
        except Exception as e:
            return f"Error listing directory: {e}"

    @tool
    def find_file_usages(filepath: str) -> str:
        """Find other files that import or depend on the given file.
        Provide the relative filepath (e.g. 'src/utils/auth.py').
        Returns a list of files that directly import it.
        """
        importers = graph.get_importers(filepath)
        if not importers:
            return f"No direct local dependencies found importing {filepath}."
        return f"Files importing '{filepath}':\n" + "\n".join(f"- {f}" for f in importers)

    @tool
    def search_codebase_keywords(query: str, dir_path: str = ".") -> str:
        """Search for an exact string or keyword across files in a directory.
        Returns files and line numbers where the keyword appears.
        Use for variable names, classes, function names, or patterns.
        """
        full_path = _project_dir if dir_path == "." else (_project_dir / dir_path)
        full_path = full_path.resolve()

        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."

        if not full_path.is_dir() and not full_path.is_file():
            return f"Error: Directory '{dir_path}' not found."

        results = []
        try:
            for root, _, files in os.walk(full_path):
                if any(x in root for x in [".git", ".claude", "node_modules", "__pycache__", "build", "dist", "target", ".dart_tool"]):
                    continue
                for file in files:
                    if file.endswith(searchable_exts):
                        fpath = Path(root) / file
                        try:
                            lines = fpath.read_text("utf-8", "replace").splitlines()
                            for i, line in enumerate(lines):
                                if query.lower() in line.lower():
                                    rel_path = fpath.relative_to(_project_dir)
                                    results.append(f"{rel_path}:{i+1}: {line.strip()[:100]}")
                                    if len(results) >= 50:
                                        return f"Results truncated at 50 matches.\n" + "\n".join(results)
                        except Exception:
                            continue
            if not results:
                return f"No matches found for '{query}' in {dir_path}"
            return "\n".join(results)
        except Exception as e:
            return f"Error searching: {e}"

    @tool
    def find_definition(symbol_name: str) -> str:
        """Find where a class, function, or variable is defined in the codebase.
        Searches for definition patterns like 'class Foo', 'def foo', 'function foo', etc.
        Returns file paths and line numbers.
        """
        patterns = [
            f"class {symbol_name}",
            f"def {symbol_name}",
            f"async def {symbol_name}",
            f"function {symbol_name}",
            f"const {symbol_name}",
            f"let {symbol_name}",
            f"var {symbol_name}",
            f"type {symbol_name}",
            f"interface {symbol_name}",
            f"enum {symbol_name}",
            f"struct {symbol_name}",
            f"trait {symbol_name}",
            f"impl {symbol_name}",
            f"func {symbol_name}",
            f"mixin {symbol_name}",
            f"extension {symbol_name}",
        ]
        results = []
        try:
            for root, _, files in os.walk(_project_dir):
                if any(x in root for x in [".git", "node_modules", "__pycache__", "build", "dist", "target", ".dart_tool"]):
                    continue
                for file in files:
                    if file.endswith(searchable_exts):
                        fpath = Path(root) / file
                        try:
                            lines = fpath.read_text("utf-8", "replace").splitlines()
                            for i, line in enumerate(lines):
                                stripped = line.strip()
                                for pattern in patterns:
                                    if stripped.startswith(pattern) or f" {pattern}" in stripped:
                                        rel_path = fpath.relative_to(_project_dir)
                                        results.append(f"{rel_path}:{i+1}: {stripped[:120]}")
                                        break
                                if len(results) >= 20:
                                    return "\n".join(results) + "\n... (truncated)"
                        except Exception:
                            continue
            if not results:
                return f"No definition found for '{symbol_name}'"
            return "\n".join(results)
        except Exception as e:
            return f"Error searching: {e}"

    @tool
    def get_file_structure(dir_path: str = ".", max_depth: int = 3) -> str:
        """Get a tree view of the project structure up to a given depth.
        Use this to understand the overall project layout.
        Provide relative path and max depth (default 3).
        """
        if dir_path == ".":
            full_path = _project_dir.resolve()
        else:
            full_path = (_project_dir / dir_path).resolve()

        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        if not full_path.is_dir():
            return f"Error: Directory '{dir_path}' not found."

        skip_dirs = {".git", "node_modules", "__pycache__", "build", "dist", "target",
                     ".dart_tool", ".next", "venv", ".venv", ".idea", ".vs"}
        lines = []

        def _walk(path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return
            dirs = [e for e in entries if e.is_dir() and e.name not in skip_dirs]
            files = [e for e in entries if e.is_file()]

            for f in files[:20]:
                lines.append(f"{prefix}{f.name}")
            if len(files) > 20:
                lines.append(f"{prefix}... ({len(files) - 20} more files)")

            for d in dirs:
                lines.append(f"{prefix}{d.name}/")
                _walk(d, prefix + "  ", depth + 1)

        lines.append(f"{dir_path}/")
        _walk(full_path, "  ", 1)
        return "\n".join(lines[:200])

    @tool
    def git_file_history(filepath: str, max_commits: int = 10) -> str:
        """Get recent git commit history for a specific file.
        Shows who changed it, when, and why. Use to understand evolution of a file.
        Provide relative filepath.
        """
        full_path = (_project_dir / filepath).resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        try:
            result = subprocess.run(
                ["git", "log", f"-{max_commits}", "--pretty=format:%h %ai %s", "--", filepath],
                cwd=str(_project_dir),
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            if not result.stdout.strip():
                return f"No git history found for '{filepath}'"
            return f"Recent commits for '{filepath}':\n{result.stdout}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def git_blame_lines(filepath: str, start_line: int, end_line: int) -> str:
        """Show git blame for specific line range of a file.
        Use when you need to know who last changed specific lines and why.
        """
        full_path = (_project_dir / filepath).resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        try:
            result = subprocess.run(
                ["git", "blame", f"-L{start_line},{end_line}", "--date=short", filepath],
                cwd=str(_project_dir),
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            return result.stdout or "No blame output"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def read_lines(filepath: str, start_line: int, end_line: int) -> str:
        """Read a specific range of lines from a file.
        Use when you only need a portion of a large file — avoids context overflow.
        Line numbers are 1-based. Returns lines with line numbers prefixed.
        """
        full_path = (_project_dir / filepath).resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        if not full_path.is_file():
            return f"Error: File '{filepath}' not found."
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, start_line) - 1
            end = min(len(lines), end_line)
            if start >= len(lines):
                return f"Error: File has only {len(lines)} lines."
            selected = lines[start:end]
            return "\n".join(f"{start + i + 1:>4} | {line}" for i, line in enumerate(selected))
        except Exception as e:
            return f"Error: {e}"

    @tool
    def regex_search(pattern: str, dir_path: str = ".", max_results: int = 30) -> str:
        """Search files using a regex pattern. More powerful than keyword search.
        Use for complex patterns like 'def .*async', 'TODO|FIXME|HACK', etc.
        Returns matching lines with file paths and line numbers.
        """
        import re as re_mod
        full_path = _project_dir if dir_path == "." else (_project_dir / dir_path)
        full_path = full_path.resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."

        try:
            compiled = re_mod.compile(pattern, re_mod.IGNORECASE)
        except re_mod.error as e:
            return f"Invalid regex: {e}"

        results = []
        try:
            for root, _, files in os.walk(full_path):
                if any(x in root for x in [".git", "node_modules", "__pycache__", "build", "dist", "target", ".dart_tool"]):
                    continue
                for file in files:
                    if file.endswith(searchable_exts):
                        fpath = Path(root) / file
                        try:
                            for i, line in enumerate(fpath.read_text("utf-8", "replace").splitlines()):
                                if compiled.search(line):
                                    rel_path = fpath.relative_to(_project_dir)
                                    results.append(f"{rel_path}:{i+1}: {line.strip()[:120]}")
                                    if len(results) >= max_results:
                                        return f"Found {len(results)}+ matches (truncated):\n" + "\n".join(results)
                        except Exception:
                            continue
            if not results:
                return f"No matches for pattern '{pattern}' in {dir_path}"
            return f"Found {len(results)} matches:\n" + "\n".join(results)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def find_references(symbol_name: str, dir_path: str = ".") -> str:
        """Find all usages/references of a symbol (function, class, variable) across the codebase.
        Unlike find_definition which finds where something is declared, this finds where it's used.
        Returns file paths, line numbers, and the line content.
        """
        full_path = _project_dir if dir_path == "." else (_project_dir / dir_path)
        full_path = full_path.resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."

        results = []
        try:
            for root, _, files in os.walk(full_path):
                if any(x in root for x in [".git", "node_modules", "__pycache__", "build", "dist", "target", ".dart_tool"]):
                    continue
                for file in files:
                    if file.endswith(searchable_exts):
                        fpath = Path(root) / file
                        try:
                            for i, line in enumerate(fpath.read_text("utf-8", "replace").splitlines()):
                                if symbol_name in line:
                                    rel_path = fpath.relative_to(_project_dir)
                                    results.append(f"{rel_path}:{i+1}: {line.strip()[:120]}")
                                    if len(results) >= 40:
                                        return f"Found {len(results)}+ references (truncated):\n" + "\n".join(results)
                        except Exception:
                            continue
            if not results:
                return f"No references found for '{symbol_name}'"
            return f"Found {len(results)} references:\n" + "\n".join(results)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def git_diff(ref: str = "HEAD~1", filepath: str = "") -> str:
        """Show git diff for recent changes. Use to understand what changed recently.
        ref: git reference like 'HEAD~1', 'HEAD~3', a branch name, or commit hash.
        filepath: optional — limit diff to a specific file.
        """
        cmd = ["git", "diff", "--stat", "-p", ref]
        if filepath:
            full_path = (_project_dir / filepath).resolve()
            if not full_path.is_relative_to(_project_dir.resolve()):
                return "Error: Path traversal not allowed."
            cmd.extend(["--", filepath])
        try:
            result = subprocess.run(
                cmd, cwd=str(_project_dir),
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"Git error: {result.stderr.strip()}"
            output = result.stdout.strip()
            if not output:
                return "No changes found."
            if len(output) > 15000:
                return output[:15000] + "\n\n... [Diff truncated] ..."
            return output
        except Exception as e:
            return f"Error: {e}"

    @tool
    def git_log_search(search_term: str, max_commits: int = 10) -> str:
        """Search git commit messages and diffs for a term.
        Use to find when a feature was added, a bug was introduced, or a file was changed.
        Searches both commit messages (-grep) and code changes (-S).
        """
        results = []
        try:
            msg_result = subprocess.run(
                ["git", "log", f"-{max_commits}", "--pretty=format:%h %ai %s", f"--grep={search_term}", "-i"],
                cwd=str(_project_dir),
                capture_output=True, text=True, timeout=10,
            )
            if msg_result.stdout.strip():
                results.append("Commits mentioning '" + search_term + "':\n" + msg_result.stdout.strip())
        except Exception:
            pass

        try:
            code_result = subprocess.run(
                ["git", "log", f"-{max_commits}", "--pretty=format:%h %ai %s", f"-S{search_term}"],
                cwd=str(_project_dir),
                capture_output=True, text=True, timeout=10,
            )
            if code_result.stdout.strip():
                results.append("Commits changing code with '" + search_term + "':\n" + code_result.stdout.strip())
        except Exception:
            pass

        if not results:
            return f"No commits found related to '{search_term}'"
        return "\n\n".join(results)

    @tool
    def file_stats(filepath: str) -> str:
        """Get stats about a file: line count, size, last modified, language.
        Use to quickly assess file complexity and recency.
        """
        full_path = (_project_dir / filepath).resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        if not full_path.is_file():
            return f"Error: File '{filepath}' not found."
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            non_blank = sum(1 for l in lines if l.strip())
            stat = full_path.stat()
            from datetime import datetime
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = stat.st_size / 1024

            parts = [
                f"File: {filepath}",
                f"Lines: {len(lines)} ({non_blank} non-blank)",
                f"Size: {size_kb:.1f} KB",
                f"Last modified: {modified}",
                f"Extension: {full_path.suffix}",
            ]

            import_count = sum(1 for l in lines if l.strip().startswith(("import ", "from ", "#include", "use ", "require")))
            class_count = sum(1 for l in lines if any(l.strip().startswith(k) for k in ("class ", "struct ", "enum ", "interface ", "trait ")))
            func_count = sum(1 for l in lines if any(l.strip().startswith(k) for k in ("def ", "func ", "fn ", "function ", "async def ", "pub fn ")))

            if import_count: parts.append(f"Imports: {import_count}")
            if class_count: parts.append(f"Classes/structs: {class_count}")
            if func_count: parts.append(f"Functions: {func_count}")

            return "\n".join(parts)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def find_files_by_name(filename_pattern: str) -> str:
        """Find files whose name matches a pattern (case-insensitive substring match).
        Use when you know part of a filename but not its full path.
        Example: 'auth' finds auth_service.py, AuthController.java, etc.
        """
        pattern_lower = filename_pattern.lower()
        results = []
        try:
            for root, dirs, files in os.walk(_project_dir):
                dirs[:] = [d for d in dirs if d not in {
                    ".git", "node_modules", "__pycache__", "build", "dist",
                    "target", ".dart_tool", ".next", "venv", ".venv",
                }]
                for file in files:
                    if pattern_lower in file.lower():
                        fpath = Path(root) / file
                        rel_path = fpath.relative_to(_project_dir)
                        results.append(str(rel_path))
                        if len(results) >= 30:
                            return f"Found {len(results)}+ files (truncated):\n" + "\n".join(results)
            if not results:
                return f"No files matching '{filename_pattern}'"
            return f"Found {len(results)} files:\n" + "\n".join(results)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_call_chain(filepath: str, function_name: str) -> str:
        """Trace who calls a function and what it calls.
        Returns callers (files that reference this function) and callees
        (functions/methods invoked inside the given function's body).
        """
        full_path = (_project_dir / filepath).resolve()
        if not full_path.is_relative_to(_project_dir.resolve()):
            return "Error: Path traversal not allowed."
        if not full_path.is_file():
            return f"Error: File '{filepath}' not found."

        parts = []

        importers = graph.get_importers(filepath)
        callers = []
        for root, _, files in os.walk(_project_dir):
            if any(x in root for x in [".git", "node_modules", "__pycache__", "build", "dist", "target"]):
                continue
            for file in files:
                if file.endswith(searchable_exts):
                    fpath = Path(root) / file
                    if fpath.resolve() == full_path:
                        continue
                    try:
                        content = fpath.read_text("utf-8", "replace")
                        if function_name in content:
                            rel = fpath.relative_to(_project_dir)
                            for i, line in enumerate(content.splitlines()):
                                if function_name in line:
                                    callers.append(f"  {rel}:{i+1}: {line.strip()[:100]}")
                                    break
                    except Exception:
                        continue
                if len(callers) >= 15:
                    break

        if callers:
            parts.append(f"Callers of {function_name}:\n" + "\n".join(callers))
        else:
            parts.append(f"No callers found for {function_name}")

        try:
            import re as re_mod
            content = full_path.read_text("utf-8", "replace")
            lines = content.splitlines()

            func_start = -1
            for i, line in enumerate(lines):
                if function_name in line and any(k in line for k in ("def ", "func ", "fn ", "function ", "void ", "class ")):
                    func_start = i
                    break

            if func_start >= 0:
                body = "\n".join(lines[func_start:func_start + 50])
                callees = set(re_mod.findall(r'\b([a-zA-Z_]\w+)\s*\(', body))
                keywords = {"if", "for", "while", "switch", "catch", "return", "print", "throw", function_name}
                callees -= keywords
                if callees:
                    parts.append(f"Functions called by {function_name}:\n  " + ", ".join(sorted(callees)))
        except Exception:
            pass

        return "\n\n".join(parts) if parts else f"Could not trace call chain for {function_name}"

    @tool
    def semantic_code_search(query: str, top_k: int = 6) -> str:
        """Search the codebase using semantic similarity (vector embeddings).
        Unlike keyword search, this finds code by MEANING — even if the exact
        words aren't present. Use for conceptual queries like:
        - 'error handling logic'
        - 'user authentication flow'
        - 'database connection setup'
        Returns the most semantically relevant code chunks with file paths.
        """
        if not collection:
            return "Error: Vector database not available."
        try:
            query_vector = embed_text(query)
            results = query_similar(collection, query_vector, n_results=top_k)
            if not results:
                return f"No semantically relevant code found for: '{query}'"
            parts = []
            for r in results:
                score = r.get('relevance', 0)
                fname = r.get('filename', '?')
                loc = r.get('location', '')
                code = r.get('code', '')[:500]
                feature = r.get('feature', '')
                layer = r.get('layer', '')
                meta = ""
                if feature and feature != "unknown":
                    meta += f" [feature: {feature}]"
                if layer and layer != "unknown":
                    meta += f" [layer: {layer}]"
                parts.append(f"--- {fname} ({loc}) [score: {score:.2f}]{meta}\n{code}")
            return f"Found {len(results)} semantically similar chunks:\n\n" + "\n\n".join(parts)
        except Exception as e:
            return f"Error in semantic search: {e}"

    @tool
    def find_similar_code(filepath: str, chunk_index: int = 0, top_k: int = 5) -> str:
        """Find code chunks that are semantically similar to a specific file/chunk.
        Use this to discover related implementations, duplicated logic, or code
        that serves a similar purpose elsewhere in the codebase.
        Provide the file path and optionally a chunk index (default 0 = first chunk).
        """
        if not collection:
            return "Error: Vector database not available."
        try:
            full_path = (_project_dir / filepath).resolve()
            if not full_path.is_relative_to(_project_dir.resolve()):
                return "Error: Path traversal not allowed."
            if not full_path.is_file():
                return f"Error: File '{filepath}' not found."

            content = full_path.read_text(encoding="utf-8", errors="replace")
            query_text = content[:800]
            query_vector = embed_text(query_text)

            results = query_similar(collection, query_vector, n_results=top_k + 2)
            results = [r for r in results if r.get('filename') != filepath][:top_k]

            if not results:
                return f"No similar code found to '{filepath}'"
            parts = []
            for r in results:
                score = r.get('relevance', 0)
                fname = r.get('filename', '?')
                code = r.get('code', '')[:400]
                parts.append(f"--- {fname} [similarity: {score:.2f}]\n{code}")
            return f"Files similar to '{filepath}':\n\n" + "\n\n".join(parts)
        except Exception as e:
            return f"Error: {e}"

    return [
        read_local_file,
        read_lines,
        list_directory,
        find_file_usages,
        search_codebase_keywords,
        regex_search,
        find_definition,
        find_references,
        find_files_by_name,
        get_file_structure,
        get_call_chain,
        file_stats,
        semantic_code_search,
        find_similar_code,
        git_file_history,
        git_blame_lines,
        git_diff,
        git_log_search,
    ]


# ---------------------------------------------------------------------------
# Streaming RAG response
# ---------------------------------------------------------------------------


async def stream_rag_response(
    collection: Any,
    graph: ImportGraph,
    query: str,
    mode: str,
    model: str = "gemini-2.5-flash",
    history: list[dict[str, str]] | None = None,
    profile: LanguageProfile | None = None,
    project_dir: Path | None = None,
    gemini_api_key: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
    """Async generator that retrieves context and streams a Gemini response."""
    if history is None:
        history = []

    _project_dir = project_dir or Path.cwd()
    highlight_language = profile.highlight_language if profile else "text"

    yield {"type": "status", "stage": "retrieving"}

    try:
        chunks = await retrieve_for_mode(collection, graph, query, mode, profile=profile)
    except Exception as exc:
        yield {"type": "error", "content": f"Retrieval failed: {exc}"}
        return

    unique_files = len(set(c.get("filename", "") for c in chunks))
    yield {"type": "status", "stage": "context_found", "chunks": len(chunks), "files": unique_files}

    claude_md, memory_content = _load_static_context(_project_dir)

    prompt = assemble_prompt(
        mode=mode,
        query=query,
        chunks=chunks,
        claude_md=claude_md,
        memory_content=memory_content,
        history=history,
        model=model,
        highlight_language=highlight_language,
    )

    messages: list[Any] = [SystemMessage(content=prompt["system"])]

    for turn in prompt["history"]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    context_block = prompt["context"]
    if context_block:
        human_content = f"## Retrieved Code Context\n\n{context_block}\n\n---\n\n{query}"
    else:
        human_content = query

    messages.append(HumanMessage(content=human_content))

    sources = [
        {"filename": c.get("filename", ""), "priority": c.get("priority", "semantic")}
        for c in chunks
    ]

    try:
        tools = _get_tools(graph, profile, collection, _project_dir)
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_api_key,
            streaming=True,
        )
        llm_with_tools = llm.bind_tools(tools)

        yield {"type": "status", "stage": "thinking"}

        MAX_STEPS = 5
        step = 0
        first_token = True

        while step < MAX_STEPS:
            step += 1
            has_tool_calls = False
            full_chunk_msg = None

            async for chunk in llm_with_tools.astream(messages):
                if full_chunk_msg is None:
                    full_chunk_msg = chunk
                else:
                    full_chunk_msg += chunk

                if chunk.content:
                    content_val = chunk.content
                    if isinstance(content_val, list):
                        text_parts = [
                            b.get("text", "") for b in content_val
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        content_str = "".join(text_parts)
                    else:
                        content_str = str(content_val)

                    if content_str:
                        if first_token:
                            yield {"type": "status", "stage": "generating"}
                            first_token = False
                        yield {"type": "token", "content": content_str}

            if full_chunk_msg:
                messages.append(full_chunk_msg)

                if full_chunk_msg.tool_calls:
                    has_tool_calls = True
                    for tool_call in full_chunk_msg.tool_calls:
                        tool_name = tool_call["name"]
                        args = tool_call["args"]
                        tool_id = tool_call["id"]

                        args_summary = ", ".join(f"{k}={repr(v)[:40]}" for k, v in args.items()) if args else ""
                        yield {"type": "status", "stage": "tool_call", "tool": tool_name, "args": args_summary}

                        selected_tool = next((t for t in tools if t.name == tool_name), None)
                        if selected_tool:
                            try:
                                result = await asyncio.to_thread(selected_tool.invoke, args)
                                result_str = str(result)
                            except Exception as e:
                                result_str = f"Error executing tool: {e}"
                        else:
                            result_str = f"Error: Tool {tool_name} not found."

                        yield {"type": "status", "stage": "tool_done", "tool": tool_name}
                        messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

                    first_token = True

            if not has_tool_calls:
                break

        yield {"type": "done", "sources": sources, "stats": prompt["stats"]}

    except Exception as exc:
        yield {"type": "error", "content": f"Gemini API error: {exc}"}
