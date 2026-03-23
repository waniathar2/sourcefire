"""LangChain RAG chain with mode-aware retrieval and Gemini API streaming."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.config import CODEBASE_PATH, DEFAULT_MODEL, GEMINI_API_KEY
from src.indexer.embeddings import embed_text
from src.retriever.search import semantic_search, get_chunks_by_filenames, parse_file_references
from src.retriever.graph import ImportGraph
from src.chain.prompts import assemble_prompt

# ---------------------------------------------------------------------------
# Architecture rules chunks (injected for feature mode)
# ---------------------------------------------------------------------------

_ARCH_RULES_CHUNK: dict[str, Any] = {
    "filename": "CLAUDE.md",
    "location": "CLAUDE.md:arch",
    "code": (
        "## Architecture\n\n"
        "Feature-first Clean Architecture. Each feature follows:\n"
        "  presentation → domain ← data\n\n"
        "Never import data layer directly from presentation.\n\n"
        "lib/\n"
        "├── main.dart / main_dev.dart / bootstrap.dart\n"
        "├── core/  (network, storage, router, theme, widgets, errors)\n"
        "└── features/\n"
        "    ├── auth/\n"
        "    ├── onboarding/\n"
        "    ├── device_setup/\n"
        "    ├── home/\n"
        "    ├── recipes/\n"
        "    ├── cooking/\n"
        "    ├── cravv/\n"
        "    └── safety/\n\n"
        "Each feature internally has: data/ domain/ presentation/"
    ),
    "feature": "core",
    "layer": "docs",
    "file_type": "docs",
    "relevance": 1.0,
    "priority": "direct",
}

# ---------------------------------------------------------------------------
# Static context loader
# ---------------------------------------------------------------------------


def _load_static_context() -> tuple[str, str]:
    """Load CLAUDE.md and memory markdown files from CODEBASE_PATH.

    Returns:
        A 2-tuple of (claude_md_content, memory_content).
        Either value is an empty string if the file/directory is not found.
    """
    # CLAUDE.md
    claude_md = ""
    claude_md_path = CODEBASE_PATH / "CLAUDE.md"
    if claude_md_path.is_file():
        try:
            claude_md = claude_md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

    # Memory files: .claude/**/memory/*.md
    memory_parts: list[str] = []
    memory_glob = list(CODEBASE_PATH.glob(".claude/**/memory/*.md"))
    for mem_path in sorted(memory_glob):
        try:
            content = mem_path.read_text(encoding="utf-8", errors="replace")
            rel = mem_path.relative_to(CODEBASE_PATH).as_posix()
            memory_parts.append(f"### {rel}\n\n{content}")
        except OSError:
            pass

    memory_content = "\n\n---\n\n".join(memory_parts)
    return claude_md, memory_content


# ---------------------------------------------------------------------------
# Mode-specific retrievers
# ---------------------------------------------------------------------------


async def _retrieve_debug(
    pool: Any,
    graph: ImportGraph,
    query: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Debug mode: parse stack trace → direct lookup → graph expansion → semantic.

    Priority tagging:
    - "direct"   — chunks fetched by explicit filename from the stack trace
    - "graph"    — chunks fetched by 1-hop import graph expansion
    - "semantic" — chunks from cosine similarity search
    """
    chunks: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()

    # 1. Parse explicit file references from the query (stack traces, error msgs).
    file_refs = parse_file_references(query)
    direct_filenames = [ref["file"] for ref in file_refs]

    if direct_filenames:
        direct_chunks = await get_chunks_by_filenames(pool, direct_filenames)
        for c in direct_chunks:
            c["priority"] = "direct"
            c.setdefault("relevance", 1.0)
            chunks.append(c)
            seen_filenames.add(c["filename"])

        # 1-hop import graph expansion from each direct file.
        graph_filenames: list[str] = []
        for fname in direct_filenames:
            graph_filenames.extend(graph.get_neighbors(fname, hops=1))

        # Exclude files already fetched directly.
        graph_filenames = [f for f in graph_filenames if f not in seen_filenames]

        if graph_filenames:
            graph_chunks = await get_chunks_by_filenames(pool, graph_filenames)
            for c in graph_chunks:
                c["priority"] = "graph"
                c.setdefault("relevance", 0.6)
                chunks.append(c)
                seen_filenames.add(c["filename"])

    # 2. Semantic search to fill remaining budget.
    semantic_chunks = await semantic_search(pool, query_vector, top_k=top_k)
    for c in semantic_chunks:
        if c["filename"] not in seen_filenames:
            c["priority"] = "semantic"
            chunks.append(c)
            seen_filenames.add(c["filename"])

    return chunks


async def _retrieve_feature(
    pool: Any,
    graph: ImportGraph,
    query: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Feature mode: semantic search → best feature → retrieve feature chunks + arch rules.

    Strategy:
    1. Run semantic search.
    2. Group results by the ``feature`` metadata field.
    3. Pick the feature with the highest average relevance.
    4. Fetch all chunks for that feature (capped at 15).
    5. Append architecture rules chunk.
    """
    _FEATURE_CAP = 15

    # 1. Semantic search for orientation.
    seed_chunks = await semantic_search(pool, query_vector, top_k=top_k)

    # 2. Group by feature, compute average relevance per feature.
    feature_scores: dict[str, list[float]] = {}
    for c in seed_chunks:
        feat = c.get("feature") or "core"
        feature_scores.setdefault(feat, []).append(float(c.get("relevance", 0.0)))

    if not feature_scores:
        # Fall back to semantic results + arch rules if nothing found.
        for c in seed_chunks:
            c["priority"] = "semantic"
        result = seed_chunks
        arch = dict(_ARCH_RULES_CHUNK)
        result.append(arch)
        return result

    # 3. Pick the best feature by average relevance.
    best_feature = max(feature_scores, key=lambda f: sum(feature_scores[f]) / len(feature_scores[f]))

    # 4. Fetch all chunks for that feature (up to cap).
    feature_chunks = await semantic_search(
        pool,
        query_vector,
        top_k=_FEATURE_CAP,
        feature=best_feature,
    )
    for c in feature_chunks:
        c["priority"] = "semantic"

    # 5. Add architecture rules chunk.
    arch = dict(_ARCH_RULES_CHUNK)
    return feature_chunks + [arch]


async def _retrieve_explain(
    pool: Any,
    graph: ImportGraph,
    query: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Explain mode: semantic search → import graph expansion in both directions.

    Strategy:
    1. Run semantic search.
    2. For each result file, collect 1-hop graph neighbors (imports + importers).
    3. Fetch neighbor chunks.
    """
    chunks: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()

    # 1. Semantic search seed.
    seed_chunks = await semantic_search(pool, query_vector, top_k=top_k)
    for c in seed_chunks:
        c["priority"] = "semantic"
        chunks.append(c)
        seen_filenames.add(c["filename"])

    # 2. Collect 1-hop neighbors for all seed files.
    neighbor_filenames: list[str] = []
    for c in seed_chunks:
        for neighbor in graph.get_neighbors(c["filename"], hops=1):
            if neighbor not in seen_filenames:
                neighbor_filenames.append(neighbor)
                seen_filenames.add(neighbor)

    # 3. Fetch neighbor chunks.
    if neighbor_filenames:
        neighbor_chunks = await get_chunks_by_filenames(pool, neighbor_filenames)
        for c in neighbor_chunks:
            c["priority"] = "graph"
            c.setdefault("relevance", 0.5)
            chunks.append(c)

    return chunks


# ---------------------------------------------------------------------------
# Public retrieval entry point
# ---------------------------------------------------------------------------


async def retrieve_for_mode(
    pool: Any,
    graph: ImportGraph,
    query: str,
    mode: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Embed *query* and dispatch to the mode-specific retriever.

    Args:
        pool:   A psycopg ConnectionPool.
        graph:  A populated ImportGraph.
        query:  The user's question or stack trace.
        mode:   One of "debug", "feature", "explain".
        top_k:  Maximum number of semantic results to seed the retriever with.

    Returns:
        A list of chunk dicts, each augmented with ``priority`` and ``relevance`` fields.
    """
    # Embedding is CPU-bound; run in executor to avoid blocking the event loop.
    loop = asyncio.get_event_loop()
    query_vector: list[float] = await loop.run_in_executor(None, embed_text, query)

    if mode == "debug":
        return await _retrieve_debug(pool, graph, query, query_vector, top_k)
    elif mode == "feature":
        return await _retrieve_feature(pool, graph, query, query_vector, top_k)
    else:
        # Default: explain mode.
        return await _retrieve_explain(pool, graph, query, query_vector, top_k)


# ---------------------------------------------------------------------------
# Streaming RAG response
# ---------------------------------------------------------------------------


async def stream_rag_response(
    pool: Any,
    graph: ImportGraph,
    query: str,
    mode: str,
    model: str = DEFAULT_MODEL,
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Async generator that retrieves context and streams a Gemini response.

    Yields dicts of one of three shapes:
    - ``{"type": "token",  "content": "<text>"}``
    - ``{"type": "done",   "sources": [...], "stats": {...}}``
    - ``{"type": "error",  "content": "<message>"}``

    Args:
        pool:    A psycopg ConnectionPool.
        graph:   A populated ImportGraph.
        query:   The user's question or stack trace.
        mode:    One of "debug", "feature", "explain".
        model:   Gemini model identifier (default: DEFAULT_MODEL).
        history: Conversation history as list of {"role": ..., "content": ...} dicts.
    """
    if history is None:
        history = []

    # 1. Retrieve relevant chunks.
    try:
        chunks = await retrieve_for_mode(pool, graph, query, mode)
    except Exception as exc:
        yield {"type": "error", "content": f"Retrieval failed: {exc}"}
        return

    # 2. Handle empty results gracefully.
    if not chunks:
        yield {
            "type": "error",
            "content": (
                "No relevant code found in the index for your query. "
                "Make sure the codebase has been indexed (`/index` command) "
                "and try rephrasing your question."
            ),
        }
        return

    # 3. Load static context (CLAUDE.md + memory).
    claude_md, memory_content = _load_static_context()

    # 4. Assemble prompt.
    prompt = assemble_prompt(
        mode=mode,
        query=query,
        chunks=chunks,
        claude_md=claude_md,
        memory_content=memory_content,
        history=history,
        model=model,
    )

    # 5. Build LangChain messages.
    messages: list[Any] = [SystemMessage(content=prompt["system"])]

    # Append trimmed conversation history.
    for turn in prompt["history"]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    # Final human message: context block + query.
    context_block = prompt["context"]
    if context_block:
        human_content = f"## Retrieved Code Context\n\n{context_block}\n\n---\n\n{query}"
    else:
        human_content = query

    messages.append(HumanMessage(content=human_content))

    # 6. Stream from Gemini.
    sources = [
        {"filename": c.get("filename", ""), "priority": c.get("priority", "semantic")}
        for c in chunks
    ]

    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GEMINI_API_KEY,
            streaming=True,
        )

        async for event in llm.astream(messages):
            token = event.content if hasattr(event, "content") else str(event)
            if token:
                yield {"type": "token", "content": token}

        yield {"type": "done", "sources": sources, "stats": prompt["stats"]}

    except Exception as exc:
        yield {"type": "error", "content": f"Gemini API error: {exc}"}
