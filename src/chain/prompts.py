"""Prompt assembly and token budget management for Cravv Observatory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import MAX_HISTORY_PAIRS, MAX_TOKEN_BUDGET, RESPONSE_HEADROOM

# ---------------------------------------------------------------------------
# System template
# ---------------------------------------------------------------------------

_SYSTEM_MD_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "system.md"
_SYSTEM_TEMPLATE: str = _SYSTEM_MD_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Priority ordering (lower = higher priority)
# ---------------------------------------------------------------------------

_PRIORITY_ORDER: dict[str, int] = {"direct": 0, "semantic": 1, "graph": 2}

# ---------------------------------------------------------------------------
# Per-mode suffixes appended to the system prompt
# ---------------------------------------------------------------------------

_MODE_SUFFIXES: dict[str, str] = {
    "debug": (
        "\n\n## Mode: Debug\n"
        "Focus on diagnosing the root cause. "
        "Trace the call chain from the error site back to its origin. "
        "Show the exact files and line regions involved. "
        "Suggest a minimal, targeted fix."
    ),
    "feature": (
        "\n\n## Mode: Feature\n"
        "Focus on where the new code should live in the feature-first clean architecture. "
        "Identify the correct layer (data / domain / presentation) and file location. "
        "Point to similar existing features as implementation references."
    ),
    "explain": (
        "\n\n## Mode: Explain\n"
        "Focus on clarity. "
        "Walk through the relevant files in dependency order (data → domain → presentation). "
        "Use analogies where helpful, but always ground explanations in actual file paths."
    ),
}

# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Chunk truncation with token budget enforcement
# ---------------------------------------------------------------------------

_MAX_CHUNK_CHARS = 6_000  # ~1500 tokens per chunk


def truncate_chunks(
    chunks: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Return a subset of *chunks* that fits within *max_tokens*.

    Strategy:
    1. Sort by priority (direct < semantic < graph) then descending relevance.
    2. Truncate any single chunk's code to ``_MAX_CHUNK_CHARS`` before counting.
    3. Drop chunks (lowest priority / relevance first) until budget is met.
    """
    # Sort: primary = priority order, secondary = relevance descending
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            _PRIORITY_ORDER.get(c.get("priority", "graph"), 2),
            -float(c.get("relevance", 0.0)),
        ),
    )

    # Truncate individual chunk code to cap per-chunk token cost
    capped: list[dict[str, Any]] = []
    for chunk in sorted_chunks:
        c = dict(chunk)
        if len(c.get("code", "")) > _MAX_CHUNK_CHARS:
            c["code"] = c["code"][:_MAX_CHUNK_CHARS]
        capped.append(c)

    # Greedy inclusion: keep adding chunks while under budget
    result: list[dict[str, Any]] = []
    used_tokens = 0
    for chunk in capped:
        chunk_tokens = estimate_tokens(chunk.get("code", ""))
        if used_tokens + chunk_tokens <= max_tokens:
            result.append(chunk)
            used_tokens += chunk_tokens
        # If this chunk doesn't fit, skip it (preserves higher-priority chunks)

    return result


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def assemble_prompt(
    mode: str,
    query: str,
    chunks: list[dict[str, Any]],
    claude_md: str,
    memory_content: str,
    history: list[dict[str, str]],
    model: str,
) -> dict[str, Any]:
    """Assemble the full prompt dict for the LLM call.

    Returns a dict with keys:
    - ``system``   — full system prompt (template + claude_md + memory + mode suffix)
    - ``context``  — formatted retrieved code chunks
    - ``query``    — the user's question (unchanged)
    - ``history``  — trimmed conversation history (≤ MAX_HISTORY_PAIRS pairs)
    - ``stats``    — token usage summary dict
    """
    # 1. Compute available context budget
    token_budget = MAX_TOKEN_BUDGET.get(model, 100_000)
    system_tokens = estimate_tokens(_SYSTEM_TEMPLATE + claude_md + memory_content)
    query_tokens = estimate_tokens(query)
    history_tokens = sum(estimate_tokens(m.get("content", "")) for m in history)
    overhead = system_tokens + query_tokens + history_tokens + RESPONSE_HEADROOM
    context_budget = max(0, token_budget - overhead)

    # 2. Truncate chunks to fit context budget
    kept_chunks = truncate_chunks(chunks, max_tokens=context_budget)

    # 3. Build system prompt
    mode_suffix = _MODE_SUFFIXES.get(mode, "")
    system_parts = [
        _SYSTEM_TEMPLATE,
        "\n\n---\n\n## Project Rules (CLAUDE.md)\n\n",
        claude_md,
        "\n\n---\n\n## Developer Memory\n\n",
        memory_content,
        mode_suffix,
    ]
    system_prompt = "".join(system_parts)

    # 4. Build context block
    context_parts: list[str] = []
    for chunk in kept_chunks:
        filename = chunk.get("filename", "unknown")
        location = chunk.get("location", "")
        relevance = chunk.get("relevance", 0.0)
        code = chunk.get("code", "")
        header = f"### {filename}"
        if location:
            header += f" ({location})"
        header += f"  [relevance: {relevance:.2f}]"
        context_parts.append(f"{header}\n```dart\n{code}\n```")

    context_block = "\n\n".join(context_parts)

    # 5. Trim history to MAX_HISTORY_PAIRS
    trimmed_history = history[-(MAX_HISTORY_PAIRS * 2):]

    # 6. Stats
    context_tokens = estimate_tokens(context_block)
    stats = {
        "model": model,
        "token_budget": token_budget,
        "system_tokens": system_tokens,
        "context_tokens": context_tokens,
        "query_tokens": query_tokens,
        "history_tokens": history_tokens,
        "total_estimated": system_tokens + context_tokens + query_tokens + history_tokens,
        "chunks_used": len(kept_chunks),
        "chunks_dropped": len(chunks) - len(kept_chunks),
    }

    return {
        "system": system_prompt,
        "context": context_block,
        "query": query,
        "history": trimmed_history,
        "stats": stats,
    }
