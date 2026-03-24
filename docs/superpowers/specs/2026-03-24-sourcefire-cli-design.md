# Sourcefire CLI — Design Spec

**Date:** 2026-03-24
**Status:** Approved
**Goal:** Transform Sourcefire (tara_rag) from a project-local FastAPI app into a globally-installable CLI tool with per-project isolation, embedded vector storage, and zero-config auto-initialization.

---

## 1. Overview

Sourcefire becomes a single command — `sourcefire` — that a developer runs from any project directory. It auto-detects the project, creates a `.sourcefire/` directory for per-project config and data, indexes the codebase into an embedded ChromaDB instance, serves the web UI, and watches for file changes to keep the index live.

### Key Principles

- **Zero config:** First run auto-generates everything via LLM analysis.
- **Zero external services:** No PostgreSQL. ChromaDB (SQLite-backed) is embedded.
- **Per-project isolation:** Each project gets its own `.sourcefire/` with config and data.
- **Single command:** `sourcefire` handles init, indexing, and serving automatically.
- **Globally installable:** `pip install sourcefire` adds the command to PATH.

---

## 2. CLI Interface

### Command

```
sourcefire [--port PORT] [--no-open] [--reinit] [--verbose]
```

- `--port PORT` — Override the default server port (default: 8000).
- `--no-open` — Don't auto-open the browser.
- `--reinit` — Force re-run the LLM-powered config generation (regenerate include/exclude patterns).
- `--verbose` — Enable verbose logging output.

No subcommands. The single command handles everything.

### Project Root Discovery

When `sourcefire` is invoked, it walks up the directory tree from `cwd` looking for an existing `.sourcefire/` directory — similar to how `git` finds `.git/`. If found, that directory's parent is used as the project root. If not found, `cwd` is treated as the project root and auto-init begins.

### Execution Flow

```
sourcefire
  |
  v
Walk up from cwd looking for .sourcefire/
  |
  +-- NOT FOUND --> Auto-Init Flow:
  |                  1. Use cwd as project root
  |                  2. Acquire lockfile (.sourcefire/.lock)
  |                  3. Create .sourcefire/ directory
  |                  4. Scan project file tree (all file paths, not content)
  |                  5. Send tree to LLM to generate include/exclude patterns
  |                  6. Write .sourcefire/config.toml with generated patterns + defaults
  |                  7. Detect language (from file extensions, using existing profile logic)
  |                  8. Full index into .sourcefire/chroma/
  |                  9. Build import graph -> .sourcefire/graph.json
  |                  10. Start FastAPI server + file watcher
  |
  +-- FOUND ----> Read .sourcefire/config.toml
                  1. Acquire lockfile (.sourcefire/.lock)
                  2. Incremental index (diff changed files against ChromaDB state)
                  3. Start FastAPI server + file watcher
```

### Lockfile

`.sourcefire/.lock` prevents concurrent instances from corrupting the ChromaDB database. On startup, attempt to acquire an exclusive file lock (using `fcntl.flock` on macOS/Linux). If the lock is held, print a clear error:

```
Error: Another sourcefire instance is already running for this project.
```

The lock is released on shutdown.

---

## 3. `.sourcefire/` Directory Structure

Created in the root of the target project directory.

```
.sourcefire/
  ├── config.toml       # Project config (LLM-generated on init, user-editable)
  ├── .lock             # Runtime lockfile (prevents concurrent instances)
  ├── chroma/           # ChromaDB persistent storage directory
  └── graph.json        # Import graph edges (adjacency list)
```

### `.sourcefire/config.toml`

```toml
config_version = 1                # Schema version for future migrations

[project]
name = "my-project"               # Auto-detected from directory name
language = "auto"                  # "auto" or forced: "python", "go", "rust", etc.

[indexer]
include = [                        # LLM-generated on first init
    "src/**/*.py",
    "lib/**/*.py",
    "tests/**/*.py",
]
exclude = [                        # LLM-generated on first init
    "__pycache__/**",
    ".venv/**",
    "node_modules/**",
    "dist/**",
    ".git/**",
    ".sourcefire/**",
]
chunk_size = 1000                  # Characters per chunk
chunk_overlap = 300                # Character overlap between chunks

[llm]
provider = "gemini"
model = "gemini-2.5-flash"
api_key_env = "GEMINI_API_KEY"     # Name of env var holding the key (NOT the key itself)

[server]
host = "127.0.0.1"
port = 8000

[retrieval]
top_k = 8
relevance_threshold = 0.3
```

### `.gitignore` consideration

On init, if `.sourcefire/chroma/` and `.sourcefire/graph.json` are not already gitignored, print a suggestion:

```
Tip: Add to your .gitignore:
  .sourcefire/chroma/
  .sourcefire/graph.json
  .sourcefire/.lock
```

The `config.toml` should be committed (it's project config). The data files should not.

---

## 4. Auto-Init: LLM-Powered Config Generation

On first run (no `.sourcefire/` found):

1. **Scan the file tree** — collect all file paths recursively (names and extensions only, not file contents). Cap at 5000 entries to stay within token limits. Include directory structure to give the LLM context about project layout.
2. **Build a prompt** with the tree and send to the configured LLM. The prompt instructs the LLM to respond with a fenced TOML code block:

   > "Given this project file tree, determine which files are source code worth indexing for a code RAG system. Respond with ONLY a TOML code block containing two arrays: `include` (glob patterns for source files, configs, and docs) and `exclude` (glob patterns for build artifacts, dependencies, generated files, and non-code assets). Be comprehensive but conservative."
   >
   > ```toml
   > include = [...]
   > exclude = [...]
   > ```

3. **Parse the LLM response** — extract the TOML code block using regex (`\`\`\`toml\n(.*?)\n\`\`\``), then parse with `tomllib`. If parsing fails, fall back to defaults.
4. **Write `.sourcefire/config.toml`** with the generated patterns plus defaults for all other fields.
5. **Populate `[project].name`** from the directory name.
6. **Detect language** using the existing `get_profile()` logic from `language_profiles.py`.

### Config patterns are authoritative

The `config.toml` include/exclude patterns fully replace (not merge with) language profile patterns for file collection. Language profiles are still used for AST-aware chunking, import extraction, and metadata inference — just not for deciding which files to scan.

### Fallback

If the LLM call fails (no API key, network error), fall back to language-profile-based defaults for include/exclude patterns. The system must never fail to init just because the LLM is unavailable.

### Re-init

`sourcefire --reinit` re-runs the LLM config generation, overwriting the `[indexer]` section of `config.toml` while preserving other user-edited sections. Useful when the project structure changes significantly.

### Empty project detection

If no indexable files are found after applying include/exclude patterns, print:

```
Error: No source files found matching the configured patterns.
Run `sourcefire --reinit` to regenerate patterns, or edit .sourcefire/config.toml manually.
```

---

## 5. Database Migration: PostgreSQL to ChromaDB

### Current State (PostgreSQL + pgvector)

- `code_embeddings` table with 384-dim vectors, metadata columns (filename, location, code, feature, layer, file_type)
- `import_graph` table with source_path/target_path edges
- Uses `psycopg` + `psycopg-pool` for connection pooling
- IVFFlat index for approximate nearest neighbor

### New State (ChromaDB)

**ChromaDB Collection: `code_chunks`**

```python
import chromadb

client = chromadb.PersistentClient(path=".sourcefire/chroma")
collection = client.get_or_create_collection(
    name="code_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

Each document in the collection:
- **id:** `"{filename}:{chunk_index}"` (unique identifier, relies on delete-then-reinsert pattern during re-indexing to avoid orphaned chunks)
- **document:** The code chunk text
- **embedding:** 384-dim vector from sentence-transformers
- **metadata:**
  - `filename` (str) — relative path
  - `location` (str) — `"filename:chunk_index"`
  - `feature` (str) — inferred feature area
  - `layer` (str) — inferred architecture layer
  - `file_type` (str) — inferred file role

**Import Graph: `graph.json`**

```json
{
  "edges": [
    {"source": "src/auth/service.py", "target": "src/auth/models.py"},
    {"source": "src/api/routes.py", "target": "src/auth/service.py"}
  ]
}
```

Stored as a simple JSON file. Loaded into the existing `ImportGraph` data structure at startup. Written to disk on shutdown and periodically (every 60s) during the session.

### Async wrapping (critical)

ChromaDB's Python client is entirely synchronous. Since the FastAPI server, file watcher, and retrieval all share one asyncio event loop, all ChromaDB operations must be wrapped in `asyncio.loop.run_in_executor(None, ...)` to avoid blocking:

```python
async def query_collection(collection, query_embedding, n_results):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    )
```

This applies to all ChromaDB calls: `query()`, `add()`, `delete()`, `upsert()`.

### Retrieval Changes

Replace the raw SQL vector search with ChromaDB's query API:

```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=top_k,
    where={"filename": {"$in": relevant_files}}  # optional graph-based filtering
)
```

ChromaDB handles cosine similarity natively (HNSW index). No manual IVFFlat setup needed.

---

## 6. File Watcher: Live Incremental Re-indexing

### Library

`watchfiles` — async-compatible, efficient filesystem watcher using Rust notify under the hood.

### Behavior

1. On startup, after indexing, start watching the project directory.
2. Apply the same include/exclude patterns from config.toml to filter events.
3. On file change (create/modify):
   - Delete all existing chunks for that file from ChromaDB (`collection.delete(where={"filename": changed_file})`)
   - Re-chunk the file
   - Re-embed the new chunks
   - Upsert into ChromaDB
   - Update import graph edges for that file in memory
4. On file delete:
   - Delete all chunks for that file from ChromaDB
   - Remove import graph edges involving that file in memory
5. Debounce: batch changes within a 1-second window to avoid thrashing during rapid saves.

### Integration

Run the watcher as a background asyncio task alongside the FastAPI server. Both share the same event loop.

### Startup Indexing Strategy

- **First run (after auto-init):** Full index — chunk, embed, and insert all files.
- **Subsequent runs:** Incremental index — compare file mtimes against ChromaDB metadata to detect changes. Only re-index files that are new, modified, or deleted since the last run. This preserves the benefit of ChromaDB's persistent storage and avoids unnecessary re-embedding.

### Graceful Shutdown

On SIGINT/SIGTERM:
1. Cancel the file watcher task
2. Flush or discard any pending debounced re-index batch
3. Write `graph.json` to disk (persist in-memory graph changes)
4. Close the ChromaDB client (ensures WAL data is flushed to SQLite)
5. Release the lockfile

---

## 7. Packaging and Installation

### Package Rename: `src/` -> `sourcefire/`

The top-level Python package is renamed from `src` to `sourcefire` to avoid namespace conflicts when installed globally. This means:

- `src/config.py` -> `sourcefire/config.py`
- `src/api/routes.py` -> `sourcefire/api/routes.py`
- `src/indexer/pipeline.py` -> `sourcefire/indexer/pipeline.py`
- etc.

All internal imports change from `from src.x import y` to `from sourcefire.x import y`.

### Static Files and Prompts (Asset Resolution)

When installed via `pip`, the working directory is the user's project — not the Sourcefire installation directory. Static files and prompts must be located via the package installation path:

```python
from importlib.resources import files

# Static files directory
STATIC_DIR = str(files("sourcefire") / "static")

# System prompt
PROMPTS_DIR = str(files("sourcefire") / "prompts")
```

The `static/` and `prompts/` directories move inside the `sourcefire/` package and are included via `package_data` in pyproject.toml:

```toml
[tool.setuptools.package-data]
sourcefire = ["static/**/*", "prompts/**/*"]
```

### pyproject.toml Changes

```toml
[project]
name = "sourcefire"
version = "0.2.0"
description = "Sourcefire — AI-powered codebase RAG from your terminal"
requires-python = ">=3.11"

dependencies = [
    "chromadb",
    "langchain",
    "langchain-google-genai",
    "fastapi",
    "uvicorn[standard]",
    "sse-starlette",
    "sentence-transformers",
    "tree-sitter",
    "tree-sitter-languages",
    "python-dotenv",
    "watchfiles",
    "tomli-w",            # TOML writing (tomllib in stdlib handles reading)
]

[project.scripts]
sourcefire = "sourcefire.cli:main"

[tool.setuptools.package-data]
sourcefire = ["static/**/*", "prompts/**/*"]
```

### Removed Dependencies

- `psycopg[binary]` — no more PostgreSQL
- `psycopg-pool` — no more connection pooling
- `cocoindex` — was unused
- `tomli` — unnecessary, Python 3.11+ has `tomllib` in stdlib

### Entry Point

`sourcefire/cli.py` — new module replacing `main.py` as the entry point:

```python
def main():
    """Sourcefire CLI entry point."""
    args = parse_args()  # --port, --no-open, --reinit, --verbose

    # Walk up from cwd to find .sourcefire/ or use cwd as root
    project_dir, sourcefire_dir = discover_project()

    # Acquire lockfile
    lock = acquire_lock(sourcefire_dir / ".lock")
    if not lock:
        print("Error: Another sourcefire instance is already running for this project.")
        sys.exit(1)

    if not sourcefire_dir.exists() or args.reinit:
        auto_init(project_dir, sourcefire_dir)

    config = load_config(sourcefire_dir / "config.toml")

    # Run FastAPI app with indexing + watching in lifespan
    run_server(config, project_dir, sourcefire_dir, args)
```

### Global Installation

```bash
pip install sourcefire
# or
pipx install sourcefire
```

After installation, `sourcefire` is available in PATH. Run from any project directory.

---

## 8. Config Module Refactor

`sourcefire/config.py` changes from environment-variable-based to TOML-file-based:

- On init: values come from LLM-generated config + defaults
- On subsequent runs: values read from `.sourcefire/config.toml`
- Environment variables still work as overrides (e.g., `GEMINI_API_KEY`)
- The `CODEBASE_PATH` env var is removed — the project directory is always discovered via `.sourcefire/` traversal or `cwd`
- The `DATABASE_URL` env var is removed — storage is always `.sourcefire/chroma/`
- `CODEBASE_PATH` references in other modules are replaced with the resolved `project_dir` passed through at startup

---

## 9. Affected Modules

| Module | Change |
|--------|--------|
| `main.py` | Replaced by `sourcefire/cli.py` |
| `src/` -> `sourcefire/` | Full package rename |
| `sourcefire/config.py` | Refactored to read from TOML config; remove `CODEBASE_PATH` and `DATABASE_URL` |
| `sourcefire/indexer/pipeline.py` | Replace psycopg with ChromaDB operations; add incremental indexing support |
| `sourcefire/indexer/embeddings.py` | No change (sentence-transformers stays) |
| `sourcefire/indexer/metadata.py` | No change (chunking logic stays) |
| `sourcefire/indexer/language_profiles.py` | No change |
| `sourcefire/retriever/search.py` | Replace SQL queries with ChromaDB query API; wrap in `run_in_executor` |
| `sourcefire/retriever/graph.py` | Load/save from graph.json instead of DB table |
| `sourcefire/chain/rag_chain.py` | Replace `pool` parameter with ChromaDB `collection`; update `CODEBASE_PATH` refs to use `project_dir`; update all tool functions to use new config |
| `sourcefire/chain/prompts.py` | Update system prompt path resolution to use `importlib.resources` |
| `sourcefire/api/routes.py` | Update dependency injection: replace `pool` with ChromaDB `collection` and `project_dir` |
| `sourcefire/api/models.py` | Update model name literals to match new defaults |
| `pyproject.toml` | Update name, deps, add scripts entry, add package-data |
| `static/`, `prompts/` | Move inside `sourcefire/` package directory |
| `.env` / `.env.example` | Remove `DATABASE_URL` and `CODEBASE_PATH`, keep API key only |

---

## 10. What Does NOT Change

- **Frontend** (static/index.html, app.js, styles.css) — unchanged (just relocated into package)
- **Embedding model** — sentence-transformers/all-MiniLM-L6-v2 stays
- **Language profiles** — all 8 language profiles stay
- **AST-aware chunking** — tree-sitter chunking stays
- **SSE streaming** — stays
- **API route structure** — FastAPI routes stay the same (paths, request/response shapes)
- **LangChain + Gemini integration** — stays (function signatures change but the RAG chain logic is preserved)

---

## 11. Error Handling

| Scenario | Behavior |
|----------|----------|
| No GEMINI_API_KEY in env | Print error: "Set GEMINI_API_KEY environment variable" and exit |
| LLM fails during auto-init | Fall back to language-profile defaults for include/exclude |
| ChromaDB write fails | Log error, continue serving with stale index |
| File watcher event on excluded file | Ignore silently |
| Port already in use | Print error with suggestion to use `--port` |
| `.sourcefire/config.toml` is malformed | Print parse error with line number, exit |
| No source files found | Print error suggesting `--reinit` or manual config edit |
| Another instance already running | Print lockfile error, exit |
| Run from subdirectory | Walk up to find `.sourcefire/`, use parent as project root |

---

## 12. Future Considerations (NOT in scope)

- Multiple LLM provider support (OpenAI, Claude, Ollama)
- `sourcefire config` subcommand for interactive config editing
- Remote/team sharing of index
- Plugin system for custom chunkers
- MCP server mode
