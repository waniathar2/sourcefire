# Sourcefire

AI-powered codebase RAG from your terminal. Index any project, ask questions, get answers with full source context.

## Install

```bash
pip install sourcefire
```

## Quick Start

```bash
cd your-project
sourcefire
```

That's it. Sourcefire will:

1. **Auto-detect** your project structure and language
2. **Generate** smart include/exclude patterns via LLM
3. **Index** your codebase into a local ChromaDB vector database
4. **Serve** a web UI where you can ask questions about your code
5. **Watch** for file changes and re-index automatically

## Requirements

- Python 3.11+
- A [Gemini API key](https://ai.google.dev/) (prompted on first run, saved to `.env`)

## How It Works

Sourcefire creates a `.sourcefire/` directory in your project root:

```
.sourcefire/
  ├── config.toml    # Project config (auto-generated, editable)
  ├── chroma/        # Vector database (local, no server needed)
  └── graph.json     # Import graph for code navigation
```

- **No PostgreSQL** — uses ChromaDB (SQLite-backed, embedded)
- **No external services** — everything runs locally
- **Per-project isolation** — each project gets its own database

## Features

- **Zero config** — first run auto-generates everything via LLM analysis
- **8 language profiles** — Python, JavaScript/TypeScript, Go, Rust, Java, Dart, C, C++
- **AST-aware chunking** — splits code at function/class boundaries using tree-sitter
- **Live re-indexing** — file watcher detects changes and re-indexes automatically
- **3 query modes** — Debug (stack traces), Feature (architecture), Explain (walkthroughs)
- **18 code exploration tools** — the LLM can read files, search code, trace call chains, git blame, and more
- **Incremental indexing** — only re-indexes files that changed since last run

## CLI Options

```
sourcefire [--port PORT] [--no-open] [--reinit] [--verbose]
```

| Flag | Description |
|------|-------------|
| `--port PORT` | Server port (default: 8000) |
| `--no-open` | Don't auto-open browser |
| `--reinit` | Regenerate include/exclude patterns via LLM |
| `--verbose` | Verbose logging |

## Configuration

Edit `.sourcefire/config.toml` to customize:

```toml
[project]
name = "my-project"
language = "auto"          # or "python", "go", "rust", etc.

[indexer]
include = ["src/**/*.py"]  # glob patterns to index
exclude = ["__pycache__/**", ".venv/**"]
chunk_size = 1000
chunk_overlap = 300

[llm]
model = "gemini-2.5-flash"
api_key_env = "GEMINI_API_KEY"

[server]
port = 8000

[retrieval]
top_k = 8
relevance_threshold = 0.3
```

## Subdirectory Support

Run `sourcefire` from any subdirectory — it walks up the tree to find `.sourcefire/`, just like `git` finds `.git/`.

## License

MIT — Created by [Athar Wani](https://github.com/waniathar2) / Cravv HQ
