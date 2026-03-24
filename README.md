# Sourcefire

You inherited a codebase. Maybe AI wrote it. Maybe a dev who left wrote it. Either way, you have no idea what's going on.

Sourcefire gives you instant context. Point it at any project, ask questions in plain English, and get answers grounded in the actual code — not hallucinations.

```bash
pip install sourcefire
cd your-project
sourcefire
```

One command. No config. It indexes your codebase, opens a web UI, and you start asking:

- *"Where does authentication happen?"*
- *"What does this error mean?"* (paste the stack trace)
- *"How would I add a new API endpoint following the existing patterns?"*
- *"Explain the data flow from request to database"*

Sourcefire reads the code so you don't have to.

## Why

AI-generated codebases are everywhere now. Cursor, Copilot, Claude — they write thousands of lines fast, but the human on the team still needs to understand what was built, where things live, and how it all connects.

Reading every file isn't realistic. Grep only works when you know what to search for. Sourcefire fills the gap: it understands your code structurally and lets you query it conversationally.

## How It Works

Sourcefire creates a `.sourcefire/` directory in your project:

```
.sourcefire/
  ├── config.toml    # Auto-generated project config
  ├── chroma/        # Local vector database
  └── graph.json     # Import graph
```

On first run, it:
1. Scans your project structure and asks an LLM to figure out what to index
2. Chunks your code at function/class boundaries (AST-aware, not dumb line splits)
3. Embeds everything into a local ChromaDB database — no server, no PostgreSQL
4. Builds an import graph so it can trace dependencies
5. Starts a web UI and watches for file changes to keep the index live

Everything stays local. No data leaves your machine except the queries you send to Gemini.

## Requirements

- Python 3.11+
- A [Gemini API key](https://ai.google.dev/) — Sourcefire prompts for it on first run and saves it to `.env`

## What You Can Ask

**Debug mode** — paste a stack trace or error, Sourcefire traces through the actual files:
> *"I'm getting a 500 error on /api/users — the traceback mentions auth_middleware.py:34"*

**Feature mode** — understand architecture and add new code in the right place:
> *"How is the payment system structured? Where would I add refund support?"*

**Explain mode** — get walkthroughs of how things connect:
> *"Walk me through what happens when a user signs up, from the route handler to the database"*

The LLM has 18 tools at its disposal — it can read files, search code, trace call chains, check git blame, find definitions, and more. It doesn't just answer from the index; it actively explores your codebase during the conversation.

## Languages

Python, JavaScript/TypeScript, Go, Rust, Java, Dart, C, C++. Falls back to plain text chunking for anything else.

## CLI

```
sourcefire [--port PORT] [--no-open] [--reinit] [--verbose]
```

| Flag | Description |
|------|-------------|
| `--port PORT` | Server port (default: 8000) |
| `--no-open` | Don't auto-open browser |
| `--reinit` | Re-generate include/exclude patterns via LLM |
| `--verbose` | Verbose logging |

Run from any subdirectory — Sourcefire walks up the tree to find `.sourcefire/`, just like `git` finds `.git/`.

## Configuration

After first run, edit `.sourcefire/config.toml` if you want to tweak anything:

```toml
[project]
name = "my-project"
language = "auto"

[indexer]
include = ["src/**/*.py"]
exclude = ["__pycache__/**", ".venv/**"]

[llm]
model = "gemini-2.5-flash"

[retrieval]
top_k = 8
```

Most people never touch this. The defaults work.

## License

MIT — Built by [Athar Wani](https://github.com/waniathar2) / Cravv HQ
