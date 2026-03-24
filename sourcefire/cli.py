"""Sourcefire CLI — single command entry point."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import sys
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sourcefire",
        description="Sourcefire — AI-powered codebase RAG from your terminal",
    )
    parser.add_argument("--port", type=int, default=None, help="Server port (default: from config or 8000)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--reinit", action="store_true", help="Regenerate .sourcefire/config.toml via LLM")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def discover_project() -> tuple[Path, Path]:
    """Walk up from cwd to find .sourcefire/, like git finds .git/.

    Returns (project_dir, sourcefire_dir).
    If not found, returns (cwd, cwd/.sourcefire).
    """
    current = Path.cwd().resolve()
    while True:
        candidate = current / ".sourcefire"
        if candidate.is_dir():
            return current, candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    cwd = Path.cwd().resolve()
    return cwd, cwd / ".sourcefire"


def acquire_lock(lock_path: Path) -> int | None:
    """Acquire an exclusive file lock. Returns fd on success, None on failure."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (OSError, BlockingIOError):
        return None


def release_lock(fd: int, lock_path: Path) -> None:
    """Release the file lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# App state (shared between main() and lifespan)
# ---------------------------------------------------------------------------

_app_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: index codebase, build graph, start watcher."""
    config = _app_state["config"]
    project_dir = _app_state["project_dir"]
    api_key = _app_state["api_key"]
    args = _app_state["args"]

    from sourcefire.db import create_client, get_collection
    from sourcefire.indexer.language_profiles import get_profile
    from sourcefire.indexer.pipeline import run_indexing
    from sourcefire.retriever.graph import ImportGraph
    from sourcefire.api.routes import init_dependencies
    from sourcefire.watcher import watch_and_reindex

    print(f"[sourcefire] Project: {project_dir.name}")
    print(f"[sourcefire] Config: {config.config_path}")

    # Detect language
    language_override = config.language if config.language != "auto" else None
    profile = get_profile(project_dir, language_override)
    lang_name = profile.language if profile else "generic"
    print(f"[sourcefire] Language: {lang_name}")

    # Create ChromaDB client
    client = create_client(config.chroma_dir)
    collection = get_collection(client)

    # Determine if this is a first run (empty collection)
    existing_count = collection.count()
    is_first_run = existing_count == 0

    # Run indexing
    if is_first_run:
        print("[sourcefire] First run — full index...")
        stats = run_indexing(collection, config, client=client, full=True)
    else:
        print("[sourcefire] Checking for changes...")
        stats = run_indexing(collection, config, client=client, full=False)

    print(f"[sourcefire] Indexed: {stats['files']} files, {stats['chunks']} chunks")

    # Build import graph
    external_prefixes = profile.external_import_prefixes if profile else ()
    graph = ImportGraph(external_prefixes=external_prefixes)

    import_edges = stats.get("import_edges", {})
    if import_edges:
        for source_file, imports in import_edges.items():
            for imp in imports:
                resolved = ImportGraph._resolve_import(source_file, imp)
                graph.add_edge(source_file, resolved)
    elif config.graph_path.is_file():
        graph = ImportGraph.load(config.graph_path, external_prefixes=external_prefixes)

    print(f"[sourcefire] Import graph: {graph.node_count} nodes")

    # Build index status
    index_status = {
        "files_indexed": stats.get("files", 0),
        "last_indexed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_status": "ready",
        "language": lang_name,
    }

    # Inject dependencies into routes
    init_dependencies(collection, graph, index_status, profile, project_dir, api_key)

    # Start file watcher
    watcher_task = asyncio.create_task(
        watch_and_reindex(config, collection, graph, profile)
    )

    # Open browser
    url = f"http://{config.host}:{config.port}"
    print(f"[sourcefire] Ready — {url}")
    if not args.no_open:
        webbrowser.open(url)

    yield

    # Shutdown
    print("[sourcefire] Shutting down...")
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass

    # Save graph
    graph.save(config.graph_path)
    print("[sourcefire] Graph saved.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

from importlib.resources import files as _resource_files

_static_dir = str(Path(_resource_files("sourcefire")) / "static")

app = FastAPI(
    title="Sourcefire",
    description="AI-powered codebase RAG. Created by Athar Wani.",
    version="0.2.0",
    lifespan=lifespan,
)

from sourcefire.api.routes import router  # noqa: E402

app.include_router(router)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(os.path.join(_static_dir, "index.html"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Sourcefire CLI entry point."""
    args = parse_args()

    project_dir, sourcefire_dir = discover_project()

    # Acquire lock
    lock_fd = acquire_lock(sourcefire_dir / ".lock")
    if lock_fd is None:
        print("Error: Another sourcefire instance is already running for this project.")
        sys.exit(1)

    # Check for API key — prompt interactively if missing
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("No GEMINI_API_KEY found in environment.")
        try:
            api_key = input("Enter your Gemini API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            release_lock(lock_fd, sourcefire_dir / ".lock")
            sys.exit(1)

        if not api_key:
            print("Error: A Gemini API key is required to run Sourcefire.")
            release_lock(lock_fd, sourcefire_dir / ".lock")
            sys.exit(1)

        # Persist to .env in project root
        env_path = project_dir / ".env"
        with open(env_path, "a") as f:
            f.write(f"\nGEMINI_API_KEY={api_key}\n")
        os.environ["GEMINI_API_KEY"] = api_key
        print(f"API key saved to {env_path}")

    # Auto-init or reinit
    needs_init = not sourcefire_dir.exists() or not (sourcefire_dir / "config.toml").exists()

    if needs_init:
        from sourcefire.init import auto_init
        config = auto_init(
            project_dir=project_dir,
            sourcefire_dir=sourcefire_dir,
            api_key=api_key,
        )
    elif args.reinit:
        from sourcefire.config import load_config
        from sourcefire.init import reinit_patterns
        config = load_config(project_dir, sourcefire_dir)
        config = reinit_patterns(config, api_key=api_key)
    else:
        from sourcefire.config import load_config
        config = load_config(project_dir, sourcefire_dir)

    # Override port from CLI
    if args.port:
        config.port = args.port

    # Store state for lifespan access
    _app_state["config"] = config
    _app_state["project_dir"] = project_dir
    _app_state["sourcefire_dir"] = sourcefire_dir
    _app_state["api_key"] = api_key
    _app_state["args"] = args
    _app_state["lock_fd"] = lock_fd

    # Run server
    try:
        uvicorn.run(
            "sourcefire.cli:app",
            host=config.host,
            port=config.port,
            reload=False,
            log_level="info" if args.verbose else "warning",
        )
    finally:
        release_lock(lock_fd, sourcefire_dir / ".lock")


if __name__ == "__main__":
    main()
