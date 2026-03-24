"""Sourcefire CLI — single command entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

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
    parser.add_argument("--uninstall", action="store_true", help="Remove global ~/.sourcefire/ config directory")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
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
        if sys.platform == "win32":
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (OSError, BlockingIOError):
        return None


def release_lock(fd: int, lock_path: Path) -> None:
    """Release the file lock."""
    try:
        if sys.platform == "win32":
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def _port_available(host: str, port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(host: str, preferred: int, max_attempts: int = 20) -> int:
    """Find an available port, starting from preferred and incrementing."""
    for offset in range(max_attempts):
        port = preferred + offset
        if _port_available(host, port):
            return port
    raise RuntimeError(f"No available port found in range {preferred}-{preferred + max_attempts - 1}")


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
        "project_name": config.project_name or project_dir.name,
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

from importlib.metadata import version as _pkg_version
from importlib.resources import files as _resource_files

_static_dir = str(Path(_resource_files("sourcefire")) / "static")

try:
    _version = _pkg_version("sourcefire")
except Exception:
    _version = "0.0.0"

app = FastAPI(
    title="Sourcefire",
    description="AI-powered codebase RAG. Created by Athar Wani.",
    version=_version,
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

    # Handle --version
    if args.version:
        from importlib.metadata import version as _get_ver
        try:
            print(f"sourcefire {_get_ver('sourcefire')}")
        except Exception:
            print("sourcefire (version unknown)")
        return

    # Handle --uninstall
    if args.uninstall:
        from sourcefire.global_config import uninstall
        uninstall()
        return

    project_dir, sourcefire_dir = discover_project()

    # Safety check: warn if running in a broad directory (home, /, etc.)
    needs_init = not sourcefire_dir.exists() or not (sourcefire_dir / "config.toml").exists()
    if needs_init:
        dangerous_dirs = {
            Path.home().resolve(),
            Path("/").resolve(),
        }
        # Also flag common broad directories
        for name in ("Documents", "Downloads", "Desktop"):
            dangerous_dirs.add((Path.home() / name).resolve())

        if project_dir.resolve() in dangerous_dirs:
            print(f"\n  WARNING: You are about to index: {project_dir.resolve()}")
            print("  This is a broad directory and may index thousands of files.\n")
            try:
                confirm = input("  Do you trust this folder? (yes/no): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            if confirm not in ("yes", "y"):
                print("Aborted. Run sourcefire from a project directory instead.")
                sys.exit(0)

    # Acquire lock
    lock_fd = acquire_lock(sourcefire_dir / ".lock")
    if lock_fd is None:
        print("Error: Another sourcefire instance is already running for this project.")
        sys.exit(1)

    # Check for API key: env var -> ~/.sourcefire/config.toml -> prompt
    from sourcefire.global_config import get_api_key, save_api_key, get_global_dir

    api_key = get_api_key()
    if not api_key:
        print("No Gemini API key found.")
        print(f"It will be saved to {get_global_dir() / 'config.toml'} (global, works across all projects).\n")
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

        save_api_key(api_key)
        print(f"API key saved to {get_global_dir() / 'config.toml'}\n")

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

    # Find available port (auto-increment if taken)
    try:
        actual_port = find_available_port(config.host, config.port)
    except RuntimeError:
        print(f"Error: No available port found starting from {config.port}.")
        release_lock(lock_fd, sourcefire_dir / ".lock")
        sys.exit(1)

    if actual_port != config.port:
        print(f"Port {config.port} is in use, using {actual_port} instead.")
    config.port = actual_port

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
