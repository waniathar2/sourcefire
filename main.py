"""Cravv Observatory -- entry point."""

import os
import sys
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg_pool import ConnectionPool

load_dotenv()

from src.api.routes import init_dependencies, router
from src.config import DATABASE_URL, CODEBASE_PATH, HOST, PORT
from src.indexer.pipeline import run_indexing
from src.retriever.graph import ImportGraph


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: index codebase, build graph, inject dependencies."""
    print(f"[main] Starting Cravv Observatory…")
    print(f"[main] Codebase: {CODEBASE_PATH}")

    # 1. Create connection pool.
    pool = ConnectionPool(DATABASE_URL, open=True)

    # 2. Run full indexing pipeline.
    print("[main] Indexing codebase…")
    stats = run_indexing(pool)
    print(f"[main] Indexing complete: {stats}")

    # 3. Build ImportGraph from import_graph DB table.
    graph = ImportGraph()
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT source_path, target_path FROM import_graph"
            ).fetchall()
        for source_path, target_path in rows:
            graph.add_edge(source_path, target_path)
        print(f"[main] Import graph loaded: {graph.node_count} nodes, {len(rows)} edges.")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] Warning: could not load import graph — {exc}")

    # 4. Build index_status dict.
    index_status = {
        "files_indexed": stats.get("files", 0),
        "last_indexed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_status": "ready",
    }

    # 5. Inject dependencies into routes module.
    init_dependencies(pool, graph, index_status)

    # 6. Print URL and optionally open browser.
    url = f"http://{HOST}:{PORT}"
    print(f"[main] Ready — {url}")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)

    yield

    # Shut down: close pool.
    pool.close()
    print("[main] Connection pool closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cravv Observatory",
    description="Local RAG codebase guide for the Cravv Flutter app.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
