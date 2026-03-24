"""Test that run_indexing always returns 'collection' key in stats dict."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb

from sourcefire.indexer.pipeline import run_indexing
from sourcefire.config import SourcefireConfig
from sourcefire.db import create_client, get_collection, reset_collection


def _make_config(tmp: Path, project: Path) -> SourcefireConfig:
    """Create a minimal SourcefireConfig pointing at temp dirs."""
    sf_dir = tmp / ".sourcefire"
    sf_dir.mkdir(parents=True, exist_ok=True)
    return SourcefireConfig(
        sourcefire_dir=sf_dir,
        project_dir=project,
        include=["*.py"],
        exclude=[],
    )


def test_collection_key_on_empty_project():
    """run_indexing should return 'collection' even when no source files found."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project = tmp_path / "project"
        project.mkdir()

        config = _make_config(tmp_path, project)
        client = create_client(config.chroma_dir)
        collection = get_collection(client)

        stats = run_indexing(collection, config, client=client, full=True)
        assert "collection" in stats, "Missing 'collection' key when no source files found"


def test_collection_key_on_up_to_date_index():
    """run_indexing incremental should return 'collection' when index is already current."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project = tmp_path / "project"
        project.mkdir()

        # Create a source file
        (project / "main.py").write_text("print('hello')\n")

        config = _make_config(tmp_path, project)
        client = create_client(config.chroma_dir)
        collection = get_collection(client)

        # Full index first
        stats = run_indexing(collection, config, client=client, full=True)
        assert "collection" in stats
        collection = stats["collection"]

        # Incremental — nothing changed
        stats2 = run_indexing(collection, config, client=client, full=False)
        assert "collection" in stats2, "Missing 'collection' key when index is up to date"


def test_collection_key_on_no_chunks_produced():
    """run_indexing should return 'collection' even when chunking produces 0 chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project = tmp_path / "project"
        project.mkdir()

        # Create an empty .py file (will produce 0 chunks)
        (project / "empty.py").write_text("")

        config = _make_config(tmp_path, project)
        client = create_client(config.chroma_dir)
        collection = get_collection(client)

        stats = run_indexing(collection, config, client=client, full=True)
        assert "collection" in stats, "Missing 'collection' key when no chunks produced"


def test_chroma_dir_nuke_on_deep_corruption():
    """When ChromaDB is corrupted beyond reset, nuking the dir should recover.

    Simulates the recovery pattern from cli.py lifespan. Since ChromaDB keeps
    in-process state that survives rmtree within the same process, we verify
    the recovery by using a fresh directory (simulating what happens across
    a server restart).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chroma_dir = tmp_path / "chroma"

        # Create a valid client and collection first
        client = create_client(chroma_dir)
        collection = get_collection(client)
        collection.add(ids=["test1"], documents=["hello"], embeddings=[[0.1] * 384])
        assert collection.count() == 1

        # Corrupt the SQLite database by overwriting it
        for f in chroma_dir.rglob("*.sqlite3"):
            f.write_bytes(b"CORRUPTED DATA NOT SQLITE")

        # Verify corruption causes failure at client creation level
        try:
            c2 = create_client(chroma_dir)
            get_collection(c2).count()
            corruption_detected = False
        except Exception:
            corruption_detected = True

        assert corruption_detected, "Expected corruption to cause an error"

        # Recovery: nuke dir and rebuild in a fresh directory
        # (In production, this is the same path after rmtree + process restart)
        recovery_dir = tmp_path / "chroma_recovered"
        shutil.rmtree(chroma_dir, ignore_errors=True)
        client3 = create_client(recovery_dir)
        coll3 = get_collection(client3)
        assert coll3.count() == 0, "Fresh collection after nuke should be empty"


def test_reset_collection_catches_all_exceptions():
    """reset_collection should not raise even if delete_collection fails."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chroma_dir = tmp_path / "chroma"
        client = create_client(chroma_dir)

        # Mock delete_collection to raise a generic exception
        original_delete = client.delete_collection
        def failing_delete(name):
            raise RuntimeError("SQLite table corrupted")
        client.delete_collection = failing_delete

        # Should not raise — catches Exception
        collection = reset_collection(client)
        assert collection is not None
        assert collection.count() == 0
