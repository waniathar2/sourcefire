from src.retriever.graph import ImportGraph


def test_add_and_get_imports():
    g = ImportGraph()
    g.add_edge("lib/a.dart", "lib/b.dart")
    g.add_edge("lib/a.dart", "lib/c.dart")
    assert set(g.get_imports("lib/a.dart")) == {"lib/b.dart", "lib/c.dart"}


def test_get_importers_reverse():
    g = ImportGraph()
    g.add_edge("lib/a.dart", "lib/b.dart")
    g.add_edge("lib/c.dart", "lib/b.dart")
    assert set(g.get_importers("lib/b.dart")) == {"lib/a.dart", "lib/c.dart"}


def test_get_neighbors_both_directions():
    g = ImportGraph()
    g.add_edge("lib/a.dart", "lib/b.dart")
    g.add_edge("lib/c.dart", "lib/a.dart")
    neighbors = g.get_neighbors("lib/a.dart")
    assert set(neighbors) == {"lib/b.dart", "lib/c.dart"}


def test_unknown_file_returns_empty():
    g = ImportGraph()
    assert g.get_imports("lib/nonexistent.dart") == []
    assert g.get_importers("lib/nonexistent.dart") == []


def test_build_from_metadata():
    file_imports = {
        "lib/a.dart": ["package:flutter/material.dart", "../b.dart"],
        "lib/b.dart": ["../c.dart"],
    }
    g = ImportGraph.from_import_map(file_imports, base_dir="lib")
    imports = g.get_imports("lib/a.dart")
    # ../b.dart from lib/a.dart resolves to b.dart (one level up from lib/)
    assert "b.dart" in imports or "lib/b.dart" in imports or "../b.dart" in imports
    # package: imports should be excluded
    assert not any(i.startswith("package:") for i in imports)
