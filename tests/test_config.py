# tests/test_config.py
from pathlib import Path


def test_config_loads():
    from src.config import CODEBASE_PATH, CHUNK_SIZE, TOP_K, MAX_TOKEN_BUDGET

    assert isinstance(CODEBASE_PATH, Path)
    assert CHUNK_SIZE == 1000
    assert TOP_K == 8
    assert "gemini-2.5-flash" in MAX_TOKEN_BUDGET


def test_included_patterns_has_dart():
    from src.config import INCLUDED_PATTERNS

    assert any("*.dart" in p for p in INCLUDED_PATTERNS)


def test_excluded_patterns_filters_generated():
    from src.config import EXCLUDED_PATTERNS

    assert "*.g.dart" in EXCLUDED_PATTERNS
    assert "*.freezed.dart" in EXCLUDED_PATTERNS
