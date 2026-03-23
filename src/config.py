import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CODEBASE_PATH = Path(os.getenv("CODEBASE_PATH", "../tara_companion")).resolve()

# Indexing
INCLUDED_PATTERNS: list[str] = [
    "lib/**/*.dart",
    "CLAUDE.md",
    "docs/**/*.md",
    ".claude/**/memory/*.md",
    "pubspec.yaml",
    "analysis_options.yaml",
]
EXCLUDED_PATTERNS: list[str] = ["*.g.dart", "*.freezed.dart", "test/**"]
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 300

# Retrieval
TOP_K: int = 8
RELEVANCE_THRESHOLD: float = 0.3

# Generation
DEFAULT_MODEL: str = "claude-sonnet-4-6"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Token budgets
MAX_TOKEN_BUDGET: dict[str, int] = {
    "claude-sonnet-4-6": 50_000,
    "claude-opus-4-6": 100_000,
}
MAX_HISTORY_PAIRS: int = 5
RESPONSE_HEADROOM: int = 8_000  # tokens reserved for Claude's response

# Infrastructure
DATABASE_URL: str = os.getenv(
    "COCOINDEX_DATABASE_URL", "postgresql://localhost:5432/tara_rag"
)
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Server
HOST: str = "127.0.0.1"
PORT: int = 8000
