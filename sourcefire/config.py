import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CODEBASE_PATH = Path(os.getenv("CODEBASE_PATH", ".")).resolve()

# Project identity (used in UI, prompts, API metadata)
PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Sourcefire")

# Language override — if not set, auto-detected from codebase marker files
LANGUAGE_OVERRIDE: str | None = os.getenv("LANGUAGE", None)

# Indexing — base patterns; language profile patterns are merged at runtime
EXTRA_INCLUDE_PATTERNS: list[str] = [
    "CLAUDE.md",
    "README.md",
]
EXTRA_EXCLUDE_PATTERNS: list[str] = [
    ".claude/**",
    "docs/superpowers/**",
]
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 300

# Retrieval
TOP_K: int = 8
RELEVANCE_THRESHOLD: float = 0.3

# Generation
DEFAULT_MODEL: str = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Token budgets
MAX_TOKEN_BUDGET: dict[str, int] = {
    "gemini-3.1-flash-lite-preview": 100_000,
    "gemini-3.1-pro-preview": 200_000,
}
MAX_HISTORY_PAIRS: int = 5
RESPONSE_HEADROOM: int = 8_000  # tokens reserved for model response

# Infrastructure
DATABASE_URL: str = os.getenv(
    "COCOINDEX_DATABASE_URL", "postgresql://localhost:5432/tara_rag"
)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Server
HOST: str = "127.0.0.1"
PORT: int = 8000
