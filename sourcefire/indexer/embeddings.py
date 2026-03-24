"""Shared embedding module using sentence-transformers directly."""

from sentence_transformers import SentenceTransformer
from sourcefire.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns a list of floats."""
    model = get_model()
    return model.encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. More efficient than calling embed_text in a loop."""
    model = get_model()
    return model.encode(texts).tolist()
