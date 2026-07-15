"""Embedding model loader.

The model is loaded once at process startup (see ``app.main`` lifespan) and
reused across requests — ``HuggingFaceEmbeddings`` is expensive to construct
(it loads the sentence-transformers weights) so we never build it per-request.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance, creating it on first call."""
    logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("embedding_model_loaded", model=settings.EMBEDDING_MODEL)
    return embeddings
