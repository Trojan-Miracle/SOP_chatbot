"""Chroma vector store singleton for SOP document chunks."""

from functools import lru_cache

from langchain_chroma import Chroma

from app.core.config import settings
from app.core.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the shared Chroma collection instance, creating it on first call.

    Chroma's client is synchronous under the hood; callers should use the
    ``a*`` methods (``aadd_documents``, ``asimilarity_search``, ...) which
    LangChain's ``VectorStore`` base class runs in a thread-pool executor by
    default, so the event loop is never blocked.
    """
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
