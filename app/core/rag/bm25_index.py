"""BM25 keyword index over the SOP chunk corpus.

Vector search alone misses exact-term queries (a SOP code, a batch number) —
paraphrase-based embedding similarity doesn't reward an exact string match
any more than a loosely related one. BM25 catches those; ``retrieve_node``
fuses its ranking with the vector ranking via reciprocal rank fusion.

The index is an in-memory rebuild-from-scratch structure (no incremental
updates) — sized for a demo corpus, not a production-scale one. Rebuilt
after every successful document ingestion.
"""

import jieba
from rank_bm25 import BM25Okapi

from app.core.logging import logger
from app.core.rag.vectorstore import get_vectorstore


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25 term matching."""
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


class BM25Index:
    """In-memory BM25 index over every chunk currently in the vector store."""

    def __init__(self) -> None:
        """Initialize an empty index — call ``rebuild()`` before using it."""
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def rebuild(self) -> None:
        """Rebuild the index from every chunk currently in the vector store.

        Synchronous (chromadb's collection API and jieba tokenization are
        both blocking) — callers in async code should wrap this in
        ``asyncio.to_thread``.
        """
        collection = get_vectorstore()._collection
        result = collection.get(include=["documents", "metadatas"])
        self._ids = result["ids"]
        self._texts = result["documents"] or []
        self._metadatas = [dict(metadata) for metadata in (result["metadatas"] or [])]

        if not self._texts:
            self._bm25 = None
            logger.info("bm25_index_rebuilt", chunk_count=0)
            return

        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts])
        logger.info("bm25_index_rebuilt", chunk_count=len(self._texts))

    def top_n(self, query: str, n: int) -> list[tuple[str, str, dict, float]]:
        """Return up to ``n`` chunks ranked by BM25 score, highest first.

        Returns:
            List of ``(chunk_id, text, metadata, bm25_score)``, filtering out
            zero-score (no keyword overlap at all) results.
        """
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            (self._ids[i], self._texts[i], self._metadatas[i], float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0
        ]


_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    """Return the shared BM25 index singleton."""
    return _bm25_index
