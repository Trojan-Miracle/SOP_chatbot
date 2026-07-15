"""Reciprocal rank fusion for combining vector and keyword search rankings.

Vector distance and BM25 score live on incompatible scales (bounded cosine
distance vs. unbounded corpus-dependent term-frequency score), so summing
them directly is meaningless. RRF sidesteps that by fusing on *rank position*
instead of raw score — the standard, scale-free way to combine heterogeneous
rankings (used by Elasticsearch, Weaviate, and others).
"""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Fuse multiple rankings of the same ID space into one score per ID.

    Args:
        rankings: Each element is a list of IDs in ranked order (best first).
        k: RRF constant — higher values flatten the influence of rank
            position (the ``+k`` in ``1/(k + rank + 1)``); 60 is the
            commonly cited default.

    Returns:
        Mapping of ID to fused score (higher is better), for IDs appearing
        in at least one ranking.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
