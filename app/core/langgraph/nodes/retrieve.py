"""Retrieval node: fetch the most relevant SOP chunks for the current query.

Hybrid search: vector similarity and BM25 keyword search each produce a
ranked candidate list, fused via reciprocal rank fusion — vector search
carries semantic/paraphrase matches, BM25 carries exact-term matches (a SOP
code, a batch number) that embeddings don't specially reward.
"""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph.state import Command

from app.core.config import settings
from app.core.logging import logger
from app.core.rag.bm25_index import get_bm25_index
from app.core.rag.fusion import reciprocal_rank_fusion
from app.core.rag.vectorstore import get_vectorstore
from app.schemas.graph import GraphState, RetrievedChunk


def _latest_question(messages: list) -> str:
    """Extract the text of the most recent human message in the conversation."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _chunk_id(metadata: dict) -> str:
    """Stable id for fusing vector and BM25 rankings of the same chunk."""
    return metadata.get("chunk_id") or f"{metadata.get('filename')}:{metadata.get('page')}"


async def retrieve_node(state: GraphState) -> Command:
    """Retrieve the top SOP chunks for the current query via hybrid search.

    On the first pass ``state.query`` is empty, so the query is derived from
    the latest human message. On subsequent passes (after a rewrite),
    ``state.query`` already holds the reformulated query.
    """
    query = state.query or _latest_question(state.messages)
    vectorstore = get_vectorstore()
    doc_filter = {"document_id": {"$in": state.document_ids}} if state.document_ids else None

    # Chroma has no native async client — LangChain's default async
    # VectorStore methods run the sync call in a thread-pool executor.
    vector_hits = await vectorstore.asimilarity_search_with_score(query, k=settings.RAG_CANDIDATE_K, filter=doc_filter)
    # Distance-based pre-filter: drop obviously off-topic candidates before
    # they reach the (more expensive, semantic) grade step. Chroma distance
    # is lower-is-better; threshold derived empirically (see config.py).
    vector_hits = [(doc, score) for doc, score in vector_hits if score <= settings.RAG_SCORE_THRESHOLD]

    candidates: dict[str, RetrievedChunk] = {}
    vector_ranking: list[str] = []
    for doc, score in vector_hits:
        cid = _chunk_id(doc.metadata)
        vector_ranking.append(cid)
        candidates[cid] = RetrievedChunk(
            content=doc.page_content,
            filename=doc.metadata.get("filename", "unknown"),
            page=doc.metadata.get("page", 0),
            score=float(score),
        )

    bm25_ranking: list[str] = []
    if settings.RAG_HYBRID_SEARCH_ENABLED:
        bm25_hits = await asyncio.to_thread(get_bm25_index().top_n, query, settings.RAG_CANDIDATE_K)
        for cid, text, metadata, bm25_score in bm25_hits:
            if state.document_ids and metadata.get("document_id") not in state.document_ids:
                continue
            bm25_ranking.append(cid)
            candidates.setdefault(
                cid,
                RetrievedChunk(
                    content=text,
                    filename=metadata.get("filename", "unknown"),
                    page=metadata.get("page", 0),
                    score=bm25_score,
                ),
            )

    if bm25_ranking:
        fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=settings.RAG_RRF_K)
        ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)
        # Vector distance and BM25 score live on incompatible scales (see
        # fusion.py) — overwrite with the fused RRF score so every chunk in
        # the final list reports a score in the same, meaningful unit
        # (higher is better here, unlike the raw vector distance above).
        for cid in ranked_ids:
            candidates[cid].score = fused[cid]
    else:
        ranked_ids = vector_ranking

    retrieved = [candidates[cid] for cid in ranked_ids[: settings.RAG_TOP_K]]

    logger.info(
        "sop_chunks_retrieved",
        query=query,
        chunk_count=len(retrieved),
        vector_candidates=len(vector_ranking),
        bm25_candidates=len(bm25_ranking),
    )
    return Command(update={"query": query, "retrieved_docs": retrieved}, goto="grade")
