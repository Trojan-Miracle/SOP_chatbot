"""Ensure all workflow adapters use the same ranking and retrieval settings."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.core.incidents.live import LivePlanner
from app.core.langgraph.nodes.retrieve import retrieve_node
from app.core.rag import retrieval
from app.schemas.graph import GraphState


def document(name, content):
    """Build chunks with stable identities and distinct sources."""
    return Document(
        page_content=content, metadata={"chunk_id": name, "document_id": name, "filename": name + ".md", "page": 2}
    )


@pytest.fixture
def backends(monkeypatch):
    """Replace only storage adapters; exercise real ranking and workflow adapters."""
    a, b, c, irrelevant = [
        document(name, text)
        for name, text in [("a", "语义命中"), ("b", "两路共同命中"), ("c", "编号关键词命中"), ("off", "不相关")]
    ]
    vector = Mock()
    vector.asimilarity_search_with_score = AsyncMock(return_value=[(a, 0.1), (b, 0.4), (irrelevant, 2.0)])
    bm25 = Mock()
    bm25.top_n.return_value = [("b", b.page_content, b.metadata, 10.0), ("c", c.page_content, c.metadata, 8.0)]
    monkeypatch.setattr(retrieval, "get_vectorstore", lambda: vector)
    monkeypatch.setattr(retrieval, "get_bm25_index", lambda: bm25)
    for key, value in {
        "RAG_HYBRID_SEARCH_ENABLED": True,
        "RAG_CANDIDATE_K": 15,
        "RAG_TOP_K": 3,
        "RAG_RRF_K": 60,
        "RAG_SCORE_THRESHOLD": 1.2,
    }.items():
        monkeypatch.setattr(retrieval.settings, key, value)
    return vector, bm25


def test_all_paths_share_ranking_and_sources(backends):
    """The graph and investigation share BM25-only hits and ranking."""

    async def scenario():
        query = "SC-100 离线"
        expected = await retrieval.search_sop(query)
        graph = await retrieve_node(GraphState(messages=[HumanMessage(content=query)]))
        investigation = await LivePlanner("test-user").search(query)
        assert [c.filename for c in expected] == ["b.md", "a.md", "c.md"]
        assert graph.update["retrieved_docs"] == expected
        assert graph.update["query"] == query and graph.goto == "grade"
        assert [(c.filename, c.page, c.content) for c in investigation] == [
            (c.filename, c.page, c.content) for c in expected
        ]
        vector, bm25 = backends
        assert vector.asimilarity_search_with_score.await_count == bm25.top_n.call_count == 3
        for call in vector.asimilarity_search_with_score.call_args_list:
            assert call.args == (query,) and call.kwargs == {"k": 15, "filter": None}

    asyncio.run(scenario())


def test_empty_results_across_paths(backends):
    """All paths preserve their empty-result contracts after distance filtering."""
    vector, bm25 = backends
    vector.asimilarity_search_with_score.return_value = [(document("off", "无关"), 2.0)]
    bm25.top_n.return_value = []

    async def scenario():
        assert await retrieval.search_sop("没有依据") == []
        assert (await retrieve_node(GraphState(query="没有依据"))).update["retrieved_docs"] == []
        assert await LivePlanner("test-user").search("没有依据") == []

    asyncio.run(scenario())


def test_document_filter_preserved(backends):
    """Graph filters reach Chroma, while unrelated keyword hits are excluded."""
    vector, _ = backends
    vector.asimilarity_search_with_score.return_value = []
    result = asyncio.run(retrieve_node(GraphState(query="编号", document_ids=["c"])))
    assert [c.filename for c in result.update["retrieved_docs"]] == ["c.md"]
    assert vector.asimilarity_search_with_score.call_args.kwargs["filter"] == {"document_id": {"$in": ["c"]}}


def test_vector_only_ablation_is_shared(backends, monkeypatch):
    """Disabling hybrid retrieval changes every caller through the same setting."""
    monkeypatch.setattr(retrieval.settings, "RAG_HYBRID_SEARCH_ENABLED", False)

    async def scenario():
        expected = await retrieval.search_sop("测试")
        assert [c.filename for c in expected] == ["a.md", "b.md"]
        assert (await retrieve_node(GraphState(query="测试"))).update["retrieved_docs"] == expected
        assert [c.filename for c in await LivePlanner("test-user").search("测试")] == ["a.md", "b.md"]
        backends[1].top_n.assert_not_called()

    asyncio.run(scenario())


def test_rewritten_query_and_top_k_preserved(backends, monkeypatch):
    """A rewritten query overrides the original message and obeys the shared Top-k."""
    monkeypatch.setattr(retrieval.settings, "RAG_TOP_K", 1)
    result = asyncio.run(retrieve_node(GraphState(query="改写后的查询", messages=[HumanMessage(content="原问题")])))
    assert [c.filename for c in result.update["retrieved_docs"]] == ["b.md"]
    assert backends[0].asimilarity_search_with_score.call_args.args == ("改写后的查询",)
