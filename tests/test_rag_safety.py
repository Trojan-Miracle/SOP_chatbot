"""RAG must abstain without evidence and expose only final-answer stream events."""

import asyncio
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.nodes import generate
from app.schemas.graph import GradeResult, GraphState, RetrievedChunk
from app.utils.graph import dump_messages
from app.schemas.chat import Message


def test_insufficient_evidence_does_not_call_model(monkeypatch):
    """Exhausted retrieval must yield a stable refusal without another LLM call."""
    model = AsyncMock()
    monkeypatch.setattr(generate.llm_service, "call", model)
    state = GraphState(
        messages=[HumanMessage(content="如何处理？")], grade=GradeResult(sufficient=False, reasoning="没有适用 SOP")
    )
    result = asyncio.run(generate.generate_node(state, {}))
    assert "资料不足" in result.update["messages"][0].content
    assert result.update["sources"] == []
    model.assert_not_called()


def test_stream_filters_intermediate_nodes(monkeypatch):
    """Internal grade/rewrite tokens cannot leak into the user-facing SSE answer."""

    class Snapshot:
        values = {"messages": [AIMessage(content="最终回答")]}

    class Graph:
        async def astream(self, *args, **kwargs):
            yield AIMessageChunk(content="内部判定"), {"langgraph_node": "grade"}
            yield AIMessageChunk(content="内部改写"), {"langgraph_node": "rewrite"}
            yield AIMessageChunk(content="最终回答"), {"langgraph_node": "generate"}

        async def aget_state(self, config):
            return Snapshot()

    async def scenario():
        agent = LangGraphAgent()
        monkeypatch.setattr(agent, "_get_graph", AsyncMock(return_value=Graph()))
        # Avoid background memory writes; this test only verifies emitted messages.
        monkeypatch.setattr("app.core.langgraph.graph.memory_service.search", AsyncMock(return_value=None))
        monkeypatch.setattr("app.core.langgraph.graph.spawn_background_task", lambda coro: coro.close())
        return [chunk async for chunk in agent.get_stream_response([Message(role="user", content="问题")], "test")]

    assert asyncio.run(scenario()) == ["最终回答"]


def test_stream_returns_non_llm_refusal(monkeypatch):
    """A refusal returned directly by a node still reaches streaming clients."""

    class Snapshot:
        values = {"messages": [AIMessage(content="资料不足，请补充 SOP。")]}

    class Graph:
        async def astream(self, *args, **kwargs):
            yield AIMessageChunk(content="内部判定"), {"langgraph_node": "grade"}

        async def aget_state(self, config):
            return Snapshot()

    async def scenario():
        agent = LangGraphAgent()
        monkeypatch.setattr(agent, "_get_graph", AsyncMock(return_value=Graph()))
        monkeypatch.setattr("app.core.langgraph.graph.memory_service.search", AsyncMock(return_value=None))
        monkeypatch.setattr("app.core.langgraph.graph.spawn_background_task", lambda coro: coro.close())
        return [chunk async for chunk in agent.get_stream_response([Message(role="user", content="问题")], "test")]

    assert asyncio.run(scenario()) == ["资料不足，请补充 SOP。"]


def test_langchain_messages_keep_api_roles():
    """Internal LangChain message types must become OpenAI roles before generation."""
    messages = dump_messages([HumanMessage(content="设备异常"), AIMessage(content="请补充型号")])
    assert messages == [{"role": "user", "content": "设备异常"}, {"role": "assistant", "content": "请补充型号"}]


def test_sufficient_evidence_uses_role_messages(monkeypatch):
    """The actual generation path passes usable role messages to the LLM service."""

    async def scenario():
        model = AsyncMock(return_value=AIMessage(content="请登记异常。"))
        monkeypatch.setattr(generate.llm_service, "call", model)
        state = GraphState(
            messages=[HumanMessage(content="设备异常如何登记？")],
            grade=GradeResult(sufficient=True, reasoning="有登记流程"),
            retrieved_docs=[RetrievedChunk(content="登记设备异常。", filename="sop.md", page=1)],
        )
        result = await generate.generate_node(state, {})
        payload = model.call_args.args[0]
        assert payload[0]["role"] == "system"
        assert payload[-1]["role"] == "user"
        assert "登记设备异常" in payload[-1]["content"]
        assert result.update["sources"] == state.retrieved_docs

    asyncio.run(scenario())
