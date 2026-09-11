> 本文保留原 RAG / ReAct 架构说明。当前新增事件闭环见 [项目首页](../README.md)；RAG 资料不足时已改为明确拒答。

# SOP Agentic RAG

An internal SOP (Standard Operating Procedure) question-answering system implementing
two complementary architectural paradigms for grounding an LLM in internal knowledge
and giving it tools:

1. **A fixed Agentic RAG graph** (`/chatbot/chat`) — upload SOP documents, ask
   questions, get answers grounded in retrieved excerpts with page-level citations.
   Every question always goes through the same retrieve → grade → generate pipeline.
2. **A lightweight ReAct tool-calling harness** (`/harness/chat`) — the same
   underlying knowledge base, but exposed as one tool among several (knowledge-base
   search, current time via a real external API, sandboxed codebase read/grep). The
   model decides for itself whether and which tools a question needs.

Adapted from [fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template),
keeping its auth/persistence/observability scaffolding and building both agents on top.

## Why two architectures

Both are implemented side by side because the right choice depends on what's being
retrieved, not because one strictly supersedes the other:

|                          | Agentic RAG graph                              | ReAct harness                                    |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------- |
| Control flow             | Fixed pipeline, same steps every query           | Model decides which tools to call, if any          |
| Retrieval                | Always runs, with a grade/rewrite quality loop   | One tool among several — only called if the model decides it's needed |
| Best fit                 | High-stakes domain where every answer must be grounded and auditable | General-purpose assistant needing several capabilities (docs, time, code, ...) |
| This project's analogue  | `app/core/langgraph/`                             | `app/core/harness/`                                |

Coding agents (Claude Code, Devin, etc.) lean on the harness side for code — grep/read
tools give exact-match recall that beats semantic retrieval for symbols and file
paths — while document QA still benefits from the graph side's mandatory
grade-before-answer discipline. Both patterns are useful; which one fits depends on
what's being retrieved and how much control flow needs to be guaranteed versus
delegated to the model.

## Architecture — Agentic RAG graph

```
retrieve -> grade -> generate            (sufficient context)
              \-> rewrite -> retrieve     (insufficient, loop up to RAG_MAX_REWRITES times)
```

- **retrieve**: hybrid search — vector similarity (Chroma/bge-m3) and BM25 keyword
  matching each produce a ranked candidate list, fused via reciprocal rank fusion.
  Vector search alone misses exact-term queries (a SOP code, a batch number) that
  paraphrase-based embeddings don't specially reward; BM25 catches those. A
  distance-based threshold (`RAG_SCORE_THRESHOLD`) also drops obviously off-topic
  candidates before they reach the LLM.
- **grade**: LLM judges whether the retrieved chunks are sufficient to answer.
- **rewrite**: LLM reformulates the query (more specific terms, expanded abbreviations)
  and loops back.
- **generate**: answers using only the retrieved chunks, citing `(来源: filename, 第N页)`
  per claim.

Optional `document_ids` filter restricts retrieval to specific uploaded documents
(useful once you have more than one SOP loaded).

## Architecture — ReAct harness

Built with `langchain.agents.create_agent` (the current, non-deprecated agent
factory — `langgraph.prebuilt.create_react_agent` is now the older path). Tools:

- `search_knowledge_base` — the same vector/BM25/RRF retrieval service as the RAG graph, but called
  at the model's discretion instead of on every turn.
- `get_current_time` — a genuine external API call (timeapi.io), not a
  `datetime.now()` wrapper, with a local fallback if the API is unreachable.
- `read_project_file` / `grep_project_files` — read-only codebase exploration,
  sandboxed to the project root (path-traversal checked — this is a real security
  boundary since it's exposed over a network-facing API, not a hypothetical one).

Uses `MemorySaver` rather than the RAG graph's Postgres checkpointer — a deliberate
simplification for this secondary demo path (session history doesn't survive a
restart), documented rather than silently swept under the rug.

## Multi-format documents & structured chunking

PDF, DOCX, TXT, and Markdown are all supported (`app/core/rag/loader.py`,
dispatched by extension in `document_service.py`). Markdown gets structure-aware
chunking (`split_markdown`, via `MarkdownHeaderTextSplitter`) — each section under a
heading becomes its own chunk instead of being cut at an arbitrary character count.
PDF/DOCX/TXT still use fixed-size `RecursiveCharacterTextSplitter` chunking, since
they don't have Markdown's built-in structural markers to split on — a real
limitation, not fully solved here. (A production version would parse PDF layout
directly, e.g. via `unstructured`, to recover section boundaries the same way.)

## Stack

- **LangGraph** — the fixed RAG graph, checkpointed to Postgres
- **`langchain.agents.create_agent`** — the ReAct harness orchestrator
- **FastAPI** — REST API, JWT auth (reused from the base template)
- **PostgreSQL** — chat history (LangGraph checkpointer), document metadata, mem0 long-term memory
- **Redis/Valkey** — rate-limiter storage backend
- **ChromaDB + `rank-bm25`** — hybrid vector/keyword retrieval (local, persisted to disk)
- **sentence-transformers (`BAAI/bge-m3`)** — multilingual (中/英) embeddings
- **DeepSeek** (`deepseek-v4-pro` / `deepseek-v4-flash`) — OpenAI-compatible LLM API
- **uv** — dependency management

## API

| Method | Path                       | Description                                          |
| ------ | -------------------------- | ----------------------------------------------------- |
| POST   | `/api/v1/documents/upload` | Upload a SOP document (PDF/DOCX/TXT/MD) for ingestion |
| GET    | `/api/v1/documents`        | List uploaded documents and ingestion status          |
| POST   | `/api/v1/chatbot/chat`     | Ask a question (Agentic RAG graph)                    |
| POST   | `/api/v1/chatbot/chat/stream` | Same, streamed via SSE                             |
| GET    | `/api/v1/chatbot/history`  | Get the session's chat history                        |
| POST   | `/api/v1/harness/chat`     | Ask a question (ReAct tool-calling harness)           |

All routes (except `/health`) require a JWT session — see `/api/v1/auth/register`,
`/api/v1/auth/login`, `/api/v1/auth/session`. A minimal test UI covering the whole
flow (login → upload → ask, both architectures side by side) is served at `/ui`.

## Setup

```bash
cp .env.example .env   # fill in OPENAI_API_KEY (DeepSeek key); set POSTGRES_HOST/VALKEY_HOST
                        # to localhost if running the app natively (not via docker compose)
uv sync --extra cache   # `cache` extra pulls in the `redis` client the rate limiter needs
docker compose up -d db valkey   # Postgres (pgvector image) + Redis-compatible cache
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

The embedding model (`bge-m3`, ~2GB) downloads on first run and is cached by
`sentence-transformers`; the app pre-warms it and the BM25 index at startup so the
first request isn't slow.

### Windows

Two Windows-specific gotchas, both due to psycopg's async driver (used by the
LangGraph Postgres checkpointer) requiring a `SelectorEventLoop`, while
Windows defaults to `ProactorEventLoop`:

- Run via `uv run python run_windows.py` instead of `uv run uvicorn app.main:app --reload`.
  Setting the asyncio event loop policy isn't enough on modern uvicorn (it
  passes an explicit `loop_factory` that bypasses the policy), so this script
  wires a custom Selector-based loop factory through uvicorn's `loop=` option.
- `reload=False` — auto-reload spawns subprocesses that need
  `ProactorEventLoop`, which conflicts with the above. Restart manually after
  code changes, or develop inside Docker/WSL2 instead (the `app` service in
  `docker-compose.yml` runs Linux, where neither loop type is an issue).

## Bugs found while building this (kept, not swept under the rug)

Real issues hit and fixed during development — listed because they were more
instructive than anything that went smoothly on the first try:

- **`starlette-prometheus` vs. a newer Starlette**: newer Starlette wraps
  `include_router()` output in a lazy `_IncludedRouter` proxy that the (older,
  unmaintained) `starlette-prometheus` middleware doesn't know how to unwrap —
  every request 500'd. Patched the middleware's route-matching to recurse into
  `original_router.routes` (`app/core/metrics.py`).
- **DeepSeek's "thinking" mode rejects forced `tool_choice`**: `with_structured_output`
  defaults to strict `json_schema` (unsupported) and, once switched to
  `method="function_calling"`, DeepSeek v4's default thinking mode then rejects the
  resulting forced `tool_choice`. Fixed by disabling thinking mode
  (`extra_body={"thinking": {"type": "disabled"}}`) for the models used in
  structured-output calls.
- **`Message.content` max-length validator applied to system prompts**: a 3000-char
  cap meant to bound user-submitted `ChatRequest` content was being reused
  internally to wrap the system prompt — which legitimately grows past that once
  retrieved SOP context is embedded in it. Fixed with `Message.model_construct(...)`
  to bypass validation for this specific, non-user-controlled internal use.
- **Langfuse tracing silently produced no traces**: the graph-level `RunnableConfig`
  (carrying the Langfuse callback) was never forwarded into the actual
  `llm.ainvoke()` calls inside `LLMService.call()` — callbacks attached at the graph
  level don't automatically propagate into arbitrary nested calls a node happens to
  make. Threaded `config` through `LLMService.call()` → `_fallback_loop()` →
  `_invoke_with_retry()` so it reaches the model call itself.
- **mem0 API breaking change**: `AsyncMemory.from_config()` became a synchronous
  classmethod in the installed mem0 version (`2.0.12`) despite the class name
  — `await`ing it raised `TypeError`.
- **Windows + psycopg async + newer uvicorn**: psycopg's async driver needs a
  `SelectorEventLoop`; Windows defaults to `ProactorEventLoop`; and modern uvicorn
  passes an explicit `loop_factory` to `asyncio.Runner` that bypasses the usual
  `asyncio.set_event_loop_policy()` workaround entirely. Solved via a custom loop
  factory registered through uvicorn's `loop=` import-string option
  (`run_windows.py`).
