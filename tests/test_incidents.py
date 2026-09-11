"""Behavioral regression tests for approval, concurrency, ownership and recovery."""

import asyncio
from pathlib import Path

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import demo
from app.core.incidents.demo import DemoPlanner
from app.core.incidents.schemas import Decision, IncidentInput
from app.core.incidents.service import IncidentService
from app.core.incidents.store import ConflictError, IncidentStore

FACTS = IncidentInput(equipment="LP-200", symptom="卡纸", location="包装区")


async def setup(path: Path, planner=None):
    """Construct a real persistent workflow with injectable planning."""
    store = IncidentStore(str(path / "incidents.sqlite3"))
    await store.initialize()
    return IncidentService(store, planner or DemoPlanner(), "demo")


def test_clarification_and_restart(tmp_path):
    """Missing information must not create a ticket; restart resumes the approval."""

    async def scenario():
        service = await setup(tmp_path)
        incomplete = await service.prepare(IncidentInput(equipment="LP-200", symptom="卡纸"), "alice")
        assert incomplete.status == "needs_info"
        assert incomplete.questions == ["请补充发生位置。"]
        ready = await service.prepare(FACTS, "alice", incomplete.id, incomplete.version)
        assert ready.status == "awaiting_approval"
        assert ready.ticket_id is None
        restarted = await setup(tmp_path)
        assert await restarted.store.get(ready.id, "alice") == ready
        completed = await restarted.store.decide(
            ready.id, "alice", Decision(expected_version=ready.version, approve=True)
        )
        assert completed.status == "completed"
        assert completed.ticket_id

    asyncio.run(scenario())


def test_concurrent_approval_is_idempotent(tmp_path):
    """Concurrent identical requests and later retries produce exactly one ticket."""

    async def scenario():
        service = await setup(tmp_path)
        record = await service.prepare(FACTS, "alice")
        decision = Decision(expected_version=record.version, approve=True)
        results = await asyncio.gather(*(service.store.decide(record.id, "alice", decision) for _ in range(5)))
        assert len({r.ticket_id for r in results}) == 1
        async with aiosqlite.connect(service.store.path) as db:
            rows = await db.execute_fetchall("SELECT COUNT(*) FROM mock_tickets")
            assert rows[0][0] == 1
        assert await service.store.decide(record.id, "alice", decision) == results[0]

    asyncio.run(scenario())


def test_revised_proposal_rejects_old_approval(tmp_path):
    """An operator cannot approve a proposal replaced by new facts."""

    async def scenario():
        service = await setup(tmp_path)
        old = await service.prepare(FACTS, "alice")
        new = await service.prepare(FACTS.model_copy(update={"location": "二楼"}), "alice", old.id, old.version)
        with pytest.raises(ConflictError):
            await service.store.decide(old.id, "alice", Decision(expected_version=old.version, approve=True))
        assert (await service.store.get(new.id, "alice")).status == "awaiting_approval"

    asyncio.run(scenario())


def test_ownership_and_terminal_state(tmp_path):
    """Other users cannot read or decide; a rejected workflow cannot execute later."""

    async def scenario():
        service = await setup(tmp_path)
        record = await service.prepare(FACTS, "alice")
        with pytest.raises(KeyError):
            await service.store.get(record.id, "bob")
        with pytest.raises(KeyError):
            await service.store.decide(record.id, "bob", Decision(expected_version=1, approve=True))
        rejected = await service.store.decide(record.id, "alice", Decision(expected_version=1, approve=False))
        assert rejected.ticket_id is None
        with pytest.raises(ConflictError):
            await service.store.decide(record.id, "alice", Decision(expected_version=2, approve=True))
        with pytest.raises(ConflictError):
            await service.prepare(FACTS, "alice", record.id, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("equipment,symptom", [("unknown", "卡纸"), ("LP-200", "温度过高"), ("SC-100", "卡纸")])
def test_unknown_scenario_blocks(tmp_path, equipment, symptom):
    """Unknown device/symptom pairs cannot reach approval."""

    async def scenario():
        service = await setup(tmp_path)
        result = await service.prepare(IncidentInput(equipment=equipment, symptom=symptom, location="一楼"), "alice")
        assert result.status == "blocked"
        with pytest.raises(ConflictError):
            await service.store.decide(result.id, "alice", Decision(expected_version=1, approve=True))

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["invented_id", "invented_quote", "empty", "insufficient", "failure"])
def test_invalid_model_results_fail_closed(tmp_path, kind):
    """Model errors and unsupported citations cannot become executable proposals."""

    class BadPlanner(DemoPlanner):
        async def propose(self, facts, evidence):
            if kind == "failure":
                raise RuntimeError("upstream unavailable")
            result = await super().propose(facts, evidence)
            if kind == "invented_id":
                result.steps[0].evidence_id = "E999"
            elif kind == "invented_quote":
                result.steps[0].quote = "这是文档中不存在的指令"
            elif kind == "empty":
                result.steps = []
            else:
                result.sufficient = False
            return result

    async def scenario():
        service = await setup(tmp_path, BadPlanner())
        record = await service.prepare(FACTS, "alice")
        assert record.status == "blocked"
        assert record.proposal is None
        assert record.ticket_id is None

    asyncio.run(scenario())


def test_concurrent_revision_cannot_overwrite_approval(tmp_path):
    """A slow in-flight plan cannot replace a completed ticket snapshot."""

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class SlowPlanner(DemoPlanner):
            async def propose(self, facts, evidence):
                started.set()
                await release.wait()
                return await super().propose(facts, evidence)

        service = await setup(tmp_path)
        record = await service.prepare(FACTS, "alice")
        slow = IncidentService(service.store, SlowPlanner(), "demo")
        task = asyncio.create_task(slow.prepare(FACTS, "alice", record.id, 1))
        await started.wait()
        await service.store.decide(record.id, "alice", Decision(expected_version=1, approve=True))
        release.set()
        with pytest.raises(ConflictError):
            await task
        assert (await service.store.get(record.id, "alice")).status == "completed"

    asyncio.run(scenario())


def test_demo_http_contract(tmp_path, monkeypatch):
    """Exercise actual request validation, routing, stale versions and rendered page."""
    monkeypatch.setattr(demo, "store", IncidentStore(str(tmp_path / "api.sqlite3")))
    with TestClient(demo.app) as client:
        assert "SOP Copilot" in client.get("/").text
        assert client.post("/api/v1/incidents", json={"equipment": "x" * 201}).status_code == 422
        response = client.post("/api/v1/incidents", json=FACTS.model_dump())
        assert response.status_code == 200
        record = response.json()
        assert record["status"] == "awaiting_approval"
        url = f"/api/v1/incidents/{record['id']}"
        assert client.get(url).json() == record
        assert client.post(url + "/decision", json={"expected_version": 9, "approve": True}).status_code == 409
        done = client.post(url + "/decision", json={"expected_version": 1, "approve": True})
        assert done.json()["status"] == "completed"
        assert client.post(url + "/decision", json={"expected_version": 1, "approve": True}).json() == done.json()
        assert client.get("/api/v1/incidents/missing").status_code == 404
