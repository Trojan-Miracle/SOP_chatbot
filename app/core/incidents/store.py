"""Async local persistence; approval and mock ticket creation share one transaction."""

from pathlib import Path

import aiosqlite

from app.core.incidents.schemas import Decision, Event, Incident


class ConflictError(Exception):
    """The submitted version or transition is no longer valid."""


class IncidentStore:
    """Persist snapshots and mock tickets on one host; no network side effects."""

    def __init__(self, path: str) -> None:
        """Keep an explicit database path for runtime and isolated tests."""
        self.path = path

    async def initialize(self) -> None:
        """Create the local workflow schema idempotently."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(
                "CREATE TABLE IF NOT EXISTS incidents "
                "(id TEXT PRIMARY KEY, owner TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS mock_tickets "
                "(incident_id TEXT PRIMARY KEY, ticket_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);"
            )
            await db.commit()

    async def get(self, incident_id: str, owner: str) -> Incident:
        """Read only records owned by the authenticated principal."""
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT payload FROM incidents WHERE id=? AND owner=?", (incident_id, owner)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            raise KeyError(incident_id)
        return Incident.model_validate_json(row[0])

    async def save(self, record: Incident, owner: str, expected_version: int | None = None) -> Incident:
        """Insert or compare-and-swap; stale planning cannot overwrite a decision."""
        async with aiosqlite.connect(self.path) as db:
            if expected_version is None:
                await db.execute(
                    "INSERT INTO incidents VALUES (?, ?, ?, ?)",
                    (record.id, owner, record.version, record.model_dump_json()),
                )
            else:
                cursor = await db.execute(
                    "UPDATE incidents SET version=?, payload=? WHERE id=? AND owner=? AND version=?",
                    (record.version, record.model_dump_json(), record.id, owner, expected_version),
                )
                if cursor.rowcount != 1:
                    raise ConflictError("记录已更新，请刷新后重试。")
            await db.commit()
        return record

    async def decide(self, incident_id: str, owner: str, decision: Decision) -> Incident:
        """Atomically decide once; a repeated identical decision returns the saved result."""
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute("SELECT payload FROM incidents WHERE id=? AND owner=?", (incident_id, owner)) as cur:
                row = await cur.fetchone()
            if row is None:
                raise KeyError(incident_id)
            record = Incident.model_validate_json(row[0])
            target = "completed" if decision.approve else "rejected"
            if record.status == target and record.version == decision.expected_version + 1:
                return record
            if record.version != decision.expected_version or record.status != "awaiting_approval":
                raise ConflictError("只能审批当前版本的待审批方案，请刷新后重试。")
            record.version += 1
            record.status = target
            record.events.append(
                Event(name="approved" if decision.approve else "rejected", detail="操作人确认当前版本方案。")
            )
            if decision.approve:
                record.ticket_id = f"SOP-{record.id}"
                record.events.append(Event(name="mock_ticket_created", detail=f"模拟工单 {record.ticket_id} 已创建。"))
                await db.execute(
                    "INSERT INTO mock_tickets VALUES (?, ?, ?)",
                    (record.id, record.ticket_id, record.model_dump_json()),
                )
            await db.execute(
                "UPDATE incidents SET version=?, payload=? WHERE id=?",
                (record.version, record.model_dump_json(), record.id),
            )
            await db.commit()
            return record
