"""Bounded planning with server-enforced clarification, evidence and approval gates."""

import asyncio
from typing import Literal, Protocol, runtime_checkable

import structlog

from app.core.incidents.schemas import Evidence, Event, Incident, IncidentInput, Proposal, ResearchResult
from app.core.incidents.research import proposal_error
from app.core.incidents.store import ConflictError, IncidentStore

logger = structlog.get_logger(__name__)


class Planner(Protocol):
    """Injected retrieval and planning boundary for live calls and offline fixtures."""

    async def retrieve(self, facts: IncidentInput) -> list[Evidence]:
        """Retrieve candidate evidence."""
        ...

    async def propose(self, facts: IncidentInput, evidence: list[Evidence]) -> Proposal:
        """Produce a typed proposal grounded in supplied evidence."""
        ...


@runtime_checkable
class InvestigatingPlanner(Protocol):
    """Optional richer capability implemented by the live research adapter."""

    async def investigate(self, facts: IncidentInput) -> ResearchResult:
        """Run bounded adaptive evidence investigation."""
        ...


def evidence_error(proposal: Proposal, evidence: list[Evidence]) -> str | None:
    """Validate source existence and verbatim quotes."""
    return proposal_error(proposal, evidence)


class IncidentService:
    """Coordinate external planning outside transactions and commit with version checks."""

    def __init__(self, store: IncidentStore, planner: Planner, mode: Literal["demo", "live"]) -> None:
        """Bind persistence and planning dependencies."""
        self.store = store
        self.planner = planner
        self.mode: Literal["demo", "live"] = mode

    async def prepare(
        self, facts: IncidentInput, owner: str, incident_id: str | None = None, expected_version: int | None = None
    ) -> Incident:
        """Collect mandatory facts and draft a proposal without executing any action."""
        if incident_id:
            record = await self.store.get(incident_id, owner)
            if record.version != expected_version or record.status in ("completed", "rejected"):
                raise ConflictError("当前版本不可修改，请刷新或新建事件。")
            record.version += 1
            record.facts = facts
        else:
            record = Incident(facts=facts, mode=self.mode)
        record.proposal, record.evidence, record.questions = None, [], []
        record.research = None
        record.status = "needs_info"
        for field, label in (("equipment", "设备型号或编号"), ("symptom", "异常现象"), ("location", "发生位置")):
            if not getattr(facts, field):
                record.questions.append(f"请补充{label}。")
        if record.questions:
            record.events.append(Event(name="clarification_required", detail=" ".join(record.questions)))
            return await self.store.save(record, owner, expected_version)
        try:
            async with asyncio.timeout(90):
                if isinstance(self.planner, InvestigatingPlanner):
                    research = await self.planner.investigate(facts)
                    if research.status == "ready" and (
                        research.proposal is None or evidence_error(research.proposal, research.evidence) is not None
                    ):
                        research.status = "verification_failed"
                        research.proposal = None
                    record.research = research
                    record.evidence = research.evidence
                    record.questions = research.questions
                    record.proposal = research.proposal if research.status == "ready" else None
                    record.status = (
                        "awaiting_approval"
                        if research.status == "ready"
                        else "needs_info"
                        if research.status == "clarify"
                        else "blocked"
                    )
                    record.events.extend(
                        Event(name="research_" + item.stage, detail=item.summary) for item in research.trace
                    )
                else:
                    record.evidence = await self.planner.retrieve(facts)
                    record.events.append(Event(name="retrieved", detail=f"检索到 {len(record.evidence)} 条候选依据。"))
                    if not record.evidence:
                        record.status = "blocked"
                        record.events.append(
                            Event(name="insufficient_evidence", detail="没有适用依据，请补充 SOP 后重试。")
                        )
                    else:
                        proposal = await self.planner.propose(facts, record.evidence)
                        error = evidence_error(proposal, record.evidence)
                        record.status = "blocked" if error else "awaiting_approval"
                        record.proposal = None if error else proposal
                        record.events.append(
                            Event(
                                name="evidence_rejected" if error else "approval_required",
                                detail=error or "引用原文校验通过，等待操作人审阅方案；尚未创建工单。",
                            )
                        )
        except Exception:
            logger.exception("incident_planning_failed", incident_id=record.id)
            record.status = "blocked"
            record.events.append(
                Event(name="planning_failed", detail="检索或模型调用失败；可保留输入重试，未执行操作。")
            )
        logger.info("incident_prepared", incident_id=record.id, status=record.status, version=record.version)
        return await self.store.save(record, owner, expected_version)
