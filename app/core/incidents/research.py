"""Adaptive evidence investigation with coverage, conflict and semantic support gates."""

import asyncio
import hashlib
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import structlog

from app.core.incidents.schemas import (
    Assessment,
    Evidence,
    IncidentInput,
    Proposal,
    ResearchPlan,
    ResearchResult,
    ResearchTrace,
    SupportReport,
)

logger = structlog.get_logger(__name__)


class ResearchBackend(Protocol):
    """Model and retrieval ports; controller policy is independent of adapters."""

    async def plan(self, facts: IncidentInput) -> ResearchPlan:
        """Identify the evidence requirements and initial search."""
        ...

    async def search(self, query: str) -> list[Evidence]:
        """Retrieve candidates for a model-selected query."""
        ...

    async def assess(
        self, facts: IncidentInput, plan: ResearchPlan, evidence: list[Evidence], queries: list[str]
    ) -> Assessment:
        """Select the next action using accumulated evidence and prior searches."""
        ...

    async def propose(self, facts: IncidentInput, evidence: list[Evidence]) -> Proposal:
        """Draft a cited proposal."""
        ...

    async def verify(self, facts: IncidentInput, proposal: Proposal, evidence: list[Evidence]) -> SupportReport:
        """Independently check each instruction against its cited evidence."""
        ...


@dataclass(frozen=True)
class ResearchBudget:
    """Per-investigation bounds on cost, latency and accumulated context."""

    searches: int = 3
    model_calls: int = 6
    timeout_seconds: float = 80
    evidence_count: int = 16
    context_chars: int = 24000
    chunk_chars: int = 3000

    def __post_init__(self) -> None:
        """Reject configurations that cannot perform a valid investigation."""
        if (
            min(
                self.searches,
                self.model_calls,
                self.timeout_seconds,
                self.evidence_count,
                self.context_chars,
                self.chunk_chars,
            )
            <= 0
        ):
            raise ValueError("research budgets must be positive")


def normalized_query(query: str) -> str:
    """Detect whitespace/case/punctuation-only repeats before spending retrieval budget."""
    return re.sub(r"[\W_]+", "", query.casefold())


def proposal_error(proposal: Proposal, evidence: list[Evidence]) -> str | None:
    """Require grounded, nonempty steps before semantic verification."""
    if not proposal.sufficient or not proposal.steps:
        return "现有资料不足以形成处理方案，请补充适用 SOP 或转交人工。"
    lookup = {item.id: item for item in evidence}
    for step in proposal.steps:
        source = lookup.get(step.evidence_id)
        if source is None or not step.quote.strip() or step.quote not in source.content:
            return "方案引用未通过原文校验，已阻止进入审批。"
    return None


def supported_proposal(report: SupportReport, proposal: Proposal) -> bool:
    """Missing, duplicate or unsupported verdicts fail closed."""
    indices = [step.step_index for step in report.steps]
    return (
        len(indices) == len(proposal.steps)
        and set(indices) == set(range(len(proposal.steps)))
        and all(step.supported for step in report.steps)
    )


class ResearchAgent:
    """Model chooses searches and stopping; deterministic policy checks every boundary."""

    def __init__(self, backend: ResearchBackend, budget: ResearchBudget | None = None) -> None:
        """Bind adapter and immutable budgets; all mutable state is invocation-local."""
        self.backend = backend
        self.budget = budget or ResearchBudget()

    async def run(self, facts: IncidentInput) -> ResearchResult:
        """Return an inspectable partial artifact even when upstream calls fail."""
        result = ResearchResult(status="abstain")
        start = perf_counter()
        try:
            async with asyncio.timeout(self.budget.timeout_seconds):
                await self._investigate(facts, result)
        except TimeoutError:
            result.status = "budget_exhausted"
            result.proposal = None
            result.trace.append(ResearchTrace(stage="timeout", summary="调查达到时间预算，保留已有证据。"))
        except Exception:
            logger.exception("research_failed")
            result.status = "failed"
            result.proposal = None
            result.trace.append(ResearchTrace(stage="failed", summary="检索或模型调用失败，未产生可审批方案。"))
        result.elapsed_ms = round((perf_counter() - start) * 1000, 2)
        return result

    def _model_slot(self, result: ResearchResult) -> bool:
        if result.model_calls >= self.budget.model_calls:
            result.status = "budget_exhausted"
            result.trace.append(ResearchTrace(stage="budget", summary="模型调用预算耗尽。"))
            return False
        result.model_calls += 1
        return True

    def _merge(self, result: ResearchResult, hits: list[Evidence]) -> None:
        known = {e.id for e in result.evidence}
        remaining = self.budget.context_chars - sum(len(e.content) for e in result.evidence)
        for hit in hits:
            identity = f"{hit.filename}\0{hit.page}\0{hit.content}"
            source_id = "E-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            if source_id in known or not hit.content.strip():
                continue
            if remaining <= 0 or len(result.evidence) >= self.budget.evidence_count:
                break
            content = hit.content[: min(self.budget.chunk_chars, remaining)]
            result.evidence.append(hit.model_copy(update={"id": source_id, "content": content}))
            known.add(source_id)
            remaining -= len(content)

    async def _investigate(self, facts: IncidentInput, result: ResearchResult) -> None:
        if not self._model_slot(result):
            return
        plan = await self.backend.plan(facts)
        result.plan = plan
        result.trace.append(ResearchTrace(stage="plan", summary="；".join(plan.questions)))
        query, queries = plan.initial_query, []
        while result.search_calls < self.budget.searches:
            normalized = normalized_query(query)
            if not normalized or normalized in {normalized_query(q) for q in queries}:
                result.status = "abstain"
                result.trace.append(ResearchTrace(stage="stalled", summary="查询为空或与已有查询重复，停止无效循环。"))
                return
            queries.append(query)
            result.search_calls += 1
            hits = await self.backend.search(query)
            before = len(result.evidence)
            self._merge(result, hits)
            result.trace.append(
                ResearchTrace(
                    stage="search",
                    query=query,
                    summary=f"返回 {len(hits)} 条，新增 {len(result.evidence) - before} 条；累计 {len(result.evidence)} 条。",
                    evidence_ids=[e.id for e in result.evidence],
                )
            )
            if not self._model_slot(result):
                return
            assessment = await self.backend.assess(facts, plan, result.evidence, queries)
            result.trace.append(ResearchTrace(stage=assessment.action, summary=assessment.summary))
            lookup = {e.id: e for e in result.evidence}
            if assessment.action == "search":
                query = assessment.next_query
                continue
            if assessment.action == "clarify":
                result.questions = [q.strip() for q in assessment.questions if q.strip()]
                result.status = "clarify" if result.questions else "abstain"
                return
            if assessment.action == "conflict":
                valid = bool(assessment.conflicts)
                for conflict in assessment.conflicts:
                    if conflict.left.evidence_id == conflict.right.evidence_id:
                        valid = False
                    for citation in (conflict.left, conflict.right):
                        source = lookup.get(citation.evidence_id)
                        if source is None or not citation.quote.strip() or citation.quote not in source.content:
                            valid = False
                result.status = "conflict" if valid else "verification_failed"
                result.conflicts = assessment.conflicts if valid else []
                return
            if assessment.action == "abstain":
                result.status = "abstain"
                return
            coverage = assessment.coverage
            complete = (
                len(coverage) == len(plan.questions)
                and {c.question_index for c in coverage} == set(range(len(plan.questions)))
                and all(eid in lookup for c in coverage for eid in c.evidence_ids)
            )
            if not complete or assessment.conflicts or not result.evidence:
                result.status = "verification_failed"
                result.trace.append(
                    ResearchTrace(stage="coverage_rejected", summary="子问题覆盖不完整或依据 ID 无效。")
                )
                return
            if not self._model_slot(result):
                return
            proposal = await self.backend.propose(facts, result.evidence)
            error = proposal_error(proposal, result.evidence)
            if error:
                result.status = "verification_failed"
                result.trace.append(ResearchTrace(stage="citation_rejected", summary=error))
                return
            if not self._model_slot(result):
                return
            report = await self.backend.verify(facts, proposal, result.evidence)
            result.verification = report
            if not supported_proposal(report, proposal):
                result.status = "verification_failed"
                result.trace.append(ResearchTrace(stage="support_rejected", summary="方案未通过逐步骤语义支持检查。"))
                return
            result.proposal = proposal
            result.status = "ready"
            result.trace.append(
                ResearchTrace(stage="verified", summary="覆盖、引用和语义支持检查通过，提交人工审阅。")
            )
            return
        result.status = "budget_exhausted"
        result.trace.append(ResearchTrace(stage="budget", summary="检索预算耗尽，仍有证据缺口。"))
