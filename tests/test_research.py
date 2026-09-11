"""Controller contracts under cross-document gaps, contradictions and model failures."""

import asyncio
from pathlib import Path

import pytest

from app.core.incidents.research import ResearchAgent, ResearchBudget
from app.core.incidents.schemas import (
    Assessment,
    Coverage,
    Evidence,
    EvidenceQuote,
    IncidentInput,
    Proposal,
    ResearchPlan,
    SourceConflict,
    Step,
    StepSupport,
    SupportReport,
)
from app.core.incidents.service import IncidentService
from app.core.incidents.store import IncidentStore

FACTS = IncidentInput(equipment="LP-200", symptom="卡纸", location="包装区")


class Backend:
    """Scripted model port, intentionally not a model-quality simulation."""

    def __init__(self):
        """Track calls so tests assert behavior at the external boundary."""
        self.queries = []
        self.draft_calls = 0
        self.verify_calls = 0

    async def plan(self, facts):
        """Require both the device procedure and referenced registration form."""
        return ResearchPlan(questions=["适用处理流程", "登记字段"], initial_query="LP-200 卡纸")

    async def search(self, query):
        """Second search reveals a referenced document absent from the first hit."""
        self.queries.append(query)
        device = Evidence(id="old-id", filename="device.md", page=1, content="卡纸时提交登记单 FORM-9。")
        if len(self.queries) == 1:
            return [device]
        return [device, Evidence(id="old-id", filename="form.md", page=1, content="登记设备编号与位置。")]

    async def assess(self, facts, plan, evidence, queries):
        """Request the missing document, then cover each requirement."""
        if len(evidence) < 2:
            return Assessment(action="search", summary="缺少登记单字段", next_query="FORM-9 登记字段")
        return Assessment(
            action="answer",
            summary="两个子问题均有证据",
            coverage=[Coverage(question_index=i, evidence_ids=[evidence[i].id]) for i in range(2)],
        )

    async def propose(self, facts, evidence):
        """Draft only after the controller has accepted complete coverage."""
        self.draft_calls += 1
        return Proposal(
            sufficient=True,
            explanation="按登记单填报",
            steps=[Step(instruction=evidence[-1].content, evidence_id=evidence[-1].id, quote=evidence[-1].content)],
        )

    async def verify(self, facts, proposal, evidence):
        """Return an independent support verdict."""
        self.verify_calls += 1
        return SupportReport(steps=[StepSupport(step_index=0, supported=True, explanation="原文支持")])


def test_cross_document_search_and_deduplication():
    """Resolve a missing referenced document without losing first-round evidence."""
    backend = Backend()
    result = asyncio.run(ResearchAgent(backend).run(FACTS))
    assert result.status == "ready"
    assert backend.queries == ["LP-200 卡纸", "FORM-9 登记字段"]
    assert len(result.evidence) == 2
    assert len({e.id for e in result.evidence}) == 2
    assert result.search_calls == 2 and result.model_calls == 5
    assert backend.draft_calls == backend.verify_calls == 1


@pytest.mark.parametrize("mode", ["missing_question", "invented_id", "duplicate_question"])
def test_coverage_must_reference_all_requirements(mode):
    """The model cannot claim completeness by omitting a requirement or forging an ID."""

    class Invalid(Backend):
        async def assess(self, facts, plan, evidence, queries):
            coverage = [Coverage(question_index=0, evidence_ids=[evidence[0].id])]
            if mode == "invented_id":
                coverage.append(Coverage(question_index=1, evidence_ids=["E-invented"]))
            elif mode == "duplicate_question":
                coverage.append(coverage[0])
            return Assessment(action="answer", summary="claimed complete", coverage=coverage)

    backend = Invalid()
    result = asyncio.run(ResearchAgent(backend).run(FACTS))
    assert result.status == "verification_failed"
    assert backend.draft_calls == 0


@pytest.mark.parametrize("mode", ["unsupported", "missing", "duplicate", "wrong_index"])
def test_verbatim_quote_does_not_bypass_semantic_review(mode):
    """Exact citations alone cannot authorize unsupported or incompletely reviewed steps."""

    class Unsupported(Backend):
        async def verify(self, facts, proposal, evidence):
            verdict = StepSupport(step_index=0, supported=mode != "unsupported", explanation="检查结果")
            return SupportReport(
                steps=[]
                if mode == "missing"
                else [verdict, verdict]
                if mode == "duplicate"
                else [verdict.model_copy(update={"step_index": 9})]
                if mode == "wrong_index"
                else [verdict]
            )

    result = asyncio.run(ResearchAgent(Unsupported()).run(FACTS))
    assert result.status == "verification_failed"
    assert result.proposal is None


@pytest.mark.parametrize("valid", [True, False])
def test_conflict_requires_two_real_verbatim_sources(valid):
    """Persist concrete competing source spans; forged conflict evidence fails closed."""

    class Conflicting(Backend):
        async def assess(self, facts, plan, evidence, queries):
            if len(evidence) < 2:
                return await super().assess(facts, plan, evidence, queries)
            return Assessment(
                action="conflict",
                summary="两个来源的适用要求不能同时满足",
                conflicts=[
                    SourceConflict(
                        description="示例冲突",
                        left=EvidenceQuote(evidence_id=evidence[0].id, quote=evidence[0].content),
                        right=EvidenceQuote(
                            evidence_id=evidence[1].id, quote=evidence[1].content if valid else "伪造原文"
                        ),
                    )
                ],
            )

    backend = Conflicting()
    result = asyncio.run(ResearchAgent(backend).run(FACTS))
    assert result.status == ("conflict" if valid else "verification_failed")
    assert bool(result.conflicts) == valid
    assert backend.draft_calls == 0


def test_repeated_query_stops_before_second_search():
    """Formatting-only rewrites do not consume repeated retrieval calls."""

    class Repeating(Backend):
        async def assess(self, facts, plan, evidence, queries):
            return Assessment(action="search", summary="retry", next_query="lp 200 卡纸!!!")

    backend = Repeating()
    result = asyncio.run(ResearchAgent(backend).run(FACTS))
    assert result.status == "abstain"
    assert len(backend.queries) == 1
    assert result.trace[-1].stage == "stalled"


def test_search_budget_does_not_fall_through_to_draft():
    """Exhaustion cannot turn partially collected evidence into an approved proposal."""
    backend = Backend()
    result = asyncio.run(ResearchAgent(backend, ResearchBudget(searches=1)).run(FACTS))
    assert result.status == "budget_exhausted"
    assert result.search_calls == 1 and backend.draft_calls == 0


def test_model_budget_reserves_no_implicit_verification_pass():
    """If there is no budget to verify a draft it must not be returned as ready."""
    result = asyncio.run(ResearchAgent(Backend(), ResearchBudget(model_calls=4)).run(FACTS))
    assert result.status == "budget_exhausted"
    assert result.model_calls == 4 and result.proposal is None


def test_context_size_is_bounded():
    """Oversized retrieved text is clipped before it reaches the model assessor."""

    class Large(Backend):
        async def search(self, query):
            return [Evidence(id=str(i), filename=f"{i}.md", page=1, content="x" * 10000) for i in range(50)]

    result = asyncio.run(ResearchAgent(Large(), ResearchBudget(context_chars=100, chunk_chars=60)).run(FACTS))
    assert sum(len(e.content) for e in result.evidence) <= 100
    assert all(len(e.content) <= 60 for e in result.evidence)


def test_timeout_preserves_partial_evidence():
    """An interrupted upstream reviewer leaves an inspectable failure artifact."""

    class Slow(Backend):
        async def assess(self, *args):
            await asyncio.sleep(1)

    result = asyncio.run(ResearchAgent(Slow(), ResearchBudget(timeout_seconds=0.02)).run(FACTS))
    assert result.status == "budget_exhausted"
    assert result.evidence and result.proposal is None


def test_backend_error_preserves_partial_evidence():
    """An upstream exception does not erase completed searches."""

    class Broken(Backend):
        async def assess(self, *args):
            raise RuntimeError("unavailable")

    result = asyncio.run(ResearchAgent(Broken()).run(FACTS))
    assert result.status == "failed" and result.evidence


def test_followup_fact_roundtrip_through_service(tmp_path: Path):
    """A model-requested fact can be supplied, replanned, and persisted for review."""

    class Followup(Backend):
        async def assess(self, facts, plan, evidence, queries):
            if not facts.context:
                return Assessment(action="clarify", summary="缺少固件版本", questions=["设备固件版本是多少？"])
            return await super().assess(facts, plan, evidence, queries)

        async def investigate(self, facts):
            self.queries = []
            return await ResearchAgent(self).run(facts)

        async def retrieve(self, facts):
            raise AssertionError("legacy retrieval path should not run")

    async def scenario():
        store = IncidentStore(str(tmp_path / "research.sqlite3"))
        await store.initialize()
        service = IncidentService(store, Followup(), "live")
        first = await service.prepare(FACTS, "alice")
        assert first.status == "needs_info" and first.questions == ["设备固件版本是多少？"]
        second = await service.prepare(
            FACTS.model_copy(update={"context": "固件2.0"}), "alice", first.id, first.version
        )
        assert second.status == "awaiting_approval"
        restored = await store.get(second.id, "alice")
        assert restored.research.search_calls == 2
        assert restored.facts.context == "固件2.0"

    asyncio.run(scenario())
