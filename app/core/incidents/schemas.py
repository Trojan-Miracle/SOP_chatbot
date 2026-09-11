"""Typed contracts shared by live and deterministic demo workflows."""

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class IncidentInput(BaseModel):
    """Facts supplied by the operator; blank fields trigger clarification."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    equipment: str = Field(default="", max_length=200)
    symptom: str = Field(default="", max_length=2000)
    location: str = Field(default="", max_length=200)
    context: str = Field(default="", max_length=4000, description="Answers to follow-up questions and observed facts")


class Evidence(BaseModel):
    """An immutable excerpt copied into the approval record."""

    id: str
    filename: str
    page: int
    content: str


class Step(BaseModel):
    """A proposed step tied to a verbatim excerpt."""

    instruction: str = Field(min_length=1, max_length=1000)
    evidence_id: str
    quote: str = Field(min_length=1, max_length=1000)


class Proposal(BaseModel):
    """Model output; sufficient is advisory until evidence validation passes."""

    sufficient: bool
    explanation: str = Field(max_length=2000)
    steps: list[Step] = Field(default_factory=list, max_length=12)


class Event(BaseModel):
    """Persisted user-visible execution event, not hidden model reasoning."""

    name: str
    detail: str
    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResearchPlan(BaseModel):
    """Decompose the incident into answerable evidence requirements."""

    questions: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(
        min_length=1, max_length=4
    )
    initial_query: str = Field(min_length=1, max_length=400)


class Coverage(BaseModel):
    """Evidence claimed to cover one zero-indexed research question."""

    question_index: int = Field(ge=0)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class EvidenceQuote(BaseModel):
    """Verbatim support for a conflict finding."""

    evidence_id: str
    quote: str = Field(min_length=1, max_length=1000)


class SourceConflict(BaseModel):
    """Two incompatible claims whose applicability cannot be resolved."""

    description: str = Field(min_length=1, max_length=500)
    left: EvidenceQuote
    right: EvidenceQuote


class Assessment(BaseModel):
    """Model-selected next action; the controller validates transitions."""

    action: Literal["answer", "search", "clarify", "conflict", "abstain"]
    summary: str = Field(max_length=1000)
    coverage: list[Coverage] = Field(default_factory=list, max_length=4)
    next_query: str = Field(default="", max_length=400)
    questions: list[str] = Field(default_factory=list, max_length=3)
    conflicts: list[SourceConflict] = Field(default_factory=list, max_length=4)


class StepSupport(BaseModel):
    """Independent semantic support verdict for a zero-indexed proposed step."""

    step_index: int = Field(ge=0)
    supported: bool
    explanation: str = Field(max_length=500)


class SupportReport(BaseModel):
    """Every step must receive exactly one support verdict."""

    steps: list[StepSupport] = Field(max_length=12)


class ResearchTrace(BaseModel):
    """Observable actions and outcomes, excluding private model reasoning."""

    stage: str
    summary: str
    query: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Bounded investigation artifact retained alongside the approval snapshot."""

    status: Literal["ready", "clarify", "conflict", "abstain", "budget_exhausted", "verification_failed", "failed"]
    plan: ResearchPlan | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    proposal: Proposal | None = None
    questions: list[str] = Field(default_factory=list)
    conflicts: list[SourceConflict] = Field(default_factory=list)
    verification: SupportReport | None = None
    trace: list[ResearchTrace] = Field(default_factory=list)
    search_calls: int = 0
    model_calls: int = 0
    elapsed_ms: float = 0


class Incident(BaseModel):
    """Versioned snapshot of one incident and its approval artifact."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    status: Literal["needs_info", "blocked", "awaiting_approval", "completed", "rejected"] = "needs_info"
    facts: IncidentInput
    questions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    proposal: Proposal | None = None
    research: ResearchResult | None = None
    events: list[Event] = Field(default_factory=list)
    ticket_id: str | None = None
    mode: Literal["demo", "live"]


class Revision(BaseModel):
    """Full replacement facts with optimistic concurrency control."""

    expected_version: int = Field(ge=1)
    facts: IncidentInput


class Decision(BaseModel):
    """Approval applies only to the exact version the operator reviewed."""

    expected_version: int = Field(ge=1)
    approve: bool
