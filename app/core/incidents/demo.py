"""Deterministic fixture adapter: exercises workflow, does not simulate model quality."""

from pathlib import Path

from app.core.incidents.schemas import Evidence, IncidentInput, Proposal, Step

SOP_DIR = Path(__file__).resolve().parents[3] / "examples" / "sops"


class DemoPlanner:
    """Match two explicit fixture scenarios; unknown situations fail closed."""

    async def retrieve(self, facts: IncidentInput) -> list[Evidence]:
        """Return a documented fixture only when its model and symptom match."""
        filename = None
        if facts.equipment.upper() == "LP-200" and facts.symptom == "卡纸":
            filename = "printer-jam.md"
        elif facts.equipment.upper() == "SC-100" and facts.symptom == "离线":
            filename = "scanner-offline.md"
        if filename is None:
            return []
        return [Evidence(id="E1", filename=filename, page=1, content=(SOP_DIR / filename).read_text())]

    async def propose(self, facts: IncidentInput, evidence: list[Evidence]) -> Proposal:
        """Copy fixture steps literally; no external LLM or tool calls occur."""
        source = evidence[0]
        steps = [line[3:] for line in source.content.splitlines() if line[:1].isdigit() and line[1:3] == ". "]
        return Proposal(
            sufficient=True,
            explanation="根据匹配的演示 SOP 生成登记方案，等待人工确认。",
            steps=[Step(instruction=step, evidence_id=source.id, quote=step) for step in steps],
        )
