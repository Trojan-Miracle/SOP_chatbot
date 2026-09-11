"""Live one-pass versus adaptive investigation comparison on public synthetic cases."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from collections.abc import Callable

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from app.core.config import settings
from app.core.rag.bm25_index import get_bm25_index
from app.core.incidents.research import ResearchAgent, ResearchBackend, proposal_error, supported_proposal
from app.core.incidents.schemas import IncidentInput, ResearchResult


# Keep CLI help and missing-configuration diagnostics usable without constructing
# authenticated model clients. Optional adapter imports stay at module scope.
backend_factory: Callable[[str], ResearchBackend] | None = None
if settings.OPENAI_API_KEY:
    from app.core.incidents.live import LivePlanner

    backend_factory = LivePlanner


class Case(BaseModel):
    """Public routing and source-recall labels, not model-generated gold answers."""

    id: str
    category: str
    facts: IncidentInput
    expected_statuses: list[str] = Field(min_length=1)
    required_sources: list[str]


async def single_pass(backend: ResearchBackend, facts: IncidentInput) -> ResearchResult:
    """Existing one-shot retrieve-and-draft baseline with the same citation gate."""
    result = ResearchResult(status="abstain", search_calls=1)
    start = perf_counter()
    try:
        async with asyncio.timeout(80):
            result.evidence = await backend.search(f"{facts.equipment} {facts.symptom}")
            if result.evidence:
                result.model_calls = 1
                proposal = await backend.propose(facts, result.evidence)
                if proposal_error(proposal, result.evidence) is None:
                    result.status, result.proposal = "ready", proposal
    except TimeoutError:
        result.status = "budget_exhausted"
    except Exception:
        result.status = "failed"
    result.elapsed_ms = round((perf_counter() - start) * 1000, 2)
    return result


async def evaluate(cases_path: Path, output: Path, runs: int, judge: bool = False) -> None:
    """Run actual LLM calls; report routing, retrieval coverage, calls and wall time."""
    if backend_factory is None:
        raise SystemExit("请先配置真实 OPENAI_API_KEY，并上传 examples/research/sops/ 全部文档。")
    cases = [Case.model_validate(c) for c in json.loads(cases_path.read_text())]
    if not cases:
        raise SystemExit("评测集不能为空。")
    await asyncio.to_thread(get_bm25_index().rebuild)
    rows = []
    for run in range(runs):
        for index, case in enumerate(cases):
            # Alternate order to avoid always giving one strategy the warm cache.
            strategies = ["single_pass", "adaptive"] if (run + index) % 2 == 0 else ["adaptive", "single_pass"]
            for strategy in strategies:
                backend = backend_factory(f"eval-{case.id}-{run}-{strategy}")
                result = (
                    await single_pass(backend, case.facts)
                    if strategy == "single_pass"
                    else await ResearchAgent(backend).run(case.facts)
                )
                support_check = None
                support_passed = None
                evaluation_error = None
                evaluation_calls = 0
                if judge and result.proposal is not None:
                    evaluation_calls = 1
                    try:
                        async with asyncio.timeout(60):
                            support_check = await backend.verify(case.facts, result.proposal, result.evidence)
                        support_passed = supported_proposal(support_check, result.proposal)
                    except Exception:
                        evaluation_error = "support_judge_failed"
                found = {e.filename for e in result.evidence}
                required = set(case.required_sources)
                rows.append(
                    {
                        "case": case.id,
                        "category": case.category,
                        "run": run,
                        "strategy": strategy,
                        "status_match": result.status in case.expected_statuses,
                        "support_passed": support_passed,
                        "support_check": support_check.model_dump() if support_check else None,
                        "evaluation_model_calls": evaluation_calls,
                        "evaluation_error": evaluation_error,
                        "required_source_recall": len(found & required) / len(required) if required else None,
                        "result": result.model_dump(mode="json"),
                    }
                )
                # Save after each trial so interrupted real-model evaluations remain inspectable.
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "dataset": str(cases_path),
                            "runs": runs,
                            "model": settings.DEFAULT_LLM_MODEL,
                            "review_model": settings.GRADE_LLM_MODEL,
                            "embedding_model": settings.EMBEDDING_MODEL,
                            "rows": rows,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
    table = Table(title="真实模型调查对照 · 合成微型数据集")
    for column in ("策略", "路由命中", "平均检索", "平均模型服务调用", "平均秒"):
        table.add_column(column)
    for strategy in ("single_pass", "adaptive"):
        trials = [r for r in rows if r["strategy"] == strategy]
        count = len(trials)
        table.add_row(
            strategy,
            f"{sum(r['status_match'] for r in trials)}/{count}",
            f"{sum(r['result']['search_calls'] for r in trials) / count:.2f}",
            f"{sum(r['result']['model_calls'] for r in trials) / count:.2f}",
            f"{sum(r['result']['elapsed_ms'] for r in trials) / count / 1000:.2f}",
        )
    Console().print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("examples/research/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/reports/research.json"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--judge", action="store_true", help="Apply the same extra support judge to both strategies")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    asyncio.run(evaluate(args.cases, args.output, args.runs, args.judge))
