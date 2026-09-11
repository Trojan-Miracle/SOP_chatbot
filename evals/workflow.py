"""Run reproducible offline workflow scenarios; this does not measure LLM accuracy."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.core.incidents.demo import DemoPlanner
from app.core.incidents.schemas import Decision, IncidentInput
from app.core.incidents.service import IncidentService
from app.core.incidents.store import ConflictError, IncidentStore

CASES = [
    ("缺少位置先追问", {"equipment": "LP-200", "symptom": "卡纸"}, "needs_info"),
    ("空输入不调用规划", {}, "needs_info"),
    ("打印机登记方案", {"equipment": "LP-200", "symptom": "卡纸", "location": "包装区"}, "awaiting_approval"),
    ("扫码器登记方案", {"equipment": "SC-100", "symptom": "离线", "location": "收货区"}, "awaiting_approval"),
    ("未知设备阻止规划", {"equipment": "X-999", "symptom": "卡纸", "location": "一楼"}, "blocked"),
    ("不适用现象阻止规划", {"equipment": "LP-200", "symptom": "起火", "location": "一楼"}, "blocked"),
]


async def evaluate(output: Path) -> None:
    """Run fixed adapter scenarios against real local persistence and transition guards."""
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        store = IncidentStore(str(Path(directory) / "eval.sqlite3"))
        await store.initialize()
        service = IncidentService(store, DemoPlanner(), "demo")
        for label, facts, expected in CASES:
            record = await service.prepare(IncidentInput(**facts), "evaluator")
            passed = record.status == expected and record.ticket_id is None
            if expected != "awaiting_approval":
                try:
                    await store.decide(record.id, "evaluator", Decision(expected_version=1, approve=True))
                    passed = False
                except ConflictError:
                    pass
            else:
                decision = Decision(expected_version=1, approve=True)
                first = await store.decide(record.id, "evaluator", decision)
                second = await store.decide(record.id, "evaluator", decision)
                passed = passed and first.ticket_id == second.ticket_id and first.status == "completed"
            rows.append({"case": label, "expected": expected, "actual": record.status, "passed": passed})
    report = {
        "mode": "deterministic_fixture",
        "measures": "workflow_contracts_not_model_quality",
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    table = Table(title="离线流程回归（不代表模型效果）")
    for label in ("场景", "期望状态", "结果"):
        table.add_column(label)
    for row in rows:
        table.add_row(row["case"], row["expected"], "PASS" if row["passed"] else "FAIL")
    Console().print(table)
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evals/reports/workflow.json"))
    asyncio.run(evaluate(parser.parse_args().output))
