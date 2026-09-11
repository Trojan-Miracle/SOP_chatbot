"""Small reproducible retrieval ablation; upload the fixture documents before running."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.table import Table

from app.core.config import settings
from app.core.rag.bm25_index import get_bm25_index
from app.core.rag.retrieval import search_sop


async def evaluate(cases_path: Path, output: Path) -> None:
    """Compare vector and hybrid retrieval on identical labeled queries."""
    cases = json.loads(cases_path.read_text())
    report = {
        "dataset": str(cases_path),
        "k": settings.RAG_TOP_K,
        "embedding_model": settings.EMBEDDING_MODEL,
        "results": {},
    }
    await asyncio.to_thread(get_bm25_index().rebuild)
    previous = settings.RAG_HYBRID_SEARCH_ENABLED
    try:
        for mode in ("vector", "hybrid"):
            settings.RAG_HYBRID_SEARCH_ENABLED = mode == "hybrid"
            rows = []
            for case in cases:
                start = perf_counter()
                chunks = await search_sop(case["query"])
                names = [chunk.filename for chunk in chunks]
                relevant = set(case["relevant"])
                hits = relevant.intersection(names)
                ranks = [i for i, name in enumerate(names, 1) if name in relevant]
                rows.append(
                    {
                        "query": case["query"],
                        "expected": sorted(relevant),
                        "retrieved": names,
                        "recall": len(hits) / len(relevant) if relevant else None,
                        "reciprocal_rank": 1 / min(ranks) if ranks else 0,
                        "negative_query": not relevant,
                        "empty_result": not names,
                        "latency_ms": round((perf_counter() - start) * 1000, 2),
                    }
                )
            positive = [r for r in rows if not r["negative_query"]]
            negative = [r for r in rows if r["negative_query"]]
            report["results"][mode] = {
                "recall_at_k": sum(r["recall"] for r in positive) / len(positive),
                "mrr_at_k": sum(r["reciprocal_rank"] for r in positive) / len(positive),
                "negative_empty_rate": sum(r["empty_result"] for r in negative) / len(negative) if negative else None,
                "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
                "cases": rows,
            }
    finally:
        settings.RAG_HYBRID_SEARCH_ENABLED = previous
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    table = Table(title="检索对照（文档级；微型合成数据集）")
    for column in ("模式", "Recall@k", "MRR@k", "负例空结果率", "平均 ms"):
        table.add_column(column)
    for mode, result in report["results"].items():
        table.add_row(
            mode,
            f"{result['recall_at_k']:.3f}",
            f"{result['mrr_at_k']:.3f}",
            str(result["negative_empty_rate"]),
            f"{result['mean_latency_ms']:.1f}",
        )
    Console().print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("examples/retrieval-cases.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/reports/retrieval.json"))
    args = parser.parse_args()
    asyncio.run(evaluate(args.cases, args.output))
