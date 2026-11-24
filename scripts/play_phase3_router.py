"""Interactive Phase 3 router playground: RoutingDecision, ranked graph paths, reranked supporting chunks.

Examples:
    .\\.venv\\Scripts\\python.exe scripts/play_phase3_router.py --query "Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, and give the exact W^{DKV} latent compression equation"
    .\\.venv\\Scripts\\python.exe scripts/play_phase3_router.py          # prompt loop

Offline (HeuristicRouterLLM, in-memory stores over the 12 seed papers). `--gemini` uses GeminiRouterLLM.
Decisions are appended to data/cache/routing_decisions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.offline_demo import build_offline_corpus  # noqa: E402
from src.retrieval.router import GeminiRouterLLM, GraphVectorEngine, QueryRouter, RetrievalResult  # noqa: E402

LINE = "=" * 100


def _s(eid: str) -> str:
    return eid.split("::", 1)[-1]


def show(res: RetrievalResult, ms: float) -> None:
    d = res.decision
    print(LINE)
    print(f" QUESTION : {d.question}")
    print(LINE)
    print(f" Route    : {d.route.value}" + (f"  (upgraded from {d.original_route.value})" if d.fallback_triggered else ""))
    print(f" Intent   : {d.query_intent.value}   confidence={d.confidence}   router={d.router_model}   "
          f"routing={d.latency_ms} ms, total={ms:.1f} ms")
    print(f" Entities : {', '.join(_s(e) for e in d.resolved_entity_ids) or '(none resolved)'}"
          + ("   [vector-seeded graph entry]" if d.vector_seeded else ""))
    if d.fallback_reasons:
        print(f" Fallback : {'; '.join(d.fallback_reasons)}")
    print(f" Graph sub-query  : {d.graph_subquery or '-'}")
    print(f" Vector sub-query : {d.vector_subquery or '-'}")
    print(f" Template : {d.cypher_template_id.value if d.cypher_template_id else '-'}   $params={json.dumps(d.template_params)}")

    tr = res.template_result
    print("\n RANKED GRAPH PATHS")
    print(LINE)
    if not tr or not tr.paths:
        print("  (none)")
    else:
        for i, path in enumerate(tr.paths[:8], start=1):
            # Start from the hop-0 endpoint that is not shared with hop 1 (the traversal origin).
            node = path[0]["source"]
            if len(path) > 1 and node in (path[1]["source"], path[1]["target"]):
                node = path[0]["target"]
            chain = _s(node)
            parts = []
            for h in path:
                fwd = h["source"] == node
                parts.append(f"-[:{h['relation_type']}]->({_s(h['target'])})" if fwd else f"<-[:{h['relation_type']}]-({_s(h['source'])})")
                node = h["target"] if fwd else h["source"]
            score = f"  score={tr.hop_scores[i - 1]:.2f}" if tr.hop_scores else ""
            print(f"  {i}. ({chain}){''.join(parts)}{score}")
            print(f"      evidence chunks: {', '.join(c for h in path for c in h['source_chunk_ids'])}")
        for agg in tr.aggregates[:5]:
            print(f"  aggregate: {json.dumps(agg)}")

    for title, chunks in (("RERANKED GRAPH-SUPPORTING CHUNKS", res.supporting_chunks), ("VECTOR CHUNKS (RRF + Rerank)", res.vector_chunks)):
        print(f"\n {title}")
        print(LINE)
        if not chunks:
            print("  (none)")
        for c in chunks:
            print(f"  {c.rank}. {c.score:7.3f}  {c.chunk_id}  [{c.section_path}]")
            print(textwrap.indent(textwrap.fill(c.raw_text.replace("\n", " "), 94), "      "))
    print(LINE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 router playground (offline by default).")
    ap.add_argument("--query", "-q")
    ap.add_argument("--graph-top-k", type=int, default=3)
    ap.add_argument("--vector-top-k", type=int, default=3)
    ap.add_argument("--gemini", action="store_true", help="Use GeminiRouterLLM (needs GEMINI_API_KEY).")
    args = ap.parse_args()

    c = build_offline_corpus()
    router = QueryRouter(c.resolver, c.graph, c.vectors, llm=GeminiRouterLLM() if args.gemini else None)
    engine = GraphVectorEngine(router)

    def run(q: str) -> None:
        t0 = time.perf_counter()
        res = engine.retrieve(q, graph_top_k=args.graph_top_k, vector_top_k=args.vector_top_k)
        show(res, (time.perf_counter() - t0) * 1000)

    if args.query:
        run(args.query)
        return 0
    try:
        while True:
            q = input("\nquestion> ").strip()
            if q in ("", "exit", "quit"):
                return 0
            run(q)
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    sys.exit(main())
