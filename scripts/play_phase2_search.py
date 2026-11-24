"""Interactive Phase 2 search playground: Dense vs BM25 vs Hybrid RRF vs RRF + Cross-Encoder, plus 1-hop graph.

Examples:
    .\\.venv\\Scripts\\python.exe scripts/play_phase2_search.py --query "What is the W^{DKV} latent compression matrix in MLA?"
    .\\.venv\\Scripts\\python.exe scripts/play_phase2_search.py --query "preference optimization" --pillar post_training
    .\\.venv\\Scripts\\python.exe scripts/play_phase2_search.py            # interactive prompt loop

Runs offline over the 12 seed papers (synthetic fixtures, mock embedder, in-memory stores).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.offline_demo import OfflineCorpus, build_offline_corpus  # noqa: E402
from src.stores.pgvector_store import RetrievedChunk  # noqa: E402

LINE = "=" * 100


def _short(eid: str) -> str:
    return eid.split("::", 1)[-1]


def _hit_line(h: RetrievedChunk) -> str:
    extra = ""
    if "dense_rank" in h.stage_scores:
        extra = f" (dense#{h.stage_scores.get('dense_rank') or '-'}, bm25#{h.stage_scores.get('lexical_rank') or '-'})"
    if "stage1_rank" in h.stage_scores:
        extra = f" (stage1 RRF #{h.stage_scores['stage1_rank']})"
    snippet = textwrap.shorten(h.raw_text.replace("\n", " "), width=92, placeholder=" ...")
    return f"  {h.rank}. {h.score:8.4f}  {h.chunk_id}{extra}\n       {h.section_path}\n       \"{snippet}\""


def search_once(c: OfflineCorpus, args: argparse.Namespace, query: str) -> None:
    v, filters = c.vectors, {k: getattr(args, k) for k in ("pillar", "min_year") if getattr(args, k) is not None}
    t0 = time.perf_counter()
    modes: Dict[str, List[RetrievedChunk]] = {
        "DENSE (cosine)": v.search_dense(query, top_k=args.top_k, ef_search=args.ef_search, filters=filters),
        "BM25 (lexical)": v.search_lexical_bm25(query, top_k=args.top_k, filters=filters),
        f"HYBRID RRF (k={args.rrf_k})": v.search_hybrid_rrf(query, top_k=args.top_k, rrf_k=args.rrf_k,
                                                           ef_search=args.ef_search, filters=filters),
        f"RRF + CROSS-ENCODER (candidates={args.candidate_k})": v.search_hybrid_reranked(
            query, top_k=args.top_k, candidate_k=max(args.candidate_k, args.top_k), rrf_k=args.rrf_k,
            ef_search=args.ef_search, filters=filters),
    }
    ms = (time.perf_counter() - t0) * 1000
    print(LINE)
    print(f" QUERY: {query}")
    print(f" top_k={args.top_k} rrf_k={args.rrf_k} ef_search={args.ef_search} filters={filters or 'none'} | 4 modes in {ms:.1f} ms")
    print(LINE)
    for name, hits in modes.items():
        print(f"\n [{name}]")
        print("\n".join(_hit_line(h) for h in hits) if hits else "  (no hits)")

    reranked = next(iter(reversed(modes.values())))
    print("\n" + LINE)
    if not reranked:
        print(" 1-HOP GRAPH EXPANSION: no top chunk")
        return
    top = reranked[0]
    print(f" 1-HOP GRAPH EXPANSION from top chunk {top.chunk_id}")
    print(f" entity_ids: {', '.join(_short(e) for e in top.entity_ids) or '(none)'}")
    print(LINE)
    seen = set()
    for eid in top.entity_ids:
        for n in c.graph.neighbors(eid):
            key = (n["source"], n["relation_type"], n["target"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  ({_short(n['source'])})-[:{n['relation_type']}]->({_short(n['target'])})   "
                  f"evidence: {', '.join(n['source_chunk_ids'])}")
    if not seen:
        print("  (no semantic edges)")
    print(LINE)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2 hybrid search playground (offline).")
    p.add_argument("--query", "-q", default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--ef-search", type=int, default=80)
    p.add_argument("--candidate-k", type=int, default=15)
    p.add_argument("--pillar", choices=["pre_training", "architecture", "mid_training", "post_training", "inference"])
    p.add_argument("--min-year", type=int, default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    corpus = build_offline_corpus()
    print(f"Indexed {corpus.vectors.count()} chunks from {len(corpus.report.papers)} seed papers "
          f"in {time.perf_counter() - t0:.2f}s (offline: synthetic text, mock embedder).")
    try:
        if args.query:
            search_once(corpus, args, args.query)
            return 0
        while True:
            q = input("\nquery> ").strip()
            if q in ("", "exit", "quit"):
                return 0
            search_once(corpus, args, q)
    except ValueError as exc:
        print(f"Invalid parameter: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    sys.exit(main())
