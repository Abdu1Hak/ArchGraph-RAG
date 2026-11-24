"""Phase 1 ingestion CLI: arXiv -> Markdown -> chunks -> budget guard -> Gemini extraction -> Neo4j.

Examples:
    # Offline demo: synthetic seed Markdown + recorded extractions + in-memory graph (no key, no network)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --offline-demo

    # Pre-flight cost projection only (downloads/convert PDFs, no model calls, no graph writes)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --dry-run-budget
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --dry-run-budget --all-50

    # Live run on the 12 seed papers (needs GEMINI_API_KEY and `docker compose up -d neo4j`)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --seed-only --max-budget-usd 1.00

Exit codes: 0 ok, 2 budget exceeded, 3 configuration/connectivity error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.arxiv_fetcher import ArxivFetcher, load_manifest, load_or_synthesize_seed_markdown  # noqa: E402
from src.ingestion.cost_estimator import BudgetExceededError, ExtractionCache  # noqa: E402
from src.ingestion.extractor import GeminiGraphExtractor  # noqa: E402
from src.ingestion.offline_fixtures import RecordedExtractionLLM  # noqa: E402
from src.ingestion.pipeline import Phase1IngestionPipeline  # noqa: E402
from src.stores.neo4j_store import InMemoryGraphStore, Neo4jGraphStore  # noqa: E402

OFFLINE_DEMO_CACHE = settings.cache_dir / "extractions_offline_demo.sqlite"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 ingestion pipeline.")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--seed-only", dest="seed_only", action="store_true", default=True,
                       help="Ingest the 12 seed_set papers (default).")
    scope.add_argument("--all-50", dest="seed_only", action="store_false", help="Ingest all 50 manifest papers.")
    p.add_argument("--papers", default="", help="Comma-separated paper_ids to restrict the selection.")
    p.add_argument("--dry-run-budget", action="store_true",
                   help="Only chunk and print the projected extraction cost; no model calls, no graph writes.")
    p.add_argument("--offline-demo", action="store_true",
                   help="Synthetic Markdown + recorded extractions + in-memory graph (no API key, no network).")
    p.add_argument("--neo4j", action="store_true", help="With --offline-demo, write to Neo4j instead of memory.")
    p.add_argument("--max-budget-usd", type=float, default=settings.max_extraction_budget_usd,
                   help=f"Abort before any API call if projected cost exceeds this (default {settings.max_extraction_budget_usd}).")
    p.add_argument("--cache-path", type=Path, default=None, help="SQLite extraction cache path.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    papers = load_manifest(seed_only=args.seed_only)
    if args.papers:
        wanted = {p.strip() for p in args.papers.split(",") if p.strip()}
        unknown = wanted - {p.paper_id for p in load_manifest(seed_only=False)}
        if unknown:
            print(f"Unknown paper_id(s): {sorted(unknown)}", file=sys.stderr)
            return 3
        papers = [p for p in load_manifest(seed_only=False) if p.paper_id in wanted]
    print(f"Selected {len(papers)} paper(s): {', '.join(p.paper_id for p in papers)}")

    fetcher = None
    if args.offline_demo:
        cache = ExtractionCache(args.cache_path or OFFLINE_DEMO_CACHE)
        extractor = GeminiGraphExtractor(cache, llm_callable=RecordedExtractionLLM())
        loader = lambda paper: load_or_synthesize_seed_markdown(paper, prefer_cached=False)  # noqa: E731
    else:
        cache = ExtractionCache(args.cache_path or settings.extraction_cache_path)
        extractor = GeminiGraphExtractor(cache)
        fetcher = ArxivFetcher()
        loader = fetcher.fetch_markdown

    store = None
    try:
        if not args.dry_run_budget:
            if args.offline_demo and not args.neo4j:
                store = InMemoryGraphStore()
            else:
                if not args.offline_demo and not settings.gemini_api_key:
                    print("GEMINI_API_KEY is not set (see .env.example). Use --offline-demo or --dry-run-budget.", file=sys.stderr)
                    return 3
                store = Neo4jGraphStore()
                try:
                    store.verify_connectivity()
                except Exception as exc:
                    print(f"Cannot reach Neo4j at {settings.neo4j_uri}: {exc}\nStart it with: docker compose up -d neo4j",
                          file=sys.stderr)
                    return 3

        pipeline = Phase1IngestionPipeline(extractor, store, loader)
        try:
            report = pipeline.run(papers, max_budget_usd=args.max_budget_usd, dry_run_budget=args.dry_run_budget)
        except BudgetExceededError as exc:
            print(f"BUDGET GUARD: {exc}", file=sys.stderr)
            return 2

        print(report.summary())
        if args.dry_run_budget:
            print(f"Dry run only: budget cap ${args.max_budget_usd:.2f} -> "
                  f"{'WITHIN' if report.estimate.projected_usd <= args.max_budget_usd else 'EXCEEDS'} budget.")
        elif isinstance(store, InMemoryGraphStore):
            print(f"In-memory graph: {store.count_nodes()} nodes, {store.count_relationships()} relationships "
                  f"(re-run to see the cache serve every chunk at $0.00).")
        return 0
    finally:
        cache.close()
        if fetcher is not None:
            fetcher.close()
        if isinstance(store, Neo4jGraphStore):
            store.close()


if __name__ == "__main__":
    sys.exit(main())
