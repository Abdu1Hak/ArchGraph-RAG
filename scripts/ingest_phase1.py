"""Phase 1 ingestion CLI: arXiv -> Markdown -> chunks -> budget guard -> Gemini extraction -> Neo4j.

Examples:
    # Offline demo: synthetic seed Markdown + recorded extractions + in-memory graph (no key, no network)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --offline-demo

    # Pre-flight cost projection only (downloads/convert PDFs, no model calls, no graph writes)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --dry-run-budget
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --dry-run-budget --all-50

    # Live run on the 12 seed papers into Neo4j AND pgvector (needs GEMINI_API_KEY and
    # `docker compose up -d neo4j postgres`)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --seed-only --index-vectors --max-budget-usd 1.00

    # Seed live Neo4j + pgvector from the 12-paper fixtures (0 LLM calls, real Gemini embeddings, ~5 s)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --seed-from-fixtures

    # Same for all 50 manifest papers (data/fixtures/full_corpus: 38 more papers + 2 seed-paper appendices)
    .\\.venv\\Scripts\\python.exe scripts/ingest_phase1.py --seed-from-fixtures --all-50

Exit codes: 0 ok, 2 budget exceeded, 3 configuration/connectivity error,
4 daily API quota reached on every fallback model (all finished work saved; swap the key and re-run).
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.arxiv_fetcher import ArxivFetcher, load_manifest, load_or_synthesize_seed_markdown  # noqa: E402
from src.ingestion.chunker import StructureAwareChunker  # noqa: E402
from src.ingestion.cost_estimator import BudgetExceededError, ExtractionCache  # noqa: E402
from src.ingestion.extractor import GeminiGraphExtractor  # noqa: E402
from src.ingestion.offline_fixtures import RecordedExtractionLLM  # noqa: E402
from src.ingestion.pipeline import IngestionQuotaExhausted, Phase1IngestionPipeline
from src.ingestion.rate_limiter import get_governor  # noqa: E402
from src.stores.neo4j_store import InMemoryGraphStore, Neo4jGraphStore  # noqa: E402

OFFLINE_DEMO_CACHE = settings.cache_dir / "extractions_offline_demo.sqlite"
OFFLINE_DEMO_CACHE_ALL50 = settings.cache_dir / "extractions_offline_demo_all50.sqlite"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 ingestion pipeline.")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--seed-only", dest="seed_only", action="store_true", default=argparse.SUPPRESS,
                       help="Ingest only seed_set papers (default when --manifest is omitted).")
    scope.add_argument("--all-50", dest="seed_only", action="store_false", default=argparse.SUPPRESS,
                       help="Ingest all manifest papers.")
    p.set_defaults(seed_only=None)
    p.add_argument("--manifest", type=Path, default=None,
                   help=f"Path to a corpus manifest YAML (default {settings.corpus_manifest_path}).")
    p.add_argument("--core-body-only", action="store_true",
                   help="Strip trailing back-matter sections (References, Acknowledgments, Limitations, Appendices, etc.) before chunking.")
    p.add_argument("--max-chunks-per-paper", type=int, default=None,
                   help="Cap the number of retained chunks per paper (default: uncapped).")
    p.add_argument("--papers", default="", help="Comma-separated paper_ids to restrict the selection.")
    p.add_argument("--dry-run-budget", action="store_true",
                   help="Only chunk and print the projected extraction cost; no model calls, no graph writes.")
    p.add_argument("--offline-demo", action="store_true",
                   help="Synthetic Markdown + recorded extractions + in-memory graph (no API key, no network).")
    p.add_argument("--neo4j", action="store_true", help="With --offline-demo, write to Neo4j instead of memory.")
    p.add_argument("--index-vectors", action="store_true",
                   help="Also index every chunk + its resolved entity_ids into PostgreSQL pgvector (GeminiEmbedder; "
                        "with --offline-demo and no --neo4j: an in-memory index with the mock embedder).")
    p.add_argument("--max-budget-usd", type=float, default=settings.max_extraction_budget_usd,
                   help=f"Abort before any API call if projected cost exceeds this (default {settings.max_extraction_budget_usd}).")
    p.add_argument("--seed-from-fixtures", action="store_true",
                   help="Seed LIVE Neo4j + pgvector from the enriched fixture corpus (12 papers, or 50 with --all-50): "
                        "recorded extractions (0 LLM calls) but real GeminiEmbedder vectors.")
    p.add_argument("--cache-path", type=Path, default=None, help="SQLite extraction cache path.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.max_chunks_per_paper is not None and args.max_chunks_per_paper < 0:
        p.error("--max-chunks-per-paper must be >= 0")
    if args.seed_only is None:
        args.seed_only = args.manifest is None
    return args


def open_vector_store(args: argparse.Namespace):
    """pgvector + GeminiEmbedder for real ingestion; mock embedder for --offline-demo (in memory unless --neo4j)."""
    from src.stores.embedders import DeterministicMockEmbedder, GeminiEmbedder
    from src.stores.pgvector_store import InMemoryVectorStore, PgVectorStore

    if args.offline_demo and not args.neo4j:
        return InMemoryVectorStore(embedder=DeterministicMockEmbedder())
    import psycopg

    # Mock vectors only for a pure --offline-demo; --seed-from-fixtures and real runs use Gemini embeddings.
    embedder = DeterministicMockEmbedder() if (args.offline_demo and not args.seed_from_fixtures) else GeminiEmbedder()
    store = PgVectorStore(connection=psycopg.connect(settings.postgres_dsn, connect_timeout=5), embedder=embedder)
    store.init_schema()
    return store


def main() -> int:
    args = parse_args()
    settings.reload()  # pick up a swapped GEMINI_API_KEY / model names from .env
    if args.seed_from_fixtures:
        args.offline_demo, args.neo4j, args.index_vectors = True, True, True
        if not settings.gemini_api_key:
            print("--seed-from-fixtures embeds with Gemini: set GEMINI_API_KEY in .env.", file=sys.stderr)
            return 3
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    papers = load_manifest(manifest_path=args.manifest, seed_only=args.seed_only)
    if args.papers:
        wanted = {p.strip() for p in args.papers.split(",") if p.strip()}
        all_manifest = load_manifest(manifest_path=args.manifest, seed_only=False)
        unknown = wanted - {p.paper_id for p in all_manifest}
        if unknown:
            print(f"Unknown paper_id(s): {sorted(unknown)}", file=sys.stderr)
            return 3
        papers = [p for p in all_manifest if p.paper_id in wanted]
    print(f"Selected {len(papers)} paper(s): {', '.join(p.paper_id for p in papers)}")

    fetcher = None
    if args.offline_demo:
        # Any non-seed paper switches to the 50-paper fixtures (38 more papers + appendices to 2 seed papers).
        full = any(not p.seed_set for p in papers)
        cache = ExtractionCache(args.cache_path or (OFFLINE_DEMO_CACHE_ALL50 if full else OFFLINE_DEMO_CACHE))
        extractor = GeminiGraphExtractor(cache, llm_callable=RecordedExtractionLLM(full_corpus=full))
        loader = lambda paper: load_or_synthesize_seed_markdown(paper, prefer_cached=False, full_corpus=full)  # noqa: E731
    else:
        cache = ExtractionCache(args.cache_path or settings.extraction_cache_path)
        extractor = GeminiGraphExtractor(cache)
        fetcher = ArxivFetcher()
        loader = fetcher.fetch_markdown

    store = None
    vectors = None
    try:
        if args.index_vectors and not args.dry_run_budget:
            # Connect BEFORE extraction so an unreachable database never wastes extraction spend.
            try:
                vectors = open_vector_store(args)
            except Exception as exc:
                print(f"Cannot open the vector store: {type(exc).__name__}: {exc}\n"
                      f"Start it with: docker compose up -d postgres (host port {settings.postgres_port})", file=sys.stderr)
                return 3
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

        failed_downloads: List[Tuple[str, str, str]] = []
        if args.dry_run_budget and fetcher is not None:
            cached_markdown: Dict[str, str] = {}
            ok_papers = []
            for p in papers:
                try:
                    cached_markdown[p.paper_id] = fetcher.fetch_markdown(p)
                    ok_papers.append(p)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    failed_downloads.append((p.paper_id, p.arxiv_id, err))
                    logging.getLogger(__name__).warning(
                        "Failed to fetch %s (%s): %s", p.paper_id, p.arxiv_id, err
                    )
            papers = ok_papers
            loader = lambda paper: cached_markdown[paper.paper_id]  # noqa: E731

        chunker = StructureAwareChunker(
            core_body_only=args.core_body_only,
            max_chunks_per_paper=args.max_chunks_per_paper,
        )
        pipeline = Phase1IngestionPipeline(extractor, store, loader, chunker=chunker, known_papers=papers)
        try:
            report = pipeline.run(papers, max_budget_usd=args.max_budget_usd, dry_run_budget=args.dry_run_budget,
                                  vector_store=vectors)
        except BudgetExceededError as exc:
            print(f"BUDGET GUARD: {exc}", file=sys.stderr)
            return 2
        except IngestionQuotaExhausted as exc:
            bar = "=" * 100
            print(f"\n{bar}\n {exc.banner()}\n{bar}", file=sys.stderr)
            print(f" stage: {exc.stage} | committed papers: {len(exc.report.committed_papers)} | next chunk: "
                  f"{exc.next_chunk_id} | exhausted models: {', '.join(getattr(exc.cause, 'models', []))}", file=sys.stderr)
            return 4

        print(report.summary())
        embed_usd = sum(c.approx_token_count for c in report.chunks) * settings.embedding_usd_per_1m / 1_000_000
        if args.dry_run_budget:
            counts = sorted(sum(1 for c in report.chunks if c.paper_id == pid) for pid in report.papers)
            if counts:
                print(
                    f"Per-paper chunks ({len(counts)} paper(s)): "
                    f"min={counts[0]}, median={statistics.median(counts):g}, max={counts[-1]}, total={sum(counts)}"
                )
            if failed_downloads:
                print(f"Failed downloads ({len(failed_downloads)}):", file=sys.stderr)
                for pid, aid, err in failed_downloads:
                    print(f"  - {pid} ({aid}): {err}", file=sys.stderr)
            print(f"Projected one-time embedding cost for {len(report.chunks)} chunks (if --index-vectors): ${embed_usd:.4f}")
            print(f"Dry run only: budget cap ${args.max_budget_usd:.2f} -> "
                  f"{'WITHIN' if report.estimate.projected_usd <= args.max_budget_usd else 'EXCEEDS'} budget.")
        if vectors is not None:
            print(f"Vector index ({type(vectors).__name__}, embedder {getattr(vectors.embedder, 'name', '?')}): "
                  f"{vectors.count()} chunks, indexed paper by paper (projected embedding cost ~${embed_usd:.4f}); "
                  "chunk_ids match the graph.")
        gov = get_governor()
        if gov.paced_requests:
            print(f"Rate pacing: {gov.paced_requests} request(s) waited {gov.total_wait_s:.1f}s in total to stay under RPM/TPM.")
        if isinstance(store, InMemoryGraphStore):
            print(f"In-memory graph: {store.count_nodes()} nodes, {store.count_relationships()} relationships "
                  f"(re-run to see the cache serve every chunk at $0.00).")
        return 0
    finally:
        cache.close()
        if fetcher is not None:
            fetcher.close()
        if isinstance(store, Neo4jGraphStore):
            store.close()
        if vectors is not None and hasattr(vectors, "close"):
            vectors.close()


if __name__ == "__main__":
    sys.exit(main())
