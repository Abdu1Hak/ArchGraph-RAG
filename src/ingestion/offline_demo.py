"""Build the offline Phase 1 graph + Phase 2 vector index over the 12 seed papers (no Docker, no API key)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ingestion.arxiv_fetcher import load_manifest, load_or_synthesize_seed_markdown
from src.ingestion.cost_estimator import ExtractionCache
from src.ingestion.entity_resolver import EntityResolver
from src.ingestion.extractor import GeminiGraphExtractor
from src.ingestion.offline_fixtures import RecordedExtractionLLM
from src.ingestion.pipeline import Phase1IngestionPipeline, PipelineReport
from src.stores.embedders import DeterministicMockEmbedder
from src.stores.neo4j_store import InMemoryGraphStore
from src.stores.pgvector_store import InMemoryVectorStore


@dataclass
class OfflineCorpus:
    report: PipelineReport
    graph: InMemoryGraphStore
    vectors: InMemoryVectorStore
    embedder: DeterministicMockEmbedder
    resolver: EntityResolver


def build_offline_corpus(vector_store: Optional[InMemoryVectorStore] = None, seed_only: bool = True) -> OfflineCorpus:
    """Run Phase 1 offline, then index chunks + resolved entity_ids for Phase 2.

    seed_only=True (default, used by every scorecard): the 12 seed papers, unchanged.
    seed_only=False: all 50 manifest papers with the full-corpus fixtures and recordings.
    """
    graph = InMemoryGraphStore()
    extractor = GeminiGraphExtractor(ExtractionCache(":memory:"), llm_callable=RecordedExtractionLLM(full_corpus=not seed_only))
    pipeline = Phase1IngestionPipeline(
        extractor, graph, lambda p: load_or_synthesize_seed_markdown(p, prefer_cached=False, full_corpus=not seed_only)
    )
    report = pipeline.run(load_manifest(seed_only=seed_only), max_budget_usd=1.0)
    embedder = DeterministicMockEmbedder()
    vectors = vector_store or InMemoryVectorStore(embedder=embedder)
    vectors.index_chunks(report.chunks, report.mentions)
    return OfflineCorpus(report, graph, vectors, vectors.embedder, pipeline.resolver)
