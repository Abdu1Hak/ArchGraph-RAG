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


def build_offline_corpus(vector_store: Optional[InMemoryVectorStore] = None) -> OfflineCorpus:
    """Run Phase 1 offline over the seed papers, then index chunks + resolved entity_ids for Phase 2."""
    graph = InMemoryGraphStore()
    extractor = GeminiGraphExtractor(ExtractionCache(":memory:"), llm_callable=RecordedExtractionLLM())
    pipeline = Phase1IngestionPipeline(
        extractor, graph, lambda p: load_or_synthesize_seed_markdown(p, prefer_cached=False)
    )
    report = pipeline.run(load_manifest(seed_only=True), max_budget_usd=1.0)
    embedder = DeterministicMockEmbedder()
    vectors = vector_store or InMemoryVectorStore(embedder=embedder)
    vectors.index_chunks(report.chunks, report.mentions)
    return OfflineCorpus(report, graph, vectors, vectors.embedder, pipeline.resolver)
