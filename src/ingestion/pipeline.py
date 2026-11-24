"""Phase 1 Ingestion Orchestrator: manifest -> Markdown -> chunks -> budget -> extract -> resolve -> graph.

Order of operations (budget guard runs before ANY model call or graph write):
1. Load Markdown per paper and chunk it (deterministic `chunk_id`s and `content_hash`es).
2. `CostBudgetEstimator.enforce_budget()` - raises `BudgetExceededError` if over the cap.
   `dry_run_budget=True` stops here and returns the estimate.
3. Upsert `(:Entity:Paper)` + `(:Chunk)-[:IN_PAPER]->` for each paper.
4. Extract each chunk (cache-first), resolve entities/relationships through ONE shared
   `EntityResolver` (seeded with `seed_aliases.yaml` + every manifest paper).
5. Upsert resolved nodes, aggregated edges, and `MENTIONED_IN` links.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, Dict, List, Optional, Sequence

from src.config import settings
from src.ingestion.arxiv_fetcher import PaperMeta, load_manifest
from src.ingestion.chunker import DocumentChunk, StructureAwareChunker
from src.ingestion.cost_estimator import CostBudgetEstimator, CostEstimate
from src.ingestion.entity_resolver import EntityResolver
from src.ingestion.extractor import ExtractionRunStats, GeminiGraphExtractor
from src.schema.ontology import ResolvedEntityNode, ResolvedGraphEdge
from src.stores.neo4j_store import GraphStore, aggregate_edges

logger = logging.getLogger(__name__)

MarkdownLoader = Callable[[PaperMeta], str]


@dataclass
class PipelineReport:
    papers: List[str] = field(default_factory=list)
    chunks: List[DocumentChunk] = field(default_factory=list)
    estimate: Optional[CostEstimate] = None
    extraction: Optional[ExtractionRunStats] = None
    nodes: List[ResolvedEntityNode] = field(default_factory=list)
    edges: List[ResolvedGraphEdge] = field(default_factory=list)
    mentions: Dict[str, List[str]] = field(default_factory=dict)
    nodes_written: int = 0
    edges_written: int = 0
    mentions_written: int = 0
    dry_run: bool = False

    @property
    def unique_edge_count(self) -> int:
        return len(aggregate_edges(self.edges))

    def summary(self) -> str:
        lines = [f"papers={len(self.papers)} chunks={len(self.chunks)}"]
        if self.estimate:
            lines.append(f"budget: {self.estimate.summary()}")
        if self.extraction:
            x = self.extraction
            lines.append(
                f"extraction: chunks={x.chunks} cache_hits={x.cache_hits} api_calls={x.api_calls} "
                f"failed={x.failed_chunks} dropped_items={x.dropped_items} "
                f"tokens in/out={x.input_tokens:,}/{x.output_tokens:,} actual~${x.actual_usd():.4f}"
            )
        if not self.dry_run:
            lines.append(
                f"graph: nodes={len(self.nodes)} unique_edges={self.unique_edge_count} "
                f"(raw edge mentions={len(self.edges)}) mention_links={self.mentions_written}"
            )
        return "\n".join(lines)


class Phase1IngestionPipeline:
    def __init__(
        self,
        extractor: GeminiGraphExtractor,
        graph_store: Optional[GraphStore],
        markdown_loader: MarkdownLoader,
        resolver: Optional[EntityResolver] = None,
        chunker: Optional[StructureAwareChunker] = None,
        known_papers: Optional[Sequence[PaperMeta]] = None,
    ) -> None:
        self.extractor = extractor
        self.graph_store = graph_store
        self.markdown_loader = markdown_loader
        self.resolver = resolver or EntityResolver(
            seed_aliases_path=settings.seed_aliases_path,
            auto_merge_threshold=settings.er_auto_merge_threshold,
            gray_zone_low_threshold=settings.er_gray_zone_low_threshold,
        )
        self.chunker = chunker or StructureAwareChunker()
        papers = list(known_papers) if known_papers is not None else load_manifest(seed_only=False)
        for p in papers:
            self.resolver.register_known_paper(p.paper_id, p.title)

    def chunk_papers(self, papers: Sequence[PaperMeta]) -> Dict[str, List[DocumentChunk]]:
        out: Dict[str, List[DocumentChunk]] = {}
        for p in papers:
            self.resolver.register_known_paper(p.paper_id, p.title)
            markdown = self.markdown_loader(p)
            out[p.paper_id] = self.chunker.chunk_paper(p.paper_id, p.title, p.year, p.pillar.value, markdown)
        return out

    def run(
        self,
        papers: Sequence[PaperMeta],
        max_budget_usd: Optional[float] = None,
        dry_run_budget: bool = False,
    ) -> PipelineReport:
        report = PipelineReport(papers=[p.paper_id for p in papers], dry_run=dry_run_budget)
        chunks_by_paper = self.chunk_papers(papers)
        report.chunks = [c for cs in chunks_by_paper.values() for c in cs]

        estimator = CostBudgetEstimator(report.chunks, self.extractor.cache)
        report.estimate = estimator.estimate() if dry_run_budget else estimator.enforce_budget(max_budget_usd)
        if dry_run_budget:
            return report
        if self.graph_store is None:
            raise ValueError("graph_store is required unless dry_run_budget=True")

        store = self.graph_store
        store.init_schema()
        for p in papers:
            store.upsert_paper_and_chunks(p, chunks_by_paper[p.paper_id])

        resolved_in_chunk: Dict[str, set] = {}
        for chunk in report.chunks:
            result = self.extractor.extract_chunk(chunk)
            ids = resolved_in_chunk.setdefault(chunk.chunk_id, set())
            for ent in result.payload.entities:
                ids.add(self.resolver.resolve_entity(ent, chunk.chunk_id, chunk.paper_id).entity_id)
            for rel in result.payload.relationships:
                edge = self.resolver.resolve_relationship(rel, chunk.chunk_id, chunk.paper_id)
                if edge is not None:
                    report.edges.append(edge)
                    ids.update({edge.source_entity_id, edge.target_entity_id})
        report.extraction = self.extractor.stats

        for chunk in report.chunks:
            found = set(self.resolver.resolve_text_mentions_in_chunk(chunk.raw_text))
            report.mentions[chunk.chunk_id] = sorted(found | resolved_in_chunk.get(chunk.chunk_id, set()))

        mentioned = {eid for ids in report.mentions.values() for eid in ids}
        report.nodes = [n for n in self.resolver.nodes.values() if n.source_chunk_ids or n.entity_id in mentioned]
        report.nodes_written = store.upsert_resolved_nodes(report.nodes)
        report.edges_written = store.upsert_resolved_edges(report.edges)
        for chunk_id, ids in report.mentions.items():
            report.mentions_written += store.upsert_chunk_mentions(chunk_id, ids)
        return report
