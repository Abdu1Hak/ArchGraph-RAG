"""FastAPI service (Phases 4 + 6): web UI, POST /query (grounded answer + graph_trace + funnel_trace), GET /health.

Run:  .\\.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8000   then open http://localhost:8000/

GET /            single-page UI (src/api/static/index.html): knowledge-graph canvas, retrieval funnel, citations
GET /examples    curated one-click questions;  GET /ui-config  modes + generation models
POST /query      optional `mode` = offline | gemini | live and `generation_model` per request

RAG_MODE=offline (default): in-memory corpus (all 50 papers; RAG_OFFLINE_CORPUS=seed for the 12 seed
papers), offline generator, no Docker or API key needed; RAG_GENERATOR=gemini swaps in GeminiGeneratorLLM.
RAG_MODE=live: Neo4j + pgvector (GeminiEmbedder) + GeminiRouterLLM + GeminiGeneratorLLM; startup fails
with a list of every unmet dependency. Ingest first with scripts/ingest_phase1.py --index-vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import yaml

from src.ingestion.arxiv_fetcher import PaperMeta
from src.ingestion.rate_limiter import AllModelsExhaustedError
from src.usage import QueryUsageTelemetry, summarize_usage, usage_scope
from pydantic import BaseModel, Field

from src.generation.synthesizer import (
    GeminiGeneratorLLM,
    GeneratorLLM,
    GroundedAnswer,
    GroundedAnswerSynthesizer,
    OfflineGroundedGeneratorLLM,
)
from src.retrieval.router import GraphVectorEngine, QueryRouter, RetrievalResult, RouterUnavailableError, RouteType
from src.stores.pgvector_store import RetrievedChunk


_MODEL_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9.\-_]{0,79}$"


EXTRA_GENERATION_MODELS = ("gemini-3.8-flash",)


def generation_models() -> List[str]:
    """Models the UI may request: GENERATION_MODEL, the fallback chain, then extra selectable models."""
    from src.config import settings

    chain = settings.text_model_chain(settings.generation_model)
    return list(dict.fromkeys(chain[:2] + list(EXTRA_GENERATION_MODELS) + chain[2:]))


class InvalidRequestError(ValueError):
    """A syntactically valid request the service cannot honor (HTTP 422)."""


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    graph_top_k: int = Field(default=3, ge=1, le=20)
    vector_top_k: int = Field(default=3, ge=1, le=20)
    mode: Optional[Literal["offline", "gemini", "live"]] = None  # None: the app's default service
    generation_model: Optional[str] = Field(default=None, pattern=_MODEL_NAME_RE)  # Gemini modes only


class CitationOut(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section_path: str
    source_origin: str
    score: Optional[float] = None
    text_snippet: str


class TraceEdge(BaseModel):
    source: str
    relation_type: str
    target: str
    hop_index: Optional[int] = None
    path_index: Optional[int] = None
    score: Optional[float] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    evidence_quote: Optional[str] = None
    evidence_quotes: List[str] = Field(default_factory=list)


class TraceNode(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    role: Literal["seed", "selected_path", "rejected_neighbor"]
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)


class GraphTrace(BaseModel):
    seed_node_ids: List[str] = Field(default_factory=list)
    selected_path_edges: List[TraceEdge] = Field(default_factory=list)
    explored_rejected_edges: List[TraceEdge] = Field(default_factory=list)
    nodes: List[TraceNode] = Field(default_factory=list)
    rejected_edges_truncated: int = 0  # neighbor edges not shown (display cap)


class ChunkSummary(BaseModel):
    chunk_id: str
    paper_id: str
    rank: int
    score: float
    source_origin: Optional[str] = None


class FunnelTrace(BaseModel):
    query: Optional[str] = None
    dense_top_k: List[ChunkSummary] = Field(default_factory=list)
    bm25_top_k: List[ChunkSummary] = Field(default_factory=list)
    rrf_top_k: List[ChunkSummary] = Field(default_factory=list)
    reranked_vector_top_k: List[ChunkSummary] = Field(default_factory=list)
    reranked_graph_supporting_top_k: List[ChunkSummary] = Field(default_factory=list)
    rrf_rank_by_chunk: Dict[str, int] = Field(default_factory=dict)  # RRF rank in the top-20 pool (rank movement)
    final_deduplicated_llm_context: List[ChunkSummary] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    route: str
    query_intent: str
    confidence: float
    fallback_triggered: bool
    graph_subquery: Optional[str] = None
    vector_subquery: Optional[str] = None
    cypher_template_id: Optional[str] = None
    template_params: Dict[str, Any] = Field(default_factory=dict)
    verbalized_graph_facts: List[str] = Field(default_factory=list)
    citations: List[CitationOut] = Field(default_factory=list)
    repair_attempts: int = 0
    cropped_sentences_count: int = 0
    used_extractive_fallback: bool = False
    abstained: bool = False
    generator_calls: int = 0
    latency_ms: float = 0.0
    routing_latency_ms: float = 0.0  # router classification + entity linking + template selection
    retrieval_latency_ms: float = 0.0  # vector channel + Cypher templates + supporting chunks (includes routing)
    generation_latency_ms: float = 0.0  # grounded synthesis + citation gate
    graph_trace: GraphTrace = Field(default_factory=GraphTrace)
    funnel_trace: FunnelTrace = Field(default_factory=FunnelTrace)
    usage_telemetry: QueryUsageTelemetry = Field(default_factory=QueryUsageTelemetry)
    mode: str = ""
    backend: str = ""
    resolved_entity_ids: List[str] = Field(default_factory=list)
    entity_links: List[Dict[str, Any]] = Field(default_factory=list)
    fallback_reasons: List[str] = Field(default_factory=list)
    citation_gate: Dict[str, Any] = Field(default_factory=dict)


_PREFIX_TYPES = {"paper": "Paper", "model_family": "ModelFamily", "arch_component": "ArchComponent",
                 "training_method": "TrainingMethod", "inference_technique": "InferenceTechnique",
                 "bottleneck": "Bottleneck", "hyperparam_metric": "HyperparameterOrMetric",
                 "benchmark_dataset": "BenchmarkOrDataset"}


def citation_gate_status(g: GroundedAnswer) -> Dict[str, Any]:
    """Outcome of the citation gate for the UI: which rung of the repair ladder produced the answer."""
    v = g.final_validation
    if g.refused:
        status = "REFUSED"
    elif g.abstained:
        status = "ABSTAINED"
    elif g.used_extractive_fallback:
        status = "EXTRACTIVE_FALLBACK"
    elif g.cropped_sentences_count:
        status = "CROPPED"
    elif g.repair_attempts:
        status = "REPAIRED_PASS"
    else:
        status = "PASS" if v is not None and v.valid else "UNVALIDATED"
    return {"status": status, "valid": bool(v.valid) if v is not None else False,
            "cited_ids": list(v.cited_ids) if v else [], "invented_ids": list(v.invented_ids) if v else [],
            "unresolved_ids": list(v.unresolved_ids) if v else [], "uncited_claims": len(v.uncited_claims) if v else 0,
            "valid_claim_count": int(getattr(v, "valid_claim_count", 0) or 0) if v else 0,
            "repair_attempts": g.repair_attempts, "cropped_sentences": g.cropped_sentences_count,
            "generator_calls": g.generator_calls}


def _summ(chunks: List[RetrievedChunk]) -> List[ChunkSummary]:
    return [ChunkSummary(chunk_id=c.chunk_id, paper_id=c.paper_id, rank=c.rank or i, score=round(float(c.score), 4))
            for i, c in enumerate(chunks, start=1)]


@dataclass
class LiveCorpus:
    graph: Any
    vectors: Any
    resolver: Any


class RagService:
    """Router -> retrieval -> grounded synthesis -> UI traces. Shared by the API, CLI and benchmark."""

    def __init__(self, corpus: Any, engine: GraphVectorEngine, synthesizer: GroundedAnswerSynthesizer) -> None:
        self.corpus, self.engine, self.synthesizer = corpus, engine, synthesizer

    @classmethod
    def offline(cls, generator: Optional[GeneratorLLM] = None, telemetry_path=None, router_llm: Any = None,
                full_corpus: bool = False, corpus: Any = None) -> "RagService":
        """In-memory corpus (12 seed papers by default; all 50 with full_corpus=True; or a prebuilt `corpus`)."""
        from src.ingestion.offline_demo import build_offline_corpus

        c = corpus if corpus is not None else build_offline_corpus(seed_only=not full_corpus)
        engine = GraphVectorEngine(QueryRouter(c.resolver, c.graph, c.vectors, llm=router_llm, telemetry_path=telemetry_path))
        return cls(c, engine, GroundedAnswerSynthesizer(generator or OfflineGroundedGeneratorLLM(), c.vectors))

    @classmethod
    def live(cls, telemetry_path=None, check_only: bool = False) -> "RagService":
        """Neo4j + pgvector (GeminiEmbedder) + GeminiRouterLLM + GeminiGeneratorLLM.

        Fails fast with one message listing every unmet dependency (API key, Neo4j, PostgreSQL, data).
        """
        from src.config import settings

        settings.reload()  # a swapped GEMINI_API_KEY in .env takes effect on restart
        from src.ingestion.entity_resolver import EntityResolver
        from src.retrieval.router import GeminiRouterLLM
        from src.stores.embedders import GeminiEmbedder
        from src.stores.neo4j_store import Neo4jGraphStore
        from src.stores.pgvector_store import PgVectorStore
        from src.stores.reranker import create_reranker

        problems: List[str] = []
        if not settings.gemini_api_key:
            problems.append("GEMINI_API_KEY is not set (copy .env.example to .env and fill it in)")
        graph = vectors = None
        try:
            graph = Neo4jGraphStore()
            graph.verify_connectivity()
            if graph.count_nodes("Entity") == 0:
                problems.append("Neo4j is reachable but empty: run scripts/ingest_phase1.py --index-vectors first")
        except Exception as exc:
            problems.append(f"Neo4j unreachable at {settings.neo4j_uri}: {type(exc).__name__}: {exc} "
                            "(start it with: docker compose up -d neo4j)")
        try:
            import psycopg

            conn = psycopg.connect(settings.postgres_dsn, connect_timeout=5)
            vectors = PgVectorStore(connection=conn, embedder=GeminiEmbedder(),
                                    default_reranker=create_reranker("api"))
            vectors.init_schema()
            mixed = vectors.count_other_embedders() if vectors.count() else 0
            if mixed:
                problems.append(f"{mixed} pgvector rows were embedded by a different embedder than "
                                f"{vectors.embedder.name}; re-run scripts/ingest_phase1.py --index-vectors to re-embed them")
            if vectors.count() == 0:
                problems.append("PostgreSQL is reachable but the chunks table is empty: run "
                                "scripts/ingest_phase1.py --index-vectors first")
        except Exception as exc:
            problems.append(f"PostgreSQL unreachable at {settings.postgres_host}:{settings.postgres_port}: "
                            f"{type(exc).__name__}: {exc} (start it with: docker compose up -d postgres)")
        if problems:
            raise RuntimeError("Live mode is not ready:\n  - " + "\n  - ".join(problems))
        if check_only:
            return None  # type: ignore[return-value]

        resolver = EntityResolver(seed_aliases_path=settings.seed_aliases_path)
        for node in graph.load_entities():  # query-time entity recognition needs every ingested name/alias
            if node.entity_id in resolver.nodes:
                if node.description and not resolver.nodes[node.entity_id].description:
                    resolver.nodes[node.entity_id].description = node.description
                for alias in [node.canonical_name] + node.aliases:
                    resolver._add_alias_to_node(node.entity_id, alias)
            else:
                resolver._register_node(node)
        resolver.allow_live_node_embed = False
        resolver.load_cached_query_node_embeddings(vectors.embedder)
        corpus = LiveCorpus(graph=graph, vectors=vectors, resolver=resolver)
        # Live mode is 100% agent-routed: no silent fallback to the offline heuristic router.
        router = QueryRouter(resolver, graph, vectors, llm=GeminiRouterLLM(allow_heuristic_fallback=False),
                             telemetry_path=telemetry_path)
        return cls(corpus, GraphVectorEngine(router), GroundedAnswerSynthesizer(GeminiGeneratorLLM(), vectors))

    def answer_with_usage(self, req: QueryRequest) -> tuple:
        """(result, grounded, latency_ms, QueryUsageTelemetry) for one question."""
        with usage_scope() as meter:
            result, grounded, ms = self.answer(req)
        return result, grounded, ms, self._usage(meter)

    def _usage(self, meter: Any) -> QueryUsageTelemetry:
        router = getattr(self.engine.router.llm, "name", "custom")
        emb = getattr(self.corpus.vectors.embedder, "name", type(self.corpus.vectors.embedder).__name__)
        offline = not meter.events and not isinstance(self.corpus, LiveCorpus)
        return summarize_usage(meter, router_name=router, generator_name=self.synthesizer.generator.name,
                               embedding_name=emb,
                               note="offline mode: heuristic router, extractive generator, mock embedder (0 API calls)"
                               if offline else "")

    def synthesizer_for(self, req: QueryRequest) -> GroundedAnswerSynthesizer:
        """Per-request generator override (Gemini generators only); the shared synthesizer is never mutated."""
        gen = self.synthesizer.generator
        model = req.generation_model
        if not model or not isinstance(gen, GeminiGeneratorLLM) or model == gen.model:
            return self.synthesizer
        from src.config import settings

        allowed = generation_models()
        if model not in allowed:
            raise InvalidRequestError(f"generation_model must be one of {allowed}")
        return GroundedAnswerSynthesizer(GeminiGeneratorLLM(model=model), self.corpus.vectors,
                                         max_repairs=self.synthesizer.max_repairs,
                                         max_passages=self.synthesizer.max_passages)

    def answer(self, req: QueryRequest, synthesizer: Optional[GroundedAnswerSynthesizer] = None) -> tuple:
        t0 = time.perf_counter()
        result = self.engine.retrieve(req.question, graph_top_k=req.graph_top_k, vector_top_k=req.vector_top_k)
        grounded = (synthesizer or self.synthesizer).synthesize(req.question, result)
        return result, grounded, (time.perf_counter() - t0) * 1000

    @property
    def backend(self) -> str:
        return "live-neo4j-pgvector" if isinstance(self.corpus, LiveCorpus) else "offline-in-memory"

    def query(self, req: QueryRequest, mode: str = "") -> QueryResponse:
        synth = self.synthesizer_for(req)
        # The UI traces re-run vector searches (possibly embedding queries), so they share the usage scope.
        with usage_scope() as meter:
            t0 = time.perf_counter()
            result = self.engine.retrieve(req.question, graph_top_k=req.graph_top_k, vector_top_k=req.vector_top_k)
            t1 = time.perf_counter()
            g = synth.synthesize(req.question, result)
            t2 = time.perf_counter()
            ms = (t2 - t0) * 1000
            graph_trace, funnel_trace = self.graph_trace(result), self.funnel_trace(req, result, g)
        usage = self._usage(meter)
        if synth is not self.synthesizer and not usage.generator_requests:
            usage.generator_model = synth.generator.name
        d = result.decision
        return QueryResponse(
            answer=g.answer, refused=g.refused, route=d.route.value, query_intent=d.query_intent.value,
            confidence=d.confidence, fallback_triggered=d.fallback_triggered, graph_subquery=d.graph_subquery,
            vector_subquery=d.vector_subquery,
            cypher_template_id=d.cypher_template_id.value if d.cypher_template_id else None,
            template_params=d.template_params, verbalized_graph_facts=g.verbalized_graph_facts,
            citations=[CitationOut(**c.__dict__) for c in g.citations], repair_attempts=g.repair_attempts,
            cropped_sentences_count=g.cropped_sentences_count, used_extractive_fallback=g.used_extractive_fallback,
            abstained=g.abstained,
            generator_calls=g.generator_calls, latency_ms=round(ms, 3),
            routing_latency_ms=round(d.latency_ms, 3), retrieval_latency_ms=round((t1 - t0) * 1000, 3),
            generation_latency_ms=round((t2 - t1) * 1000, 3),
            graph_trace=graph_trace, funnel_trace=funnel_trace,
            usage_telemetry=usage, mode=mode, backend=self.backend,
            resolved_entity_ids=list(d.resolved_entity_ids), entity_links=list(d.entity_links),
            fallback_reasons=list(d.fallback_reasons), citation_gate=citation_gate_status(g),
        )

    MAX_REJECTED_EDGES = 60
    MAX_REJECTED_PER_NODE = 8

    def _node_meta(self, entity_id: str, role: str) -> TraceNode:
        node = getattr(self.corpus.resolver, "nodes", {}).get(entity_id)
        if node is not None:
            return TraceNode(entity_id=entity_id, canonical_name=node.canonical_name, entity_type=node.entity_type.value,
                             role=role, aliases=list(node.aliases)[:8], description=node.description or "",
                             source_chunk_ids=list(node.source_chunk_ids)[:12])
        props = getattr(self.corpus.graph, "nodes", {}).get(entity_id, {})  # in-memory graph fallback
        prefix = entity_id.split("::", 1)[0]
        return TraceNode(entity_id=entity_id, canonical_name=props.get("canonical_name") or props.get("title") or entity_id,
                         entity_type=props.get("entity_type") or _PREFIX_TYPES.get(prefix, prefix), role=role,
                         aliases=list(props.get("aliases", []))[:8], description=props.get("description") or "",
                         source_chunk_ids=list(props.get("source_chunk_ids", []))[:12])

    def graph_trace(self, result: RetrievalResult) -> GraphTrace:
        d, tr = result.decision, result.template_result
        if d.route == RouteType.OUT_OF_SCOPE or tr is None:
            seeds = list(d.resolved_entity_ids)
            return GraphTrace(seed_node_ids=seeds, nodes=[self._node_meta(n, "seed") for n in seeds])
        seeds = list(d.resolved_entity_ids) or list(d.template_params.get("seed_entity_ids", []))
        selected: Dict[tuple, TraceEdge] = {}
        for pi, path in enumerate(tr.paths[:6]):
            for hi, h in enumerate(path):
                key = (h["source"], h["relation_type"], h["target"])
                score = tr.hop_scores[pi] if tr.hop_scores and pi < len(tr.hop_scores) else None
                if key not in selected or hi < (selected[key].hop_index or 0):
                    quotes = list(h.get("evidence_quotes") or [])
                    selected[key] = TraceEdge(source=key[0], relation_type=key[1], target=key[2], hop_index=hi,
                                              path_index=pi, score=score, source_chunk_ids=list(h["source_chunk_ids"]),
                                              evidence_quote=quotes[0] if quotes else None, evidence_quotes=quotes)
        nodes = list(dict.fromkeys(seeds + [n for k in selected for n in (k[0], k[2])]))
        rejected: Dict[tuple, TraceEdge] = {}
        truncated = 0
        for n in nodes:
            shown = 0
            for e in self.corpus.graph.neighbors(n):
                key = (e["source"], e["relation_type"], e["target"])
                if key in selected or key in rejected:
                    continue
                if shown >= self.MAX_REJECTED_PER_NODE or len(rejected) >= self.MAX_REJECTED_EDGES:
                    truncated += 1
                    continue
                quotes = list(e.get("evidence_quotes") or [])
                rejected[key] = TraceEdge(source=key[0], relation_type=key[1], target=key[2],
                                          source_chunk_ids=list(e.get("source_chunk_ids") or []),
                                          evidence_quote=quotes[0] if quotes else None, evidence_quotes=quotes)
                shown += 1
        path_nodes = {n for k in selected for n in (k[0], k[2])}
        roles: Dict[str, str] = {n: "seed" for n in seeds}
        for n in path_nodes:
            roles.setdefault(n, "selected_path")
        for k in rejected:
            for n in (k[0], k[2]):
                roles.setdefault(n, "rejected_neighbor")
        return GraphTrace(seed_node_ids=seeds, selected_path_edges=list(selected.values()),
                          explored_rejected_edges=list(rejected.values()),
                          nodes=[self._node_meta(n, r) for n, r in roles.items()], rejected_edges_truncated=truncated)

    def funnel_trace(self, req: QueryRequest, result: RetrievalResult, g: GroundedAnswer) -> FunnelTrace:
        d = result.decision
        if d.route == RouteType.OUT_OF_SCOPE:
            return FunnelTrace()
        ft = FunnelTrace(reranked_vector_top_k=_summ(result.vector_chunks),
                         reranked_graph_supporting_top_k=_summ(result.supporting_chunks))
        if True:  # the vector channel runs on every in-scope route
            q, v, k = d.vector_subquery or req.question, self.corpus.vectors, req.vector_top_k
            filters = d.metadata_filters or None
            ft.query = q
            ft.dense_top_k = _summ(v.search_dense(q, top_k=k, filters=filters))
            ft.bm25_top_k = _summ(v.search_lexical_bm25(q, top_k=k, filters=filters))
            ft.rrf_top_k = _summ(v.search_hybrid_rrf(q, top_k=k, filters=filters))
            pool = v.search_hybrid_rrf(q, top_k=20, filters=filters)
            ft.rrf_rank_by_chunk = {c.chunk_id: i for i, c in enumerate(pool, start=1)}
        if g.context:
            ft.final_deduplicated_llm_context = [
                ChunkSummary(chunk_id=p.chunk.chunk_id, paper_id=p.chunk.paper_id, rank=i, score=round(p.final_score, 4),
                             source_origin=p.source_origin) for i, p in enumerate(g.context.passages, start=1)]
        return ft

    def full_graph(self, include_chunks: bool = False) -> Dict[str, Any]:
        g = self.corpus.graph
        data = g.export_graph(include_chunks=include_chunks)
        nodes = [{"entity_id": n["entity_id"], "canonical_name": n["canonical_name"], "entity_type": n["entity_type"],
                  "category": n.get("category"), "aliases": list(n.get("aliases") or [])[:8],
                  "chunk_count": int(n.get("chunk_count") or 0), "paper_id": n.get("paper_id"),
                  **({"section_path": n.get("section_path")} if n["entity_type"] == "Chunk" else {})}
                 for n in data["nodes"]]
        edges = [{"source": e["source"], "relation_type": e["relation_type"], "target": e["target"],
                  "source_chunk_ids": list(e.get("source_chunk_ids") or []),
                  "evidence_quote": (e.get("evidence_quotes") or [None])[0]} for e in data["edges"]]
        stats = {"papers": g.count_nodes("Paper"), "entities": g.count_nodes("Entity"), "chunks": g.count_nodes("Chunk"),
                 "relationships": g.count_relationships(), "returned_nodes": len(nodes), "returned_edges": len(edges)}
        return {"backend": self.backend, "include_chunks": include_chunks, "nodes": nodes, "edges": edges, "stats": stats}

    def health(self) -> Dict[str, Any]:
        g, v = self.corpus.graph, self.corpus.vectors
        backend = "live-neo4j-pgvector" if isinstance(self.corpus, LiveCorpus) else "offline-in-memory"
        return {"status": "ok", "backend": backend, "generator": self.synthesizer.generator.name,
                "router": getattr(self.engine.router.llm, "name", "custom"),
                "reranker": getattr(getattr(v, "default_reranker", None), "name", "token-interaction"),
                "graph": {"entities": g.count_nodes("Entity"), "chunks": g.count_nodes("Chunk"),
                          "relationships": g.count_relationships()},
                "vector": {"chunks": v.count()}}


STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_research_corpus(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the 35-paper Frontier LLM Systems manifest grouped into 5 UI pillars."""
    from src.config import settings

    path = manifest_path or (settings.corpus_manifest_path.parent / "corpus_35_papers.yaml")
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "data" / "manifest" / "corpus_35_papers.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    papers = [PaperMeta.model_validate(item) for item in raw.get("papers", [])]
    ids = [p.paper_id for p in papers]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate paper_id(s) in research corpus: {sorted(dupes)}")

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for p in papers:
        group_id = p.pillar_group or p.pillar.value
        by_group.setdefault(group_id, []).append({
            "paper_id": p.paper_id,
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "year": p.year,
            "org": p.org or "",
            "pillar": p.pillar.value,
            "abs_url": p.abs_url,
            "pdf_url": p.pdf_url,
        })

    pillars_out: List[Dict[str, Any]] = []
    seen_groups = set()
    for meta in raw.get("pillars", []):
        gid = meta["id"]
        seen_groups.add(gid)
        items = sorted(by_group.get(gid, []), key=lambda x: (x["year"], x["title"].lower()))
        pillars_out.append({
            "id": gid,
            "name": meta["name"],
            "summary": meta.get("summary", ""),
            "count": len(items),
            "papers": items,
        })
    for gid, items in by_group.items():
        if gid not in seen_groups:
            sorted_items = sorted(items, key=lambda x: (x["year"], x["title"].lower()))
            pillars_out.append({
                "id": gid,
                "name": gid.replace("_", " ").title(),
                "summary": "",
                "count": len(sorted_items),
                "papers": sorted_items,
            })

    years = [p.year for p in papers]
    year_range = f"{min(years)}–{max(years)}" if years else "2017–2026"
    return {
        "title": raw.get("title", "Research Corpus — Frontier LLM Systems & Architecture Engineering (2017–2026)"),
        "subtitle": raw.get("subtitle", ""),
        "total_papers": len(papers),
        "year_range": year_range,
        "pillars": pillars_out,
    }

MODES: Dict[str, str] = {
    "live": "Live: Neo4j + pgvector + Gemini Agents",
    "gemini": "Gemini Agent + In-Memory",
    "offline": "Offline Instant",
}

EXAMPLES: List[Dict[str, str]] = [
    {"id": "norm-placement", "tag": "Pre-LN vs Post-LN", "label": "Post-LN -> Pre-LN -> Pre-RMSNorm",
     "question": "How did the post-norm layer of Attention Is All You Need evolve into the pre-norm layer and "
                 "LLaMA's Pre-RMSNorm, and why did Post-LN need a learning-rate warm-up?"},
    {"id": "critic-bridge", "tag": "Implicit Bridge", "label": "PPO -> Critic -> GRPO -> R1-Zero",
     "question": "Which bottleneck connects PPO and GRPO, and which DeepSeek model was trained with GRPO "
                 "without supervised fine-tuning?"},
    {"id": "mha-gqa-mla", "tag": "3-Hop Lineage", "label": "MHA -> GQA -> MLA + W^{DKV}",
     "question": "Trace the lineage of attention from Multi-Head Attention through GQA to MLA, and what does "
                 "the W^{DKV} projection compress?"},
    {"id": "kv-cache", "tag": "Bottleneck Trace", "label": "KV-cache mitigations",
     "question": "Which techniques mitigate the KV cache bottleneck?"},
    {"id": "aggregation", "tag": "Aggregation", "label": "Components shared across model families",
     "question": "Which architectural components show up in more than one model family in this corpus?"},
    {"id": "refusal", "tag": "Refusal Gate", "label": "Unindexed subject (Gemini 1.5 Pro)",
     "question": "What context length does Gemini 1.5 Pro support, and which attention variant does it use?"},
]


def create_app(service: Optional[RagService] = None) -> FastAPI:
    """`service` (tests, benchmarks) answers requests without `mode`; `mode` selects a lazily built backend."""
    app = FastAPI(title="LLM Architecture KG-RAG", version="0.6.0")
    lock = threading.Lock()
    state: Dict[str, Any] = {"svc": service, "modes": {}, "corpus": None}

    def offline_corpus() -> Any:
        if state["corpus"] is None:  # shared by "offline" and "gemini" modes; all 50 papers unless RAG_OFFLINE_CORPUS=seed
            from src.ingestion.offline_demo import build_offline_corpus

            state["corpus"] = build_offline_corpus(seed_only=os.getenv("RAG_OFFLINE_CORPUS", "all50").lower() == "seed")
        return state["corpus"]

    def build(mode: str) -> RagService:
        if mode == "live":
            return RagService.live()
        if mode == "gemini":
            from src.config import settings
            from src.retrieval.router import GeminiRouterLLM

            settings.reload()
            if not settings.gemini_api_key:
                raise RuntimeError("Gemini mode needs GEMINI_API_KEY in .env")
            return RagService.offline(generator=GeminiGeneratorLLM(), corpus=offline_corpus(),
                                      router_llm=GeminiRouterLLM(allow_heuristic_fallback=False))
        return RagService.offline(corpus=offline_corpus())

    def svc(mode: Optional[str] = None) -> RagService:
        with lock:  # backends are built lazily on first use
            if mode is None:
                if state["svc"] is None:
                    env_mode = os.getenv("RAG_MODE", "offline").lower()
                    if env_mode == "live":
                        state["svc"] = RagService.live()
                    else:
                        gen = GeminiGeneratorLLM() if os.getenv("RAG_GENERATOR", "").lower() == "gemini" else None
                        state["svc"] = RagService.offline(generator=gen, corpus=offline_corpus())
                return state["svc"]
            if mode not in state["modes"]:
                state["modes"][mode] = build(mode)
            return state["modes"][mode]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return svc().health()

    @app.get("/examples")
    def examples() -> List[Dict[str, str]]:
        return EXAMPLES

    @app.get("/corpus")
    def corpus() -> Dict[str, Any]:
        return load_research_corpus()

    @app.get("/ui-config")
    def ui_config() -> Dict[str, Any]:
        from src.config import settings

        return {"modes": [{"id": k, "label": v} for k, v in MODES.items()],
                "default_mode": os.getenv("UI_DEFAULT_MODE", "live").lower(),
                "generation_models": generation_models(),
                "default_generation_model": settings.generation_model}

    @app.get("/graph/full")
    def graph_full(mode: Optional[Literal["offline", "gemini", "live"]] = None, include_chunks: bool = False) -> Dict[str, Any]:
        """The whole knowledge graph (Neo4j in live mode, the in-memory store otherwise) for the explorer view."""
        try:
            service_for_mode = svc(mode)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return service_for_mode.full_graph(include_chunks=include_chunks)

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        try:
            service_for_mode = svc(req.mode)
        except RuntimeError as exc:  # live / gemini dependencies missing: report every unmet one
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            return service_for_mode.query(req, mode=req.mode or "default")
        except InvalidRequestError as exc:  # e.g. generation_model outside the configured model chain
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RouterUnavailableError, AllModelsExhaustedError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
