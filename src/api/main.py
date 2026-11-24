"""FastAPI service (Phase 4): POST /query (grounded answer + graph_trace + funnel_trace), GET /health.

Run:  .\\.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8000

V1 serves the offline corpus (12 seed papers, in-memory graph + vector stores, offline generator)
so it runs without Docker or an API key. Set RAG_GENERATOR=gemini to use GeminiGeneratorLLM.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.generation.synthesizer import (
    GeminiGeneratorLLM,
    GeneratorLLM,
    GroundedAnswer,
    GroundedAnswerSynthesizer,
    OfflineGroundedGeneratorLLM,
)
from src.retrieval.router import GraphVectorEngine, QueryRouter, RetrievalResult, RouteType
from src.stores.pgvector_store import RetrievedChunk


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    graph_top_k: int = Field(default=3, ge=1, le=20)
    vector_top_k: int = Field(default=3, ge=1, le=20)


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
    score: Optional[float] = None
    source_chunk_ids: List[str] = Field(default_factory=list)


class GraphTrace(BaseModel):
    seed_node_ids: List[str] = Field(default_factory=list)
    selected_path_edges: List[TraceEdge] = Field(default_factory=list)
    explored_rejected_edges: List[TraceEdge] = Field(default_factory=list)


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
    graph_trace: GraphTrace = Field(default_factory=GraphTrace)
    funnel_trace: FunnelTrace = Field(default_factory=FunnelTrace)


def _summ(chunks: List[RetrievedChunk]) -> List[ChunkSummary]:
    return [ChunkSummary(chunk_id=c.chunk_id, paper_id=c.paper_id, rank=c.rank or i, score=round(float(c.score), 4))
            for i, c in enumerate(chunks, start=1)]


class RagService:
    """Router -> retrieval -> grounded synthesis -> UI traces. Shared by the API, CLI and benchmark."""

    def __init__(self, corpus: Any, engine: GraphVectorEngine, synthesizer: GroundedAnswerSynthesizer) -> None:
        self.corpus, self.engine, self.synthesizer = corpus, engine, synthesizer

    @classmethod
    def offline(cls, generator: Optional[GeneratorLLM] = None, telemetry_path=None) -> "RagService":
        from src.ingestion.offline_demo import build_offline_corpus

        c = build_offline_corpus()
        engine = GraphVectorEngine(QueryRouter(c.resolver, c.graph, c.vectors, telemetry_path=telemetry_path))
        return cls(c, engine, GroundedAnswerSynthesizer(generator or OfflineGroundedGeneratorLLM(), c.vectors))

    def answer(self, req: QueryRequest) -> tuple:
        t0 = time.perf_counter()
        result = self.engine.retrieve(req.question, graph_top_k=req.graph_top_k, vector_top_k=req.vector_top_k)
        grounded = self.synthesizer.synthesize(req.question, result)
        return result, grounded, (time.perf_counter() - t0) * 1000

    def query(self, req: QueryRequest) -> QueryResponse:
        result, g, ms = self.answer(req)
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
            graph_trace=self.graph_trace(result), funnel_trace=self.funnel_trace(req, result, g),
        )

    def graph_trace(self, result: RetrievalResult) -> GraphTrace:
        d, tr = result.decision, result.template_result
        if d.route == RouteType.OUT_OF_SCOPE or tr is None:
            return GraphTrace(seed_node_ids=list(d.resolved_entity_ids))
        seeds = list(d.resolved_entity_ids) or list(d.template_params.get("seed_entity_ids", []))
        selected: Dict[tuple, TraceEdge] = {}
        for pi, path in enumerate(tr.paths[:6]):
            for hi, h in enumerate(path):
                key = (h["source"], h["relation_type"], h["target"])
                score = tr.hop_scores[pi] if tr.hop_scores and pi < len(tr.hop_scores) else None
                if key not in selected or hi < (selected[key].hop_index or 0):
                    selected[key] = TraceEdge(source=key[0], relation_type=key[1], target=key[2], hop_index=hi,
                                              score=score, source_chunk_ids=list(h["source_chunk_ids"]))
        nodes = list(dict.fromkeys(seeds + [n for k in selected for n in (k[0], k[2])]))
        rejected: Dict[tuple, TraceEdge] = {}
        for n in nodes:
            for e in self.corpus.graph.neighbors(n):
                key = (e["source"], e["relation_type"], e["target"])
                if key not in selected and key not in rejected:
                    rejected[key] = TraceEdge(source=key[0], relation_type=key[1], target=key[2],
                                              source_chunk_ids=list(e["source_chunk_ids"]))
        return GraphTrace(seed_node_ids=seeds, selected_path_edges=list(selected.values()),
                          explored_rejected_edges=list(rejected.values()))

    def funnel_trace(self, req: QueryRequest, result: RetrievalResult, g: GroundedAnswer) -> FunnelTrace:
        d = result.decision
        if d.route == RouteType.OUT_OF_SCOPE:
            return FunnelTrace()
        ft = FunnelTrace(reranked_vector_top_k=_summ(result.vector_chunks),
                         reranked_graph_supporting_top_k=_summ(result.supporting_chunks))
        if d.route in (RouteType.VECTOR_ONLY, RouteType.HYBRID_PARALLEL):
            q, v, k = d.vector_subquery or req.question, self.corpus.vectors, req.vector_top_k
            filters = d.metadata_filters or None
            ft.query = q
            ft.dense_top_k = _summ(v.search_dense(q, top_k=k, filters=filters))
            ft.bm25_top_k = _summ(v.search_lexical_bm25(q, top_k=k, filters=filters))
            ft.rrf_top_k = _summ(v.search_hybrid_rrf(q, top_k=k, filters=filters))
        if g.context:
            ft.final_deduplicated_llm_context = [
                ChunkSummary(chunk_id=p.chunk.chunk_id, paper_id=p.chunk.paper_id, rank=i, score=round(p.final_score, 4),
                             source_origin=p.source_origin) for i, p in enumerate(g.context.passages, start=1)]
        return ft

    def health(self) -> Dict[str, Any]:
        g, v = self.corpus.graph, self.corpus.vectors
        return {"status": "ok", "backend": "offline-in-memory", "generator": self.synthesizer.generator.name,
                "router": getattr(self.engine.router.llm, "name", "custom"),
                "graph": {"entities": g.count_nodes("Entity"), "chunks": g.count_nodes("Chunk"),
                          "relationships": g.count_relationships()},
                "vector": {"chunks": v.count()}}


def create_app(service: Optional[RagService] = None) -> FastAPI:
    app = FastAPI(title="LLM Architecture KG-RAG", version="0.4.0")
    lock = threading.Lock()
    state: Dict[str, Optional[RagService]] = {"svc": service}

    def svc() -> RagService:
        with lock:  # build the offline corpus lazily on first request (~0.2 s)
            if state["svc"] is None:
                gen = GeminiGeneratorLLM() if os.getenv("RAG_GENERATOR", "").lower() == "gemini" else None
                state["svc"] = RagService.offline(generator=gen)
            return state["svc"]

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return svc().health()

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        return svc().query(req)

    return app


app = create_app()
