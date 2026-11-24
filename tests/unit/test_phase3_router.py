"""Unit tests for Phase 3: router, sub-query split, fallbacks, Cypher templates, reranked graph chunks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.ingestion.offline_demo import build_offline_corpus
from src.retrieval.cypher_templates import CYPHER, CypherTemplateCatalog, CypherTemplateId as T, validate_params
from src.retrieval.router import (
    GeminiRouterLLM,
    GraphVectorEngine,
    HeuristicRouterLLM,
    QueryIntent,
    QueryRouter,
    RouteType as R,
    RoutingDecision,
)
from src.stores.neo4j_store import Neo4jGraphStore

MHA, GQA, MLA = ("arch_component::multi_head_attention", "arch_component::grouped_query_attention",
                 "arch_component::multi_head_latent_attention")


@pytest.fixture(scope="module")
def corpus():
    return build_offline_corpus()


@pytest.fixture()
def engine(corpus, tmp_path):
    return GraphVectorEngine(QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "t.jsonl"))


def test_compound_question_is_split_and_both_sides_run(engine) -> None:
    r = engine.retrieve("Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, "
                        "and give the exact W^{DKV} latent compression equation")
    d = r.decision
    assert d.route == R.HYBRID_PARALLEL and d.cypher_template_id == T.T1_COMPONENT_LINEAGE
    assert d.graph_subquery.endswith("Multi-head Latent Attention") and d.vector_subquery.startswith("give the exact W^{DKV}")
    assert d.template_params == {"root_entity_id": MHA}
    assert any(len(p) == 2 and p[-1]["source"] == MLA for p in r.template_result.paths)
    assert r.vector_chunks[0].chunk_id == "deepseek_2024_v2::s02::c00"


@pytest.mark.parametrize("q, route, tid", [
    ("Explain the DPO loss equation.", R.VECTOR_ONLY, None),
    ("Compare the architecture of DeepSeek-V2 and DeepSeek-V3", R.GRAPH_ONLY, T.T2_MODEL_ARCHITECTURE_DIFF),
    ("Which techniques mitigate the KV cache bottleneck?", R.GRAPH_ONLY, T.T3_BOTTLENECK_SOLVERS),
    ("What links Llama 3 to Multi-Head Attention?", R.GRAPH_ONLY, T.T4_BRIDGE_PATH_2_TO_3_HOP),
    ("Which components are most widely adopted across the corpus since 2024?", R.GRAPH_ONLY, T.T5_CORPUS_AGGREGATION),
    ("Write me a recipe for chocolate cake.", R.OUT_OF_SCOPE, None),
])
def test_routes_and_templates(engine, q, route, tid) -> None:
    d = engine.router.route(q)
    assert (d.route, d.cypher_template_id) == (route, tid)
    if tid == T.T5_CORPUS_AGGREGATION:
        assert d.template_params["min_year"] == 2024


def test_low_confidence_upgrade_and_vector_seeded_t6(engine) -> None:
    d = engine.router.route("What's the deal with GQA?")
    assert d.fallback_triggered and d.original_route == R.VECTOR_ONLY and d.route == R.HYBRID_PARALLEL
    assert d.cypher_template_id == T.T6_DYNAMIC_SUBGRAPH_FALLBACK and d.template_params["seed_entity_ids"] == [GQA]
    d2 = engine.router.route("What is connected to the technique that pages memory like an operating system?")
    assert d2.resolved_entity_ids == [] and d2.vector_seeded
    assert "arch_component::paged_attention" in d2.template_params["seed_entity_ids"]


def test_telemetry_jsonl_round_trips(engine) -> None:
    engine.router.route("Explain the DPO loss equation.")
    engine.router.route("Trace the lineage of Grouped-Query Attention")
    lines = engine.router.telemetry_path.read_text(encoding="utf-8").splitlines()
    assert [RoutingDecision.model_validate_json(l).route for l in lines] == [R.VECTOR_ONLY, R.GRAPH_ONLY]


def test_supporting_chunks_are_fetched_and_reranked(engine) -> None:
    r = engine.retrieve("Which techniques mitigate the KV cache bottleneck?", graph_top_k=2)
    assert len(r.template_result.source_chunk_ids) > 2 and len(r.supporting_chunks) == 2
    assert r.supporting_chunks[0].score >= r.supporting_chunks[1].score
    assert r.supporting_chunks[0].chunk_id == "kwon_2023_pagedattention::s02::c00"


def test_t4_three_hop_and_t6_rerank(corpus) -> None:
    cat = CypherTemplateCatalog()
    t4 = cat.run(corpus.graph, T.T4_BRIDGE_PATH_2_TO_3_HOP, {"entity_a_id": "model_family::deepseek_v3", "entity_b_id": MHA})
    assert [[h["relation_type"] for h in p] for p in t4.paths] == [["ADOPTS_COMPONENT", "EXTENDS", "EXTENDS"]]
    t6 = cat.run(corpus.graph, T.T6_DYNAMIC_SUBGRAPH_FALLBACK, {"seed_entity_ids": ["arch_component::paged_attention"]},
                 query="KV cache memory bottleneck")
    assert t6.hop_scores == sorted(t6.hop_scores, reverse=True) and t6.paths[0][0]["relation_type"] == "MITIGATES"


class _Tx:
    def __init__(self, log): self.log = log
    def run(self, q, p=None): self.log.append((q, p)); return []


class _Session:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return None
    def execute_read(self, fn): return fn(_Tx(self.log))


class _Driver:
    def __init__(self): self.log = []
    def session(self, database=None): return _Session(self.log)


def test_cypher_injection_is_bound_as_params() -> None:
    evil = "RoPE' OR 1=1 DETACH DELETE n //"
    driver = _Driver()
    CypherTemplateCatalog().run(Neo4jGraphStore(driver=driver), T.T1_COMPONENT_LINEAGE, {"root_entity_id": evil})
    q, p = driver.log[0]
    assert evil not in q and p["root_entity_id"] == evil and "$root_entity_id" in q
    assert all("DELETE" not in c and "CREATE" not in c and "MERGE" not in c for c in CYPHER.values())
    with pytest.raises(ValueError):
        validate_params(T.T5_CORPUS_AGGREGATION, {"relation_type": "DROP"})
    with pytest.raises(ValueError):
        validate_params(T.T6_DYNAMIC_SUBGRAPH_FALLBACK, {"seed_entity_ids": evil})


def test_gemini_router_structured_output_and_fallback() -> None:
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(output_text=json.dumps({
            "query_intent": "BOTTLENECK_TRACE", "route": "GRAPH_ONLY", "confidence": 0.93,
            "standalone_query": "q", "extracted_entities": ["KV cache"]}))

    llm = GeminiRouterLLM(client=SimpleNamespace(interactions=SimpleNamespace(create=create)), model="gemini-3.8-flash")
    out = llm.classify("Which techniques mitigate the KV cache bottleneck?")
    assert out.query_intent == QueryIntent.BOTTLENECK_TRACE and calls[0]["store"] is False
    assert calls[0]["response_format"]["mime_type"] == "application/json"

    def broken(**kw):
        raise RuntimeError("quota")

    bad = GeminiRouterLLM(client=SimpleNamespace(interactions=SimpleNamespace(create=broken)))
    fb = bad.classify("Trace the lineage of grouped query attention")
    assert fb.route == R.GRAPH_ONLY and fb.confidence <= 0.7


def test_query_router_injects_entity_detector_into_gemini_fallback(corpus, tmp_path) -> None:
    def broken(**kw):
        raise RuntimeError("quota")

    llm = GeminiRouterLLM(client=SimpleNamespace(interactions=SimpleNamespace(create=broken)))
    router = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, llm=llm, telemetry_path=tmp_path / "t.jsonl")
    d = router.route("Trace the lineage of GQA")  # 'GQA' alone is only recognizable via the resolver
    assert d.route == R.HYBRID_PARALLEL and d.fallback_triggered and d.resolved_entity_ids == [GQA]


def test_heuristic_router_is_fast() -> None:
    import time

    llm = HeuristicRouterLLM()
    t0 = time.perf_counter()
    for _ in range(200):
        llm.classify("Trace how MHA evolved through GQA to MLA, and give the exact W^{DKV} equation")
    assert (time.perf_counter() - t0) / 200 < 0.05
