"""Unit tests for Phase 3: router, sub-query split, fallbacks, Cypher templates, reranked graph chunks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.ingestion.offline_demo import build_offline_corpus
from src.retrieval.cypher_templates import CYPHER, CypherTemplateCatalog, CypherTemplateId as T, validate_params
from src.retrieval.evidence_gap import EvidenceGapRefiner
from src.retrieval.ppr import PersonalizedPageRankRetriever
from src.retrieval.router import (
    GeminiRouterLLM,
    GraphVectorEngine,
    HeuristicRouterLLM,
    QueryIntent,
    QueryRouter,
    RetrievalResult,
    RouteType as R,
    RoutingDecision,
)
from src.stores.neo4j_store import InMemoryGraphStore, Neo4jGraphStore
from src.stores.pgvector_store import RetrievedChunk

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


# ---------------------------------------------------------------------------
# HippoRAG PPR + Bounded 2-Turn Evidence-Gap Refinement + Ablations
# ---------------------------------------------------------------------------


def test_ppr_ranks_multihop_chain_and_excludes_plumbing_edges() -> None:
    """PPR propagates across multi-hop semantic chains, down-weights INTRODUCES, and excludes IN_PAPER/MENTIONED_IN."""
    export_calls = 0

    class FakeGraph:
        def __init__(self) -> None:
            self.nodes = [
                {"entity_id": "paper::p1", "canonical_name": "Paper 1", "entity_type": "Paper"},
                {"entity_id": "model_family::m1", "canonical_name": "Model 1", "entity_type": "ModelFamily"},
                {"entity_id": "arch_component::a1", "canonical_name": "Comp A1", "entity_type": "ArchComponent"},
                {"entity_id": "arch_component::a2", "canonical_name": "Comp A2", "entity_type": "ArchComponent"},
                {"entity_id": "bottleneck::b1", "canonical_name": "KV Bottleneck", "entity_type": "Bottleneck"},
                {"entity_id": "model_family::distractor", "canonical_name": "Distractor", "entity_type": "ModelFamily"},
            ]
            self.edges = [
                {"source": "paper::p1", "relation_type": "INTRODUCES", "target": "model_family::m1",
                 "source_chunk_ids": ["c0"], "evidence_quotes": ["Paper 1 introduces Model 1"]},
                {"source": "paper::p1", "relation_type": "INTRODUCES", "target": "model_family::distractor",
                 "source_chunk_ids": ["c9"], "evidence_quotes": ["Paper 1 introduces Distractor"]},
                {"source": "model_family::m1", "relation_type": "ADOPTS_COMPONENT", "target": "arch_component::a1",
                 "source_chunk_ids": ["c1"], "evidence_quotes": ["Model 1 adopts Comp A1"]},
                {"source": "arch_component::a1", "relation_type": "EXTENDS", "target": "arch_component::a2",
                 "source_chunk_ids": ["c2"], "evidence_quotes": ["Comp A1 extends Comp A2"]},
                {"source": "arch_component::a2", "relation_type": "MITIGATES", "target": "bottleneck::b1",
                 "source_chunk_ids": ["c3"], "evidence_quotes": ["Comp A2 mitigates KV Bottleneck"]},
                # Plumbing edges (defensively skipped even if a backend returns them).
                {"source": "c1", "relation_type": "IN_PAPER", "target": "paper::p1",
                 "source_chunk_ids": [], "evidence_quotes": []},
                {"source": "model_family::m1", "relation_type": "MENTIONED_IN", "target": "c1",
                 "source_chunk_ids": [], "evidence_quotes": []},
            ]

        def count_nodes(self, label=None) -> int:
            return len(self.nodes)

        def count_relationships(self) -> int:
            return len(self.edges)

        def export_graph(self, include_chunks: bool = False):
            nonlocal export_calls
            export_calls += 1
            return {"nodes": list(self.nodes), "edges": list(self.edges)}

    g = FakeGraph()
    ppr = PersonalizedPageRankRetriever()
    digraph = ppr.build_graph(g)
    assert export_calls == 1 and ppr.cache_builds == 1
    assert digraph["paper::p1"]["model_family::m1"]["weight"] == pytest.approx(0.35)
    assert digraph["model_family::m1"]["arch_component::a1"]["weight"] == pytest.approx(1.0)
    assert digraph["arch_component::a1"]["model_family::m1"]["weight"] == pytest.approx(0.75)
    assert "c1" not in digraph

    # Cache hit on repeated calls when node/edge counts are unchanged.
    scores = ppr.score_nodes(g, endpoint_a=["model_family::m1", "paper::p1"], endpoint_b=["bottleneck::b1"])
    assert export_calls == 1 and ppr.cache_builds == 1
    assert scores["arch_component::a1"] > scores["model_family::distractor"]
    assert scores["arch_component::a2"] > scores["model_family::distractor"]
    # Paper seeds are excluded whenever concept seeds exist.
    assert scores["model_family::m1"] > scores["paper::p1"]

    paths = ppr.retrieve_paths(g, endpoint_a=["model_family::m1"], endpoint_b=["bottleneck::b1"], top_k=6, max_hops=2)
    assert paths
    rels = {h["relation_type"] for p in paths for h in p}
    assert not (rels & {"IN_PAPER", "MENTIONED_IN"})
    assert any(len(p) == 2 for p in paths)
    assert all("ppr_score" in h and "source_chunk_ids" in h and "evidence_quotes" in h for p in paths for h in p)

    # Mutating counts invalidates the cache.
    g.nodes.append({"entity_id": "arch_component::a3", "canonical_name": "Comp A3", "entity_type": "ArchComponent"})
    g.edges.append({"source": "arch_component::a2", "relation_type": "REQUIRES", "target": "arch_component::a3",
                    "source_chunk_ids": ["c4"], "evidence_quotes": ["Comp A2 requires Comp A3"]})
    ppr.build_graph(g)
    assert export_calls == 2 and ppr.cache_builds == 2


def test_ppr_on_inmemory_graph_store_excludes_plumbing(corpus) -> None:
    ppr = PersonalizedPageRankRetriever()
    paths = ppr.retrieve_paths(
        corpus.graph,
        endpoint_a=["model_family::deepseek_v3"],
        endpoint_b=["bottleneck::kv_cache_memory_bandwidth"],
        top_k=5,
        max_hops=2,
    )
    assert paths
    rels = {h["relation_type"] for p in paths for h in p}
    assert not (rels & {"IN_PAPER", "MENTIONED_IN"})
    assert all(h["source_chunk_ids"] and h["evidence_quotes"] and h["ppr_score"] > 0 for p in paths for h in p)


def _chunk(cid: str, paper_id: str, text: str, entity_ids=()) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id=paper_id,
        paper_title=paper_id,
        pub_year=2024,
        pillar="architecture",
        section_path="Method",
        raw_text=text,
        context_prefixed_text=f"[Paper: {paper_id} | Section: Method]\n{text}",
        content_hash=cid,
        entity_ids=tuple(entity_ids),
        score=5.0,
        rank=1,
    )


def test_evidence_gap_refiner_triggers_only_on_missing_evidence_and_is_bounded() -> None:
    refiner = EvidenceGapRefiner(max_turns=99)  # hard-bounded at 2
    assert refiner.max_turns == 2

    d_hybrid = RoutingDecision(
        question="Trace MLA lineage, and give the exact W^{DKV} equation",
        standalone_query="Trace MLA lineage, and give the exact W^{DKV} equation",
        route=R.HYBRID_PARALLEL,
        original_route=R.HYBRID_PARALLEL,
        query_intent=QueryIntent.LINEAGE_MULTI_HOP,
        confidence=0.9,
        resolved_entity_ids=[MLA],
    )
    # Missing W^{DKV} + second clause -> triggers Pass 2 (turn 1 -> gap)
    c_only_lineage = _chunk(
        "c1", "deepseek_2024_v2",
        "Multi-head Latent Attention extends Grouped-Query Attention by compressing KV pairs.",
        entity_ids=[MLA, GQA],
    )
    gap = refiner.assess(
        "Trace MLA lineage, and what quality trade-off does GQA strike?",
        d_hybrid,
        [c_only_lineage],
        current_turn=1,
    )
    assert gap.has_gap and gap.gap_query

    # Hard-bounded at max_turns=2: current_turn=2 must never trigger a 3rd turn
    assert refiner.assess("Trace MLA lineage, and what quality trade-off does GQA strike?", d_hybrid, [c_only_lineage], current_turn=2).has_gap is False

    # Missing W^{DKV} symbolic equation -> triggers
    gap_sym = refiner.assess(
        "Trace MLA lineage, and give the exact W^{DKV} latent compression equation",
        d_hybrid,
        [c_only_lineage],
        current_turn=1,
    )
    assert gap_sym.has_gap and "W^{DKV}" in (gap_sym.gap_query or "")

    # Already covered in Pass 1 -> stays at 1 turn
    c_complete = _chunk(
        "c2", "deepseek_2024_v2",
        "Multi-head Latent Attention (MLA) extends GQA via latent compression $c_t^{KV} = W^{DKV} h_t$.",
        entity_ids=[MLA, GQA],
    )
    no_gap = refiner.assess(
        "Trace MLA lineage, and give the exact W^{DKV} latent compression equation",
        d_hybrid,
        [c_complete],
        current_turn=1,
    )
    assert not no_gap.has_gap

    # OUT_OF_SCOPE is always skipped
    d_oos = RoutingDecision(
        question="What is the weather in Paris?",
        standalone_query="What is the weather in Paris?",
        route=R.OUT_OF_SCOPE,
        original_route=R.OUT_OF_SCOPE,
        query_intent=QueryIntent.UNSUPPORTED_DOMAIN,
        confidence=0.95,
    )
    assert not refiner.assess(d_oos.question, d_oos, [], current_turn=1).has_gap


def test_engine_ablation_toggles_and_two_turn_bound(corpus, tmp_path) -> None:
    q_covered = "Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, and give the exact W^{DKV} latent compression equation"
    q_gap = "The rotary extension method that needs 10x fewer tokens improves on which interpolation approach, which model family did it stretch to 128k tokens, and does it alter the architecture?"

    # Full engine: covered compound query stays at 1 turn; missing-clause query uses 2 turns and retrieves s04
    full_eng = GraphVectorEngine(QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "full.jsonl"))
    r_cov = full_eng.retrieve(q_covered)
    assert r_cov.turns_used == 1 and not r_cov.evidence_gap_triggered and r_cov.gap_query is None

    r_gap = full_eng.retrieve(q_gap)
    assert r_gap.evidence_gap_triggered and r_gap.turns_used == 2 and r_gap.gap_query
    assert "peng_2023_yarn::s04::c00" in r_gap.context_chunk_ids

    # enable_ppr=False ablation
    no_ppr = GraphVectorEngine(
        QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "noppr.jsonl"),
        enable_ppr=False,
    )
    r_np = no_ppr.retrieve("Which bottleneck does the component adopted by DeepSeek-V3 mitigate?")
    assert r_np.ppr_paths_added == 0

    # enable_evidence_gap=False ablation
    no_gap = GraphVectorEngine(
        QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "nogap.jsonl"),
        enable_evidence_gap=False,
    )
    r_ng = no_gap.retrieve(q_gap)
    assert not r_ng.evidence_gap_triggered and r_ng.turns_used == 1 and r_ng.gap_query is None

    # RetrievalResult defaults remain backward-compatible
    res = RetrievalResult(decision=r_cov.decision)
    assert (res.evidence_gap_triggered, res.gap_query, res.turns_used, res.ppr_paths_added) == (False, None, 1, 0)
