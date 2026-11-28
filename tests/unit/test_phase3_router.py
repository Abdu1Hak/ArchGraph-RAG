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


def test_entity_resolver_before_router_and_llm_template_assignment(corpus, tmp_path) -> None:
    from src.retrieval.router import RouterLLMOutput

    seen_candidates = []

    class SmartRouterLLM:
        name = "mock-smart-router"

        def classify(self, question: str, resolved_entities=()):
            seen_candidates.append(list(resolved_entities))
            return RouterLLMOutput(
                query_intent=QueryIntent.LINEAGE_MULTI_HOP,
                route=R.GRAPH_ONLY,
                confidence=0.95,
                standalone_query=question,
                graph_subquery=question,
                extracted_entities=[],
                cypher_template_id=T.T4_BRIDGE_PATH_2_TO_3_HOP,
                template_params={"entity_a_id": MHA, "entity_b_id": MLA},
            )

    r_smart = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, llm=SmartRouterLLM(), telemetry_path=tmp_path / "smart.jsonl")
    d_smart = r_smart.route("How do Multi-Head Attention and Multi-head Latent Attention connect?")
    assert seen_candidates and {e["entity_id"] for e in seen_candidates[0]} >= {MHA, MLA}
    assert all({"entity_id", "canonical_name", "entity_type"} <= set(e) for e in seen_candidates[0])
    assert d_smart.cypher_template_id == T.T4_BRIDGE_PATH_2_TO_3_HOP
    assert d_smart.template_params == {"entity_a_id": MHA, "entity_b_id": MLA}
    assert d_smart.template_assigned_by == "llm"
    assert d_smart.routing_trace["routing"]["template_assigned_by"] == "llm"

    # Invalid LLM-assigned entity_id must fall back cleanly to _select_template
    class BadEntityRouterLLM:
        name = "mock-bad-entity"

        def classify(self, question: str, resolved_entities=()):
            return RouterLLMOutput(
                query_intent=QueryIntent.LINEAGE_MULTI_HOP,
                route=R.GRAPH_ONLY,
                confidence=0.95,
                standalone_query=question,
                graph_subquery=question,
                cypher_template_id=T.T1_COMPONENT_LINEAGE,
                template_params={"root_entity_id": "arch_component::does_not_exist"},
            )

    r_bad = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, llm=BadEntityRouterLLM(), telemetry_path=tmp_path / "bad.jsonl")
    d_bad = r_bad.route("Trace the lineage of Multi-Head Attention")
    assert d_bad.template_assigned_by == "heuristic_fallback"
    assert d_bad.cypher_template_id == T.T1_COMPONENT_LINEAGE
    assert d_bad.template_params == {"root_entity_id": MHA}

    # Legacy 1-arg classify(self, question) mocks remain supported
    class LegacyRouterLLM:
        name = "legacy-1arg"

        def classify(self, question: str):
            return RouterLLMOutput(
                query_intent=QueryIntent.SINGLE_FACT_DEFINITION,
                route=R.VECTOR_ONLY,
                confidence=0.9,
                standalone_query=question,
                vector_subquery=question,
            )

    r_leg = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, llm=LegacyRouterLLM(), telemetry_path=tmp_path / "leg.jsonl")
    d_leg = r_leg.route("Explain the DPO loss equation.")
    assert d_leg.route == R.VECTOR_ONLY and d_leg.template_assigned_by == "none"

    # GeminiRouterLLM includes RESOLVED_ENTITIES in prompt input when provided
    gem_calls = []

    def create(**kw):
        gem_calls.append(kw)
        return SimpleNamespace(output_text=json.dumps({
            "query_intent": "LINEAGE_MULTI_HOP", "route": "GRAPH_ONLY", "confidence": 0.94,
            "standalone_query": "q", "cypher_template_id": "T1_COMPONENT_LINEAGE",
            "template_params": {"root_entity_id": MHA},
        }))

    gem = GeminiRouterLLM(client=SimpleNamespace(interactions=SimpleNamespace(create=create)))
    out = gem.classify("Trace MHA", resolved_entities=[{"entity_id": MHA, "canonical_name": "Multi-Head Attention", "entity_type": "ArchComponent"}])
    assert "RESOLVED_ENTITIES:" in gem_calls[0]["input"] and MHA in gem_calls[0]["input"]
    assert out.cypher_template_id == T.T1_COMPONENT_LINEAGE and out.template_params == {"root_entity_id": MHA}


def test_paper_title_expansion_gating_and_multitype_disambiguation(corpus, tmp_path) -> None:
    r = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "gate.jsonl")

    # Concrete concept already resolved -> link paper node ONLY, do not dump extra INTRODUCES concepts
    d_concrete = r.route("How does Grouped-Query Attention extend Attention Is All You Need?")
    assert "paper::vaswani_2017_transformer" in d_concrete.resolved_entity_ids
    assert GQA in d_concrete.resolved_entity_ids
    assert MHA not in d_concrete.resolved_entity_ids
    assert "model_family::transformer" not in d_concrete.resolved_entity_ids
    assert d_concrete.routing_trace["entity_resolution"]["paper_expansions"] == 0

    # Paper-focused query with 0 concrete non-paper hits -> expand INTRODUCES concepts
    d_paper_only = r.route("What was proposed in Attention Is All You Need?")
    assert "paper::vaswani_2017_transformer" in d_paper_only.resolved_entity_ids
    assert {"model_family::transformer", MHA} <= set(d_paper_only.resolved_entity_ids)
    assert d_paper_only.routing_trace["entity_resolution"]["paper_expansions"] >= 2

    # Duplicate canonical names across entity types carry distinct entity_type in resolved_entities & entity_links
    from src.ingestion.entity_resolver import EntityResolver
    from src.schema.ontology import EntityType, ResolvedEntityNode

    res_dup = EntityResolver(seed_aliases_path=None)
    g_dup = InMemoryGraphStore()
    n1 = ResolvedEntityNode(entity_id="arch_component::transformer", entity_type=EntityType.ARCH_COMPONENT, canonical_name="Transformer")
    n2 = ResolvedEntityNode(entity_id="model_family::transformer", entity_type=EntityType.MODEL_FAMILY, canonical_name="Transformer")
    res_dup._register_node(n1)
    res_dup._register_node(n2)
    g_dup.upsert_resolved_nodes([n1, n2])
    r_dup = QueryRouter(res_dup, g_dup, corpus.vectors, telemetry_path=tmp_path / "dup.jsonl")
    d_dup = r_dup.route("Compare Transformer components")
    by_id = {e["entity_id"]: e for e in d_dup.resolved_entities}
    assert by_id["arch_component::transformer"]["entity_type"] == "ArchComponent"
    assert by_id["model_family::transformer"]["entity_type"] == "ModelFamily"


def test_diversity_aware_top3_path_selection(corpus, tmp_path) -> None:
    from src.retrieval.cypher_templates import TemplateResult

    def hop(src, src_name, rel, tgt, tgt_name, cid, quote):
        return {
            "source": src, "source_name": src_name, "relation_type": rel,
            "target": tgt, "target_name": tgt_name,
            "source_chunk_ids": [cid], "evidence_quotes": [quote],
        }

    # Paths 0..2 repeat the same MHA/GQA edge & chunk; path 3 covers MLA + KV cache clause
    paths = [
        [hop(MHA, "Multi-Head Attention", "EXTENDS", GQA, "Grouped-Query Attention", "c_dup", "Multi-Head Attention grouped query attention")],
        [hop(MHA, "Multi-Head Attention", "EXTENDS", GQA, "Grouped-Query Attention", "c_dup", "Multi-Head Attention grouped query attention"),
         hop(GQA, "Grouped-Query Attention", "REQUIRES", "training_method::uptraining", "Uptraining", "c_dup", "uptraining")],
        [hop(MHA, "Multi-Head Attention", "EXTENDS", GQA, "Grouped-Query Attention", "c_dup", "Multi-Head Attention grouped query attention"),
         hop(GQA, "Grouped-Query Attention", "ADOPTS_COMPONENT", "model_family::llama_2", "LLaMA-2", "c_dup", "llama")],
        [hop(MLA, "Multi-head Latent Attention", "MITIGATES", "bottleneck::kv_cache_memory_bandwidth", "KV Cache Bottleneck", "c_mla", "Multi-head Latent Attention mitigates KV cache memory")],
    ]
    q = "Compare Multi-Head Attention and Multi-head Latent Attention, and which bottleneck does MLA mitigate?"

    eng_off = GraphVectorEngine(
        QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "off.jsonl"),
        enable_diverse_path_selection=False,
    )
    tr_off = TemplateResult(T.T6_DYNAMIC_SUBGRAPH_FALLBACK, {}, list(paths))
    eng_off._rank_paths(tr_off, q, resolved_entity_ids=[MHA, MLA])

    eng_on = GraphVectorEngine(
        QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "on.jsonl"),
        enable_diverse_path_selection=True,
    )
    tr_on = TemplateResult(T.T6_DYNAMIC_SUBGRAPH_FALLBACK, {}, list(paths))
    ps_trace = eng_on._rank_paths(tr_on, q, resolved_entity_ids=[MHA, MLA])

    assert any(h["source"] == MLA for p in tr_on.paths[:3] for h in p)
    assert ps_trace["diverse_selection_enabled"] is True
    assert ps_trace["unique_entities_covered"] >= eng_off._path_nodes(tr_off.paths[:3]).__len__()


def test_pipeline_trace_covers_all_six_flow_patterns(corpus, tmp_path) -> None:
    eng = GraphVectorEngine(QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "pt.jsonl"))

    # 1. HYBRID_DIRECT
    r1 = eng.retrieve("Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, and give the exact W^{DKV} latent compression equation")
    pt1 = r1.pipeline_trace
    assert pt1["flow_pattern"]["id"] == 1 and pt1["flow_pattern"]["code"] == "HYBRID_DIRECT"
    assert pt1["entity_resolution"]["latency_ms"] >= 0 and pt1["routing"]["latency_ms"] >= 0
    assert pt1["vector_pass1"]["ran"] and pt1["cypher_template"]["ran"]
    assert pt1["hipporag_ppr"]["alpha"] == pytest.approx(0.85) and pt1["hipporag_ppr"]["top_heatmap_nodes"]
    assert pt1["graph_chunk_fetch"]["ran"] and pt1["graph_chunk_fetch"]["kept_count"] > 0
    assert pt1["floor_short_circuit"]["triggered"] is False

    # 2. HYBRID_TWO_TURN
    r2 = eng.retrieve("The rotary extension method that needs 10x fewer tokens improves on which interpolation approach, which model family did it stretch to 128k tokens, and does it alter the architecture?")
    pt2 = r2.pipeline_trace
    assert pt2["flow_pattern"]["id"] == 2 and pt2["flow_pattern"]["code"] == "HYBRID_TWO_TURN"
    assert pt2["evidence_gap"]["checked"] and pt2["evidence_gap"]["triggered"]
    assert pt2["evidence_gap"]["pass2_vector_new_chunks"] >= 1

    # 3. VECTOR_SEEDED_T6
    r3 = eng.retrieve("What is connected to the technique that pages memory like an operating system?")
    pt3 = r3.pipeline_trace
    assert pt3["flow_pattern"]["id"] == 3 and pt3["flow_pattern"]["code"] == "VECTOR_SEEDED_T6"
    assert pt3["vector_seeded_entry"]["triggered"] and pt3["vector_seeded_entry"]["borrowed_entities"]

    # 4. AGGREGATION_SHORT_CIRCUIT
    r4 = eng.retrieve("Which components are most widely adopted across the corpus since 2024?")
    pt4 = r4.pipeline_trace
    assert pt4["flow_pattern"]["id"] == 4 and pt4["flow_pattern"]["code"] == "AGGREGATION_SHORT_CIRCUIT"
    assert pt4["endpoint_bridging"]["status"] == "short_circuited"
    assert pt4["hipporag_ppr"]["status"] == "short_circuited"
    assert pt4["floor_short_circuit"]["triggered"] is True

    # 5. VECTOR_ONLY_FAST
    r5 = eng.retrieve("Explain the DPO loss equation.")
    pt5 = r5.pipeline_trace
    assert pt5["flow_pattern"]["id"] == 5 and pt5["flow_pattern"]["code"] == "VECTOR_ONLY_FAST"
    assert pt5["cypher_template"]["ran"] is False and pt5["floor_short_circuit"]["triggered"] is True

    # 6. REFUSAL_SHORT_CIRCUIT (both subject_gate and router_llm)
    r6a = eng.retrieve("How many parameters does GPT-4 have?")
    assert r6a.pipeline_trace["flow_pattern"]["id"] == 6 and r6a.pipeline_trace["routing"]["refused_at"] == "subject_gate"
    assert r6a.pipeline_trace["routing"]["unverified_subjects"] == ["GPT 4"]
    assert r6a.pipeline_trace["vector_pass1"]["ran"] is False

    r6b = eng.retrieve("Write me a recipe for chocolate cake.")
    assert r6b.pipeline_trace["flow_pattern"]["id"] == 6 and r6b.pipeline_trace["routing"]["refused_at"] == "router_llm"
