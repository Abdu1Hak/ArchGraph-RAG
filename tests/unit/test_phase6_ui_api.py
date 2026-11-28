"""Phase 6: query-time Tier 1-3 entity linking, the 50-paper fixture corpus, and the web UI / API traces."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import EXAMPLES, InvalidRequestError, QueryRequest, RagService, create_app
from src.generation.synthesizer import GeminiGeneratorLLM
from src.ingestion.arxiv_fetcher import load_manifest
from src.ingestion.offline_demo import build_offline_corpus
from src.ingestion.offline_fixtures import SYNTHETIC_SEED_MARKDOWN
from src.retrieval.router import QueryRouter


@pytest.fixture(scope="module")
def full():
    return build_offline_corpus(seed_only=False)


@pytest.fixture(scope="module")
def full_svc(full, tmp_path_factory):
    return RagService.offline(corpus=full, telemetry_path=tmp_path_factory.mktemp("p6") / "r.jsonl")


def _router(c, tmp_path, **kw):
    return QueryRouter(c.resolver, c.graph, c.vectors, telemetry_path=tmp_path / "t.jsonl", **kw)


# 1. Query-time entity linking ------------------------------------------------


def test_tier1_paper_title_links_paper_and_introduced_concepts(full, tmp_path) -> None:
    r = _router(full, tmp_path)
    links = full.resolver.link_query_entity_detailed("Attention is all you need", introduces_fn=r._introduced_by,
                                                     candidate_filter=full.graph.has_entity)
    ids = [l.entity_id for l in links]
    assert ids[0] == "paper::vaswani_2017_transformer" and links[0].tier == "TIER_1_EXACT"
    assert {"model_family::transformer", "arch_component::multi_head_attention", "arch_component::post_ln"} <= set(ids)
    assert full.resolver.link_query_entity("Attention is all you need", allow_papers=False) == []


def test_tier2_lexical_overlap_and_contrast_guard(full) -> None:
    links = full.resolver.link_query_entity_detailed("pre norm transformer")
    assert [(l.entity_id, l.tier) for l in links] == [("arch_component::pre_ln", "TIER_2_LEXICAL")]
    assert "arch_component::post_ln" not in full.resolver.link_query_entity("pre normalization transformer")
    assert full.resolver.link_query_entity("training technique") == []  # 'training' is not 'Uptraining'


def test_version_guard_blocks_gpt4_to_gpt3(full) -> None:
    assert full.resolver.link_query_entity("GPT-4") == []
    assert full.resolver.link_query_entity("GPT-3") == ["model_family::gpt_3"]


def _fake_embedder(mapping):
    """Phrase/node-text prefix -> one-hot vector; anything else -> a hashed, nearly orthogonal vector."""
    def embed(texts):
        out = []
        for t in texts:
            hit = next((v for k, v in mapping.items() if t.startswith(k)), None)
            if hit is None:
                rng = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                hit = rng.standard_normal(64)
            out.append(np.asarray(hit, dtype=np.float32))
        return out
    return embed


def test_tier3_semantic_link_threshold_and_version_guard(full) -> None:
    e1, e2 = np.eye(64)[0], np.eye(64)[1]
    near = e1 * 0.9 + np.eye(64)[2] * 0.436  # cosine ~0.90 with e1
    weak = e1 * 0.6 + np.eye(64)[2] * 0.8    # cosine 0.60 with e1
    embed = _fake_embedder({"Pre-LN:": e1, "Llama-3:": e2, "normalization moved inside": near,
                            "vague normalization idea": weak, "Llama 4 normalization": e2})
    hit = full.resolver.link_query_entity_detailed("normalization moved inside residual sublayers", embed_fn=embed)
    assert [(l.entity_id, l.tier) for l in hit] == [("arch_component::pre_ln", "TIER_3_SEMANTIC")] and hit[0].score >= 0.78
    assert full.resolver.link_query_entity("vague normalization idea", embed_fn=embed) == []  # below 0.78
    assert full.resolver.link_query_entity("Llama 4 normalization", embed_fn=embed) == []    # v:4 vs v:3


def test_router_links_unmatched_phrases_and_records_tiers(full, tmp_path) -> None:
    d = _router(full, tmp_path).route("Why does pre normalization stabilize gradients?")
    assert "arch_component::pre_ln" in d.resolved_entity_ids
    assert any(l["entity_id"] == "arch_component::pre_ln" and l["tier"] == "TIER_2_LEXICAL" for l in d.entity_links)


def test_mock_embedder_opts_out_of_tier3(full, tmp_path) -> None:
    assert _router(full, tmp_path).semantic_embed_fn is None
    assert _router(full, tmp_path, semantic_embed_fn=lambda t: [[1.0]] * len(t)).semantic_embed_fn is not None


# 2. 50-paper corpus -----------------------------------------------------------


def test_seed_corpus_is_unchanged() -> None:
    c = build_offline_corpus()
    # Same numbers the Phase 5 scorecard header reports for the 12-paper corpus before Phase 6.
    assert (len(c.report.papers), c.graph.count_nodes("Chunk"), c.graph.count_relationships()) == (12, 70, 248)
    assert not any("Post-LN" in t for t in SYNTHETIC_SEED_MARKDOWN.values())


def test_full_corpus_covers_all_50_papers_with_real_sections(full) -> None:
    papers = {p.paper_id for p in load_manifest(seed_only=False)}
    assert set(full.report.papers) == papers and len(papers) == 50
    texts = {c.paper_id for c in full.report.chunks if "synthetic placeholder" not in c.raw_text}
    assert texts == papers
    with_edges = {e.paper_id for e in full.report.edges}
    assert with_edges == papers
    assert not [e for e in full.resolver.nodes if e.startswith("paper::") and e[7:] not in papers]


def test_norm_placement_chain(full) -> None:
    g = full.graph
    for edge in [("model_family::transformer", "ADOPTS_COMPONENT", "arch_component::post_ln"),
                 ("arch_component::pre_ln", "REPLACES_IN", "arch_component::post_ln"),
                 ("arch_component::pre_rmsnorm", "EXTENDS", "arch_component::pre_ln"),
                 ("model_family::llama_1", "ADOPTS_COMPONENT", "arch_component::pre_rmsnorm"),
                 ("arch_component::rmsnorm", "EXTENDS", "arch_component::layernorm")]:
        rel = g.get_relationship(*edge)
        assert rel and rel["evidence_quotes"] and rel["source_chunk_ids"], edge


# 3. API + UI ------------------------------------------------------------------


def test_ui_and_static_assets_are_served(full_svc) -> None:
    client = TestClient(create_app(full_svc))
    page = client.get("/")
    assert page.status_code == 200 and all(s in page.text for s in ("Search Subgraph", "Full 35-Paper Graph", 'id="hide-labels"', "Retrieval Flow", 'id="composer"'))
    for asset in re.findall(r'(?:src|href)="(/static/[^"]+)"', page.text):
        assert client.get(asset).status_code == 200, asset


def test_examples_and_ui_config(full_svc) -> None:
    client = TestClient(create_app(full_svc))
    ex = client.get("/examples").json()
    assert len(ex) == 6 and {"norm-placement", "critic-bridge", "mha-gqa-mla"} <= {e["id"] for e in ex}
    assert all(e.get("tag") for e in ex)
    cfg = client.get("/ui-config").json()
    assert {m["id"] for m in cfg["modes"]} == {"offline", "gemini", "live"} and cfg["generation_models"]
    assert cfg["default_mode"] == "live" and cfg["default_generation_model"] == cfg["generation_models"][0]
    assert cfg["generation_models"] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


def test_graph_full_without_chunks(full_svc) -> None:
    j = TestClient(create_app(full_svc)).get("/graph/full", params={"mode": "offline"}).json()
    assert j["stats"]["papers"] == 50 and j["stats"]["chunks"] > 0 and j["stats"]["relationships"] > len(j["edges"])
    types = {n["entity_type"] for n in j["nodes"]}
    assert "Chunk" not in types and {"Paper", "ModelFamily", "ArchComponent", "TrainingMethod", "InferenceTechnique", "Bottleneck"} <= types
    assert len(j["nodes"]) == j["stats"]["entities"]
    ids = {n["entity_id"] for n in j["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in j["edges"])
    assert not {e["relation_type"] for e in j["edges"]} & {"MENTIONED_IN", "IN_PAPER"}
    pre_ln = next(e for e in j["edges"] if (e["source"], e["relation_type"]) == ("arch_component::pre_ln", "REPLACES_IN"))
    assert pre_ln["evidence_quote"] and pre_ln["source_chunk_ids"]
    node = next(n for n in j["nodes"] if n["entity_id"] == "arch_component::rmsnorm")
    assert set(node) >= {"entity_id", "canonical_name", "entity_type", "category", "aliases", "chunk_count"} and node["chunk_count"] > 0


def test_graph_full_with_chunks(full_svc) -> None:
    j = TestClient(create_app(full_svc)).get("/graph/full", params={"mode": "offline", "include_chunks": True}).json()
    chunks = [n for n in j["nodes"] if n["entity_type"] == "Chunk"]
    assert len(chunks) == j["stats"]["chunks"] and all(c["paper_id"] and c["section_path"] for c in chunks)
    assert chunks[0]["canonical_name"] == chunks[0]["entity_id"]
    rels = {e["relation_type"] for e in j["edges"]}
    assert {"MENTIONED_IN", "IN_PAPER"} <= rels and len(j["edges"]) == j["stats"]["relationships"]
    ids = {n["entity_id"] for n in j["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in j["edges"])


def test_query_reports_latency_breakdown_and_rank_pool(full_svc) -> None:
    j = TestClient(create_app(full_svc)).post("/query", json={"question": EXAMPLES[3]["question"]}).json()
    assert 0 < j["routing_latency_ms"] <= j["retrieval_latency_ms"] <= j["latency_ms"] and j["generation_latency_ms"] >= 0
    assert j["funnel_trace"]["rrf_rank_by_chunk"]


def test_query_graph_trace_nodes_roles_and_evidence(full_svc) -> None:
    j = TestClient(create_app(full_svc)).post("/query", json={"question": EXAMPLES[0]["question"]}).json()
    gt = j["graph_trace"]
    roles = {n["entity_id"]: n["role"] for n in gt["nodes"]}
    assert all(roles[s] == "seed" for s in gt["seed_node_ids"])
    endpoints = {n for e in gt["selected_path_edges"] + gt["explored_rejected_edges"] for n in (e["source"], e["target"])}
    assert endpoints <= set(roles) and "selected_path" in roles.values() and "rejected_neighbor" in roles.values()
    assert all(e["evidence_quote"] for e in gt["selected_path_edges"])
    assert {"arch_component::post_ln", "arch_component::pre_ln", "arch_component::pre_rmsnorm"} <= set(roles)
    assert j["citation_gate"]["status"] == "PASS" and j["mode"] == "default"
    node = next(n for n in gt["nodes"] if n["entity_id"] == "arch_component::pre_ln")
    assert node["entity_type"] == "ArchComponent" and "Pre-Norm" in node["aliases"]
    assert all(c["full_text"] for c in j["citations"])
    assert any(c["pdf_url"] and c["pdf_url"].startswith("https://arxiv.org/pdf/") for c in j["citations"])


def test_mode_and_generation_model_validation(full_svc, monkeypatch) -> None:
    client = TestClient(create_app(full_svc))
    assert client.post("/query", json={"question": "What is GQA?", "mode": "bogus"}).status_code == 422
    assert client.post("/query", json={"question": "What is GQA?", "generation_model": "x; rm -rf"}).status_code == 422
    assert client.post("/query", json={"question": "What is GQA?", "mode": "offline"}).json()["mode"] == "offline"

    def no_live(*a, **k):
        raise RuntimeError("Live mode is not ready:\n  - Neo4j unreachable")
    monkeypatch.setattr(RagService, "live", classmethod(no_live))
    r = client.post("/query", json={"question": "What is GQA?", "mode": "live"})
    assert r.status_code == 503 and "Neo4j unreachable" in r.json()["detail"]


def test_generation_model_override_is_per_request_and_allow_listed(full_svc) -> None:
    gem = RagService(full_svc.corpus, full_svc.engine, type(full_svc.synthesizer)(GeminiGeneratorLLM(client=object()), full_svc.corpus.vectors))
    with pytest.raises(InvalidRequestError):
        gem.synthesizer_for(QueryRequest(question="What is GQA?", generation_model="not-a-configured-model"))
    from src.config import settings
    other = [m for m in settings.text_model_chain(settings.generation_model) if m != gem.synthesizer.generator.model]
    if other:
        s = gem.synthesizer_for(QueryRequest(question="What is GQA?", generation_model=other[0]))
        assert s is not gem.synthesizer and s.generator.model == other[0] and gem.synthesizer.generator.model != other[0]


def test_corpus_35_manifest_is_valid_and_exact() -> None:
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[2] / "data" / "manifest" / "corpus_35_papers.yaml"
    papers = load_manifest(manifest_path=manifest, seed_only=False)
    assert len(papers) == 35
    assert len({p.paper_id for p in papers}) == 35
    assert len({p.arxiv_id for p in papers}) == 35
    groups = {p.pillar_group for p in papers}
    assert groups == {"core_block", "attention_sparsity", "scaling_context", "post_training_rlvr", "inference_kernels"}
    assert all(p.abs_url.startswith("https://arxiv.org/abs/") and p.pdf_url.endswith(".pdf") for p in papers)


def test_corpus_endpoint_and_ui_sidebar(full_svc) -> None:
    client = TestClient(create_app(full_svc))
    resp = client.get("/corpus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Research Corpus" and "—" not in body["title"]
    assert body["total_papers"] == 35
    assert body["year_range"] == "2017–2026"
    assert len(body["pillars"]) == 5
    assert sum(p["count"] for p in body["pillars"]) == 35
    for pillar in body["pillars"]:
        assert pillar["count"] == len(pillar["papers"])
        for paper in pillar["papers"]:
            assert paper["abs_url"].startswith("https://arxiv.org/abs/")
            assert paper["pdf_url"].startswith("https://arxiv.org/pdf/")

    page = client.get("/")
    assert page.status_code == 200
    for marker in ('id="sidebar"', "Research Corpus", "/static/corpus.js"):
        assert marker in page.text, marker
    for asset in re.findall(r'(?:src|href)="(/static/[^"]+)"', page.text):
        assert client.get(asset).status_code == 200, asset


def test_tier3_live_node_embed_guard_and_sqlite_cache_load(tmp_path) -> None:
    import numpy as np
    from src.ingestion.entity_resolver import EntityResolver
    from src.schema.ontology import EntityType, ResolvedEntityNode
    from src.stores.embedders import GeminiEmbedder

    resolver = EntityResolver(seed_aliases_path=None)
    pre_ln = ResolvedEntityNode(
        entity_id="arch_component::pre_ln",
        entity_type=EntityType.ARCH_COMPONENT,
        canonical_name="Pre-LN",
        aliases=["Pre-Norm"],
        description="Layer normalization placed inside the residual block before attention.",
    )
    resolver._register_node(pre_ln)
    resolver.allow_live_node_embed = False

    calls = []

    def tracking_embed(texts):
        calls.append(list(texts))
        return [np.ones(8, dtype=np.float32) for _ in texts]

    # When allow_live_node_embed=False and cache is empty, embed_fn is NEVER called even for the phrase
    assert resolver.link_query_entity_detailed("normalization moved inside residual", embed_fn=tracking_embed) == []
    assert calls == []

    # Pre-warm via GeminiEmbedder SQLite cache, then load with 0 API calls into a fresh resolver
    node_text = resolver.query_node_text(pre_ln)
    e1 = np.eye(8, dtype=np.float32)[0]
    e_phrase = e1 * 0.9 + np.eye(8, dtype=np.float32)[1] * 0.436

    class FakeEmbedClient:
        class models:
            @staticmethod
            def embed_content(model, contents, config):
                from types import SimpleNamespace

                return SimpleNamespace(
                    embeddings=[SimpleNamespace(values=e1.tolist()) for _ in contents],
                    usage_metadata=SimpleNamespace(prompt_token_count=10),
                )

    db_path = tmp_path / "embed_cache.sqlite"
    emb = GeminiEmbedder(client=FakeEmbedClient(), dim=8, cache_path=db_path)
    emb.embed_documents([node_text])
    assert emb.api_calls == 1

    # New GeminiEmbedder reading same SQLite file with no live client calls
    emb_cached = GeminiEmbedder(client=object(), dim=8, cache_path=db_path)
    resolver2 = EntityResolver(seed_aliases_path=None)
    resolver2._register_node(pre_ln)
    resolver2.allow_live_node_embed = False
    loaded = resolver2.load_cached_query_node_embeddings(emb_cached)
    assert loaded == 1
    assert emb_cached.api_calls == 0

    hits = resolver2.link_query_entity_detailed(
        "normalization moved inside residual",
        embed_fn=lambda texts: [e_phrase for _ in texts],
    )
    assert hits and hits[0].entity_id == "arch_component::pre_ln"
    assert hits[0].tier == "TIER_3_SEMANTIC"


def test_clean_section_and_passage_and_no_revisit_frames(full_svc) -> None:
    from src.generation.synthesizer import clean_passage_for_prompt, clean_section_path

    sec = "**_m_ HC: Manifold-Constrained Hyper-Connections** > **2.3 Relative position embedding* > **"
    assert clean_section_path(sec) == "m HC: Manifold-Constrained Hyper-Connections > 2.3 Relative position embedding"
    assert clean_section_path("core_block > snake_case_name") == "core_block > snake_case_name"

    raw = (
        "In addition to attention sub-layers.\n\n"
        "<!-- Start of picture text -->\nFFN( x ) = max(0 , xW 1 + b 1) W 2 + b 2 (2)<br><!-- End of picture text -->\n\n"
        "The dimensionality is _d_ model = 512."
    )
    cleaned = clean_passage_for_prompt(raw)
    assert "<!--" not in cleaned and "<br>" not in cleaned
    assert "FFN(" in cleaned and ("xW_1 + b_1" in cleaned or "xW 1 + b 1" in cleaned)

    client = TestClient(create_app(full_svc))
    page = client.get("/")
    assert page.status_code == 200 and "?v=9" in page.text
    core_js = client.get("/static/core.js").text
    assert "Showing in frames below" not in core_js and "revisit-btn" not in core_js


def test_query_response_pipeline_trace_and_refusal_rungs(full_svc, tmp_path) -> None:
    from src.api.main import load_research_corpus

    client = TestClient(create_app(full_svc))

    # Hybrid query: full pipeline_trace + resolved_entities + 6-rung citation_gate
    j0 = client.post("/query", json={"question": EXAMPLES[0]["question"]}).json()
    assert j0["resolved_entities"] and all(
        {"entity_id", "canonical_name", "entity_type", "tier", "phrase", "score"} <= set(e)
        for e in j0["resolved_entities"]
    )
    assert all("entity_type" in l for l in j0["entity_links"])
    pt0 = j0["pipeline_trace"]
    expected_keys = {
        "flow_pattern", "entity_resolution", "routing", "vector_pass1", "vector_seeded_entry",
        "cypher_template", "endpoint_bridging", "hipporag_ppr", "floor_short_circuit",
        "path_selection", "graph_chunk_fetch", "evidence_gap", "context_merge", "citation_gate",
    }
    assert expected_keys <= set(pt0)
    assert pt0["flow_pattern"]["code"] == "HYBRID_DIRECT"
    assert pt0["hipporag_ppr"]["alpha"] == pytest.approx(0.85)
    assert pt0["graph_chunk_fetch"]["ran"] and pt0["graph_chunk_fetch"]["kept_count"] > 0
    assert [r["id"] for r in pt0["citation_gate"]["rungs"]] == [
        "generate", "validate", "repair", "crop", "fallback", "deliver"
    ]
    assert pt0["citation_gate"]["status"] == "PASS" and pt0["citation_gate"]["short_circuited_refusal"] is False

    # Aggregation short-circuit query
    j4 = client.post("/query", json={"question": EXAMPLES[4]["question"]}).json()
    pt4 = j4["pipeline_trace"]
    assert pt4["flow_pattern"]["code"] == "AGGREGATION_SHORT_CIRCUIT"
    assert pt4["floor_short_circuit"]["triggered"] is True

    # Refusal short-circuit query (unverified subject Gemini 1.5 Pro)
    j5 = client.post("/query", json={"question": EXAMPLES[5]["question"]}).json()
    assert j5["refused"] is True and j5["unverified_subjects"]
    pt5 = j5["pipeline_trace"]
    assert pt5["flow_pattern"]["code"] == "REFUSAL_SHORT_CIRCUIT"
    assert pt5["citation_gate"]["short_circuited_refusal"] is True
    assert pt5["citation_gate"]["rungs"][0]["status"] == "skip"
    assert pt5["citation_gate"]["rungs"][-1]["status"] == "fail"

    # Default title fallback in load_research_corpus() omits em dash
    mini = tmp_path / "mini.yaml"
    mini.write_text("papers: []\npillars: []\n", encoding="utf-8")
    assert load_research_corpus(mini)["title"] == "Research Corpus"
