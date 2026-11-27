"""Tests for the five Phase 5 architectural fixes and the live-mode wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.benchmark_runner import load_questions
from src.generation.synthesizer import GroundedAnswerSynthesizer, OfflineGroundedGeneratorLLM, question_clauses
from src.ingestion.offline_demo import build_offline_corpus
from src.retrieval.cypher_templates import CypherTemplateCatalog, CypherTemplateId as T
from src.retrieval.router import GraphVectorEngine, QueryRouter, RouteType as R
from src.retrieval.subject_gate import extract_subject_names, unverified_subjects

HELDOUT = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "phase5_heldout_questions.json"


@pytest.fixture(scope="module")
def corpus():
    return build_offline_corpus()


@pytest.fixture()
def engine(corpus, tmp_path):
    return GraphVectorEngine(QueryRouter(corpus.resolver, corpus.graph, corpus.vectors, telemetry_path=tmp_path / "t.jsonl"))


# Fix 1: structural routing -------------------------------------------------


def test_false_premise_pair_upgrades_vector_only(engine) -> None:
    d = engine.router.route("How many key-value heads does LLaMA-1 use in its grouped-query attention layers?")
    assert d.original_route == R.VECTOR_ONLY and d.route == R.HYBRID_PARALLEL and d.structural_upgrade
    assert any("false premise" in s for s in d.structural_signals)


def test_directly_linked_single_fact_stays_vector_only(engine) -> None:
    d = engine.router.route("What is the exact RoPE base frequency used by Llama 3?")
    assert d.route == R.VECTOR_ONLY and not d.structural_upgrade


def test_implicit_bridge_phrase_upgrades(engine) -> None:
    d = engine.router.route("How many parameters does the predecessor of DeepSeek-V3 have?")
    assert d.route == R.HYBRID_PARALLEL and any("bridge" in s for s in d.structural_signals)


def test_natural_aggregation_phrasing_selects_t5_with_filters(engine) -> None:
    d = engine.router.route("Which architectural components show up in more than one model family in this corpus?")
    assert d.cypher_template_id == T.T5_CORPUS_AGGREGATION and d.template_params["min_sources"] == 2
    d2 = engine.router.route("Which methods in the corpus build on or modify rotary position embeddings?")
    assert d2.template_params["relation_type"] == "EXTENDS" and d2.template_params["anchor_entity_id"] == "arch_component::rope"


def test_t5_anchor_is_transitive_and_direction_preserving(corpus) -> None:
    cat = CypherTemplateCatalog()
    up = cat.run(corpus.graph, T.T5_CORPUS_AGGREGATION, {"relation_type": "EXTENDS", "anchor_entity_id": "arch_component::multi_head_attention"})
    assert {p[0]["source"] for p in up.paths} >= {"arch_component::grouped_query_attention", "arch_component::multi_head_latent_attention"}
    down = cat.run(corpus.graph, T.T5_CORPUS_AGGREGATION, {"relation_type": "ADOPTS_COMPONENT", "anchor_entity_id": "model_family::deepseek_v3"})
    assert {p[0]["source"] for p in down.paths} == {"model_family::deepseek_v3"}


# Fix 2: vector channel always on + cascade ---------------------------------


def test_graph_only_route_still_runs_vector_channel(engine) -> None:
    r = engine.retrieve("Which techniques mitigate the KV cache bottleneck?")
    assert r.decision.route == R.GRAPH_ONLY and r.vector_chunks


def test_weak_template_cascades_to_t6_and_transformer_seeds_mha(engine) -> None:
    q = ("Trace the design ancestry of the attention in DeepSeek's 671B-parameter MoE model back to the "
         "original Transformer's mechanism, naming each intermediate step.")
    r = engine.retrieve(q)
    d = r.decision
    assert "arch_component::multi_head_attention" in d.resolved_entity_ids
    assert d.cypher_template_id == T.T1_COMPONENT_LINEAGE and T.T6_DYNAMIC_SUBGRAPH_FALLBACK in d.cascade_template_ids
    nodes = {n for p in r.template_result.paths for h in p for n in (h["source"], h["target"])}
    assert {"arch_component::grouped_query_attention", "arch_component::multi_head_latent_attention"} <= nodes


def test_t3_traverses_requires_to_reach_bottleneck(corpus) -> None:
    res = CypherTemplateCatalog().run(corpus.graph, T.T3_BOTTLENECK_SOLVERS, {"bottleneck_id": "model_family::deepseek_r1"})
    chains = [[h["relation_type"] for h in p] for p in res.paths]
    assert ["REQUIRES", "ADOPTS_COMPONENT", "MITIGATES"] in chains


# Fix 3: per-hop stratified supporting chunks -------------------------------


def test_stratified_selection_covers_every_hop_of_the_top_path(engine) -> None:
    r = engine.retrieve("Which bottleneck does the component adopted by DeepSeek-V3 mitigate?")
    top = r.template_result.paths[0]
    hop_chunks = [set(h["source_chunk_ids"]) for h in top]
    picked = {c.chunk_id for c in r.supporting_chunks}
    assert len(top) >= 2 and all(hc & picked for hc in hop_chunks)


# Fix 4: score-ranked, clause-covering generator ----------------------------


def test_generator_keeps_third_relevant_sentence_of_one_passage(engine, corpus) -> None:
    q = "By what factor are attention logits scaled in the original Transformer, and what goes wrong without that scaling?"
    a = GroundedAnswerSynthesizer(OfflineGroundedGeneratorLLM(), corpus.vectors).synthesize(q, engine.retrieve(q))
    assert "1/sqrt(d_k)" in a.answer and "small gradients" in a.answer and a.final_validation.valid


def test_question_clauses() -> None:
    assert question_clauses("Which activation did LLaMA use, and how many KV groups does it use?") == [
        "Which activation did LLaMA use", "how many KV groups does it use"]


# Fix 5: subject-entity verification gate -----------------------------------


def test_subject_name_extraction() -> None:
    assert extract_subject_names("How does Mixtral 8x22B route tokens?") == ["Mixtral 8x22B", "Mixtral"]
    assert extract_subject_names("What context length does Gemini 1.5 Pro support?") == ["Gemini 1.5", "Gemini"]
    assert extract_subject_names("BLEU on WMT 2014 for Llama 3 8B") == ["Llama 3"]
    assert extract_subject_names("Meta's 2024 herd and DeepMind's recipe") == []


def test_gate_refuses_only_unindexed_subjects(corpus, engine) -> None:
    assert unverified_subjects("How many parameters does GPT-4 have?", corpus.vectors) == ["GPT 4"]
    assert unverified_subjects("How many parameters does GPT-3 have?", corpus.vectors) == []
    d = engine.router.route("How many parameters does GPT-4 have?")
    assert d.route == R.OUT_OF_SCOPE and d.unverified_subjects == ["GPT 4"]
    r = engine.retrieve("How many parameters does GPT-4 have?")
    a = GroundedAnswerSynthesizer(OfflineGroundedGeneratorLLM(), corpus.vectors).synthesize(r.decision.question, r)
    assert a.refused and "GPT 4" in a.answer and not a.citations


def test_gate_never_refuses_an_answerable_benchmark_question(corpus) -> None:
    questions = load_questions() + load_questions(HELDOUT)
    refused = [q.id for q in questions if q.answerable and unverified_subjects(q.question, corpus.vectors)]
    assert refused == []


def test_pgvector_text_exists_binds_pattern() -> None:
    from src.stores.pgvector_store import PgVectorStore

    log = []

    class Cur:
        description = None
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def execute(self, sql, params=None): log.append((sql, params))
        def fetchall(self): return []

    class Conn:
        def transaction(self):
            class Tx:
                def __enter__(s): return s
                def __exit__(s, *a): return None
            return Tx()
        def cursor(self): return Cur()

    evil = "gpt 4'); drop table chunks; --"
    assert PgVectorStore(connection=Conn()).text_exists(evil) is False
    sql, params = log[0]
    assert evil not in sql and "%(pat)s" in sql and r"drop\ table" in params["pat"]


# Live mode -----------------------------------------------------------------


def test_live_mode_reports_every_unmet_dependency(monkeypatch) -> None:
    import psycopg

    from src.api.main import RagService
    from src.config import settings
    from src.stores import neo4j_store

    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(type(settings), "reload", lambda self: self)  # live() re-reads .env; keep the blank key

    def no_neo4j(self):
        raise ConnectionError("refused")

    def no_pg(*a, **k):
        raise psycopg.OperationalError("timeout")

    monkeypatch.setattr(neo4j_store.Neo4jGraphStore, "verify_connectivity", no_neo4j)
    monkeypatch.setattr(psycopg, "connect", no_pg)
    with pytest.raises(RuntimeError) as exc:
        RagService.live()
    msg = str(exc.value)
    assert "GEMINI_API_KEY" in msg and "Neo4j unreachable" in msg and "PostgreSQL unreachable" in msg
