"""Production readiness: .env parsing, rate-limit governor + fallback chain, bibliography stripping,
paper-by-paper commit with resumable quota exit, and per-query usage telemetry. No real API calls."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from src.config import Settings
from src.ingestion.arxiv_fetcher import load_manifest, load_or_synthesize_seed_markdown
from src.ingestion.chunker import StructureAwareChunker, strip_bibliography
from src.ingestion.cost_estimator import ExtractionCache
from src.ingestion.extractor import GeminiGraphExtractor
from src.ingestion.offline_fixtures import RecordedExtractionLLM
from src.ingestion.pipeline import IngestionQuotaExhausted, Phase1IngestionPipeline
from src.ingestion.rate_limiter import (
    AllModelsExhaustedError,
    ModelLimits,
    QuotaKind,
    RateLimitGovernor,
    classify_quota_error,
    set_governor,
)
from src.stores.embedders import GeminiEmbedder
from src.stores.neo4j_store import InMemoryGraphStore
from src.stores.pgvector_store import InMemoryVectorStore
from src.usage import summarize_usage, usage_scope

E2E = ("vaswani_2017_transformer", "ainslie_2023_gqa", "deepseek_2024_v2")
DAILY_BODY = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
MINUTE_BODY = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}})


class Fake429(Exception):
    """Shaped like the Interactions client's GenAiError (status_code + JSON body)."""

    def __init__(self, body: str) -> None:
        super().__init__("Resource has been exhausted")
        self.status_code, self.body = 429, body


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: List[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def governor(limits: Dict[str, ModelLimits] = None, **kw) -> RateLimitGovernor:
    clk = FakeClock()
    g = RateLimitGovernor(limits=limits, clock=clk, sleep=clk.sleep, **kw)
    g.fake_clock = clk  # type: ignore[attr-defined]
    return g


def interaction(text: str, inp: int = 100, out: int = 20) -> Any:
    return SimpleNamespace(output_text=text, usage=SimpleNamespace(total_input_tokens=inp, total_output_tokens=out,
                                                                   total_tokens=inp + out))


# ---------------------------------------------------------------------------
# 1. .env parsing
# ---------------------------------------------------------------------------


def test_env_parsing_fallback_models_and_quote_stripping(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="  AIza-test-key-123  "\nGENERATION_MODEL=\'gemini-3.5-flash-lite\'\n'
                   "FALLBACK_MODEL_1= gemini-3.5-flash-lite \nFALLBACK_MODEL_2=gemini-3.1-flash-lite\n"
                   "FALLBACK_MODEL_3=gemma-4-31b-it\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "stale-key-exported-in-shell")
    s = Settings(_env_file=str(env))
    assert s.gemini_api_key == "AIza-test-key-123"  # quotes + whitespace stripped, .env beats the shell
    assert s.generation_model == "gemini-3.5-flash-lite"
    assert s.fallback_models == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemma-4-31b-it"]
    assert s.text_model_chain("gemini-3.1-flash-lite") == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemma-4-31b-it"]


def test_settings_reload_picks_up_swapped_key(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=first-key\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", str(env))
    s = Settings()
    assert s.gemini_api_key == "first-key"
    env.write_text("GEMINI_API_KEY='second-key'\n", encoding="utf-8")
    assert s.reload().gemini_api_key == "second-key"


def test_defaults_match_free_tier_models() -> None:
    s = Settings(_env_file=None)
    assert {s.extraction_model, s.router_model, s.generation_model} == {"gemini-3.5-flash-lite"}
    assert s.embedding_model == "gemini-embedding-001"


# ---------------------------------------------------------------------------
# 2. Governor: pacing, 429 classification, fallback, exhaustion
# ---------------------------------------------------------------------------


def test_default_ceilings_sit_below_free_tier_limits() -> None:
    g = RateLimitGovernor()
    assert g.limits_for("gemini-3.5-flash-lite") == ModelLimits(13, 200_000)
    assert g.limits_for("gemini-3.1-flash-lite") == ModelLimits(13, 200_000)
    assert g.limits_for("gemma-4-31b-it") == ModelLimits(25, 13_000)
    assert g.limits_for("gemini-embedding-001") == ModelLimits(80, 24_000)


def test_rpm_sliding_window_pacing() -> None:
    g = governor({"m": ModelLimits(max_rpm=3, max_tpm=10**9)})
    for _ in range(3):
        assert g.acquire("m", 10) == 0.0
    g.fake_clock.t += 15  # 15 s later the window is still full
    waited = g.acquire("m", 10)
    assert waited == pytest.approx(45.0)  # exactly until the oldest requests leave the 60 s window
    assert g.window_usage("m")[0] == 1  # the three old requests expired together; only the new one remains


def test_tpm_sliding_window_pacing_and_actual_tokens() -> None:
    g = governor({"m": ModelLimits(max_rpm=100, max_tpm=1_000)})
    g.acquire("m", 600)
    g.fake_clock.t += 10
    g.acquire("m", 300)
    g.record_actual("m", 350)  # server reported more than estimated
    g.fake_clock.t += 5
    assert g.wait_needed("m", 200) == pytest.approx(45.0)  # 600 + 350 + 200 > 1000 until the first entry expires
    assert g.acquire("m", 200) == pytest.approx(45.0)


def test_classify_daily_vs_minute_quota_errors() -> None:
    assert classify_quota_error(Fake429(DAILY_BODY)) == (QuotaKind.DAILY, None)
    assert classify_quota_error(Fake429(MINUTE_BODY)) == (QuotaKind.MINUTE, 7.0)
    assert classify_quota_error(ValueError("bad json")) is None
    api_style = SimpleNamespace(code=429, details={"error": {"message": "Quota exceeded for quota metric 'per day'"}})
    err = Exception("429 RESOURCE_EXHAUSTED")
    err.code, err.details = api_style.code, api_style.details  # type: ignore[attr-defined]
    assert classify_quota_error(err)[0] == QuotaKind.DAILY


def test_fallback_chain_on_daily_quota_and_retry_on_minute_spike() -> None:
    g = governor()
    calls: List[str] = []
    state = {"minute_left": 1}

    def fn(model: str) -> str:
        calls.append(model)
        if model == "gemini-3.5-flash-lite":
            raise Fake429(DAILY_BODY)
        if model == "gemini-3.1-flash-lite" and state["minute_left"]:
            state["minute_left"] -= 1
            raise Fake429(MINUTE_BODY)
        return f"ok from {model}"

    chain = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemma-4-31b-it"]
    result, used = g.call_with_fallback(chain, fn, 100)
    assert (result, used) == ("ok from gemini-3.1-flash-lite", "gemini-3.1-flash-lite")
    assert calls == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite"]
    assert 7.0 in g.fake_clock.sleeps  # honoured the server's retryDelay
    # An exhausted model is skipped for the rest of the process.
    assert g.call_with_fallback(chain, lambda m: m, 1)[1] == "gemini-3.1-flash-lite"


def test_all_models_exhausted_raises() -> None:
    g = governor()
    with pytest.raises(AllModelsExhaustedError) as exc:
        g.call_with_fallback(["a", "b", "c"], lambda m: (_ for _ in ()).throw(Fake429(DAILY_BODY)), 1)
    assert exc.value.models == ["a", "b", "c"] and g.exhausted == {"a", "b", "c"}


def test_persistent_minute_429s_step_to_next_model() -> None:
    g = governor(max_minute_retries=2)

    def fn(model: str) -> str:
        if model == "a":
            raise Fake429(MINUTE_BODY)
        return model

    assert g.call_with_fallback(["a", "b"], fn, 1) == ("b", "b")


def test_cache_hits_bypass_the_governor(tmp_path) -> None:
    g = governor({"gemini-3.5-flash-lite": ModelLimits(1, 10**9)})
    set_governor(g)
    client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kw: interaction('{"entities": [], "relationships": []}')))
    ex = GeminiGraphExtractor(ExtractionCache(":memory:"), client=client, model_name="gemini-3.5-flash-lite")
    chunk = StructureAwareChunker().chunk_paper("p", "P", 2024, "architecture", "# A\n\nSome text here.\n")[0]
    ex.extract_chunk(chunk)
    for _ in range(5):
        assert ex.extract_chunk(chunk).cache_hit
    assert g.fake_clock.sleeps == [] and g.window_usage("gemini-3.5-flash-lite")[0] == 1


# ---------------------------------------------------------------------------
# 3. Bibliography stripping
# ---------------------------------------------------------------------------


PAPER_MD = """# Some Paper

## 1 Introduction

We study grouped-query attention for fast decoding.

## 2 Method

GQA shares key-value heads across groups of query heads.

## **References**

[1] J. Ainslie et al. GQA: Training generalized multi-query transformer models. 2023.
[2] N. Shazeer. Fast transformer decoding: one write-head is all you need. 2019.

## A Additional Experiments

Uptraining with 5% of the compute recovers quality.
"""


def test_bibliography_is_stripped_but_appendix_kept() -> None:
    text, removed = strip_bibliography(PAPER_MD)
    assert removed >= 4 and "[1] J. Ainslie" not in text and "Uptraining with 5%" in text
    chunks = StructureAwareChunker().chunk_paper("p", "Some Paper", 2023, "architecture", PAPER_MD)
    joined = " ".join(c.raw_text for c in chunks)
    assert "Shazeer" not in joined and "Uptraining with 5%" in joined
    assert not any("References" in c.section_path for c in chunks)


def test_bibliography_variants() -> None:
    for heading in ("# References", "## 7 References", "**Bibliography**", "REFERENCES", "### References:"):
        md = "# T\n\nbody one\n\nbody two\n\nbody three\n\n" + heading + "\n\n[1] Some cited work. 2020.\n"
        text, removed = strip_bibliography(md)
        assert "Some cited work" not in text and removed > 0, heading
    early = "# References\n\nThis is not a bibliography; it is section one.\n\nbody\n\nbody\n\nbody\n"
    assert strip_bibliography(early)[1] == 0


# ---------------------------------------------------------------------------
# 4. Paper-by-paper commit, quota exit and resume
# ---------------------------------------------------------------------------


class ScriptedGeminiClient:
    """Fake Interactions client: replays recorded extractions; models listed in `daily_exhausted` 429."""

    def __init__(self, daily_exhausted=(), fail_after: int = None) -> None:
        self.recorded = RecordedExtractionLLM()
        self.daily_exhausted = set(daily_exhausted)
        self.fail_after = fail_after
        self.calls: List[str] = []
        self.interactions = SimpleNamespace(create=self.create)

    def create(self, **kw) -> Any:
        model = kw["model"]
        if model in self.daily_exhausted or (self.fail_after is not None and len(self.calls) >= self.fail_after):
            raise Fake429(DAILY_BODY)
        self.calls.append(model)
        return interaction(self.recorded(kw["system_instruction"], kw["input"], {}), 900, 150)


class FakeEmbedClient:
    def __init__(self, fail_after: int = None) -> None:
        self.requests = 0
        self.fail_after = fail_after
        self.models = SimpleNamespace(embed_content=self.embed_content)

    def embed_content(self, model, contents, config) -> Any:
        if self.fail_after is not None and self.requests >= self.fail_after:
            raise Fake429(DAILY_BODY.replace("GenerateRequestsPerDay", "EmbedContentRequestsPerDay"))
        self.requests += 1
        vecs = []
        for t in contents:
            v = np.zeros(768, dtype=np.float32)
            v[hash(t) % 768] = 1.0
            vecs.append(SimpleNamespace(values=v.tolist()))
        return SimpleNamespace(embeddings=vecs, usage_metadata=None)


def _papers():
    return [p for p in load_manifest() if p.paper_id in E2E]


def _pipeline(cache, client, graph):
    ex = GeminiGraphExtractor(cache, client=client, model_name="gemini-3.5-flash-lite")
    return Phase1IngestionPipeline(ex, graph, lambda p: load_or_synthesize_seed_markdown(p, prefer_cached=False))


def _graph_signature(store: InMemoryGraphStore):
    return (store.count_nodes(), store.count_relationships(), sorted("|".join(k) for k in store.rels))


def test_fallback_model_is_used_and_recorded_in_cache(tmp_path) -> None:
    cache = ExtractionCache(tmp_path / "c.sqlite")
    client = ScriptedGeminiClient(daily_exhausted={"gemini-3.5-flash-lite"})
    _pipeline(cache, client, InMemoryGraphStore()).run(_papers(), max_budget_usd=1.0)
    assert set(client.calls) == {"gemini-3.1-flash-lite"}
    rows = cache._conn.execute("SELECT DISTINCT model FROM extractions").fetchall()
    assert [r[0] for r in rows] == ["gemini-3.1-flash-lite"]


def test_quota_exit_commits_completed_papers_and_resume_loses_no_work(tmp_path) -> None:
    papers = _papers()
    total = len(StructureAwareChunker().chunk_paper(papers[0].paper_id, papers[0].title, papers[0].year,
                                                    papers[0].pillar.value, load_or_synthesize_seed_markdown(papers[0], prefer_cached=False)))
    # Reference: one uninterrupted run.
    ref_graph, ref_vec = InMemoryGraphStore(), InMemoryVectorStore(embedder=GeminiEmbedder(
        client=FakeEmbedClient(), cache_path=tmp_path / "ref_e.sqlite"))
    _pipeline(ExtractionCache(":memory:"), ScriptedGeminiClient(), ref_graph).run(papers, max_budget_usd=1.0, vector_store=ref_vec)

    # Key #1: every model's daily quota runs out after `total + 2` extractions (inside paper 2).
    cache = ExtractionCache(tmp_path / "c.sqlite")
    graph = InMemoryGraphStore()
    emb_cache = tmp_path / "e.sqlite"
    vec = InMemoryVectorStore(embedder=GeminiEmbedder(client=FakeEmbedClient(), cache_path=emb_cache))
    first = ScriptedGeminiClient(fail_after=total + 2)
    with pytest.raises(IngestionQuotaExhausted) as exc:
        _pipeline(cache, first, graph).run(papers, max_budget_usd=1.0, vector_store=vec)
    e = exc.value
    assert e.stage == "extraction" and e.completed == total + 2 and len(cache) == total + 2
    assert e.report.committed_papers == [papers[0].paper_id]
    assert "DAILY API QUOTA REACHED" in e.banner() and f"resume from chunk {total + 3}" in e.banner()
    assert graph.count_nodes("Chunk") > 0 and graph.count_relationships() > 0          # paper 1 + partial paper 2 in Neo4j
    assert {r.paper_id for r in vec.rows.values()} == {papers[0].paper_id}             # paper 1 in pgvector
    assert graph.get_relationship(f"paper::{papers[0].paper_id}", "INTRODUCES", "model_family::transformer")

    # Key #2 (fresh process state): same command resumes; finished chunks cost 0 API calls.
    set_governor(governor())
    second = ScriptedGeminiClient()
    emb2 = FakeEmbedClient()
    vec.embedder = GeminiEmbedder(client=emb2, cache_path=emb_cache)
    report = _pipeline(cache, second, graph).run(papers, max_budget_usd=1.0, vector_store=vec)
    assert len(second.calls) == report.extraction.chunks - (total + 2)                 # only the remaining chunks
    assert report.extraction.cache_hits == total + 2
    assert _graph_signature(graph) == _graph_signature(ref_graph)                      # identical to the uninterrupted run
    assert set(vec.rows) == set(ref_vec.rows)


def test_embedding_quota_exit_keeps_graph_and_extraction(tmp_path) -> None:
    cache = ExtractionCache(tmp_path / "c.sqlite")
    graph = InMemoryGraphStore()
    vec = InMemoryVectorStore(embedder=GeminiEmbedder(client=FakeEmbedClient(fail_after=1), cache_path=tmp_path / "e.sqlite",
                                                      max_batch_tokens=10**9))
    with pytest.raises(IngestionQuotaExhausted) as exc:
        _pipeline(cache, ScriptedGeminiClient(), graph).run(_papers(), max_budget_usd=1.0, vector_store=vec)
    assert exc.value.stage == "embedding" and exc.value.report.committed_papers == [E2E[0]]
    assert len(cache) > exc.value.completed  # the paper whose embedding failed is still cached for the resume


# ---------------------------------------------------------------------------
# 5. Per-query usage telemetry
# ---------------------------------------------------------------------------


def _live_like_service(tmp_path, router_text: str):
    from src.api.main import RagService
    from src.generation.synthesizer import GeminiGeneratorLLM, GroundedAnswerSynthesizer
    from src.ingestion.offline_demo import build_offline_corpus
    from src.retrieval.router import GeminiRouterLLM, GraphVectorEngine, QueryRouter

    c = build_offline_corpus()
    emb_client = FakeEmbedClient()
    c.vectors.embedder = GeminiEmbedder(client=emb_client, cache_path=tmp_path / "q.sqlite")
    router_client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kw: interaction(router_text, 410, 60)))

    def gen_create(**kw):
        cid = re.search(r"^\[([^\]|]+) \|", kw["input"], flags=re.M).group(1).strip()
        return interaction(f"The retrieved passage supports this cited claim [{cid}].", 2200, 90)

    gen = GeminiGeneratorLLM(client=SimpleNamespace(interactions=SimpleNamespace(create=gen_create)), model="gemini-3.5-flash-lite")
    router = GeminiRouterLLM(client=router_client, model="gemini-3.5-flash-lite", allow_heuristic_fallback=False)
    engine = GraphVectorEngine(QueryRouter(c.resolver, c.graph, c.vectors, llm=router, telemetry_path=tmp_path / "r.jsonl"))
    return RagService(c, engine, GroundedAnswerSynthesizer(gen, c.vectors)), emb_client


ROUTER_JSON = json.dumps({"query_intent": "BOTTLENECK_TRACE", "route": "GRAPH_ONLY", "confidence": 0.92,
                          "standalone_query": "Which techniques mitigate the KV cache bottleneck?",
                          "extracted_entities": ["KV cache"]})


def test_query_usage_telemetry_counts_every_component(tmp_path) -> None:
    from src.api.main import QueryRequest

    svc, emb = _live_like_service(tmp_path, ROUTER_JSON)
    result, g, _, u = svc.answer_with_usage(QueryRequest(question="Which techniques mitigate the KV cache bottleneck?"))
    assert result.decision.router_model == "gemini-3.5-flash-lite"
    assert (u.router_requests, u.router_input_tokens, u.router_output_tokens) == (1, 410, 60)
    assert u.generator_requests == g.generator_calls == 1
    assert (u.generator_input_tokens, u.generator_output_tokens) == (2200, 90)
    assert u.embedding_requests_used == emb.requests >= 1 and u.embedding_tokens_est > 0
    # The same question again: the query embedding now comes from the SQLite cache (0 embedding requests).
    _, _, _, u2 = svc.answer_with_usage(QueryRequest(question="Which techniques mitigate the KV cache bottleneck?"))
    assert u2.embedding_requests_used == 0 and u2.embedding_cache_hits >= 1 and u2.router_requests == 1
    assert u.llm_requests_used == 2 and u.total_api_requests_used == 2 + emb.requests
    assert u.total_input_tokens == 410 + 2200 + u.embedding_tokens_est and u.total_output_tokens == 150
    assert u.total_tokens == u.total_input_tokens + u.total_output_tokens
    assert u.router_model == u.generator_model == "gemini-3.5-flash-lite" and u.embedding_model == "gemini-embedding-001"


def test_api_response_carries_usage_telemetry(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    svc, _ = _live_like_service(tmp_path, ROUTER_JSON)
    j = TestClient(create_app(svc)).post("/query", json={"question": "Which techniques mitigate the KV cache bottleneck?"}).json()
    ut = j["usage_telemetry"]
    assert ut["router_input_tokens"] == 410 and ut["generator_output_tokens"] == 90 and ut["total_api_requests_used"] >= 3


def test_live_router_never_falls_back_to_heuristic(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from src.api.main import QueryRequest, create_app
    from src.retrieval.router import RouterUnavailableError

    svc, _ = _live_like_service(tmp_path, "this is not json")
    with pytest.raises(RouterUnavailableError):
        svc.answer_with_usage(QueryRequest(question="Which techniques mitigate the KV cache bottleneck?"))
    r = TestClient(create_app(svc)).post("/query", json={"question": "Which techniques mitigate the KV cache bottleneck?"})
    assert r.status_code == 503


def test_embedder_batches_respect_token_budget(tmp_path) -> None:
    client = FakeEmbedClient()
    emb = GeminiEmbedder(client=client, cache_path=tmp_path / "e.sqlite", max_batch_tokens=50)
    texts = ["word " * 40 for _ in range(4)]  # ~50 tokens each
    with usage_scope() as meter:
        emb.embed_documents(texts, [f"h{i}" for i in range(4)])
        emb.embed_documents(texts, [f"h{i}" for i in range(4)])
    u = summarize_usage(meter)
    assert client.requests == 4 and u.embedding_requests_used == 4 and u.embedding_cache_hits == 4
