"""Unit tests for Phase 2: embedders, in-memory hybrid store, reranker, pgvector SQL contract, bridges."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import pytest

from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.offline_demo import build_offline_corpus
from src.stores.embedders import DeterministicMockEmbedder, GeminiEmbedder
from src.stores.pgvector_store import (
    InMemoryVectorStore,
    PgVectorStore,
    TokenInteractionReranker,
    filter_sql,
    rrf_fuse,
    schema_statements,
    symbol_tokens,
    tokenize,
)


@pytest.fixture(scope="module")
def corpus():
    return build_offline_corpus()


def _chunks(paper_id: str, year: int, pillar: str, texts: List[str]):
    md = "\n\n".join(f"# S{i}\n\n{t}" for i, t in enumerate(texts))
    return StructureAwareChunker().chunk_paper(paper_id, paper_id.title(), year, pillar, md)


def test_tokenize_and_symbols() -> None:
    assert tokenize("The RoPE base of 500,000!") == ["rope", "base", "500000"]
    assert symbol_tokens("What is W^{DKV} and 500,000?") == ["w^{dkv}", "500,000"]


def test_mock_embedder_is_deterministic_unit_and_concept_aware() -> None:
    e1, e2 = DeterministicMockEmbedder(), DeterministicMockEmbedder()
    a = e1.embed_query("grouped query attention reduces the KV cache")
    assert np.allclose(a, e2.embed_query("grouped query attention reduces the KV cache"))
    assert a.shape == (768,) and abs(float(np.linalg.norm(a)) - 1) < 1e-5
    kv = e1.embed_query("key value heads share the cache")
    rlhf = e1.embed_query("reward model preference policy")
    assert float(a @ kv) > float(a @ rlhf)


def test_gemini_embedder_uses_cache_and_output_dimensionality(tmp_path) -> None:
    calls: List[dict] = []

    def embed_content(**kw: Any) -> Any:
        calls.append(kw)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * 766) for _ in kw["contents"]])

    client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    emb = GeminiEmbedder(client=client, model="gemini-embedding-001", dim=768, cache_path=tmp_path / "e.sqlite")
    v1 = emb.embed_documents(["a", "b"], ["h1", "h2"])
    v2 = emb.embed_documents(["a", "b"], ["h1", "h2"])
    assert len(calls) == 1 and calls[0]["config"].output_dimensionality == 768
    assert calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT" and calls[0]["model"] == "gemini-embedding-001"
    assert np.allclose(v1[0], v2[0]) and abs(float(np.linalg.norm(v1[0])) - 1) < 1e-6


def test_rrf_fuse_math() -> None:
    fused = rrf_fuse([["a", "b"], ["b", "c"]], rrf_k=60)
    assert [cid for cid, _, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61) and fused[0][2] == [2, 1]


def test_in_memory_modes_filters_and_idempotent_index() -> None:
    emb = DeterministicMockEmbedder()
    store = InMemoryVectorStore(embedder=emb)
    chunks = (_chunks("p_a", 2023, "architecture", ["Grouped query attention shares key value heads."])
              + _chunks("p_b", 2024, "post_training", ["DPO optimizes the policy with a preference loss."]))
    assert store.index_chunks(chunks, {chunks[0].chunk_id: ["arch_component::gqa"]}) == 2
    assert store.index_chunks(chunks) == 0 and store.count() == 2 and emb.calls == 2
    for fn in (store.search_dense, store.search_lexical_bm25, store.search_hybrid_rrf, store.search_hybrid_reranked):
        hits = fn("preference policy DPO", top_k=1)
        assert hits[0].paper_id == "p_b" and hits[0].rank == 1
    assert all(h.pillar == "architecture" for h in store.search_hybrid_rrf("policy", top_k=2, filters={"pillar": "architecture"}))
    assert all(h.pub_year >= 2024 for h in store.search_dense("attention", top_k=2, filters={"min_year": 2024}))
    assert [r.chunk_id for r in store.fetch_chunks_by_ids([chunks[1].chunk_id, "nope", chunks[0].chunk_id])] == \
        [chunks[1].chunk_id, chunks[0].chunk_id]
    with pytest.raises(ValueError):
        store.search_hybrid_reranked("x", top_k=5, candidate_k=2)


def test_reranker_promotes_exact_symbol_match() -> None:
    scores = TokenInteractionReranker().score("What is W^{DKV}?", ["latent attention compresses keys", "we use W^{DKV} h_t"])
    assert scores[1] > scores[0]


class _Cur:
    def __init__(self, log): self.log, self.description = log, None
    def __enter__(self): return self
    def __exit__(self, *a): return None
    def execute(self, sql, params=None): self.log.append((sql, params))
    def fetchall(self): return []


class _Conn:
    def __init__(self): self.log: List[Tuple[str, Any]] = []
    def transaction(self):
        log = self.log
        class T:
            def __enter__(s): log.append(("BEGIN", None))
            def __exit__(s, *a): log.append(("COMMIT", None))
        return T()
    def cursor(self): return _Cur(self.log)
    def close(self): pass


def test_pgvector_schema_and_query_sql_contract() -> None:
    ddl = "\n".join(schema_statements(768))
    assert "vector(768)" in ddl and "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)" in ddl
    assert "USING gin (tsv)" in ddl and "GENERATED ALWAYS AS (to_tsvector('english', context_prefixed_text))" in ddl
    conn = _Conn()
    pg = PgVectorStore(connection=conn)
    pg.search_hybrid_rrf("drop'; --", top_k=3, rrf_k=60, ef_search=90, filters={"pillar": "inference", "paper_ids": ["x"]})
    sqls = [s for s, _ in conn.log]
    assert sqls[:2] == ["BEGIN", "SET LOCAL hnsw.ef_search = 90"]
    sql, params = conn.log[2]
    assert "ts_rank_cd" in sql and "FULL OUTER JOIN" in sql and "drop'; --" not in sql
    assert params["q"] == "drop'; --" and params["rrf_k"] == 60 and params["f_paper_ids"] == ["x"]
    assert filter_sql({"min_year": 2024}) == (" AND c.pub_year >= %(f_min_year)s", {"f_min_year": 2024})
    with pytest.raises(ValueError):
        pg.search_dense("q", ef_search=0)


def test_offline_corpus_bridges(corpus) -> None:
    g, v = corpus.graph, corpus.vectors
    path = g.find_paths("arch_component::multi_head_latent_attention", "arch_component::multi_head_attention", "EXTENDS", 2)[0]
    texts = v.fetch_chunks_by_ids([cid for hop in path for cid in hop["source_chunk_ids"]])
    assert len(texts) == 2 and all(any(q in t.raw_text for hop in path for q in hop["evidence_quotes"]) for t in texts)
    top = v.search_hybrid_reranked("What is the W^{DKV} latent compression matrix in MLA?", top_k=1)[0]
    assert top.paper_id == "deepseek_2024_v2" and "arch_component::multi_head_latent_attention" in top.entity_ids
    assert any(n["neighbor"] == "arch_component::grouped_query_attention"
               for n in g.neighbors("arch_component::multi_head_latent_attention"))
