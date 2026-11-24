"""Phase 2 scorecard: shared-key integrity, retrieval ablation (Recall@k), graph<->text bridges, SQL contract.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/benchmark_phase2.py

Offline: the 12 seed papers go through Phase1IngestionPipeline (synthetic fixtures + recorded
extractions + InMemoryGraphStore), then chunks and their resolved entity_ids are indexed into
InMemoryVectorStore with DeterministicMockEmbedder. PgVectorStore SQL is checked through a
recording psycopg-style connection (no PostgreSQL needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.offline_demo import OfflineCorpus, build_offline_corpus  # noqa: E402
from src.stores.embedders import DeterministicMockEmbedder  # noqa: E402
from src.stores.pgvector_store import InMemoryVectorStore, PgVectorStore, RetrievedChunk, schema_statements, tokenize  # noqa: E402

LINE, SUBLINE = "=" * 96, "-" * 96
MLA = "arch_component::multi_head_latent_attention"
GQA = "arch_component::grouped_query_attention"
MHA = "arch_component::multi_head_attention"

# (query_id, kind, query, relevant paper_id, substring identifying the relevant chunk)
LABELED_QUERIES: List[Tuple[str, str, str, str, str]] = [
    ("Q01", "symbol", "What is the W^{DKV} latent compression matrix in MLA?", "deepseek_2024_v2", "W^{DKV}"),
    ("Q02", "symbol", "Which RoPE base frequency of 500,000 does Llama 3 use?", "dubey_2024_llama3", "500,000"),
    ("Q03", "acronym", "Which models use the SwiGLU activation function?", "touvron_2023_llama1", "SwiGLU"),
    ("Q04", "acronym", "How does GRPO estimate the baseline without a critic?", "deepseek_2025_r1", "(GRPO)"),
    ("Q05", "acronym", "auxiliary-loss-free load balancing of MoE experts", "deepseek_2024_v3", "auxiliary-loss-free"),
    ("Q06", "semantic", "How many tokens per parameter is compute-optimal for Chinchilla?", "hoffmann_2022_chinchilla", "20 tokens per parameter"),
    ("Q07", "semantic", "How does paging reduce KV cache memory waste in LLM serving?", "kwon_2023_pagedattention", "fixed-size blocks"),
    ("Q08", "semantic", "How can preference fine-tuning avoid fitting an explicit reward model?", "rafailov_2023_dpo", "binary cross-entropy"),
    ("Q09", "semantic", "How to extend the context window of rotary position models cheaply?", "peng_2023_yarn", "10x fewer tokens"),
    ("Q10", "semantic", "attention with an intermediate number of key-value heads", "ainslie_2023_gqa", "intermediate number of key-value heads"),
    # Zero-keyword paraphrases: no token in common with ANY chunk (asserted at runtime), so BM25 cannot match.
    ("Q11", "paraphrase", "an allocator borrowing ideas from os pages", "kwon_2023_pagedattention", "fixed-size blocks"),
    ("Q12", "paraphrase", "equal growth of dataset volume and weights when flops double", "hoffmann_2022_chinchilla", "scaled equally"),
]
MODES = ("Dense-Only", "BM25-Only", "Hybrid RRF", "RRF + Rerank")


@dataclass
class CaseResult:
    case_id: str
    title: str
    passed: bool
    lines: List[str] = field(default_factory=list)


def run_mode(v: InMemoryVectorStore, mode: str, q: str, k: int) -> List[RetrievedChunk]:
    if mode == "Dense-Only":
        return v.search_dense(q, top_k=k)
    if mode == "BM25-Only":
        return v.search_lexical_bm25(q, top_k=k)
    if mode == "Hybrid RRF":
        return v.search_hybrid_rrf(q, top_k=k)
    return v.search_hybrid_reranked(q, top_k=k, candidate_k=15)


# ---------------------------------------------------------------------------
# Bucket 1: shared_key_integrity
# ---------------------------------------------------------------------------


def bucket_shared_key(c: OfflineCorpus) -> List[CaseResult]:
    g, v = c.graph, c.vectors
    edge_chunk_ids = sorted({cid for (s, rt, t), p in g.rels.items() if rt not in ("IN_PAPER", "MENTIONED_IN")
                             for cid in p.get("source_chunk_ids", [])})
    fetched = {r.chunk_id for r in v.fetch_chunks_by_ids(edge_chunk_ids)}
    sk1 = CaseResult("SK-01", "100% of graph edge source_chunk_ids resolve via fetch_chunks_by_ids()",
                     bool(edge_chunk_ids) and fetched == set(edge_chunk_ids),
                     [f"{len(fetched)}/{len(edge_chunk_ids)} edge chunk ids fetched"])

    all_eids = {e for r in v.rows.values() for e in r.entity_ids}
    missing = sorted(e for e in all_eids if e not in g.nodes)
    tagged = sum(1 for r in v.rows.values() if r.entity_ids)
    sk2 = CaseResult("SK-02", "100% of chunk entity_ids exist as graph nodes", bool(all_eids) and not missing,
                     [f"{len(all_eids) - len(missing)}/{len(all_eids)} distinct entity_ids found; {tagged}/{v.count()} chunks tagged",
                      f"missing={missing[:5]}"] if missing else
                     [f"{len(all_eids)}/{len(all_eids)} distinct entity_ids found; {tagged}/{v.count()} chunks tagged"])

    graph_chunks = {cid: n for cid, n in g.nodes.items() if "Chunk" in n["labels"]}
    hash_ok = all(graph_chunks.get(r.chunk_id, {}).get("content_hash") == r.content_hash for r in v.rows.values())
    sk3 = CaseResult("SK-03", "Vector chunk_ids == graph (:Chunk) ids, content_hash identical",
                     set(graph_chunks) == set(v.rows) and hash_ok,
                     [f"vector rows={v.count()} graph chunks={len(graph_chunks)} content_hash match={hash_ok}"])

    before_count, before_calls = v.count(), c.embedder.calls
    snapshot = {cid: (r, v.vectors[cid].tobytes()) for cid, r in v.rows.items()}
    embedded = [v.index_chunks(c.report.chunks, c.report.mentions) for _ in range(2)]
    same = all(snapshot[cid] == (r, v.vectors[cid].tobytes()) for cid, r in v.rows.items())
    sk4 = CaseResult("SK-04", "Re-indexing the same chunks twice is idempotent (no new rows, no re-embedding)",
                     v.count() == before_count and embedded == [0, 0] and c.embedder.calls == before_calls and same,
                     [f"rows {before_count} -> {v.count()}; chunks re-embedded per pass={embedded}; rows/vectors identical={same}"])
    return [sk1, sk2, sk3, sk4]


# ---------------------------------------------------------------------------
# Bucket 2: retrieval_ablation_recall_at_k
# ---------------------------------------------------------------------------


def bucket_retrieval(c: OfflineCorpus) -> Tuple[List[CaseResult], Dict[str, Dict[str, List[float]]]]:
    v = c.vectors
    table: Dict[str, Dict[str, List[float]]] = {m: {"r1": [], "r3": [], "kind": []} for m in MODES}
    results = []
    for qid, kind, q, paper, gold in LABELED_QUERIES:
        relevant = {r.chunk_id for r in v.rows.values() if r.paper_id == paper and gold in r.context_prefixed_text}
        if kind == "paraphrase":
            vocab = {t for r in v.rows.values() for t in tokenize(r.context_prefixed_text)}
            assert not set(tokenize(q)) & vocab, f"{qid} shares tokens with the corpus: {set(tokenize(q)) & vocab}"
        ranks = {}
        for m in MODES:
            hits = [h.chunk_id for h in run_mode(v, m, q, 3)]
            r1 = len(relevant & set(hits[:1])) / len(relevant) if relevant else 0.0
            r3 = len(relevant & set(hits[:3])) / len(relevant) if relevant else 0.0
            table[m]["r1"].append(r1); table[m]["r3"].append(r3); table[m]["kind"].append(kind)
            ranks[m] = next((i + 1 for i, h in enumerate(hits) if h in relevant), None)
        passed = bool(relevant) and ranks["RRF + Rerank"] is not None and ranks["RRF + Rerank"] <= 3
        shown = "  ".join(f"{m}={ranks[m] or '>3'}" for m in MODES)
        results.append(CaseResult(qid, f"[{kind}] {q}", passed, [f"relevant={sorted(relevant)} | rank of relevant: {shown}"]))
    return results, table


def print_ablation(table: Dict[str, Dict[str, List[float]]]) -> None:
    def avg(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n RETRIEVAL ABLATION ({len(LABELED_QUERIES)} labeled queries incl. 2 zero-keyword paraphrases, 12 seed papers)")
    print(LINE)
    print(f" {'Mode':<16}{'R@1 all':>10}{'R@3 all':>10}{'R@1 symbol/acro':>18}{'R@3 symbol/acro':>18}{'R@1 sem+para':>15}{'R@3 sem+para':>15}")
    print(SUBLINE)
    for m in MODES:
        t = table[m]
        sym = [i for i, k in enumerate(t["kind"]) if k in ("symbol", "acronym")]
        sem = [i for i, k in enumerate(t["kind"]) if k in ("semantic", "paraphrase")]
        print(f" {m:<16}{avg(t['r1']):>10.2f}{avg(t['r3']):>10.2f}"
              f"{avg([t['r1'][i] for i in sym]):>18.2f}{avg([t['r3'][i] for i in sym]):>18.2f}"
              f"{avg([t['r1'][i] for i in sem]):>15.2f}{avg([t['r3'][i] for i in sem]):>15.2f}")
    print(LINE)


# ---------------------------------------------------------------------------
# Bucket 3: bidirectional_bridge_hops
# ---------------------------------------------------------------------------


def bucket_bridges(c: OfflineCorpus) -> List[CaseResult]:
    g, v = c.graph, c.vectors
    paths = [p for p in g.find_paths(MLA, MHA, "EXTENDS", 2) if p[0]["target"] == GQA]
    hop_ids = [cid for hop in (paths[0] if paths else []) for cid in hop["source_chunk_ids"]]
    fetched = v.fetch_chunks_by_ids(hop_ids)
    quotes_ok = bool(paths) and all(
        any(q in r.raw_text for q in hop["evidence_quotes"])
        for hop in paths[0] for r in fetched if r.chunk_id in hop["source_chunk_ids"])
    br1 = CaseResult("BR-01", "Graph -> Text: 2-hop MLA -> GQA -> MHA, fetch every hop's verbatim chunk",
                     bool(paths) and len(fetched) == len(hop_ids) == 2 and quotes_ok,
                     [f"path={'found' if paths else 'MISSING'}; hop chunks={hop_ids}; fetched={len(fetched)}; evidence quote in text={quotes_ok}"])

    phase1_text = {ch.chunk_id: ch.raw_text for ch in c.report.chunks}
    verbatim = bool(fetched) and all(r.raw_text == phase1_text[r.chunk_id] == g.get_chunk_text(r.chunk_id) for r in fetched)
    order_ok = [r.chunk_id for r in v.fetch_chunks_by_ids(list(reversed(hop_ids)) + ["no::such::chunk"])] == list(reversed(hop_ids))
    br2 = CaseResult("BR-02", "Graph -> Text returns Phase 1 text byte-identical, in request order, skipping unknown ids",
                     verbatim and order_ok, [f"verbatim(vector == phase1 == graph)={verbatim}; order preserved, unknown skipped={order_ok}"])

    def vector_to_graph(query: str, must_reach: str) -> Tuple[bool, List[str]]:
        top = v.search_hybrid_reranked(query, top_k=1)[0]
        neighbors = [n for eid in top.entity_ids for n in g.neighbors(eid)]
        reached = {n["neighbor"] for n in neighbors}
        shown = sorted({f"{n['source'].split('::')[1]} -[{n['relation_type']}]-> {n['target'].split('::')[1]}" for n in neighbors})
        return must_reach in reached, [f"top chunk={top.chunk_id} entity_ids={list(top.entity_ids)}",
                                       f"1-hop: {shown[:6]}{' ...' if len(shown) > 6 else ''}"]

    ok3, l3 = vector_to_graph("How does PagedAttention reduce KV cache memory waste?", "bottleneck::kv_cache_memory_bandwidth")
    br3 = CaseResult("BR-03", "Vector -> Graph: PagedAttention chunk -> entity_ids -> 1 hop reaches the KV-cache bottleneck", ok3, l3)
    ok4, l4 = vector_to_graph("What is the W^{DKV} latent compression matrix in MLA?", GQA)
    br4 = CaseResult("BR-04", "Vector -> Graph: W^{DKV} chunk -> MLA -> 1 hop reaches GQA (EXTENDS)", ok4, l4)
    return [br1, br2, br3, br4]


# ---------------------------------------------------------------------------
# Bucket 4: hnsw_sql_and_filter_contract
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, log): self.log, self.description = log, None
    def __enter__(self): return self
    def __exit__(self, *a): return None
    def execute(self, sql, params=None): self.log.append((sql, params))
    def fetchall(self): return []


class _Txn:
    def __init__(self, log): self.log = log
    def __enter__(self): self.log.append(("BEGIN", None)); return self
    def __exit__(self, *a): self.log.append(("COMMIT", None))


class RecordingConnection:
    """psycopg-like connection that records (sql, params) per transaction."""

    def __init__(self): self.log: List[Tuple[str, Any]] = []
    def transaction(self): return _Txn(self.log)
    def cursor(self): return _Cursor(self.log)
    def close(self): pass


def bucket_sql(c: OfflineCorpus) -> List[CaseResult]:
    v = c.vectors
    q = "How does preference optimization work?"
    post = v.search_hybrid_reranked(q, top_k=5, filters={"pillar": "post_training"})
    sq1 = CaseResult("SQ-01", "pillar filter: every hit (all 4 modes) is post_training",
                     bool(post) and all(r.pillar == "post_training" for m in MODES
                                        for r in run_mode_f(v, m, q, {"pillar": "post_training"})),
                     [f"reranked hits={[r.paper_id for r in post]}"])
    f2 = {"min_year": 2024, "paper_ids": ["deepseek_2024_v2", "deepseek_2024_v3", "ainslie_2023_gqa"]}
    hits = [r for m in MODES for r in run_mode_f(v, m, "multi-head latent attention KV cache", f2)]
    sq2 = CaseResult("SQ-02", "min_year + paper_ids filters compose (ainslie 2023 excluded by min_year)",
                     bool(hits) and all(r.pub_year >= 2024 and r.paper_id in f2["paper_ids"] for r in hits)
                     and not any(r.paper_id == "ainslie_2023_gqa" for r in hits),
                     [f"papers hit={sorted({r.paper_id for r in hits})}"])

    ddl = "\n".join(schema_statements(768))
    ddl_ok = all(s in ddl for s in ("vector(768)", "USING hnsw (embedding vector_cosine_ops)", "m = 16",
                                    "ef_construction = 128", "USING gin (tsv)", "entity_ids            TEXT[]",
                                    "chunk_id              TEXT PRIMARY KEY", "to_tsvector('english'"))
    conn = RecordingConnection()
    pg = PgVectorStore(connection=conn, embedder=DeterministicMockEmbedder())
    pg.init_schema()
    pg.index_chunks(c.report.chunks[:2], c.report.mentions)
    upserts = [(s, p) for s, p in conn.log if s.strip().startswith("INSERT INTO chunks")]
    upsert_ok = len(upserts) == 2 and all("ON CONFLICT (chunk_id) DO UPDATE" in s and "%(embedding)s::vector" in s
                                          and c.report.chunks[0].raw_text not in s for s, _ in upserts)
    sq3 = CaseResult("SQ-03", "DDL: vector(768), HNSW vector_cosine_ops m=16/ef_construction=128, GIN tsv; upsert ON CONFLICT",
                     ddl_ok and upsert_ok, [f"ddl={ddl_ok}; {len(upserts)} parameterized upserts ON CONFLICT (chunk_id)={upsert_ok}"])

    conn.log.clear()
    sneaky = "x'); DROP TABLE chunks; --"
    pg.search_dense(sneaky, top_k=3, ef_search=80, filters={"pillar": "architecture"})
    pg.search_hybrid_rrf(sneaky, top_k=3, rrf_k=60, ef_search=120, filters={"min_year": 2023, "paper_ids": ["a"]})
    pg.search_lexical_bm25(sneaky, top_k=3)
    sqls = [s for s, _ in conn.log]
    set_local = [i for i, s in enumerate(sqls) if s.startswith("SET LOCAL hnsw.ef_search")]
    in_txn = all(sqls[i - 1] == "BEGIN" and sqls[i + 1].lstrip().startswith(("SELECT", "WITH")) for i in set_local)
    hybrid = next((s for s in sqls if s.lstrip().startswith("WITH q AS")), "")
    contract = (sqls.count("SET LOCAL hnsw.ef_search = 80") == 1 and "SET LOCAL hnsw.ef_search = 120" in sqls and in_txn
                and "ts_rank_cd" in hybrid and "FULL OUTER JOIN" in hybrid and "%(rrf_k)s" in hybrid
                and "<=> %(qvec)s::vector" in hybrid and "c.paper_id = ANY(%(f_paper_ids)s)" in hybrid
                and not any(sneaky in s for s in sqls))
    try:
        pg.search_dense("q", ef_search=10**6)
        rejects = False
    except ValueError:
        rejects = True
    sq4 = CaseResult("SQ-04", "Query SQL: SET LOCAL hnsw.ef_search inside the txn, ts_rank_cd + RRF fusion, all values bound",
                     contract and rejects,
                     [f"SET LOCAL in same txn before SELECT={in_txn}; RRF SQL uses ts_rank_cd/FULL OUTER JOIN/%(rrf_k)s={'ts_rank_cd' in hybrid}",
                      f"query text never interpolated={not any(sneaky in s for s in sqls)}; ef_search out of range rejected={rejects}"])
    return [sq1, sq2, sq3, sq4]


def run_mode_f(v: InMemoryVectorStore, mode: str, q: str, filters: Dict[str, Any]) -> List[RetrievedChunk]:
    if mode == "Dense-Only":
        return v.search_dense(q, top_k=5, filters=filters)
    if mode == "BM25-Only":
        return v.search_lexical_bm25(q, top_k=5, filters=filters)
    if mode == "Hybrid RRF":
        return v.search_hybrid_rrf(q, top_k=5, filters=filters)
    return v.search_hybrid_reranked(q, top_k=5, filters=filters)


def main() -> int:
    t0 = time.perf_counter()
    corpus = build_offline_corpus()
    build_s = time.perf_counter() - t0
    buckets: Dict[str, List[CaseResult]] = {}
    buckets["shared_key_integrity"] = bucket_shared_key(corpus)
    buckets["retrieval_ablation_recall_at_k"], table = bucket_retrieval(corpus)
    buckets["bidirectional_bridge_hops"] = bucket_bridges(corpus)
    buckets["hnsw_sql_and_filter_contract"] = bucket_sql(corpus)
    total_s = time.perf_counter() - t0

    print(LINE)
    print(" PHASE 2 SCORECARD")
    print(f" Corpus: {len(corpus.report.papers)} seed papers -> {corpus.vectors.count()} chunks | "
          f"graph {corpus.graph.count_nodes()} nodes / {corpus.graph.count_relationships()} rels | "
          f"build {build_s:.2f}s, all buckets {total_s:.2f}s")
    print(" Mode  : offline | InMemoryVectorStore + DeterministicMockEmbedder | recording psycopg connection for SQL")
    print(LINE)
    print(f" {'Bucket':<36}{'Passed / Total':>16}{'Score':>12}")
    print(SUBLINE)
    tp = tt = 0
    for name, rs in buckets.items():
        p, t = sum(r.passed for r in rs), len(rs)
        tp, tt = tp + p, tt + t
        print(f" {name:<36}{f'{p} / {t}':>16}{100.0 * p / t:>11.1f}%")
    print(SUBLINE)
    print(f" {'OVERALL PHASE 2 SCORE':<36}{f'{tp} / {tt}':>16}{100.0 * tp / tt:>11.1f}%")
    print("   (retrieval cases pass when RRF + Rerank ranks the relevant chunk in the top 3)")
    print(LINE)
    print_ablation(table)
    print("\n CASE DETAILS")
    print(LINE)
    for rs in buckets.values():
        for r in rs:
            print(f" [{'PASS' if r.passed else 'FAIL'}] {r.case_id}: {r.title}")
            for line in r.lines:
                print(f"        {line}")
    print(LINE)
    return 0 if tp == tt else 1


if __name__ == "__main__":
    sys.exit(main())
