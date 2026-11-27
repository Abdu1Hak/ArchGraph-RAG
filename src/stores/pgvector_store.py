"""Shared-Key Vector & Hybrid Lexical Index (Phase 2): pgvector + BM25/ts_rank_cd + RRF + Stage-2 rerank.

Shared key: every row is keyed by the Phase 1 `chunk_id`, the same string stored in Neo4j
`r.source_chunk_ids` and `(:Chunk {chunk_id})`. Graph -> Text is `fetch_chunks_by_ids()`;
Vector -> Graph is a hit's `entity_ids` (resolved Phase 1 entity ids) fed back into the graph.

Two stores, one interface:
- `PgVectorStore`: PostgreSQL 16 + pgvector (HNSW `vector_cosine_ops`, m=16, ef_construction=128),
  `tsvector` GIN for lexical `ts_rank_cd`, RRF fused in a single SQL statement.
- `InMemoryVectorStore`: exact numpy cosine + Okapi BM25 + the same RRF, for offline runs.

Retrieval modes: dense, lexical (BM25), hybrid RRF, and hybrid RRF + Stage-2 reranking of the
top `candidate_k` with a pluggable `CrossEncoderReranker`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.ingestion.chunker import DocumentChunk
from src.stores.embedders import DeterministicMockEmbedder, Embedder, GeminiEmbedder  # noqa: F401 (re-export)
from src.stores.reranker import (  # noqa: F401 (re-export)
    CrossEncoderReranker,
    GeminiApiReranker,
    OnnxCrossEncoderReranker,
    TokenInteractionReranker,
    create_reranker,
    symbol_tokens,
    tokenize,
)

DEFAULT_RRF_K = 60


def embedder_id(embedder: Any) -> str:
    """Stable identity of the embedding function (vectors from different embedders are not comparable)."""
    return getattr(embedder, "name", None) or f"{type(embedder).__name__}@{getattr(embedder, 'dim', '?')}"


def _norm_for_lookup(text: str) -> str:
    t = re.sub(r"[-_\u2013\u2014]", " ", text.lower())
    return re.sub(r"\s+", " ", t)


@dataclass(frozen=True)
class SearchFilters:
    pillar: Optional[str] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    paper_ids: Optional[Tuple[str, ...]] = None

    @classmethod
    def coerce(cls, f: Union["SearchFilters", Mapping[str, Any], None]) -> "SearchFilters":
        if f is None:
            return cls()
        if isinstance(f, SearchFilters):
            return f
        ids = f.get("paper_ids")
        return cls(pillar=f.get("pillar"), min_year=f.get("min_year"), max_year=f.get("max_year"),
                   paper_ids=tuple(ids) if ids else None)

    def matches(self, row: "RetrievedChunk") -> bool:
        return ((self.pillar is None or row.pillar == self.pillar)
                and (self.min_year is None or row.pub_year >= self.min_year)
                and (self.max_year is None or row.pub_year <= self.max_year)
                and (self.paper_ids is None or row.paper_id in self.paper_ids))


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    pub_year: int
    pillar: str
    section_path: str
    raw_text: str
    context_prefixed_text: str
    content_hash: str
    entity_ids: Tuple[str, ...] = ()
    score: float = 0.0
    rank: int = 0
    mode: str = ""
    stage_scores: Dict[str, Any] = field(default_factory=dict)


def _row_from_chunk(c: DocumentChunk, entity_ids: Sequence[str]) -> RetrievedChunk:
    return RetrievedChunk(c.chunk_id, c.paper_id, c.paper_title, c.pub_year, c.pillar, c.section_path,
                          c.raw_text, c.context_prefixed_text, c.content_hash, tuple(sorted(set(entity_ids))))


def _check_params(top_k: int, ef_search: int, rrf_k: int = DEFAULT_RRF_K) -> None:
    if not (1 <= int(top_k) <= 1000):
        raise ValueError(f"top_k must be in [1, 1000], got {top_k}")
    if not (1 <= int(ef_search) <= 1000):
        raise ValueError(f"ef_search must be in [1, 1000], got {ef_search}")
    if int(rrf_k) < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")


def rrf_fuse(rankings: Sequence[Sequence[str]], rrf_k: int = DEFAULT_RRF_K) -> List[Tuple[str, float, List[Optional[int]]]]:
    """Reciprocal Rank Fusion: score(d) = sum_i 1 / (rrf_k + rank_i(d)); ties broken by chunk_id."""
    scores: Dict[str, float] = {}
    ranks: Dict[str, List[Optional[int]]] = {}
    for i, ranking in enumerate(rankings):
        for r, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r)
            ranks.setdefault(cid, [None] * len(rankings))[i] = r
    return [(cid, s, ranks[cid]) for cid, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _VectorStoreBase:
    embedder: Embedder

    def search_hybrid_rrf(self, query_text: str, top_k: int = 5, rrf_k: int = DEFAULT_RRF_K, ef_search: int = 80,
                          filters: Any = None) -> List[RetrievedChunk]:
        raise NotImplementedError

    def search_hybrid_reranked(
        self,
        query_text: str,
        top_k: int = 5,
        candidate_k: int = 15,
        rrf_k: int = DEFAULT_RRF_K,
        ef_search: int = 80,
        filters: Any = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> List[RetrievedChunk]:
        """Stage 1: hybrid RRF top `candidate_k`. Stage 2: rerank (query, context_prefixed_text) pairs."""
        if candidate_k < top_k:
            raise ValueError("candidate_k must be >= top_k")
        reranker = reranker or getattr(self, "default_reranker", None) or TokenInteractionReranker()
        cands = self.search_hybrid_rrf(query_text, top_k=candidate_k, rrf_k=rrf_k, ef_search=ef_search, filters=filters)
        scores = reranker.score(query_text, [c.context_prefixed_text for c in cands])
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i].rank))
        return [
            replace(cands[i], score=float(scores[i]), rank=r, mode=f"hybrid_rrf+{reranker.name}",
                    stage_scores={**cands[i].stage_scores, "stage1_rank": cands[i].rank, "stage1_rrf": cands[i].score})
            for r, i in enumerate(order[:top_k], start=1)
        ]


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class InMemoryVectorStore(_VectorStoreBase):
    """Exact cosine + Okapi BM25 + RRF over an in-process dict. Same interface as `PgVectorStore`."""

    def __init__(self, embedder: Optional[Embedder] = None, bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 default_reranker: Optional[CrossEncoderReranker] = None) -> None:
        self.embedder = embedder or DeterministicMockEmbedder()
        self.default_reranker = default_reranker
        self.rows: Dict[str, RetrievedChunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.k1, self.b = bm25_k1, bm25_b
        self._bm25: Optional[Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, int], float]] = None
        self._norm_text: Optional[Dict[str, str]] = None
        self.row_embedder: Dict[str, str] = {}

    def init_schema(self) -> None:
        return None

    def count(self) -> int:
        return len(self.rows)

    def index_chunks(self, chunks: Sequence[DocumentChunk], entity_ids_by_chunk: Optional[Mapping[str, Sequence[str]]] = None) -> int:
        """Upsert by chunk_id; re-embeds only chunks whose content_hash changed. Returns #embedded."""
        ents = entity_ids_by_chunk or {}
        model = embedder_id(self.embedder)
        changed = [c for c in chunks if c.chunk_id not in self.rows or self.rows[c.chunk_id].content_hash != c.content_hash
                   or self.row_embedder.get(c.chunk_id) != model]
        vecs = self.embedder.embed_documents([c.context_prefixed_text for c in changed], [c.content_hash for c in changed]) if changed else []
        for c, v in zip(changed, vecs):
            self.vectors[c.chunk_id] = np.asarray(v, dtype=np.float32)
            self.row_embedder[c.chunk_id] = model
        for c in chunks:
            self.rows[c.chunk_id] = _row_from_chunk(c, ents.get(c.chunk_id, ()))
        self._bm25 = None
        self._norm_text = None
        return len(changed)

    def _candidates(self, filters: Any) -> List[RetrievedChunk]:
        f = SearchFilters.coerce(filters)
        return [r for r in self.rows.values() if f.matches(r)]

    def _bm25_stats(self):
        if self._bm25 is None:
            tfs = {cid: {} for cid in self.rows}
            lengths, df = {}, {}
            for cid, row in self.rows.items():
                toks = tokenize(row.context_prefixed_text)
                lengths[cid] = len(toks)
                for t in toks:
                    tfs[cid][t] = tfs[cid].get(t, 0) + 1
                for t in tfs[cid]:
                    df[t] = df.get(t, 0) + 1
            avg = sum(lengths.values()) / max(len(lengths), 1)
            self._bm25 = (tfs, lengths, df, avg)
        return self._bm25

    def _dense_ranked(self, query_text: str, filters: Any) -> List[Tuple[RetrievedChunk, float]]:
        q = np.asarray(self.embedder.embed_query(query_text), dtype=np.float32)
        scored = [(r, float(np.dot(q, self.vectors[r.chunk_id]))) for r in self._candidates(filters)]
        return sorted(scored, key=lambda x: (-x[1], x[0].chunk_id))

    def _lexical_ranked(self, query_text: str, filters: Any) -> List[Tuple[RetrievedChunk, float]]:
        tfs, lengths, df, avg = self._bm25_stats()
        n = len(self.rows)
        q_terms = list(dict.fromkeys(tokenize(query_text)))
        scored = []
        for r in self._candidates(filters):
            tf, s = tfs[r.chunk_id], 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * lengths[r.chunk_id] / avg))
            if s > 0:
                scored.append((r, s))
        return sorted(scored, key=lambda x: (-x[1], x[0].chunk_id))

    @staticmethod
    def _emit(scored: List[Tuple[RetrievedChunk, float]], top_k: int, mode: str) -> List[RetrievedChunk]:
        return [replace(r, score=s, rank=i, mode=mode) for i, (r, s) in enumerate(scored[:top_k], start=1)]

    def search_dense(self, query_text: str, top_k: int = 5, ef_search: int = 80, filters: Any = None) -> List[RetrievedChunk]:
        _check_params(top_k, ef_search)
        return self._emit(self._dense_ranked(query_text, filters), top_k, "dense")

    def search_lexical_bm25(self, query_text: str, top_k: int = 5, filters: Any = None) -> List[RetrievedChunk]:
        _check_params(top_k, 80)
        return self._emit(self._lexical_ranked(query_text, filters), top_k, "bm25")

    def search_hybrid_rrf(self, query_text: str, top_k: int = 5, rrf_k: int = DEFAULT_RRF_K, ef_search: int = 80,
                          filters: Any = None) -> List[RetrievedChunk]:
        _check_params(top_k, ef_search, rrf_k)
        depth = max(top_k * 4, 20)
        dense = [r.chunk_id for r, _ in self._dense_ranked(query_text, filters)[:depth]]
        lexical = [r.chunk_id for r, _ in self._lexical_ranked(query_text, filters)[:depth]]
        return [
            replace(self.rows[cid], score=s, rank=i, mode="hybrid_rrf",
                    stage_scores={"dense_rank": ranks[0], "lexical_rank": ranks[1]})
            for i, (cid, s, ranks) in enumerate(rrf_fuse([dense, lexical], rrf_k)[:top_k], start=1)
        ]

    def fetch_chunks_by_ids(self, chunk_ids: Sequence[str]) -> List[RetrievedChunk]:
        """Graph -> Text bridge: rows in the requested order; unknown ids are skipped."""
        return [self.rows[c] for c in dict.fromkeys(chunk_ids) if c in self.rows]

    def text_exists(self, normalized_phrase: str) -> bool:
        """Whole-token phrase lookup over all indexed chunk text (lower-case, hyphens/underscores as spaces)."""
        if self._norm_text is None or len(self._norm_text) != len(self.rows):
            self._norm_text = {cid: _norm_for_lookup(r.context_prefixed_text) for cid, r in self.rows.items()}
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])")
        return any(pat.search(t) for t in self._norm_text.values())


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector store
# ---------------------------------------------------------------------------

_COLUMNS = ("chunk_id", "paper_id", "paper_title", "pub_year", "pillar", "section_path", "raw_text",
            "context_prefixed_text", "content_hash", "entity_ids")
_SELECT_COLS = ", ".join(f"c.{c}" for c in _COLUMNS)
# OR-semantics tsquery: plainto_tsquery ANDs every word, which returns nothing for natural-language questions.
_TSQUERY = "to_tsquery('english', replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | '))"


def schema_statements(dim: int = 768) -> List[str]:
    dim = int(dim)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""CREATE TABLE IF NOT EXISTS chunks (
    chunk_id              TEXT PRIMARY KEY,
    paper_id              TEXT NOT NULL,
    paper_title           TEXT NOT NULL,
    pub_year              INTEGER NOT NULL,
    pillar                TEXT NOT NULL,
    section_path          TEXT NOT NULL,
    raw_text              TEXT NOT NULL,
    context_prefixed_text TEXT NOT NULL,
    content_hash          TEXT NOT NULL,
    entity_ids            TEXT[] NOT NULL DEFAULT '{{}}',
    embedding             vector({dim}) NOT NULL,
    embedding_model       TEXT NOT NULL DEFAULT '',
    tsv                   tsvector GENERATED ALWAYS AS (to_tsvector('english', context_prefixed_text)) STORED,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
)""",
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)",
        "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)",
        "CREATE INDEX IF NOT EXISTS chunks_entity_ids_gin ON chunks USING gin (entity_ids)",
        "CREATE INDEX IF NOT EXISTS chunks_paper_id ON chunks (paper_id)",
        "CREATE INDEX IF NOT EXISTS chunks_pillar_year ON chunks (pillar, pub_year)",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT ''",
    ]


_UPSERT_SQL = f"""
INSERT INTO chunks ({", ".join(_COLUMNS)}, embedding, embedding_model)
VALUES ({", ".join(f"%({c})s" for c in _COLUMNS)}, %(embedding)s::vector, %(embedding_model)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS if c != "chunk_id")},
    embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model, updated_at = now()
"""


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


def filter_sql(filters: Any) -> Tuple[str, Dict[str, Any]]:
    """Parameterized WHERE fragment (always starts with ' AND ' or is empty) and its params."""
    f = SearchFilters.coerce(filters)
    parts, params = [], {}
    if f.pillar is not None:
        parts.append("c.pillar = %(f_pillar)s"); params["f_pillar"] = f.pillar
    if f.min_year is not None:
        parts.append("c.pub_year >= %(f_min_year)s"); params["f_min_year"] = int(f.min_year)
    if f.max_year is not None:
        parts.append("c.pub_year <= %(f_max_year)s"); params["f_max_year"] = int(f.max_year)
    if f.paper_ids is not None:
        parts.append("c.paper_id = ANY(%(f_paper_ids)s)"); params["f_paper_ids"] = list(f.paper_ids)
    return ("".join(f" AND {p}" for p in parts), params)


class PgVectorStore(_VectorStoreBase):
    """psycopg v3 store. Pass `connection` (tests) or a DSN (defaults to settings.postgres_dsn)."""

    def __init__(self, connection: Any = None, dsn: Optional[str] = None, embedder: Optional[Embedder] = None,
                 dim: Optional[int] = None, default_reranker: Optional[CrossEncoderReranker] = None) -> None:
        if connection is None:
            import psycopg

            from src.config import settings

            connection = psycopg.connect(dsn or settings.postgres_dsn)
        self.conn = connection
        self.embedder = embedder or DeterministicMockEmbedder()
        self.default_reranker = default_reranker
        self.dim = int(dim or getattr(self.embedder, "dim", 768))

    def close(self) -> None:
        self.conn.close()

    def _run(self, statements: Sequence[Tuple[str, Optional[Dict[str, Any]]]], fetch: bool = False) -> List[Dict[str, Any]]:
        """Execute statements in one transaction (so SET LOCAL applies); return last result as dicts."""
        rows: List[Dict[str, Any]] = []
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                for sql, params in statements:
                    cur.execute(sql, params)
                if fetch and cur.description:
                    names = [d[0] for d in cur.description]
                    rows = [dict(zip(names, r)) for r in cur.fetchall()]
        return rows

    def init_schema(self) -> None:
        self._run([(s, None) for s in schema_statements(self.dim)])

    def count(self) -> int:
        rows = self._run([("SELECT count(*) AS n FROM chunks", None)], fetch=True)
        return int(rows[0]["n"]) if rows else 0

    def index_chunks(self, chunks: Sequence[DocumentChunk], entity_ids_by_chunk: Optional[Mapping[str, Sequence[str]]] = None) -> int:
        ents = entity_ids_by_chunk or {}
        model = embedder_id(self.embedder)
        existing = {r["chunk_id"]: (r["content_hash"], r["embedding_model"]) for r in self._run(
            [("SELECT chunk_id, content_hash, embedding_model FROM chunks WHERE chunk_id = ANY(%(ids)s)",
              {"ids": [c.chunk_id for c in chunks]})], fetch=True)}
        # Re-embed when the text changed OR the row was embedded by a different embedder (e.g. mock -> Gemini).
        changed = [c for c in chunks if existing.get(c.chunk_id) != (c.content_hash, model)]
        vecs = self.embedder.embed_documents([c.context_prefixed_text for c in changed], [c.content_hash for c in changed]) if changed else []
        stmts: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        for c, v in zip(changed, vecs):
            row = _row_from_chunk(c, ents.get(c.chunk_id, ()))
            params = {col: getattr(row, col) for col in _COLUMNS}
            params["entity_ids"] = list(row.entity_ids)
            params["embedding"] = _vec_literal(v)
            params["embedding_model"] = model
            stmts.append((_UPSERT_SQL, params))
        changed_ids = {c.chunk_id for c in changed}
        for c in chunks:
            if c.chunk_id not in changed_ids:
                stmts.append(("UPDATE chunks SET entity_ids = %(e)s WHERE chunk_id = %(id)s",
                              {"e": sorted(set(ents.get(c.chunk_id, ()))), "id": c.chunk_id}))
        if stmts:
            self._run(stmts)
        return len(changed)

    def count_other_embedders(self, model: Optional[str] = None) -> int:
        """Rows whose vectors came from a different embedder than `model` (default: this store's)."""
        rows = self._run([("SELECT count(*) AS n FROM chunks WHERE embedding_model <> %(m)s",
                           {"m": model or embedder_id(self.embedder)})], fetch=True)
        return int(rows[0]["n"]) if rows else 0

    def _to_rows(self, rows: List[Dict[str, Any]], mode: str, score_key: str = "score") -> List[RetrievedChunk]:
        out = []
        for i, r in enumerate(rows, start=1):
            extra = {k: r[k] for k in ("dense_rank", "lexical_rank") if k in r}
            out.append(RetrievedChunk(*(r[c] for c in _COLUMNS[:-1]), entity_ids=tuple(r["entity_ids"] or ()),
                                      score=float(r[score_key]), rank=i, mode=mode, stage_scores=extra))
        return out

    def dense_statements(self, query_vec: np.ndarray, top_k: int, ef_search: int, filters: Any) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        _check_params(top_k, ef_search)
        fsql, params = filter_sql(filters)
        params.update({"qvec": _vec_literal(query_vec), "k": int(top_k)})
        return [
            # SET cannot take bind parameters; ef_search is validated as an int in [1, 1000] above.
            (f"SET LOCAL hnsw.ef_search = {int(ef_search)}", None),
            (f"SELECT {_SELECT_COLS}, 1 - (c.embedding <=> %(qvec)s::vector) AS score FROM chunks c "
             f"WHERE TRUE{fsql} ORDER BY c.embedding <=> %(qvec)s::vector, c.chunk_id LIMIT %(k)s", params),
        ]

    def lexical_statement(self, query_text: str, top_k: int, filters: Any) -> Tuple[str, Dict[str, Any]]:
        _check_params(top_k, 80)
        fsql, params = filter_sql(filters)
        params.update({"q": query_text, "k": int(top_k)})
        return (f"SELECT {_SELECT_COLS}, ts_rank_cd(c.tsv, q.tsq) AS score FROM chunks c, (SELECT {_TSQUERY} AS tsq) q "
                f"WHERE c.tsv @@ q.tsq{fsql} ORDER BY score DESC, c.chunk_id LIMIT %(k)s", params)

    def hybrid_statements(self, query_text: str, query_vec: np.ndarray, top_k: int, rrf_k: int, ef_search: int,
                          filters: Any) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        _check_params(top_k, ef_search, rrf_k)
        fsql, params = filter_sql(filters)
        params.update({"q": query_text, "qvec": _vec_literal(query_vec), "k": int(top_k),
                       "depth": max(int(top_k) * 4, 20), "rrf_k": int(rrf_k)})
        sql = f"""
WITH q AS (SELECT {_TSQUERY} AS tsq),
dense AS (
    SELECT chunk_id, row_number() OVER (ORDER BY dist, chunk_id) AS r FROM (
        SELECT c.chunk_id, c.embedding <=> %(qvec)s::vector AS dist FROM chunks c
        WHERE TRUE{fsql} ORDER BY dist LIMIT %(depth)s) d
),
lex AS (
    SELECT chunk_id, row_number() OVER (ORDER BY rank_cd DESC, chunk_id) AS r FROM (
        SELECT c.chunk_id, ts_rank_cd(c.tsv, q.tsq) AS rank_cd FROM chunks c, q
        WHERE c.tsv @@ q.tsq{fsql} ORDER BY rank_cd DESC LIMIT %(depth)s) l
),
fused AS (
    SELECT coalesce(d.chunk_id, l.chunk_id) AS chunk_id,
           coalesce(1.0 / (%(rrf_k)s + d.r), 0) + coalesce(1.0 / (%(rrf_k)s + l.r), 0) AS score,
           d.r AS dense_rank, l.r AS lexical_rank
    FROM dense d FULL OUTER JOIN lex l ON d.chunk_id = l.chunk_id
)
SELECT {_SELECT_COLS}, f.score, f.dense_rank, f.lexical_rank
FROM fused f JOIN chunks c ON c.chunk_id = f.chunk_id
ORDER BY f.score DESC, c.chunk_id LIMIT %(k)s"""
        return [(f"SET LOCAL hnsw.ef_search = {int(ef_search)}", None), (sql, params)]

    def search_dense(self, query_text: str, top_k: int = 5, ef_search: int = 80, filters: Any = None) -> List[RetrievedChunk]:
        stmts = self.dense_statements(self.embedder.embed_query(query_text), top_k, ef_search, filters)
        return self._to_rows(self._run(stmts, fetch=True), "dense")

    def search_lexical_bm25(self, query_text: str, top_k: int = 5, filters: Any = None) -> List[RetrievedChunk]:
        return self._to_rows(self._run([self.lexical_statement(query_text, top_k, filters)], fetch=True), "bm25")

    def search_hybrid_rrf(self, query_text: str, top_k: int = 5, rrf_k: int = DEFAULT_RRF_K, ef_search: int = 80,
                          filters: Any = None) -> List[RetrievedChunk]:
        stmts = self.hybrid_statements(query_text, self.embedder.embed_query(query_text), top_k, rrf_k, ef_search, filters)
        return self._to_rows(self._run(stmts, fetch=True), "hybrid_rrf")

    def text_exists(self, normalized_phrase: str) -> bool:
        """Whole-token phrase lookup over all indexed chunk text, done in SQL (value bound as a parameter)."""
        pattern = "(^|[^a-z0-9])" + re.escape(normalized_phrase) + "([^a-z0-9]|$)"
        rows = self._run([("SELECT 1 AS hit FROM chunks WHERE regexp_replace(lower(context_prefixed_text), '[-_[:space:]]+', ' ', 'g') "
                           "~ %(pat)s LIMIT 1", {"pat": pattern})], fetch=True)
        return bool(rows)

    def fetch_chunks_by_ids(self, chunk_ids: Sequence[str]) -> List[RetrievedChunk]:
        ids = list(dict.fromkeys(chunk_ids))
        rows = self._run([(f"SELECT {_SELECT_COLS}, 0.0 AS score FROM chunks c WHERE c.chunk_id = ANY(%(ids)s)",
                           {"ids": ids})], fetch=True)
        by_id = {r.chunk_id: r for r in self._to_rows(rows, "fetch")}
        return [by_id[c] for c in ids if c in by_id]
