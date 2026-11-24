"""Content-Addressed Extraction Cache & Pre-Flight Cost Budget Guard (Phase 1B).

Design Decisions:
-----------------
1. Content-addressed cache key (`DocumentChunk.content_hash`):
   `sha256(schema_version :: prompt_version :: whitespace-normalized context_prefixed_text)`.
   Re-running ingestion over unchanged papers costs $0.00. Bumping
   `EXTRACTION_SCHEMA_VERSION` or `EXTRACTION_PROMPT_VERSION` (src/schema/ontology.py)
   changes every key, so stale extractions are never served after a prompt/schema change.

2. Budget guard runs BEFORE any API call:
   `CostBudgetEstimator.enforce_budget()` partitions chunks into cached ($0.00) and uncached,
   projects uncached cost from token heuristics, and raises `BudgetExceededError` if the
   projection exceeds the cap. Identical chunks (same content_hash) are billed once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from src.config import settings
from src.ingestion.chunker import DocumentChunk, estimate_tokens


class BudgetExceededError(RuntimeError):
    """Raised before any API call when the projected uncached extraction cost exceeds the cap."""

    def __init__(self, projected_usd: float, max_budget_usd: float, uncached_chunks: int) -> None:
        self.projected_usd = projected_usd
        self.max_budget_usd = max_budget_usd
        self.uncached_chunks = uncached_chunks
        super().__init__(
            f"Projected extraction cost ${projected_usd:.4f} for {uncached_chunks} uncached chunk(s) "
            f"exceeds budget ${max_budget_usd:.4f}; no API calls were made."
        )


@dataclass(frozen=True)
class CachedExtraction:
    """One cached extraction row."""

    content_hash: str
    chunk_id: str
    paper_id: str
    model: str
    payload_json: str
    input_tokens: int
    output_tokens: int
    created_at: str


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS extractions (
    content_hash  TEXT PRIMARY KEY,
    chunk_id      TEXT NOT NULL,
    paper_id      TEXT NOT NULL,
    model         TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_extractions_paper ON extractions(paper_id);
"""


class ExtractionCache:
    """SQLite-backed content-addressed cache of chunk extractions.

    Pass `":memory:"` for an ephemeral cache (tests, benchmarks).
    """

    def __init__(self, db_path: Union[str, Path, None] = None) -> None:
        path = settings.extraction_cache_path if db_path is None else db_path
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA_SQL)

    def get(self, content_hash: str) -> Optional[CachedExtraction]:
        row = self._conn.execute(
            "SELECT * FROM extractions WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return CachedExtraction(**dict(row)) if row else None

    def contains_many(self, content_hashes: Iterable[str]) -> set:
        """Return the subset of `content_hashes` present in the cache (chunked IN queries)."""
        hashes = list(dict.fromkeys(content_hashes))
        found: set = set()
        for i in range(0, len(hashes), 500):
            batch = hashes[i : i + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT content_hash FROM extractions WHERE content_hash IN ({placeholders})", batch
            ).fetchall()
            found.update(r["content_hash"] for r in rows)
        return found

    def put(
        self,
        content_hash: str,
        chunk_id: str,
        paper_id: str,
        model: str,
        payload_json: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO extractions
                    (content_hash, chunk_id, paper_id, model, payload_json, input_tokens, output_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    chunk_id=excluded.chunk_id, paper_id=excluded.paper_id, model=excluded.model,
                    payload_json=excluded.payload_json, input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens, created_at=excluded.created_at
                """,
                (
                    content_hash, chunk_id, paper_id, model, payload_json,
                    int(input_tokens), int(output_tokens), datetime.now(timezone.utc).isoformat(),
                ),
            )

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExtractionCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class CostEstimate:
    """Pre-flight projection for one extraction run."""

    cached_chunks: List[DocumentChunk] = field(default_factory=list)
    uncached_chunks: List[DocumentChunk] = field(default_factory=list)
    duplicate_chunks: int = 0
    projected_input_tokens: int = 0
    projected_output_tokens: int = 0
    projected_usd: float = 0.0
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0

    @property
    def total_chunks(self) -> int:
        return len(self.cached_chunks) + len(self.uncached_chunks) + self.duplicate_chunks

    def summary(self) -> str:
        return (
            f"chunks={self.total_chunks} cached={len(self.cached_chunks)} "
            f"uncached={len(self.uncached_chunks)} duplicates={self.duplicate_chunks} | "
            f"input~{self.projected_input_tokens:,} tok @ ${self.input_usd_per_1m}/1M, "
            f"output~{self.projected_output_tokens:,} tok @ ${self.output_usd_per_1m}/1M | "
            f"projected=${self.projected_usd:.4f}"
        )


class CostBudgetEstimator:
    """Partition chunks by cache status and project the USD cost of the uncached remainder."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        cache: ExtractionCache,
        input_usd_per_1m: Optional[float] = None,
        output_usd_per_1m: Optional[float] = None,
        prompt_overhead_tokens: Optional[int] = None,
        output_tokens_per_chunk: Optional[int] = None,
    ) -> None:
        self.chunks = list(chunks)
        self.cache = cache
        self.input_usd_per_1m = settings.extraction_input_usd_per_1m if input_usd_per_1m is None else input_usd_per_1m
        self.output_usd_per_1m = settings.extraction_output_usd_per_1m if output_usd_per_1m is None else output_usd_per_1m
        self.prompt_overhead_tokens = (
            settings.extraction_prompt_overhead_tokens if prompt_overhead_tokens is None else prompt_overhead_tokens
        )
        self.output_tokens_per_chunk = (
            settings.extraction_output_tokens_per_chunk if output_tokens_per_chunk is None else output_tokens_per_chunk
        )

    def estimate(self) -> CostEstimate:
        """Recompute the projection against the current cache contents."""
        cached_hashes = self.cache.contains_many(c.content_hash for c in self.chunks)
        est = CostEstimate(input_usd_per_1m=self.input_usd_per_1m, output_usd_per_1m=self.output_usd_per_1m)
        seen: Dict[str, bool] = {}
        for chunk in self.chunks:
            if chunk.content_hash in seen:
                est.duplicate_chunks += 1
                continue
            seen[chunk.content_hash] = True
            if chunk.content_hash in cached_hashes:
                est.cached_chunks.append(chunk)
                continue
            est.uncached_chunks.append(chunk)
            chunk_tokens = chunk.approx_token_count or estimate_tokens(chunk.context_prefixed_text)
            est.projected_input_tokens += chunk_tokens + self.prompt_overhead_tokens
            est.projected_output_tokens += self.output_tokens_per_chunk
        est.projected_usd = (
            est.projected_input_tokens * self.input_usd_per_1m / 1_000_000
            + est.projected_output_tokens * self.output_usd_per_1m / 1_000_000
        )
        return est

    def enforce_budget(self, max_budget_usd: Optional[float] = None) -> CostEstimate:
        """Return the estimate, or raise `BudgetExceededError` if it exceeds `max_budget_usd`."""
        cap = settings.max_extraction_budget_usd if max_budget_usd is None else max_budget_usd
        est = self.estimate()
        if est.projected_usd > cap:
            raise BudgetExceededError(est.projected_usd, cap, len(est.uncached_chunks))
        return est
