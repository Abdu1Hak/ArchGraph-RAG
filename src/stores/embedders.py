"""Embedders for the Phase 2 vector index: Gemini (cached) and a deterministic offline mock."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sqlite3
from typing import Dict, List, Optional, Protocol, Sequence, Union

import numpy as np

from src.config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str], content_hashes: Optional[Sequence[str]] = None) -> List[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def _stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


# Concept groups: a document's dense vector is dominated by which groups its words touch.
# Acronyms (SwiGLU, GRPO, MLA, DKV) and formula/number tokens are deliberately NOT in any
# group, so exact-symbol queries must be won by lexical search - as with real embedders,
# which blur rare identifiers.
CONCEPT_GROUPS: Dict[str, tuple] = {
    "attention_kv": ("attention", "head", "heads", "query", "key", "value", "kv", "cache", "caches",
                     "latent", "grouped", "compress", "compression", "compressing", "decoding", "decoder"),
    "positional": ("position", "positional", "rotary", "rotation", "embedding", "embeddings", "sinusoidal",
                   "relative", "absolute", "context", "window", "extend", "extension", "interpolation", "frequency"),
    "scaling": ("scaling", "scaled", "compute", "optimal", "tokens", "parameters", "budget", "model", "size",
                "training", "doubling", "trillion",
                # paraphrase synonyms a real embedder would know (absent from the corpus text)
                "flops", "dataset", "volume", "weights", "grow", "growth", "double"),
    "post_training": ("preference", "reward", "policy", "reinforcement", "learning", "alignment", "reasoning",
                      "critic", "baseline", "fine", "tuning", "supervised", "rlhf", "optimization"),
    "inference": ("serving", "memory", "throughput", "fragmentation", "waste",
                  "batching", "latency", "inference", "speed"),
    "virtual_memory": ("paging", "virtual", "operating", "contiguous", "blocks", "pages", "os", "allocator",
                       "noncontiguous"),
    "architecture_ffn": ("activation", "function", "feed", "forward", "network", "normalization", "layer",
                         "experts", "expert", "mixture", "routing", "balancing", "load"),
}
_WORD_RE = re.compile(r"[a-z]+")


class DeterministicMockEmbedder:
    """Deterministic 768-d unit vectors from concept-group hits plus a weak hashed bag of words."""

    def __init__(self, dim: int = 768, bow_weight: float = 0.35) -> None:
        self.dim = dim
        self.bow_weight = bow_weight
        self._basis: Dict[str, np.ndarray] = {}
        self._word_to_group = {w: g for g, words in CONCEPT_GROUPS.items() for w in words}
        self.calls = 0

    def _vec(self, name: str) -> np.ndarray:
        if name not in self._basis:
            self._basis[name] = _unit(np.random.default_rng(_stable_seed(name)).standard_normal(self.dim))
        return self._basis[name]

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            group = self._word_to_group.get(word)
            if group:
                v += self._vec(f"group::{group}")
            elif len(word) >= 4:
                v += self.bow_weight * self._vec(f"word::{word}")
        return _unit(v) if np.any(v) else self._vec("empty")

    def embed_documents(self, texts: Sequence[str], content_hashes: Optional[Sequence[str]] = None) -> List[np.ndarray]:
        self.calls += len(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class GeminiEmbedder:
    """`client.models.embed_content` with a SQLite cache keyed by (content_hash, model, dim, task)."""

    def __init__(
        self,
        client: object = None,
        model: Optional[str] = None,
        dim: Optional[int] = None,
        cache_path: Union[str, Path, None] = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self.batch_size = batch_size
        self._client = client
        path = cache_path if cache_path is not None else settings.cache_dir / "embeddings.sqlite"
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        with self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
        self.api_calls = 0

    @property
    def client(self) -> object:
        if self._client is None:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set; use DeterministicMockEmbedder offline.")
            from google import genai

            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def _key(self, content_hash: str, task: str) -> str:
        return f"{content_hash}::{self.model}::{self.dim}::{task}"

    def _call(self, texts: List[str], task: str) -> List[np.ndarray]:
        from google.genai import types

        self.api_calls += 1
        resp = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dim, task_type=task),
        )
        # Truncated (MRL) Gemini embeddings are not unit-norm; normalize for cosine search.
        return [_unit(np.asarray(e.values, dtype=np.float32)) for e in resp.embeddings]

    def _embed_cached(self, texts: Sequence[str], hashes: Sequence[str], task: str) -> List[np.ndarray]:
        out: List[Optional[np.ndarray]] = [None] * len(texts)
        missing: List[int] = []
        for i, h in enumerate(hashes):
            row = self._db.execute("SELECT vector FROM embeddings WHERE cache_key = ?", (self._key(h, task),)).fetchone()
            if row:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing.append(i)
        for start in range(0, len(missing), self.batch_size):
            idx = missing[start : start + self.batch_size]
            vecs = self._call([texts[i] for i in idx], task)
            with self._db:
                for i, v in zip(idx, vecs):
                    out[i] = v
                    self._db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (self._key(hashes[i], task), v.tobytes()))
        return [v for v in out if v is not None]

    def embed_documents(self, texts: Sequence[str], content_hashes: Optional[Sequence[str]] = None) -> List[np.ndarray]:
        hashes = list(content_hashes) if content_hashes else [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        return self._embed_cached(texts, hashes, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_cached([text], [hashlib.sha256(text.encode()).hexdigest()], "RETRIEVAL_QUERY")[0]
