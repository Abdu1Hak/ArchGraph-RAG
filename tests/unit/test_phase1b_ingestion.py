"""Unit tests for Phase 1B: cache & budget guard, arXiv fetcher, extractor, graph stores, pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

from src.ingestion.arxiv_fetcher import ArxivFetcher, PaperMeta, PdfDownloadError, load_manifest, load_or_synthesize_seed_markdown
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.cost_estimator import BudgetExceededError, CostBudgetEstimator, ExtractionCache
from src.ingestion.extractor import EXTRACTION_SYSTEM_PROMPT, GeminiGraphExtractor
from src.ingestion.offline_fixtures import SYNTHETIC_BANNER, RecordedExtractionLLM
from src.ingestion.pipeline import Phase1IngestionPipeline
from src.schema.ontology import (
    ChunkExtractionPayload,
    EntityType,
    ExtractionParseError,
    RelationType,
    ResolvedEntityNode,
    ResolvedGraphEdge,
    sanitize_and_filter_extraction,
)
from src.stores.neo4j_store import InMemoryGraphStore, Neo4jGraphStore, SCHEMA_STATEMENTS, aggregate_edges


VALID_REL = {
    "source_name": "Grouped-Query Attention", "source_type": "ArchComponent", "relation_type": "EXTENDS",
    "target_name": "Multi-Head Attention", "target_type": "ArchComponent",
    "evidence_quote": "Grouped-query attention extends multi-head attention.", "confidence": 0.9,
}


def _chunk(text: str = "Grouped-query attention extends multi-head attention.", paper_id: str = "p1"):
    return StructureAwareChunker().chunk_paper(paper_id, "Paper", 2024, "architecture", f"# M\n\n{text}\n")[0]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_seed_filter_and_t5_replacement() -> None:
    seeds, all_papers = load_manifest(seed_only=True), load_manifest(seed_only=False)
    assert len(seeds) == 12 and all(p.seed_set for p in seeds)
    assert len(all_papers) == 50
    t5 = next(p for p in all_papers if p.paper_id == "raffel_2019_t5")
    assert t5.arxiv_id == "1910.10683" and t5.year == 2019 and not t5.seed_set
    assert all(p.paper_id != "radford_2019_gpt2" for p in all_papers)


# ---------------------------------------------------------------------------
# Cache & budget
# ---------------------------------------------------------------------------


def test_cache_roundtrip_and_upsert(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    with ExtractionCache(db) as cache:
        cache.put("h1", "c1", "p1", "m", '{"entities": [], "relationships": []}', 10, 5)
        cache.put("h1", "c1", "p1", "m2", '{"entities": [], "relationships": []}', 11, 6)
        assert len(cache) == 1
    with ExtractionCache(db) as reopened:  # persisted on disk
        row = reopened.get("h1")
        assert row is not None and row.model == "m2" and row.input_tokens == 11
        assert reopened.contains_many(["h1", "h2"]) == {"h1"}


def test_budget_estimator_partitions_and_enforces() -> None:
    cache = ExtractionCache(":memory:")
    a, b = _chunk("Alpha text here."), _chunk("Beta text here.")
    cache.put(a.content_hash, a.chunk_id, a.paper_id, "m", "{}")
    est = CostBudgetEstimator([a, b, b], cache, input_usd_per_1m=0.15, output_usd_per_1m=0.60).estimate()
    assert [c.chunk_id for c in est.cached_chunks] == [a.chunk_id]
    assert len(est.uncached_chunks) == 1 and est.duplicate_chunks == 1
    expected = (b.approx_token_count + 650) * 0.15 / 1e6 + 350 * 0.60 / 1e6
    assert est.projected_usd == pytest.approx(expected)
    with pytest.raises(BudgetExceededError):
        CostBudgetEstimator([b], cache).enforce_budget(expected / 2)
    assert CostBudgetEstimator([a], cache).enforce_budget(0.0).projected_usd == 0.0


# ---------------------------------------------------------------------------
# Sanitizer & extractor
# ---------------------------------------------------------------------------


def test_sanitizer_raises_only_for_non_object_responses() -> None:
    with pytest.raises(ExtractionParseError):
        sanitize_and_filter_extraction("{not json")
    with pytest.raises(ExtractionParseError):
        sanitize_and_filter_extraction("[]")
    assert sanitize_and_filter_extraction("{}") == ChunkExtractionPayload()


def test_extractor_calls_interactions_api_with_schema_and_store_false() -> None:
    calls: List[Dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"entities": [], "relationships": [VALID_REL]}),
            usage=SimpleNamespace(total_input_tokens=900, total_output_tokens=120),
        )

    client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    ex = GeminiGraphExtractor(ExtractionCache(":memory:"), client=client, model_name="gemini-3.8-flash")
    chunk = _chunk()
    res = ex.extract_chunk(chunk)
    assert len(calls) == 1
    kw = calls[0]
    assert kw["model"] == "gemini-3.8-flash" and kw["store"] is False
    assert kw["system_instruction"] == EXTRACTION_SYSTEM_PROMPT
    assert kw["input"] == chunk.context_prefixed_text
    assert kw["response_format"]["mime_type"] == "application/json"
    assert kw["response_format"]["schema"] == ChunkExtractionPayload.model_json_schema()
    assert res.input_tokens == 900 and res.output_tokens == 120 and len(res.payload.relationships) == 1
    assert ex.extract_chunk(chunk).cache_hit and len(calls) == 1


def test_extractor_without_key_or_callable_fails_clearly() -> None:
    ex = GeminiGraphExtractor(ExtractionCache(":memory:"), api_key="")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        ex.extract_chunk(_chunk())


def test_extractor_retry_adds_corrective_note() -> None:
    inputs: List[str] = []

    def llm(system: str, user: str, schema: Dict[str, Any]) -> str:
        inputs.append(user)
        return "oops" if len(inputs) == 1 else json.dumps({"relationships": [VALID_REL]})

    res = GeminiGraphExtractor(ExtractionCache(":memory:"), llm_callable=llm).extract_chunk(_chunk())
    assert res.api_calls == 2 and not res.failed
    assert "not a valid JSON object" in inputs[1] and "not a valid JSON object" not in inputs[0]


# ---------------------------------------------------------------------------
# arXiv fetcher
# ---------------------------------------------------------------------------


PAPER = PaperMeta(paper_id="test_paper", arxiv_id="2405.04434", title="T", year=2024, pillar="architecture")


def _fetcher(tmp_path: Path, handler) -> ArxivFetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "test-agent"})
    return ArxivFetcher(tmp_path / "pdf", tmp_path / "md", max_retries=3, request_delay_s=0, http_client=client,
                        sleep=lambda s: None)


def test_fetcher_retries_then_caches_pdf(tmp_path: Path) -> None:
    seen: List[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(503) if len(seen) == 1 else httpx.Response(200, content=b"%PDF-1.7 fake")

    f = _fetcher(tmp_path, handler)
    path = f.download_pdf(PAPER)
    assert path == tmp_path / "pdf" / "test_paper.pdf" and path.read_bytes().startswith(b"%PDF")
    assert str(seen[0].url) == "https://arxiv.org/pdf/2405.04434.pdf" and seen[0].headers["User-Agent"] == "test-agent"
    f.download_pdf(PAPER)
    assert len(seen) == 2  # second call served from disk


def test_fetcher_rejects_non_pdf(tmp_path: Path) -> None:
    f = _fetcher(tmp_path, lambda req: httpx.Response(200, content=b"<html>captcha</html>"))
    with pytest.raises(PdfDownloadError):
        f.download_pdf(PAPER)
    assert not (tmp_path / "pdf" / "test_paper.pdf").exists()


def test_pdf_to_markdown_and_markdown_cache(tmp_path: Path) -> None:
    import pymupdf

    pdf = tmp_path / "pdf" / "test_paper.pdf"
    pdf.parent.mkdir(parents=True)
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Grouped-Query Attention reduces KV cache size.")
    doc.save(str(pdf))
    doc.close()
    f = _fetcher(tmp_path, lambda req: pytest.fail("network must not be used when the PDF is cached"))
    md = f.fetch_markdown(PAPER)
    assert "Grouped-Query Attention" in md
    assert (tmp_path / "md" / "test_paper.md").read_text(encoding="utf-8") == md


def test_offline_markdown_never_writes_cache(tmp_path: Path) -> None:
    gqa = next(p for p in load_manifest() if p.paper_id == "ainslie_2023_gqa")
    md = load_or_synthesize_seed_markdown(gqa, markdown_dir=tmp_path)
    assert SYNTHETIC_BANNER in md and not any(tmp_path.iterdir())
    (tmp_path / "ainslie_2023_gqa.md").write_text("real text", encoding="utf-8")
    assert load_or_synthesize_seed_markdown(gqa, markdown_dir=tmp_path) == "real text"
    assert SYNTHETIC_BANNER in load_or_synthesize_seed_markdown(gqa, markdown_dir=tmp_path, prefer_cached=False)


# ---------------------------------------------------------------------------
# Graph stores
# ---------------------------------------------------------------------------


def _edge(chunk: str, quote: str) -> ResolvedGraphEdge:
    return ResolvedGraphEdge(source_entity_id="a::x", target_entity_id="a::y", relation_type=RelationType.EXTENDS,
                             source_chunk_id=chunk, paper_id="p", evidence_quote=quote, confidence=0.8)


def test_aggregate_edges_collapses_same_triple() -> None:
    rows = aggregate_edges([_edge("c1", "q1 is long enough"), _edge("c2", "q2 is long enough"), _edge("c1", "q1 is long enough")])
    assert len(rows) == 1
    row = next(iter(rows.values()))
    assert row["source_chunk_ids"] == ["c1", "c2"] and row["evidence_quotes"] == ["q1 is long enough", "q2 is long enough"]


def test_in_memory_store_requires_endpoints_like_match() -> None:
    store = InMemoryGraphStore()
    assert store.upsert_resolved_edges([_edge("c1", "quote long enough")]) == 0
    store.upsert_resolved_nodes([
        ResolvedEntityNode(entity_id="a::x", entity_type=EntityType.ARCH_COMPONENT, canonical_name="X"),
        ResolvedEntityNode(entity_id="a::y", entity_type=EntityType.ARCH_COMPONENT, canonical_name="Y"),
    ])
    assert store.upsert_resolved_edges([_edge("c1", "quote long enough")]) == 1


class _Tx:
    def __init__(self, log): self.log = log
    def run(self, q, p=None): self.log.append((q, p or {})); return []


class _Session:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return None
    def execute_write(self, fn): return fn(_Tx(self.log))
    execute_read = execute_write


class _Driver:
    def __init__(self): self.log = []
    def session(self, database=None): return _Session(self.log)
    def close(self): pass


def test_neo4j_store_sends_constraints_and_grouped_parameterized_merges() -> None:
    driver = _Driver()
    store = Neo4jGraphStore(driver=driver)
    store.init_schema()
    assert [q for q, _ in driver.log] == list(SCHEMA_STATEMENTS)
    nodes = [
        ResolvedEntityNode(entity_id="arch_component::x", entity_type=EntityType.ARCH_COMPONENT, canonical_name="X", aliases=["X", "X"]),
        ResolvedEntityNode(entity_id="model_family::y", entity_type=EntityType.MODEL_FAMILY, canonical_name="Y"),
    ]
    store.upsert_resolved_nodes(nodes)
    node_writes = driver.log[3:]
    assert len(node_writes) == 2  # one statement per secondary label
    assert {"SET e:`ArchComponent`" in q for q, _ in node_writes} == {True, False}
    arch_rows = next(p["rows"] for q, p in node_writes if "ArchComponent" in q)
    assert arch_rows[0]["aliases"] == ["X"]
    store.upsert_resolved_edges([_edge("c1", "q1 long enough"), _edge("c2", "q2 long enough")])
    q, params = driver.log[-1]
    assert "MERGE (s)-[r:`EXTENDS`]->(t)" in q and len(params["rows"]) == 1
    assert params["rows"][0]["source_chunk_ids"] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _pipeline(cache, llm, store):
    loader = lambda p: load_or_synthesize_seed_markdown(p, prefer_cached=False)  # noqa: E731
    return Phase1IngestionPipeline(GeminiGraphExtractor(cache, llm_callable=llm), store, loader)


def _e2e_papers():
    wanted = ("vaswani_2017_transformer", "ainslie_2023_gqa", "deepseek_2024_v2")
    return [p for p in load_manifest() if p.paper_id in wanted]


def test_pipeline_budget_guard_blocks_before_side_effects() -> None:
    llm, store = RecordedExtractionLLM(), InMemoryGraphStore()
    with pytest.raises(BudgetExceededError):
        _pipeline(ExtractionCache(":memory:"), llm, store).run(_e2e_papers(), max_budget_usd=0.0)
    assert llm.calls == 0 and store.count_nodes() == 0


def test_pipeline_is_idempotent_and_builds_two_hop_chain() -> None:
    cache, llm, store = ExtractionCache(":memory:"), RecordedExtractionLLM(), InMemoryGraphStore()
    first = _pipeline(cache, llm, store).run(_e2e_papers(), max_budget_usd=1.0)
    snapshot = (store.count_nodes(), store.count_relationships(), json.dumps({"|".join(k): v for k, v in store.rels.items()}))
    second = _pipeline(cache, llm, store).run(_e2e_papers(), max_budget_usd=1.0)
    assert (store.count_nodes(), store.count_relationships(), json.dumps({"|".join(k): v for k, v in store.rels.items()})) == snapshot
    assert second.extraction.api_calls == 0 and llm.calls == len(first.chunks)

    paths = store.find_paths("arch_component::multi_head_latent_attention", "arch_component::multi_head_attention", "EXTENDS", 2)
    assert len(paths) == 1 and paths[0][0]["target"] == "arch_component::grouped_query_attention"
    for hop in paths[0]:
        assert hop["source_chunk_ids"]
        assert any(q in store.get_chunk_text(cid) for cid in hop["source_chunk_ids"] for q in hop["evidence_quotes"])
    assert store.get_relationship("paper::deepseek_2024_v2", "INTRODUCES", "model_family::deepseek_v2") is not None
