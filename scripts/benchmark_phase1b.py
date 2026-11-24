"""Phase 1B scorecard: cache & budget guard, resilient extraction, idempotent graph merge, E2E audit.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/benchmark_phase1b.py [--live-neo4j | --no-live-neo4j]

Runs fully offline: in-memory SQLite extraction cache, `RecordedExtractionLLM` instead of Gemini,
synthetic seed-paper Markdown, `InMemoryGraphStore`, and a recording Neo4j driver that captures
every Cypher statement `Neo4jGraphStore` would send.

Live Neo4j (optional, not part of the 21 scored cases): by default the script probes
NEO4J_URI for 2 seconds; if reachable it writes ONLY ids prefixed `bench1b`, verifies them,
and deletes them again. `--no-live-neo4j` skips the probe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.arxiv_fetcher import PaperMeta, load_manifest, load_or_synthesize_seed_markdown  # noqa: E402
from src.ingestion.chunker import DocumentChunk, StructureAwareChunker, compute_content_hash  # noqa: E402
from src.ingestion.cost_estimator import BudgetExceededError, CostBudgetEstimator, ExtractionCache  # noqa: E402
from src.ingestion.extractor import GeminiGraphExtractor  # noqa: E402
from src.ingestion.offline_fixtures import RecordedExtractionLLM  # noqa: E402
from src.ingestion.pipeline import Phase1IngestionPipeline, PipelineReport  # noqa: E402
from src.schema.ontology import (  # noqa: E402
    EntityType,
    RelationType,
    ResolvedEntityNode,
    ResolvedGraphEdge,
)
from src.stores.neo4j_store import InMemoryGraphStore, Neo4jGraphStore, SCHEMA_STATEMENTS, _label  # noqa: E402

LINE = "=" * 92
SUBLINE = "-" * 92
E2E_PAPERS = ("vaswani_2017_transformer", "ainslie_2023_gqa", "deepseek_2024_v2")
MLA = "arch_component::multi_head_latent_attention"
GQA = "arch_component::grouped_query_attention"
MHA = "arch_component::multi_head_attention"


@dataclass
class CaseResult:
    bucket: str
    case_id: str
    title: str
    passed: bool
    lines: List[str] = field(default_factory=list)
    hint: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_PAPERS = load_manifest(seed_only=False)
SEED_PAPERS = load_manifest(seed_only=True)
PAPER_BY_ID = {p.paper_id: p for p in ALL_PAPERS}


def synthetic_loader(paper: PaperMeta) -> str:
    return load_or_synthesize_seed_markdown(paper, prefer_cached=False)


def chunk_papers(papers: List[PaperMeta]) -> List[DocumentChunk]:
    chunker = StructureAwareChunker()
    return [
        c for p in papers
        for c in chunker.chunk_paper(p.paper_id, p.title, p.year, p.pillar.value, synthetic_loader(p))
    ]


def make_pipeline(cache: ExtractionCache, llm: Callable, store: Any) -> Phase1IngestionPipeline:
    extractor = GeminiGraphExtractor(cache, llm_callable=llm, min_confidence=0.65)
    return Phase1IngestionPipeline(extractor, store, synthetic_loader, known_papers=ALL_PAPERS)


class ScriptedLLM:
    """Returns pre-scripted raw responses in order (repeats the last one)."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = responses
        self.calls = 0

    def __call__(self, system_instruction: str, user_input: str, schema: Dict[str, Any]) -> str:
        out = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return out


def one_chunk(text: str = "Grouped-query attention extends multi-head attention by sharing key-value heads.") -> DocumentChunk:
    return StructureAwareChunker().chunk_paper("bench_paper", "Bench Paper", 2024, "architecture", f"# Method\n\n{text}\n")[0]


def rel(src: str, st: str, rt: str, tgt: str, tt: str, quote: str, conf: float = 0.9) -> Dict[str, Any]:
    return {"source_name": src, "source_type": st, "relation_type": rt, "target_name": tgt,
            "target_type": tt, "evidence_quote": quote, "confidence": conf}


VALID_REL = rel("Grouped-Query Attention", "ArchComponent", "EXTENDS", "Multi-Head Attention", "ArchComponent",
                "Grouped-query attention extends multi-head attention by sharing key-value heads.")


class RecordingTx:
    def __init__(self, log: List[Tuple[str, Dict[str, Any]]]) -> None:
        self.log = log

    def run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        self.log.append((query, params or {}))
        return []


class RecordingSession:
    def __init__(self, log: List[Tuple[str, Dict[str, Any]]]) -> None:
        self.log = log

    def __enter__(self) -> "RecordingSession":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute_write(self, fn: Callable) -> Any:
        return fn(RecordingTx(self.log))

    execute_read = execute_write


class RecordingDriver:
    """Stand-in for `neo4j.Driver` that records every (query, params) sent in a transaction."""

    def __init__(self) -> None:
        self.log: List[Tuple[str, Dict[str, Any]]] = []

    def session(self, database: Optional[str] = None) -> RecordingSession:
        return RecordingSession(self.log)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Bucket 1: cache_and_budget_guard
# ---------------------------------------------------------------------------


def cb01_cache_hit_zero_api() -> Tuple[bool, List[str], str]:
    cache, llm = ExtractionCache(":memory:"), ScriptedLLM([json.dumps({"entities": [], "relationships": [VALID_REL]})])
    ex = GeminiGraphExtractor(cache, llm_callable=llm)
    chunk = one_chunk()
    first, second = ex.extract_chunk(chunk), ex.extract_chunk(chunk)
    ok = (not first.cache_hit and first.api_calls == 1 and second.cache_hit and second.api_calls == 0
          and llm.calls == 1 and second.payload == first.payload)
    return ok, [f"pass1: cache_hit={first.cache_hit} api_calls={first.api_calls}",
                f"pass2: cache_hit={second.cache_hit} api_calls={second.api_calls} | LLM invoked {llm.calls}x total"], \
        "Extractor must check ExtractionCache by content_hash before calling the model."


def cb02_second_pass_free() -> Tuple[bool, List[str], str]:
    cache, llm = ExtractionCache(":memory:"), RecordedExtractionLLM()
    chunks = chunk_papers(SEED_PAPERS)
    est1 = CostBudgetEstimator(chunks, cache).estimate()
    GeminiGraphExtractor(cache, llm_callable=llm).extract_chunks(chunks)
    calls_after_pass1 = llm.calls
    est2 = CostBudgetEstimator(chunks, cache).estimate()
    ex2 = GeminiGraphExtractor(cache, llm_callable=llm)
    ex2.extract_chunks(chunks)
    ok = (est1.projected_usd > 0 and est2.projected_usd == 0.0 and not est2.uncached_chunks
          and ex2.stats.api_calls == 0 and llm.calls == calls_after_pass1)
    return ok, [f"{len(SEED_PAPERS)} seed papers -> {len(chunks)} chunks",
                f"pass1 projected=${est1.projected_usd:.6f} ({len(est1.uncached_chunks)} uncached)",
                f"pass2 projected=${est2.projected_usd:.6f} ({len(est2.uncached_chunks)} uncached), api_calls={ex2.stats.api_calls}"], \
        "Second pass must be fully served from cache."


def cb03_budget_blocks_before_api() -> Tuple[bool, List[str], str]:
    cache, llm, store = ExtractionCache(":memory:"), RecordedExtractionLLM(), InMemoryGraphStore()
    pipe = make_pipeline(cache, llm, store)
    papers = [PAPER_BY_ID[p] for p in E2E_PAPERS]
    try:
        pipe.run(papers, max_budget_usd=0.0001)
    except BudgetExceededError as exc:
        ok = llm.calls == 0 and store.count_nodes() == 0 and not store.schema_initialized
        return ok, [f"BudgetExceededError: {exc}", f"LLM calls={llm.calls}, graph nodes={store.count_nodes()}"], \
            "Guard must raise before any extraction call or graph write."
    return False, ["run completed without raising"], "enforce_budget() did not raise for a $0.0001 cap."


def cb04_projection_math() -> Tuple[bool, List[str], str]:
    chunks = chunk_papers([PAPER_BY_ID[p] for p in E2E_PAPERS])
    est = CostBudgetEstimator(chunks, ExtractionCache(":memory:"), input_usd_per_1m=0.15, output_usd_per_1m=0.60,
                              prompt_overhead_tokens=650, output_tokens_per_chunk=350).estimate()
    exp_in = sum(c.approx_token_count + 650 for c in chunks)
    exp_out = 350 * len(chunks)
    exp_usd = exp_in * 0.15 / 1e6 + exp_out * 0.60 / 1e6
    ok = est.projected_input_tokens == exp_in and est.projected_output_tokens == exp_out and abs(est.projected_usd - exp_usd) < 1e-12
    within = CostBudgetEstimator(chunks, ExtractionCache(":memory:")).enforce_budget(exp_usd * 2)
    ok = ok and within.projected_usd > 0
    return ok, [f"input={est.projected_input_tokens} (expected {exp_in}), output={est.projected_output_tokens} (expected {exp_out})",
                f"projected=${est.projected_usd:.6f} (expected ${exp_usd:.6f}); cap 2x passes"], \
        "Projection must be (chunk tokens + 650) x input rate + 350 x output rate per uncached chunk."


def cb05_version_bump_and_dedup() -> Tuple[bool, List[str], str]:
    text = "[Paper: X (2024)] Some chunk text."
    base = compute_content_hash(text)
    bumped_prompt = compute_content_hash(text, prompt_version="kg-prompt-v999")
    bumped_schema = compute_content_hash(text, schema_version="kg-schema-v999")
    ws_variant = compute_content_hash(text.replace(" ", "   "))
    chunk = one_chunk()
    est = CostBudgetEstimator([chunk, chunk, chunk], ExtractionCache(":memory:")).estimate()
    ok = len({base, bumped_prompt, bumped_schema}) == 3 and ws_variant == base and len(est.uncached_chunks) == 1 and est.duplicate_chunks == 2
    return ok, [f"prompt bump changes key: {base != bumped_prompt}; schema bump changes key: {base != bumped_schema}; "
                f"whitespace-insensitive: {ws_variant == base}",
                f"3 identical chunks -> uncached={len(est.uncached_chunks)} duplicates={est.duplicate_chunks}"], \
        "content_hash must include schema+prompt versions; identical chunks are billed once."


# ---------------------------------------------------------------------------
# Bucket 2: resilient_extraction_validation
# ---------------------------------------------------------------------------


def _extract(responses: List[str], min_conf: float = 0.65) -> Tuple[Any, ScriptedLLM, ExtractionCache]:
    cache, llm = ExtractionCache(":memory:"), ScriptedLLM(responses)
    ex = GeminiGraphExtractor(cache, llm_callable=llm, min_confidence=min_conf)
    return ex.extract_chunk(one_chunk()), llm, cache


def rx01_malformed_then_retry() -> Tuple[bool, List[str], str]:
    good = json.dumps({"entities": [], "relationships": [VALID_REL]})
    res, llm, cache = _extract(['{"entities": [ {"entity_type": "ArchComp', good])
    ok = not res.failed and res.api_calls == 2 and len(res.payload.relationships) == 1 and len(cache) == 1
    return ok, [f"api_calls={res.api_calls} failed={res.failed} relationships={len(res.payload.relationships)} cached_rows={len(cache)}"], \
        "Truncated JSON must trigger exactly one retry, then parse and cache."


def rx02_malformed_twice_not_cached() -> Tuple[bool, List[str], str]:
    res, llm, cache = _extract(["not json at all", "```json\n[1, 2]\n```"])
    ok = res.failed and res.api_calls == 2 and len(cache) == 0 and not res.payload.relationships
    return ok, [f"api_calls={res.api_calls} failed={res.failed} cached_rows={len(cache)}"], \
        "After 1 retry the chunk must be marked failed and NOT cached (so a rerun retries it)."


def rx03_self_loop_stripped() -> Tuple[bool, List[str], str]:
    loop = rel("RoPE", "ArchComponent", "EXTENDS", "rope", "ArchComponent", "RoPE extends RoPE in a loop somehow.")
    res, _, _ = _extract([json.dumps({"entities": [], "relationships": [loop, VALID_REL]})])
    kept = [(r.source_name, r.target_name) for r in res.payload.relationships]
    ok = kept == [("Grouped-Query Attention", "Multi-Head Attention")] and any("Self-loop" in d["reason"] for d in res.dropped_items)
    return ok, [f"kept={kept}", f"dropped={[d['reason'][:60] for d in res.dropped_items]}"], \
        "Self-loop must be dropped individually; sibling triple must survive."


def rx04_low_confidence_stripped() -> Tuple[bool, List[str], str]:
    weak = rel("SwiGLU", "ArchComponent", "REPLACES_IN", "GeLU", "ArchComponent", "SwiGLU may replace GeLU, perhaps.", conf=0.30)
    ents = [{"entity_type": "ArchComponent", "canonical_name": "GQA", "confidence": 0.95, "category": "attention"},
            {"entity_type": "ArchComponent", "canonical_name": "Maybe Gate", "confidence": 0.20}]
    res, _, cache = _extract([json.dumps({"entities": ents, "relationships": [weak, VALID_REL]})])
    cached = json.loads(cache.get(one_chunk().content_hash).payload_json)
    ok = (len(res.payload.relationships) == 1 and [e.canonical_name for e in res.payload.entities] == ["GQA"]
          and len(cached["relationships"]) == 2)
    return ok, [f"returned rels={len(res.payload.relationships)} entities={[e.canonical_name for e in res.payload.entities]}",
                f"cache keeps unfiltered payload: rels={len(cached['relationships'])} (re-filterable without re-extraction)"], \
        "Low-confidence items are dropped from the returned payload only."


def rx05_prompt_injected_entity_type() -> Tuple[bool, List[str], str]:
    ents = [{"entity_type": "SystemOverride", "canonical_name": "Ignore previous instructions", "confidence": 0.99},
            {"entity_type": "ArchComponent", "canonical_name": "Grouped-Query Attention", "confidence": 0.95}]
    bad_rel = rel("X Model", "AdminNode", "EXTENDS", "Multi-Head Attention", "ArchComponent", "Injected relation with fake type.")
    res, _, _ = _extract([json.dumps({"entities": ents, "relationships": [bad_rel, VALID_REL]})])
    names = [e.canonical_name for e in res.payload.entities]
    ok = names == ["Grouped-Query Attention"] and len(res.payload.relationships) == 1
    return ok, [f"entities kept={names}", f"relationships kept={len(res.payload.relationships)}, dropped={len(res.dropped_items)}"], \
        "Out-of-ontology EntityType must be rejected per item."


def rx06_invalid_relation_and_fenced_json() -> Tuple[bool, List[str], str]:
    bad_rt = rel("GQA", "ArchComponent", "DELETE_ALL", "MHA", "ArchComponent", "Some quote that is long enough.")
    missing = {"source_name": "GQA", "relation_type": "EXTENDS"}
    body = json.dumps({"entities": "not-a-list", "relationships": [bad_rt, missing, VALID_REL]})
    res, _, _ = _extract([f"```json\n{body}\n```"])
    ok = not res.failed and res.api_calls == 1 and len(res.payload.relationships) == 1 and len(res.dropped_items) == 3
    return ok, [f"api_calls={res.api_calls} kept rels={len(res.payload.relationships)} dropped={len(res.dropped_items)}",
                f"reasons={[d['reason'][:50] for d in res.dropped_items]}"], \
        "Code-fenced JSON must parse; invalid RelationType, incomplete item and non-list field are dropped individually."


# ---------------------------------------------------------------------------
# Bucket 3: idempotent_graph_merge
# ---------------------------------------------------------------------------

_E2E_CACHE: Dict[str, Any] = {}


def triple_ingest() -> Dict[str, Any]:
    """Ingest the 3 E2E papers 3 times (fresh resolver each run, shared cache + store)."""
    if "triple" in _E2E_CACHE:
        return _E2E_CACHE["triple"]
    cache, llm, store = ExtractionCache(":memory:"), RecordedExtractionLLM(), InMemoryGraphStore()
    papers = [PAPER_BY_ID[p] for p in E2E_PAPERS]
    snapshots, reports = [], []
    for _ in range(3):
        reports.append(make_pipeline(cache, llm, store).run(papers, max_budget_usd=1.0))
        snapshots.append({
            "nodes": store.count_nodes(), "rels": store.count_relationships(),
            "dups": store.duplicate_relationship_groups(),
            "rel_props": json.dumps({"|".join(k): v for k, v in store.rels.items()}, sort_keys=True),
            "node_props": json.dumps({k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()}
                                      for k, v in store.nodes.items()}, sort_keys=True),
        })
    _E2E_CACHE["triple"] = {"store": store, "snapshots": snapshots, "reports": reports, "llm": llm}
    return _E2E_CACHE["triple"]


def ig01_no_duplicate_nodes() -> Tuple[bool, List[str], str]:
    t = triple_ingest()
    counts = [s["nodes"] for s in t["snapshots"]]
    same_props = len({s["node_props"] for s in t["snapshots"]}) == 1
    ok = len(set(counts)) == 1 and same_props
    return ok, [f"node counts after runs 1/2/3 = {counts}; node properties byte-identical across runs: {same_props}"], \
        "Re-ingesting must MERGE nodes and keep alias/provenance lists de-duplicated."


def ig02_no_duplicate_edges() -> Tuple[bool, List[str], str]:
    t = triple_ingest()
    counts = [s["rels"] for s in t["snapshots"]]
    same_props = len({s["rel_props"] for s in t["snapshots"]}) == 1
    dups = [s["dups"] for s in t["snapshots"]]
    ok = len(set(counts)) == 1 and same_props and dups == [0, 0, 0] and t["llm"].calls == len(t["reports"][0].chunks)
    return ok, [f"relationship counts after runs 1/2/3 = {counts}; duplicate (s,type,t) groups = {dups}",
                f"edge properties identical across runs: {same_props}; LLM calls over 3 runs = {t['llm'].calls} "
                f"(= {len(t['reports'][0].chunks)} chunks, runs 2-3 fully cached)"], \
        "Edges must MERGE on (source, type, target) with set-like provenance lists."


def _edge(chunk: str, quote: str, src: str = "bench1b::src", tgt: str = "bench1b::tgt") -> ResolvedGraphEdge:
    return ResolvedGraphEdge(source_entity_id=src, target_entity_id=tgt, relation_type=RelationType.EXTENDS,
                             source_chunk_id=chunk, paper_id="bench1b_paper", evidence_quote=quote, confidence=0.9)


def _bench_nodes() -> List[ResolvedEntityNode]:
    return [ResolvedEntityNode(entity_id="bench1b::src", entity_type=EntityType.ARCH_COMPONENT, canonical_name="Bench Src",
                               aliases=["BS"], source_chunk_ids=["c1"]),
            ResolvedEntityNode(entity_id="bench1b::tgt", entity_type=EntityType.ARCH_COMPONENT, canonical_name="Bench Tgt",
                               source_chunk_ids=["c2"])]


def check_multi_chunk_accumulation(store: Any) -> Tuple[bool, List[str]]:
    store.upsert_resolved_nodes(_bench_nodes())
    store.upsert_resolved_edges([_edge("c1", "Bench Src extends Bench Tgt (chunk 1).")])
    store.upsert_resolved_edges([_edge("c2", "Bench Src builds on Bench Tgt (chunk 2).")])
    store.upsert_resolved_edges([_edge("c1", "Bench Src extends Bench Tgt (chunk 1).")])  # replay
    r = store.get_relationship("bench1b::src", "EXTENDS", "bench1b::tgt") or {}
    ok = r.get("source_chunk_ids") == ["c1", "c2"] and len(r.get("evidence_quotes", [])) == 2
    return ok, [f"source_chunk_ids={r.get('source_chunk_ids')} evidence_quotes={len(r.get('evidence_quotes', []))}"]


def ig03_multi_chunk_evidence() -> Tuple[bool, List[str], str]:
    store = InMemoryGraphStore()
    ok, lines = check_multi_chunk_accumulation(store)
    ok = ok and store.count_relationships("EXTENDS") == 1
    return ok, lines + [f"EXTENDS edges between the pair = {store.count_relationships('EXTENDS')}"], \
        "Evidence from c1 and c2 must accumulate on ONE edge."


def ig04_node_set_merge() -> Tuple[bool, List[str], str]:
    store = InMemoryGraphStore()
    a = ResolvedEntityNode(entity_id="bench1b::n", entity_type=EntityType.MODEL_FAMILY, canonical_name="Llama-3",
                           aliases=["Llama 3", "LLaMA-3"], source_chunk_ids=["c1"], paper_ids=["p1"], parameter_scales=["8B"])
    b = a.model_copy(update={"aliases": ["LLaMA-3", "Llama3"], "source_chunk_ids": ["c2", "c1"], "paper_ids": ["p1", "p2"],
                             "parameter_scales": ["70B", "8B"]})
    for n in (a, b, a, b):
        store.upsert_resolved_nodes([n])
    node = store.nodes["bench1b::n"]
    ok = (node["aliases"] == ["Llama 3", "LLaMA-3", "Llama3"] and node["source_chunk_ids"] == ["c1", "c2"]
          and node["paper_ids"] == ["p1", "p2"] and node["parameter_scales"] == ["8B", "70B"]
          and node["labels"] == {"Entity", "ModelFamily"} and store.count_nodes() == 1)
    return ok, [f"aliases={node['aliases']} chunks={node['source_chunk_ids']} papers={node['paper_ids']} "
                f"scales={node['parameter_scales']} labels={sorted(node['labels'])}"], \
        "Node lists must merge as ordered de-duplicated sets with the :ModelFamily secondary label."


def ig05_paper_introduces_model_family() -> Tuple[bool, List[str], str]:
    store = triple_ingest()["store"]
    paper_id, mf_id = "paper::deepseek_2024_v2", "model_family::deepseek_v2"
    r = store.get_relationship(paper_id, "INTRODUCES", mf_id)
    paper, mf = store.nodes.get(paper_id, {}), store.nodes.get(mf_id, {})
    ok = (r is not None and {"Entity", "Paper"} <= paper.get("labels", set())
          and {"Entity", "ModelFamily"} <= mf.get("labels", set()) and paper.get("arxiv_id") == "2405.04434"
          and r.get("source_chunk_ids"))
    return ok, [f"({paper_id} {sorted(paper.get('labels', []))})-[:INTRODUCES]->({mf_id} {sorted(mf.get('labels', []))})",
                f"edge={'present' if r else 'MISSING'} source_chunk_ids={r.get('source_chunk_ids') if r else None}"], \
        "Extracted Paper 'DeepSeek-V2' must resolve to the manifest paper node, distinct from the ModelFamily."


def ig06_cypher_contract() -> Tuple[bool, List[str], str]:
    driver = RecordingDriver()
    store = Neo4jGraphStore(driver=driver)
    store.init_schema()
    sneaky = "Evil'}) DETACH DELETE n //"
    paper = PAPER_BY_ID["deepseek_2024_v2"]
    chunk = one_chunk()
    store.upsert_paper_and_chunks(paper, [chunk])
    store.upsert_resolved_nodes([ResolvedEntityNode(entity_id="arch_component::x", entity_type=EntityType.ARCH_COMPONENT,
                                                    canonical_name=sneaky, aliases=[sneaky])])
    store.upsert_resolved_edges([_edge("c1", f"{sneaky} quote long enough", "arch_component::x", "arch_component::y")])
    store.upsert_chunk_mentions(chunk.chunk_id, ["arch_component::x"])
    writes = [q for q, _ in driver.log if not q.startswith("CREATE CONSTRAINT")]
    schema_ok = [q for q, _ in driver.log[:3]] == list(SCHEMA_STATEMENTS)
    all_merge = all("MERGE" in q for q in writes)
    no_literals = not any(sneaky in q or chunk.chunk_id in q or paper.paper_id in q for q, _ in driver.log)
    labels_ok = any("SET e:`ArchComponent`" in q for q in writes) and any("[r:`EXTENDS`]" in q for q in writes)
    try:
        _label("ArchComponent`) DETACH DELETE (n")
        rejects_bad_label = False
    except ValueError:
        rejects_bad_label = True
    ok = schema_ok and all_merge and no_literals and labels_ok and rejects_bad_label and len(writes) == 4
    return ok, [f"constraints={schema_ok} statements={len(driver.log)} all writes MERGE={all_merge} "
                f"values only as parameters={no_literals}",
                f"labels/rel types from enums={labels_ok}; injected label rejected={rejects_bad_label}"], \
        "Neo4jGraphStore must create 3 constraints, MERGE every write and pass all values as parameters."


# ---------------------------------------------------------------------------
# Bucket 4: end_to_end_pipeline_audit
# ---------------------------------------------------------------------------


def e2e_report() -> Tuple[PipelineReport, InMemoryGraphStore]:
    t = triple_ingest()
    return t["reports"][0], t["store"]


def e2e01_pipeline_completes() -> Tuple[bool, List[str], str]:
    cache, llm = ExtractionCache(":memory:"), RecordedExtractionLLM()
    dry = make_pipeline(cache, llm, None).run([PAPER_BY_ID[p] for p in E2E_PAPERS], dry_run_budget=True)
    report, store = e2e_report()
    in_paper_ok = all(store.get_relationship(c.chunk_id, "IN_PAPER", f"paper::{c.paper_id}") is not None for c in report.chunks)
    ok = (dry.estimate.projected_usd > 0 and llm.calls == 0 and report.extraction.failed_chunks == 0
          and in_paper_ok and set(report.papers) == set(E2E_PAPERS))
    return ok, [f"dry-run: {dry.estimate.summary()} | LLM calls={llm.calls}",
                f"run: {len(report.chunks)} chunks, failed={report.extraction.failed_chunks}, every chunk IN_PAPER its manifest paper={in_paper_ok}"], \
        "Dry-run must make 0 calls; full run must link every chunk to its manifest Paper."


def e2e02_two_hop_chain() -> Tuple[bool, List[str], str]:
    _, store = e2e_report()
    paths = store.find_paths(MLA, MHA, "EXTENDS", 2)
    via_gqa = [p for p in paths if p[0]["target"] == GQA]
    ok = len(via_gqa) == 1
    chain = " -> ".join([via_gqa[0][0]["source"]] + [h["target"] for h in via_gqa[0]]) if via_gqa else "none"
    return ok, [f"2-hop EXTENDS paths MLA->MHA: {len(paths)}; via GQA: {chain}"], \
        "(MLA)-[:EXTENDS]->(GQA)-[:EXTENDS]->(MHA) must be traversable in exactly 2 hops."


def e2e03_hop_provenance() -> Tuple[bool, List[str], str]:
    _, store = e2e_report()
    paths = [p for p in store.find_paths(MLA, MHA, "EXTENDS", 2) if p[0]["target"] == GQA]
    if not paths:
        return False, ["no chain"], "Chain missing (see E2E-02)."
    lines, ok = [], True
    expected_paper = {MLA: "deepseek_2024_v2", GQA: "ainslie_2023_gqa"}
    for hop in paths[0]:
        hop_ok = bool(hop["source_chunk_ids"])
        for cid in hop["source_chunk_ids"]:
            text = store.get_chunk_text(cid) or ""
            quote_in_chunk = any(q in text for q in hop["evidence_quotes"])
            from_paper = cid.startswith(expected_paper[hop["source"]] + "::")
            hop_ok = hop_ok and quote_in_chunk and from_paper
            lines.append(f"{hop['source'].split('::')[1]} -> {hop['target'].split('::')[1]}: chunk={cid} "
                         f"quote_in_chunk={quote_in_chunk} from_expected_paper={from_paper}")
        ok = ok and hop_ok
    return ok, lines, "Every hop needs >=1 source_chunk_id whose stored chunk text contains the evidence quote."


def e2e04_cross_paper_resolution() -> Tuple[bool, List[str], str]:
    _, store = e2e_report()
    mha, gqa = store.nodes.get(MHA, {}), store.nodes.get(GQA, {})
    arch = [n for n in store.nodes.values() if "ArchComponent" in n["labels"]]
    gqa_like = [n.get("canonical_name") for n in arch if "gqa" in {a.lower() for a in n.get("aliases", [])} or
                n.get("canonical_name", "").lower() in {"gqa", "grouped-query attention", "grouped query attention"}]
    ok = ({"vaswani_2017_transformer", "ainslie_2023_gqa"} <= set(mha.get("paper_ids", []))
          and {"ainslie_2023_gqa", "deepseek_2024_v2"} <= set(gqa.get("paper_ids", [])) and len(gqa_like) == 1)
    return ok, [f"MHA paper_ids={mha.get('paper_ids')}", f"GQA paper_ids={gqa.get('paper_ids')}; ArchComponent nodes named/aliased GQA={gqa_like}"], \
        "Surface variants across papers must resolve to single canonical nodes."


BUCKETS: Dict[str, List[Tuple[str, str, Callable[[], Tuple[bool, List[str], str]]]]] = {
    "cache_and_budget_guard": [
        ("CB-01", "Cache hit returns payload with 0 API calls", cb01_cache_hit_zero_api),
        ("CB-02", "Second pass over the 12 seed papers projects $0.00 and makes 0 calls", cb02_second_pass_free),
        ("CB-03", "Budget guard blocks the run before any API call or graph write", cb03_budget_blocks_before_api),
        ("CB-04", "Projection = (tokens + 650) x $0.15/1M + 350 x $0.60/1M per uncached chunk", cb04_projection_math),
        ("CB-05", "Schema/prompt version bump invalidates keys; duplicate chunks billed once", cb05_version_bump_and_dedup),
    ],
    "resilient_extraction_validation": [
        ("RX-01", "Malformed JSON -> 1 automatic retry -> parsed and cached", rx01_malformed_then_retry),
        ("RX-02", "Malformed twice -> chunk marked failed and not cached", rx02_malformed_twice_not_cached),
        ("RX-03", "Self-loop stripped, sibling triple survives", rx03_self_loop_stripped),
        ("RX-04", "Low-confidence triple/entity stripped, siblings survive", rx04_low_confidence_stripped),
        ("RX-05", "Prompt-injected EntityType rejected, siblings survive", rx05_prompt_injected_entity_type),
        ("RX-06", "Invalid RelationType / incomplete item dropped; fenced JSON parsed", rx06_invalid_relation_and_fenced_json),
    ],
    "idempotent_graph_merge": [
        ("IG-01", "3x ingestion of the same papers -> 0 duplicate nodes", ig01_no_duplicate_nodes),
        ("IG-02", "3x ingestion of the same papers -> 0 duplicate edges", ig02_no_duplicate_edges),
        ("IG-03", "Multi-chunk evidence accumulates source_chunk_ids [c1, c2] on one edge", ig03_multi_chunk_evidence),
        ("IG-04", "Node aliases/provenance/scales merge as de-duplicated sets", ig04_node_set_merge),
        ("IG-05", "Manifest Paper -[:INTRODUCES]-> ModelFamily of the same name", ig05_paper_introduces_model_family),
        ("IG-06", "Neo4j Cypher contract: constraints, MERGE-only, parameterized values", ig06_cypher_contract),
    ],
    "end_to_end_pipeline_audit": [
        ("E2E-01", "Dry-run (0 calls) + full offline run on 3 seed papers", e2e01_pipeline_completes),
        ("E2E-02", "(MLA)-[:EXTENDS]->(GQA)-[:EXTENDS]->(MHA) in 2 hops", e2e02_two_hop_chain),
        ("E2E-03", "100% source_chunk_id provenance on every hop", e2e03_hop_provenance),
        ("E2E-04", "Cross-paper entity resolution yields single MHA / GQA nodes", e2e04_cross_paper_resolution),
    ],
}


# ---------------------------------------------------------------------------
# Optional live Neo4j check
# ---------------------------------------------------------------------------


def run_live_neo4j() -> Tuple[Optional[bool], List[str]]:
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
                                      connection_timeout=2.0, max_transaction_retry_time=2.0)
        driver.verify_connectivity()
    except Exception as exc:  # unreachable, auth error, driver missing
        return None, [f"skipped: Neo4j not reachable at {settings.neo4j_uri} ({type(exc).__name__})"]

    store = Neo4jGraphStore(driver=driver)
    prefixes = ["bench1b", "paper::bench1b"]
    lines: List[str] = []
    try:
        store.delete_by_prefix(prefixes)
        store.init_schema()
        paper = PaperMeta(paper_id="bench1b_paper", arxiv_id="0000.00000", title="Bench 1B Paper", year=2024,
                          pillar="architecture", seed_set=False)
        chunk = StructureAwareChunker().chunk_paper("bench1b_paper", paper.title, 2024, "architecture",
                                                    "# Method\n\nBench Src extends Bench Tgt (chunk 1).\n")[0]
        counts = []
        for _ in range(3):
            store.upsert_paper_and_chunks(paper, [chunk])
            ok_acc, acc_lines = check_multi_chunk_accumulation(store)
            store.upsert_chunk_mentions(chunk.chunk_id, ["bench1b::src"])
            rows = store._read(
                "MATCH (n) WHERE any(p IN $p WHERE n.entity_id STARTS WITH p OR n.chunk_id STARTS WITH p) "
                "OPTIONAL MATCH (n)-[r]->() RETURN count(DISTINCT n) AS nodes, count(r) AS rels", {"p": prefixes})
            counts.append((rows[0]["nodes"], rows[0]["rels"]))
        constraints = {r["name"] for r in store._read("SHOW CONSTRAINTS YIELD name RETURN name", {})}
        has_constraints = {"entity_id_unique", "chunk_id_unique", "paper_id_unique"} <= constraints
        passed = ok_acc and len(set(counts)) == 1 and has_constraints and store.duplicate_relationship_groups() == 0
        lines = [f"(nodes, rels) in bench1b namespace after 3 runs = {counts}", acc_lines[0],
                 f"constraints present={has_constraints}"]
        return passed, lines
    except Exception as exc:
        return False, [f"error: {type(exc).__name__}: {exc}"]
    finally:
        try:
            store.delete_by_prefix(prefixes)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1B benchmark scorecard (offline).")
    live = parser.add_mutually_exclusive_group()
    live.add_argument("--live-neo4j", dest="live", action="store_true", default=None, help="Require the live Neo4j check.")
    live.add_argument("--no-live-neo4j", dest="live", action="store_false", help="Skip the live Neo4j probe.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.CRITICAL)  # RX cases trigger expected warnings/errors

    results: Dict[str, List[CaseResult]] = {}
    for bucket, cases in BUCKETS.items():
        results[bucket] = []
        for case_id, title, fn in cases:
            try:
                passed, lines, hint = fn()
            except Exception as exc:
                passed, lines, hint = False, [f"EXCEPTION: {type(exc).__name__}: {exc}"], "Case raised."
            results[bucket].append(CaseResult(bucket, case_id, title, bool(passed), lines, hint))

    print(LINE)
    print(" PHASE 1B SCORECARD")
    print(" Mode : offline | in-memory SQLite cache | RecordedExtractionLLM | InMemoryGraphStore + recording Neo4j driver")
    print(LINE)
    print(f" {'Bucket':<34}{'Passed / Total':>16}{'Score':>12}")
    print(SUBLINE)
    tp = tt = 0
    for bucket, rs in results.items():
        p, t = sum(r.passed for r in rs), len(rs)
        tp, tt = tp + p, tt + t
        print(f" {bucket:<34}{f'{p} / {t}':>16}{100.0 * p / t:>11.1f}%")
    print(SUBLINE)
    print(f" {'OVERALL PHASE 1B SCORE':<34}{f'{tp} / {tt}':>16}{100.0 * tp / tt:>11.1f}%")
    print(LINE)

    print("\n CASE DETAILS")
    print(LINE)
    for rs in results.values():
        for r in rs:
            print(f" [{'PASS' if r.passed else 'FAIL'}] {r.case_id}: {r.title}")
            for line in r.lines:
                print(f"        {line}")
            if not r.passed:
                print(f"        Root Cause Hint: {r.hint}")
    print(LINE)

    live_ok: Optional[bool] = None
    if args.live is not False:
        print("\n OPTIONAL LIVE NEO4J CHECK (not scored; namespaced 'bench1b' data, deleted afterwards)")
        print(LINE)
        live_ok, lines = run_live_neo4j()
        status = "SKIPPED" if live_ok is None else ("PASS" if live_ok else "FAIL")
        print(f" [{status}] idempotent MERGE x3 + multi-chunk accumulation + constraints on {settings.neo4j_uri}")
        for line in lines:
            print(f"        {line}")
        print(LINE)

    failed = tp != tt or live_ok is False or (args.live is True and live_ok is None)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
