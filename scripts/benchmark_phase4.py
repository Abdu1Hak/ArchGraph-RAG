"""Phase 4 scorecard: verbalization & dedup, citation gate & crop-then-fallback, E2E grounded QA, FastAPI traces.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/benchmark_phase4.py

Offline: 12 seed papers, HeuristicRouterLLM, OfflineGroundedGeneratorLLM (and scripted generators
for the citation-gate cases), in-memory stores, FastAPI TestClient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import QueryRequest, RagService, create_app  # noqa: E402
from src.generation.citation_validator import validate_citations  # noqa: E402
from src.generation.synthesizer import (  # noqa: E402
    GroundedAnswerSynthesizer, OfflineGroundedGeneratorLLM, assemble_context, merge_and_rank)
from src.retrieval.cypher_templates import CypherTemplateCatalog, CypherTemplateId as T  # noqa: E402
from src.retrieval.router import RetrievalResult, RouteType as R  # noqa: E402
from src.retrieval.verbalizer import verbalize_hop, verbalize_template_result  # noqa: E402
from src.stores.pgvector_store import RetrievedChunk  # noqa: E402

LINE, SUBLINE = "=" * 100, "-" * 100
COMPOUND = ("Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, "
            "and give the exact W^{DKV} latent compression equation")
MHA, GQA, MLA = ("arch_component::multi_head_attention", "arch_component::grouped_query_attention",
                 "arch_component::multi_head_latent_attention")
DSV2, GQA_C, L3 = "deepseek_2024_v2::s02::c00", "ainslie_2023_gqa::s02::c00", "dubey_2024_llama3::s01::c00"


@dataclass
class Case:
    case_id: str
    title: str
    passed: bool
    lines: List[str] = field(default_factory=list)


class ScriptedGenerator:
    name = "scripted"

    def __init__(self, responses: List[str]) -> None:
        self.responses, self.calls, self.prompts = responses, 0, []

    def generate(self, system_instruction: str, prompt: str) -> str:
        self.prompts.append(prompt)
        out = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return out


def _rc(cid: str, score: float, text: str = "Some passage text about attention.") -> RetrievedChunk:
    return RetrievedChunk(cid, cid.split("::")[0], "Paper", 2024, "architecture", "Sec", text, text, "h", (), score)


# ---------------------------------------------------------------------------
# Bucket 1
# ---------------------------------------------------------------------------


def bucket_verbalization(svc: RagService) -> List[Case]:
    c, out = svc.corpus, []
    hop = c.graph.find_paths(MLA, GQA, "EXTENDS", 1)[0][0]
    hop["source_name"], hop["target_name"] = "Multi-head Latent Attention", "Grouped-Query Attention"
    f = verbalize_hop(hop)
    out.append(Case("VB-01", "Raw triple -> fluent sentence with names, relation, evidence quote and [chunk_id]",
                    f.text.startswith("Multi-head Latent Attention extends Grouped-Query Attention") and f"[{DSV2}]" in f.text
                    and '"Multi-head Latent Attention (MLA) extends' in f.text, [f.text]))

    t4 = CypherTemplateCatalog().run(c.graph, T.T4_BRIDGE_PATH_2_TO_3_HOP, {"entity_a_id": "model_family::deepseek_v3", "entity_b_id": MHA})
    path_fact = next(x for x in verbalize_template_result(t4) if x.kind == "path")
    out.append(Case("VB-02", "3-hop path -> one chain sentence citing every hop's chunk",
                    "DeepSeek-V3 adopts Multi-head Latent Attention, which extends Grouped-Query Attention, which extends Multi-Head Attention"
                    in path_fact.text and len(path_fact.chunk_ids) == 3, [path_fact.text]))

    t5 = CypherTemplateCatalog().run(c.graph, T.T5_CORPUS_AGGREGATION, {"relation_type": "ADOPTS_COMPONENT"})
    aggs = verbalize_template_result(t5)
    top = aggs[0].text if aggs else ""
    out.append(Case("VB-03", "T5 aggregates -> summary bullets with counts and all supporting [chunk_id]s",
                    bool(aggs) and all(a.chunk_ids and all(f"[{i}]" in a.text for i in a.chunk_ids) for a in aggs)
                    and "is adopted by 2 sources" in top, [top, f"{len(aggs)} aggregate facts"]))

    r = svc.engine.retrieve(COMPOUND)
    ctx = assemble_context(COMPOUND, r)
    ids = [p.chunk.chunk_id for p in ctx.passages]
    both = [p for p in ctx.passages if p.source_origin == "both"]
    dsv2_text = svc.corpus.vectors.fetch_chunks_by_ids([DSV2])[0].raw_text.strip()
    out.append(Case("VB-04", "Graph + vector overlap deduplicated by chunk_id (each passage once in the prompt)",
                    len(ids) == len(set(ids)) and DSV2 in [p.chunk.chunk_id for p in both] and ctx.prompt.count(dsv2_text) == 1,
                    [f"supporting={[x.chunk_id for x in r.supporting_chunks]}", f"vector={[x.chunk_id for x in r.vector_chunks]}",
                     f"merged={[(p.chunk.chunk_id, p.source_origin, round(p.final_score, 3)) for p in ctx.passages]}"]))

    ranked = merge_and_rank([_rc("p::s01::c00", 0.40), _rc("t::s01::c00", 0.60)],
                            [_rc("q::s01::c00", 0.92), _rc("p::s01::c00", 0.40), _rc("u::s01::c00", 0.60)])
    order = [(p.chunk.chunk_id, p.source_origin, round(p.final_score, 2)) for p in ranked]
    no_labels = not any(w in ctx.prompt.upper() for w in ("HIGH-PRIORITY", "MUST-USE", "MUST USE"))
    out.append(Case("VB-05", "Order follows relevance: 0.92 single-source > 0.40 corroborated (+0.05); tie broken by corroboration; no priority labels",
                    order[0][0] == "q::s01::c00" and order[-1] == ("p::s01::c00", "both", 0.45) and no_labels
                    and [o[0] for o in order[1:3]] == ["t::s01::c00", "u::s01::c00"],
                    [f"order={order}", f"no HIGH-PRIORITY/MUST-USE labels in prompt={no_labels}"]))
    return out


# ---------------------------------------------------------------------------
# Bucket 2
# ---------------------------------------------------------------------------


def bucket_citations(svc: RagService) -> List[Case]:
    r = svc.engine.retrieve(COMPOUND)
    good1 = f"GQA extends multi-head attention by sharing key-value heads [{GQA_C}]."
    good2 = f"MLA compresses keys and values with W^{{DKV}} into a latent vector [{DSV2}]."
    fake = "GPT-4 uses a secret attention variant [openai_2023_gpt4::s01::c00]."
    off = f"MLA reduces the KV cache by 93.3% [deepseek_2024_v2::s02::c99]."
    uncited = "MLA was also adopted by many later frontier models without any trade-offs."
    out = []

    def run(responses: List[str]):
        gen = ScriptedGenerator(responses)
        return GroundedAnswerSynthesizer(gen, svc.corpus.vectors).synthesize(COMPOUND, r), gen

    a, g = run([f"{good1} {good2}"])
    out.append(Case("CG-01", "Valid citations pass on attempt 1 (0 repairs, 1 generator call)",
                    a.final_validation.valid and a.repair_attempts == 0 and g.calls == 1 and len(a.citations) == 2,
                    [f"calls={g.calls} repairs={a.repair_attempts} citations={[x.chunk_id for x in a.citations]}"]))

    a, g = run([f"{good1} {fake}", f"{good1} {good2}"])
    v0 = a.attempt_validations[0]
    out.append(Case("CG-02", "Invented id [openai_2023_gpt4::s01::c00] -> 1 repair retry -> valid",
                    "openai_2023_gpt4::s01::c00" in v0.invented_ids and a.repair_attempts == 1 and g.calls == 2
                    and a.final_validation.valid and "REPAIR REQUEST" in g.prompts[1] and "openai_2023_gpt4" in g.prompts[1],
                    [f"attempt1: {v0.summary()}", f"final: {a.final_validation.summary()}"]))

    a, g = run([f"{good1} {off}", f"{good1} {good2}"])
    v0 = a.attempt_validations[0]
    out.append(Case("CG-03", "Off-by-one id [deepseek_2024_v2::s02::c99] detected -> 1 repair retry -> valid",
                    "deepseek_2024_v2::s02::c99" in v0.invented_ids and a.repair_attempts == 1 and a.final_validation.valid,
                    [f"attempt1: {v0.summary()}", f"final: {a.final_validation.summary()}"]))

    t0 = time.perf_counter()
    for _ in range(200):
        v = validate_citations(f"{good1} {uncited}", r.context_chunk_ids)
    per_ms = (time.perf_counter() - t0) * 1000 / 200
    out.append(Case("CG-04", "Uncited claim sentence violates claim coverage; validation runs in < 1 ms",
                    not v.valid and v.uncited_claims == [uncited] and per_ms < 1.0,
                    [f"uncited={v.uncited_claims}", f"validate_citations: {per_ms:.3f} ms/call"]))

    a, g = run([f"{good1} {fake} {good2}"])  # same bad answer after the retry
    kept = a.answer.splitlines()
    out.append(Case("CG-05", "Persistent bad sentence after 1 retry -> Step A crops only it, keeps the 2 valid sentences",
                    a.repair_attempts == 1 and g.calls == 2 and a.cropped_sentences_count == 1 and kept == [good1, good2]
                    and not a.used_extractive_fallback and a.final_validation.valid,
                    [f"calls={g.calls} cropped={a.cropped_sentences_count}", f"answer={kept}"]))

    a, g = run([f"{fake} {off} {uncited}"])
    ids_ok = a.final_validation.valid and not a.final_validation.invented_ids
    out.append(Case("CG-06", "All sentences bad after retry -> Step B extractive fallback with 100% valid citations",
                    a.used_extractive_fallback and a.cropped_sentences_count == 3 and ids_ok and len(a.citations) >= 2,
                    [f"cropped={a.cropped_sentences_count} fallback={a.used_extractive_fallback} {a.final_validation.summary()}",
                     f"citations={[x.chunk_id for x in a.citations]}"]))
    return out


# ---------------------------------------------------------------------------
# Bucket 3
# ---------------------------------------------------------------------------

E2E = [
    ("QA-01", "1-hop VECTOR_ONLY", "What is the exact RoPE base frequency used by Llama 3?", R.VECTOR_ONLY, [L3], "500,000"),
    ("QA-02", "2-hop GRAPH_ONLY", "What links Llama 3 to Multi-Head Attention?", R.GRAPH_ONLY, [L3, GQA_C], None),
    ("QA-03", "3-hop GRAPH_ONLY", "What connects DeepSeek-V3 to Multi-Head Attention?", R.GRAPH_ONLY,
     ["deepseek_2024_v3::s02::c00", DSV2, GQA_C], None),
    ("QA-04", "HYBRID_PARALLEL compound", COMPOUND, R.HYBRID_PARALLEL, [GQA_C, DSV2], "W^{DKV}"),
]


def bucket_e2e(svc: RagService) -> List[Case]:
    out, first_try = [], []
    for cid, label, q, route, gold, must in E2E:
        result, g, ms = svc.answer(QueryRequest(question=q))
        cited = [x.chunk_id for x in g.citations]
        first_try.append(g.attempt_validations[0].valid)
        ok = (result.decision.route == route and not g.refused and g.final_validation.valid and all(x in cited for x in gold)
              and (must is None or must in g.answer))
        out.append(Case(cid, f"[{label}] {q}", ok, [f"route={result.decision.route.value} cited={cited} {ms:.1f} ms",
                                                    *[f"  {l}" for l in g.answer.splitlines()[:6]]]))
    gen_before = svc.synthesizer.generator.calls
    result, g, _ = svc.answer(QueryRequest(question="Write me a recipe for chocolate cake."))
    out.append(Case("QA-05", "OUT_OF_SCOPE -> refused=True, 0 generator calls, citations=[]",
                    g.refused and svc.synthesizer.generator.calls == gen_before and g.generator_calls == 0 and not g.citations,
                    [f"answer={g.answer[:90]}..."]))
    out.append(Case("QA-06", "All 4 in-scope answers pass the citation gate on attempt 1 (0 invented, 100% coverage)",
                    all(first_try), [f"attempt-1 valid={first_try}"]))
    return out


# ---------------------------------------------------------------------------
# Bucket 4
# ---------------------------------------------------------------------------


def bucket_api(svc: RagService) -> List[Case]:
    client = TestClient(create_app(svc))
    out = []
    h = client.get("/health")
    hj = h.json()
    out.append(Case("API-01", "GET /health returns readiness and store counts",
                    h.status_code == 200 and hj["status"] == "ok" and hj["vector"]["chunks"] == hj["graph"]["chunks"] > 0,
                    [str(hj)]))

    resp = client.post("/query", json={"question": COMPOUND, "graph_top_k": 3, "vector_top_k": 3})
    j = resp.json()
    core = ("answer", "refused", "route", "query_intent", "confidence", "fallback_triggered", "graph_subquery",
            "vector_subquery", "cypher_template_id", "verbalized_graph_facts", "citations", "repair_attempts",
            "cropped_sentences_count", "latency_ms")
    cites_ok = all(c["text_snippet"] and c["section_path"] and c["paper_id"] for c in j.get("citations", []))
    out.append(Case("API-02", "POST /query returns all core fields with resolved citations",
                    resp.status_code == 200 and all(k in j for k in core) and j["route"] == "HYBRID_PARALLEL"
                    and cites_ok and len(j["citations"]) >= 2,
                    [f"status={resp.status_code} route={j.get('route')} template={j.get('cypher_template_id')} "
                     f"citations={[c['chunk_id'] for c in j.get('citations', [])]} latency={j.get('latency_ms')} ms"]))

    gt = j["graph_trace"]
    sel = {(e["source"], e["relation_type"], e["target"]): e for e in gt["selected_path_edges"]}
    rej = {(e["source"], e["relation_type"], e["target"]) for e in gt["explored_rejected_edges"]}
    want = {(GQA, "EXTENDS", MHA), (MLA, "EXTENDS", GQA)}
    out.append(Case("API-03", "graph_trace: green seeds, selected MHA<-GQA<-MLA path with hop_index, light-red rejected neighbors",
                    MHA in gt["seed_node_ids"] and want <= set(sel) and all(sel[k]["source_chunk_ids"] for k in want)
                    and {sel[k]["hop_index"] for k in want} == {0, 1} and rej and not rej & set(sel),
                    [f"seeds={gt['seed_node_ids']}", f"selected={sorted(sel)}", f"rejected ({len(rej)}): {sorted(rej)[:4]} ..."]))

    ft = j["funnel_trace"]
    stages = ("dense_top_k", "bm25_top_k", "rrf_top_k", "reranked_vector_top_k", "reranked_graph_supporting_top_k",
              "final_deduplicated_llm_context")
    final_ids = [x["chunk_id"] for x in ft["final_deduplicated_llm_context"]]
    expected_final = list(dict.fromkeys([x["chunk_id"] for x in ft["reranked_vector_top_k"]] + [x["chunk_id"] for x in ft["reranked_graph_supporting_top_k"]]))
    oos = client.post("/query", json={"question": "What's the weather in Paris tomorrow?"}).json()
    bad = client.post("/query", json={"question": ""})
    out.append(Case("API-04", "funnel_trace: all 6 stages populated; final context = dedup(vector + graph); OOS empty; 422 on bad input",
                    all(ft[s] for s in stages) and len(final_ids) == len(set(final_ids)) and set(final_ids) == set(expected_final)
                    and oos["refused"] and not oos["funnel_trace"]["dense_top_k"] and bad.status_code == 422,
                    [", ".join(f"{s}={len(ft[s])}" for s in stages), f"final={final_ids}",
                     f"OOS refused={oos['refused']} empty question -> HTTP {bad.status_code}"]))
    return out


def main() -> int:
    t0 = time.perf_counter()
    svc = RagService.offline(telemetry_path=Path(tempfile.mkdtemp()) / "routing.jsonl")
    build = time.perf_counter() - t0
    buckets = {
        "verbalization_and_score_deduplication": bucket_verbalization(svc),
        "citation_gate_and_crop_fallback": bucket_citations(svc),
        "end_to_end_grounded_qa_and_refusal": bucket_e2e(svc),
        "fastapi_endpoint_and_ui_traces": bucket_api(svc),
    }
    print(LINE)
    print(" PHASE 4 SCORECARD")
    print(f" Offline | HeuristicRouterLLM + OfflineGroundedGeneratorLLM | build {build:.2f}s, buckets {time.perf_counter() - t0 - build:.2f}s")
    print(LINE)
    print(f" {'Bucket':<44}{'Passed / Total':>16}{'Score':>12}")
    print(SUBLINE)
    tp = tt = 0
    for name, cs in buckets.items():
        p, t = sum(x.passed for x in cs), len(cs)
        tp, tt = tp + p, tt + t
        print(f" {name:<44}{f'{p} / {t}':>16}{100.0 * p / t:>11.1f}%")
    print(SUBLINE)
    print(f" {'OVERALL PHASE 4 SCORE':<44}{f'{tp} / {tt}':>16}{100.0 * tp / tt:>11.1f}%")
    print(LINE)
    print("\n CASE DETAILS")
    print(LINE)
    for cs in buckets.values():
        for x in cs:
            print(f" [{'PASS' if x.passed else 'FAIL'}] {x.case_id}: {x.title}")
            for line in x.lines:
                print(f"        {line}")
    print(LINE)
    return 0 if tp == tt else 1


if __name__ == "__main__":
    sys.exit(main())
