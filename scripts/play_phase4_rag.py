"""Interactive end-to-end Graph+Vector RAG CLI (Phase 4) with PATH & QUOTA TELEMETRY.

Examples:
    .\\.venv\\Scripts\\python.exe scripts/play_phase4_rag.py --query "Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, and give the exact W^{DKV} latent compression equation"
    .\\.venv\\Scripts\\python.exe scripts/play_phase4_rag.py --live -q "Which techniques mitigate the KV cache bottleneck?"
    .\\.venv\\Scripts\\python.exe scripts/play_phase4_rag.py            # prompt loop

Offline by default (12 seed papers, heuristic router, extractive offline generator, 0 API calls).
`--live`: Neo4j + pgvector + GeminiEmbedder + GeminiRouterLLM (agent router, no heuristic fallback)
+ GeminiGeneratorLLM. `--gemini`: offline corpus with GeminiGeneratorLLM.
Every query prints the path taken and the exact API requests / tokens it consumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import QueryRequest, RagService  # noqa: E402
from src.generation.synthesizer import GeminiGeneratorLLM  # noqa: E402
from src.ingestion.rate_limiter import AllModelsExhaustedError  # noqa: E402
from src.retrieval.router import GeminiRouterLLM, RouterUnavailableError  # noqa: E402

LINE, SUB = "=" * 100, "-" * 100


def _s(eid: str) -> str:
    return eid.split("::", 1)[-1]


def _path_text(path) -> str:
    node = path[0]["source"]
    if len(path) > 1 and node in (path[1]["source"], path[1]["target"]):
        node = path[0]["target"]
    out = f"({_s(node)})"
    for h in path:
        fwd = h["source"] == node
        out += f"-[:{h['relation_type']}]->({_s(h['target'])})" if fwd else f"<-[:{h['relation_type']}]-({_s(h['source'])})"
        node = h["target"] if fwd else h["source"]
    return out


def show(svc: RagService, req: QueryRequest, show_prompt: bool) -> None:
    result, g, ms, usage = svc.answer_with_usage(req)
    d = result.decision
    print(LINE)
    print(f" QUESTION: {req.question}")
    print(LINE)
    print(" ANSWER" + ("  [REFUSED]" if g.refused else "  [ABSTAINED]" if g.abstained else ""))
    print(LINE)
    print(textwrap.indent(g.answer, "  "))
    if g.citations:
        print("\n CITATIONS")
        print(SUB)
        for c in g.citations:
            score = f"{c.score:.3f}" if c.score is not None else "-"
            print(f"  [{c.chunk_id}] {c.section_path}  (source: {c.source_origin}, score {score})")
            print(textwrap.indent(textwrap.fill(c.text_snippet.replace("\n", " "), 92), "      "))

    print("\n" + LINE)
    print(" PATH & QUOTA TELEMETRY")
    print(LINE)
    print(" 1) PATH TAKEN")
    print(SUB)
    upgraded = (" (low-confidence upgrade from " + d.original_route.value + ")" if d.fallback_triggered else
                " (structural upgrade from " + d.original_route.value + ")" if d.structural_upgrade else "")
    print(f"  Agent router     : {d.router_model}  |  routing {d.latency_ms} ms, end-to-end {ms:.1f} ms")
    print(f"  Route            : {d.route.value}{upgraded}")
    print(f"  Intent           : {d.query_intent.value}   confidence={d.confidence}")
    print(f"  Graph sub-query  : {d.graph_subquery or '-'}")
    print(f"  Vector sub-query : {d.vector_subquery or '-'}")
    tmpl = d.cypher_template_id.value if d.cypher_template_id else "-"
    cascade = (" + cascade " + ", ".join(t.value for t in d.cascade_template_ids)) if d.cascade_template_ids else ""
    print(f"  Cypher template  : {tmpl}{cascade}   $params={json.dumps(d.template_params)}")
    if d.resolved_entity_ids:
        print(f"  Seed entities    : {', '.join(_s(e) for e in d.resolved_entity_ids)}" + ("  [vector-seeded]" if d.vector_seeded else ""))
    for r in d.fallback_reasons:
        print(f"  Decision note    : {r}")
    tr = result.template_result
    if tr and tr.paths:
        print("  Ranked graph paths traversed:")
        for i, p in enumerate(tr.paths[:5], start=1):
            score = f" score={tr.hop_scores[i - 1]:.2f}" if tr.hop_scores and i - 1 < len(tr.hop_scores) else ""
            print(f"    {i}. {_path_text(p)}{score}")
    print("  Graph-supporting chunks (per-hop stratified): "
          + (", ".join(f"{c.chunk_id} ({c.score:.2f})" for c in result.supporting_chunks) or "-"))
    print("  Vector chunks (RRF + rerank)                : "
          + (", ".join(f"{c.chunk_id} ({c.score:.2f})" for c in result.vector_chunks) or "-"))
    if g.context:
        print("  Final deduplicated LLM context             : "
              + ", ".join(f"{p.chunk.chunk_id}[{p.source_origin}]" for p in g.context.passages))
    v = g.final_validation
    print(f"  Citation gate    : valid={v.valid if v else 'n/a'}  invented={v.invented_ids if v else []}  "
          f"generator calls={g.generator_calls}  repairs={g.repair_attempts}  cropped={g.cropped_sentences_count}  "
          f"extractive_fallback={g.used_extractive_fallback}")

    u = usage
    print(SUB)
    print(" 2) CREDITS & TOKENS CONSUMED")
    print(SUB)
    print(f"  {'Component':<22}{'Model':<28}{'API req':>9}{'Input tok':>12}{'Output tok':>12}{'Total tok':>12}")
    rows = [
        ("Agent Router", u.router_model, u.router_requests, u.router_input_tokens, u.router_output_tokens),
        ("Vector Embedder", u.embedding_model, u.embedding_requests_used, u.embedding_tokens_est, 0),
        ("Answer Generator", u.generator_model, u.generator_requests, u.generator_input_tokens, u.generator_output_tokens),
    ]
    for name, model, req_n, inp, out in rows:
        print(f"  {name:<22}{model[:27]:<28}{req_n:>9}{inp:>12,}{out:>12,}{inp + out:>12,}")
    print(f"  {'Total Query Footprint':<22}{'':<28}{u.total_api_requests_used:>9}{u.total_input_tokens:>12,}"
          f"{u.total_output_tokens:>12,}{u.total_tokens:>12,}")
    print(f"  LLM requests={u.llm_requests_used}  embedding requests={u.embedding_requests_used} "
          f"(cache hits {u.embedding_cache_hits}; embedding tokens are estimates)  rate-limit wait={u.rate_limit_wait_s}s")
    if u.note:
        print(f"  Note: {u.note}")
    if show_prompt and g.context:
        print("\n PROMPT\n" + LINE + "\n" + g.context.prompt)
    print(LINE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 end-to-end RAG CLI (offline by default).")
    ap.add_argument("--query", "-q")
    ap.add_argument("--graph-top-k", type=int, default=3)
    ap.add_argument("--vector-top-k", type=int, default=3)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--gemini", action="store_true", help="Offline corpus, Gemini generator.")
    ap.add_argument("--live", action="store_true", help="Live mode: Neo4j + pgvector + Gemini agent router/generator.")
    args = ap.parse_args()
    if args.live:
        try:
            svc = RagService.live()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 3
    else:
        svc = RagService.offline(
            generator=GeminiGeneratorLLM() if args.gemini else None,
            router_llm=GeminiRouterLLM(allow_heuristic_fallback=False) if args.gemini else None,
        )

    def run(q: str) -> int:
        try:
            show(svc, QueryRequest(question=q, graph_top_k=args.graph_top_k, vector_top_k=args.vector_top_k), args.show_prompt)
            return 0
        except AllModelsExhaustedError as exc:
            print(f"DAILY API QUOTA REACHED for {', '.join(exc.models)}. Swap GEMINI_API_KEY in .env and re-run.", file=sys.stderr)
            return 4
        except RouterUnavailableError as exc:
            print(f"Agent router unavailable: {exc}", file=sys.stderr)
            return 5

    if args.query:
        return run(args.query)
    try:
        while True:
            q = input("\nquestion> ").strip()
            if q in ("", "exit", "quit"):
                return 0
            run(q)
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    sys.exit(main())
