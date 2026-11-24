"""Interactive end-to-end Graph+Vector RAG CLI (Phase 4).

Examples:
    .\\.venv\\Scripts\\python.exe scripts/play_phase4_rag.py --query "Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, and give the exact W^{DKV} latent compression equation"
    .\\.venv\\Scripts\\python.exe scripts/play_phase4_rag.py --show-prompt -q "What links Llama 3 to Multi-Head Attention?"
    .\\.venv\\Scripts\\python.exe scripts/play_phase4_rag.py            # prompt loop

Offline by default (12 seed papers, heuristic router, extractive offline generator). `--gemini` uses
GeminiGeneratorLLM (needs GEMINI_API_KEY).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import QueryRequest, RagService  # noqa: E402
from src.generation.synthesizer import GeminiGeneratorLLM  # noqa: E402

LINE = "=" * 100


def show(svc: RagService, req: QueryRequest, show_prompt: bool) -> None:
    result, g, ms = svc.answer(req)
    d = result.decision
    print(LINE)
    print(f" QUESTION: {req.question}")
    print(f" route={d.route.value} intent={d.query_intent.value} template={d.cypher_template_id.value if d.cypher_template_id else '-'} "
          f"confidence={d.confidence}{' (fallback)' if d.fallback_triggered else ''} | {ms:.1f} ms")
    if d.graph_subquery or d.vector_subquery:
        print(f" graph sub-query : {d.graph_subquery or '-'}\n vector sub-query: {d.vector_subquery or '-'}")
    print(LINE)
    print(" ANSWER" + ("  [REFUSED]" if g.refused else ""))
    print(LINE)
    print(textwrap.indent(g.answer, "  "))
    if not g.refused:
        v = g.final_validation
        print(f"\n citation gate: valid={v.valid if v else None} generator_calls={g.generator_calls} repairs={g.repair_attempts} "
              f"cropped={g.cropped_sentences_count} extractive_fallback={g.used_extractive_fallback}")
        print("\n CITATIONS")
        print(LINE)
        for c in g.citations:
            score = f"{c.score:.3f}" if c.score is not None else "-"
            print(f"  [{c.chunk_id}] {c.section_path}  (source: {c.source_origin}, score {score})")
            print(textwrap.indent(textwrap.fill(c.text_snippet.replace("\n", " "), 92), "      "))
        if g.context:
            print("\n DEDUPLICATED LLM CONTEXT (by relevance)")
            print(LINE)
            for i, p in enumerate(g.context.passages, start=1):
                print(f"  {i}. {p.chunk.chunk_id}  final={p.final_score:.3f} relevance={p.relevance_score:.3f} source={p.source_origin}")
        if show_prompt and g.context:
            print("\n PROMPT\n" + LINE + "\n" + g.context.prompt)
    print(LINE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 end-to-end RAG CLI (offline by default).")
    ap.add_argument("--query", "-q")
    ap.add_argument("--graph-top-k", type=int, default=3)
    ap.add_argument("--vector-top-k", type=int, default=3)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--gemini", action="store_true")
    args = ap.parse_args()
    svc = RagService.offline(generator=GeminiGeneratorLLM() if args.gemini else None)

    def run(q: str) -> None:
        show(svc, QueryRequest(question=q, graph_top_k=args.graph_top_k, vector_top_k=args.vector_top_k), args.show_prompt)

    if args.query:
        run(args.query)
        return 0
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
