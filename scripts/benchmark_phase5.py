"""Phase 5 Stress-Test Benchmark: 60 pathway-agnostic questions, 3 systems (+2 sensitivity rows).

Usage:
    .\\.venv\\Scripts\\python.exe scripts/benchmark_phase5.py [--questions PATH] [--out PATH] [--reranker {token-interaction,onnx,api}] [--ablate]

Offline and deterministic: 12 seed papers (synthetic fixtures), DeterministicMockEmbedder,
HeuristicRouterLLM, OfflineGroundedGeneratorLLM. All systems share the same generator, citation
gate, corpus, embedder and reranker; only retrieval differs. Full output -> data/benchmark/phase5_results.json.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.evaluation.benchmark_runner import (  # noqa: E402
    DEFAULT_QUESTIONS,
    STRATA,
    ArchGraphRAG,
    build_systems,
    load_questions,
    run_benchmark,
)
from src.generation.synthesizer import OfflineGroundedGeneratorLLM  # noqa: E402
from src.ingestion.offline_demo import build_offline_corpus  # noqa: E402
from src.retrieval.router import GraphVectorEngine  # noqa: E402

LINE, SUB = "=" * 112, "-" * 112
HEADLINE = ("S1", "S2", "S3")
SHORT = {
    "S1": "Plain Dense",
    "S2": "Tuned Hybrid",
    "S3": "ArchGraph-RAG",
    "S2@k6": "Hybrid top_k=6",
    "S3@3p": "ArchGraph <=3 passages",
    "S3-no-ppr": "S3 (PPR OFF)",
    "S3-no-2turn": "S3 (2-Turn OFF)",
    "S3-base": "S3 (Both OFF)",
}
DEFAULT_OUT = PROJECT_ROOT / "data" / "benchmark" / "phase5_results.json"


def _wire_reranker(vectors: Any, reranker: Any) -> None:
    """Bind `reranker` as the default Stage-2 reranker for all `search_hybrid_reranked` calls."""
    orig_search = vectors.search_hybrid_reranked

    def search_with_reranker(*a: Any, **kw: Any) -> Any:
        if kw.get("reranker") is None:
            kw["reranker"] = reranker
        return orig_search(*a, **kw)

    vectors.reranker = reranker
    vectors.search_hybrid_reranked = search_with_reranker


def _build_ablation_systems(corpus: Any, reranker: Any) -> List[ArchGraphRAG]:
    sig = inspect.signature(GraphVectorEngine.__init__)
    specs = [
        ("S3-no-ppr", "ArchGraph-RAG (PPR=OFF, 2-Turn=ON)", {"enable_ppr": False, "enable_evidence_gap": True}),
        ("S3-no-2turn", "ArchGraph-RAG (PPR=ON, 2-Turn=OFF)", {"enable_ppr": True, "enable_evidence_gap": False}),
        ("S3-base", "ArchGraph-RAG (PPR=OFF, 2-Turn=OFF)", {"enable_ppr": False, "enable_evidence_gap": False}),
    ]
    out: List[ArchGraphRAG] = []
    for key, label, raw_kw in specs:
        filtered = {k: v for k, v in raw_kw.items() if k in sig.parameters}
        sys_obj = ArchGraphRAG(
            corpus,
            OfflineGroundedGeneratorLLM(),
            graph_top_k=3,
            vector_top_k=3,
            reranker=reranker,
            engine_kwargs=filtered,
        )
        sys_obj.key = key
        sys_obj.label = label
        out.append(sys_obj)
    return out


def fmt(v: Any, w: int = 14) -> str:
    return f"{'-' if v is None else f'{v:.1f}%':>{w}}"


def table(title: str, rows: List[tuple], systems: List[str], res: Dict[str, Any], key: str, note: str = "") -> None:
    print(f"\n {title}")
    print(LINE)
    print(f" {'':<34}" + "".join(f"{SHORT[s]:>{19}}" for s in systems))
    print(SUB)
    for label, sub in rows:
        print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary'][key].get(sub) if isinstance(res['systems'][s]['summary'][key], dict)
                                             else res['systems'][s]['summary'][key], 19) for s in systems))
    if note:
        print(SUB + f"\n {note}")
    print(LINE)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def sign_test_p(a_only: int, b_only: int) -> float:
    """Exact two-sided binomial (McNemar exact) test on discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def significance(res: Dict[str, Any], questions) -> Dict[str, Any]:
    per = {s: {p["qid"]: p["strict"] for p in v["per_question"]} for s, v in res["systems"].items()}
    out: Dict[str, Any] = {"wilson_95": {}, "paired_sign_test": {}}
    for s, scores in per.items():
        out["wilson_95"][s] = wilson(int(sum(scores.values())), len(scores))
    for a, b in (("S3", "S2"), ("S3", "S1"), ("S2", "S1"), ("S2@k6", "S3")):
        a_only = sum(1 for q in per[a] if per[a][q] > per[b][q])
        b_only = sum(1 for q in per[a] if per[a][q] < per[b][q])
        out["paired_sign_test"][f"{SHORT[a]} vs {SHORT[b]}"] = {"a_only": a_only, "b_only": b_only, "p": round(sign_test_p(a_only, b_only), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--reranker", choices=["token-interaction", "onnx", "api"], default="token-interaction")
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    corpus = build_offline_corpus()
    reranker = None
    if args.reranker != "token-interaction":
        from src.stores.reranker import create_reranker

        reranker = create_reranker(args.reranker)
        _wire_reranker(corpus.vectors, reranker)

    questions = load_questions(args.questions)
    systems = build_systems(corpus, reranker=reranker)
    ablation_keys: List[str] = []
    if args.ablate:
        ablations = _build_ablation_systems(corpus, reranker=reranker)
        ablation_keys = [s.key for s in ablations]
        systems.extend(ablations)

    res = run_benchmark(corpus, questions, systems=systems)
    elapsed = time.perf_counter() - t0
    qmap = {q.id: q for q in questions}
    counts = {s: sum(q.stratum == s for q in questions) for s in STRATA}
    systems_all = list(res["systems"])

    print(LINE)
    print(" PHASE 5 STRESS-TEST BENCHMARK REPORT")
    print(f" {len(questions)} questions ({', '.join(f'{v} {k}' for k, v in counts.items())}) | corpus: "
          f"{res['corpus']['papers']} papers, {res['corpus']['chunks']} chunks, {res['corpus']['graph_entities']} entities, "
          f"{res['corpus']['graph_relationships']} relationships | reranker={args.reranker} | run {elapsed:.2f}s")
    print(" Offline: synthetic fixtures + mock embedder + heuristic router + extractive offline generator (same generator for all systems).")
    print(f" Benchmark validity (oracle): {res['oracle']['oracle_satisfiable']}/{res['oracle']['answerable']} answerable questions are fully "
          "answerable from their required chunks.")
    for s in systems_all:
        print(f"   {s:<12} {res['systems'][s]['label']}")
    print(LINE)

    strata_rows = [(f"{s} (n={counts[s]})", s) for s in STRATA]
    print("\n HEADLINE: STRICT ALL-HOPS ANSWER ACCURACY BY STRATUM")
    print(LINE)
    print(f" {'':<34}" + "".join(f"{SHORT[s]:>19}" for s in HEADLINE))
    print(SUB)
    for label, sub in strata_rows:
        print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary']['strict_by_stratum'][sub], 19) for s in HEADLINE))
    print(SUB)
    n_multi = counts["two_hop"] + counts["three_hop"]
    for label, key in ((f"Multi-hop (2+3 hop, n={n_multi})", "strict_multi_hop"), (f"All answerable (n={res['oracle']['answerable']})", "strict_answerable"),
                       (f"OVERALL (n={len(questions)})", "strict_overall")):
        print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary'][key], 19) for s in HEADLINE))
    print(LINE)

    table("SECONDARY: PROPORTIONAL CONCEPT/HOP RECALL (k/N concept groups satisfied)",
          [(f"{s} (n={counts[s]})", s) for s in STRATA if s != "out_of_scope"], list(HEADLINE), res, "concept_recall_by_stratum")
    print("\n SECONDARY: EVIDENCE & CITATIONS (answerable questions)")
    print(LINE)
    print(f" {'':<34}" + "".join(f"{SHORT[s]:>19}" for s in HEADLINE))
    print(SUB)
    for label, key in (("Concept recall (all answerable)", "concept_recall_answerable"),
                       ("Evidence chunk recall (in context)", "evidence_recall_answerable"),
                       ("Citation gate valid (answered)", "citation_gate_valid_rate"),
                       ("Citation support precision", "citation_support_precision"),
                       ("Faithfulness (answerable)", "faithfulness_answerable"),
                       ("False refusal/abstention rate", "false_refusal_rate_answerable")):
        print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary'][key], 19) for s in HEADLINE))
    print(SUB + "\n Citation support precision = cited chunks that are required evidence or contain a required concept / all cited chunks.")
    print(LINE)

    print("\n SUMMARY BY SYSTEM (accuracy, concept recall, and claim-level faithfulness)")
    print(LINE)
    print(f" {'System':<28}{'Strict All':>14}{'Strict Ans':>14}{'Multi-hop':>14}{'Concept Rec':>14}{'Faithfulness':>16}")
    print(SUB)
    for s in systems_all:
        m = res["systems"][s]["summary"]
        print(
            f" {SHORT.get(s, s):<28}"
            f"{fmt(m['strict_overall'], 14)}"
            f"{fmt(m['strict_answerable'], 14)}"
            f"{fmt(m['strict_multi_hop'], 14)}"
            f"{fmt(m['concept_recall_answerable'], 14)}"
            f"{fmt(m['faithfulness_answerable'], 16)}"
        )
    print(LINE)

    stress_counts = {t: sum(q.stress_type == t for q in questions) for t in ("direct", "paraphrased", "implicit_bridge", "caveat_trap", "cross_pillar", "out_of_scope")}
    table("STRICT ACCURACY BY STRESS TYPE", [(f"{t} (n={n})", t) for t, n in stress_counts.items()], list(HEADLINE), res, "strict_by_stress_type")

    print("\n SENSITIVITY: IS THE GRAPH ADVANTAGE JUST MORE CONTEXT? (strict accuracy)")
    print(LINE)
    sens = ["S2", "S2@k6", "S3", "S3@3p"]
    print(f" {'':<34}" + "".join(f"{SHORT[s]:>24}" for s in sens))
    print(SUB)
    for label, sub in strata_rows[:4]:
        print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary']['strict_by_stratum'][sub], 24) for s in sens))
    print(f" {'All answerable':<34}" + "".join(fmt(res['systems'][s]['summary']['strict_answerable'], 24) for s in sens))
    print(SUB + "\n Hybrid top_k=6 reads as many passages as ArchGraph-RAG can (3 graph + 3 vector); 'ArchGraph <=3 passages' caps it to the baselines' budget.")
    print(LINE)

    if ablation_keys:
        print("\n ABLATION: S3 PPR & 2-TURN EVIDENCE-GAP ON vs OFF")
        print(LINE)
        abl = ["S3"] + ablation_keys
        print(f" {'':<34}" + "".join(f"{SHORT.get(s, s):>19}" for s in abl))
        print(SUB)
        for label, sub in strata_rows[:4]:
            print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary']['strict_by_stratum'][sub], 19) for s in abl))
        for label, key in (
            ("All answerable", "strict_answerable"),
            ("OVERALL", "strict_overall"),
            ("Faithfulness (answerable)", "faithfulness_answerable"),
        ):
            print(f" {label:<34}" + "".join(fmt(res['systems'][s]['summary'][key], 19) for s in abl))
        print(LINE)

    print("\n PATHWAY DISTRIBUTION (ArchGraph-RAG; logged, never graded)")
    print(LINE)
    print(f" {'pathway':<52}{'questions':>12}{'passing':>12}{'pass rate':>12}")
    print(SUB)
    for p, row in res["systems"]["S3"]["summary"]["pathway_distribution"].items():
        print(f" {p:<52}{row['questions']:>12}{row['passing']:>12}{100.0 * row['passing'] / row['questions']:>11.1f}%")
    print(LINE)

    oc = res["one_time_costs"]
    print("\n LATENCY & COST (offline latency measured in-process; USD projected from token counts at configured rates)")
    print(LINE)
    print(f" {'':<34}{'p50 ms':>12}{'p95 ms':>12}{'gen calls/q':>14}{'$/query':>14}{'one-time $':>14}")
    print(SUB)
    for s in systems_all:
        m = res["systems"][s]["summary"]
        print(f" {SHORT.get(s, s):<34}{m['latency_ms_p50']:>12.2f}{m['latency_ms_p95']:>12.2f}{m['generator_calls_mean']:>14.2f}"
              f"{m['projected_cost_per_query_usd']:>14.6f}{m['one_time_cost_usd']:>14.6f}")
    print(SUB)
    print(f" One-time: embedding {res['corpus']['chunks']} chunks = ${oc['embedding_usd']:.6f} (all systems); "
          f"LLM graph extraction = ${oc['graph_extraction_usd']:.6f} (ArchGraph-RAG only; Phase 1B estimator).")
    print(f" Rates: router ${settings.router_input_usd_per_1m}/{settings.router_output_usd_per_1m}, generation "
          f"${settings.generation_input_usd_per_1m}/{settings.generation_output_usd_per_1m}, embedding ${settings.embedding_usd_per_1m} "
          "per 1M tokens in/out (assumed; set in .env). Synthetic chunks are far shorter than real papers.")
    print(LINE)

    print("\n FAILURE ANALYSIS: WHY EACH SYSTEM MISSED (answerable questions, strict = 0)")
    print(LINE)
    for s in HEADLINE:
        causes: Dict[str, int] = {}
        for p in res["systems"][s]["per_question"]:
            if p["strict"] < 1 and qmap[p["qid"]].answerable:
                cause = p["failure"].split(":")[0]
                causes[cause] = causes.get(cause, 0) + 1
        print(f" {SHORT[s]:<16} " + "; ".join(f"{k} x{v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])))
    print(LINE)

    s1, s2, s3 = ({p["qid"]: p for p in res["systems"][k]["per_question"]} for k in HEADLINE)
    print("\n HARDEST QUESTIONS: ArchGraph-RAG misses (with what the baselines did)")
    print(LINE)
    for qid, p in s3.items():
        if p["strict"] >= 1:
            continue
        q = qmap[qid]
        print(f" [{qid}] {q.stratum}/{q.stress_type}  S1={'PASS' if s1[qid]['strict'] else 'fail'}  "
              f"S2={'PASS' if s2[qid]['strict'] else 'fail'}  S3=fail  pathway={p['pathway']}")
        print(f"   Q: {q.question}")
        print(f"   why: {p['failure']}")
        for m in p["missing_groups"]:
            print(f"     missing {m['group'][:3]} -> {m['diagnosis']}")
        if p["evidence_recall"] is not None:
            print(f"   evidence recall {p['evidence_recall']:.2f}; context={p['context_chunk_ids'][:6]}")
    print(LINE)

    wins = [qid for qid in s3 if s3[qid]["strict"] > s2[qid]["strict"]]
    losses = [qid for qid in s3 if s3[qid]["strict"] < s2[qid]["strict"]]
    print(f"\n HEAD-TO-HEAD ArchGraph-RAG vs Tuned Hybrid: S3 wins {len(wins)} {wins} | S3 loses {len(losses)} {losses}")
    print(LINE)

    stats = significance(res, questions)
    res["statistics"] = stats
    print("\n STATISTICAL CONTEXT (strict accuracy, all 60 questions)")
    print(LINE)
    for s in systems_all:
        lo, hi = stats["wilson_95"][s]
        print(f" {SHORT.get(s, s):<26} {res['systems'][s]['summary']['strict_overall']:>6.1f}%   Wilson 95% CI [{lo:5.1f}%, {hi:5.1f}%]")
    print(SUB)
    for pair, row in stats["paired_sign_test"].items():
        print(f" {pair:<26} discordant: first-only {row['a_only']}, second-only {row['b_only']}  exact two-sided p = {row['p']:.3f}")
    print(LINE)

    before_path = args.out.with_name(args.out.stem + "_before.json")
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        print(f"\n BEFORE vs AFTER (same frozen questions; before = {before_path.name})")
        print(LINE)
        print(f" {'Strict accuracy':<34}" + "".join(f"{SHORT[s] + ' before':>27}{'after':>9}" for s in HEADLINE))
        print(SUB)
        rows = [(f"{s} (n={counts[s]})", "strict_by_stratum", s) for s in STRATA] + [
            ("All answerable", "strict_answerable", None), ("OVERALL", "strict_overall", None)]
        for label, key, sub in rows:
            cells = ""
            for s in HEADLINE:
                b = before["systems"][s]["summary"][key]
                a = res["systems"][s]["summary"][key]
                b, a = (b[sub], a[sub]) if sub else (b, a)
                cells += f"{b:>26.1f}%{a:>8.1f}%"
            print(f" {label:<34}" + cells)
        print(LINE)
        b3 = {p["qid"]: p["strict"] for p in before["systems"]["S3"]["per_question"]}
        a3 = {p["qid"]: p["strict"] for p in res["systems"]["S3"]["per_question"]}
        fixed = [q for q in a3 if a3[q] > b3.get(q, 0)]
        broke = [q for q in a3 if a3[q] < b3.get(q, 0)]
        print(f" ArchGraph-RAG newly passing ({len(fixed)}): {fixed}")
        print(f" ArchGraph-RAG newly failing ({len(broke)}): {broke}")
        print(LINE)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f" Full results saved to {args.out.relative_to(PROJECT_ROOT) if args.out.is_relative_to(PROJECT_ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
