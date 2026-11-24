"""Phase 5: pathway-agnostic 3-system comparison runner.

Systems (identical generator, citation gate, corpus, embedder and reranker; only retrieval differs):
  S1 Plain Dense Vector RAG      - search_dense(top_k)
  S2 Tuned Hybrid Vector RAG     - Dense + BM25 RRF + cross-encoder rerank (top_k)
  S3 ArchGraph-RAG               - router + Cypher templates + reranked graph chunks + hybrid vector
Sensitivity rows (not headline): S2 with top_k=6 (does S3 win only because it reads more passages?)
and S3 capped at 3 passages (budget-matched to S1/S2).

Grading never looks at the route or template a system chose. Each answer is scored only against
the question's `concept_groups` (any synonym in a group satisfies it), the citation gate, and
refusal behavior for out-of-scope questions. See `grade()` for exact rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.generation.citation_validator import CITATION_RE
from src.generation.synthesizer import (
    GENERATION_SYSTEM_PROMPT,
    GroundedAnswer,
    GroundedAnswerSynthesizer,
    OfflineGroundedGeneratorLLM,
)
from src.ingestion.chunker import estimate_tokens
from src.ingestion.cost_estimator import CostBudgetEstimator, ExtractionCache
from src.retrieval.router import (
    ROUTER_SYSTEM_PROMPT,
    GraphVectorEngine,
    QueryIntent,
    QueryRouter,
    RetrievalResult,
    RouteType,
    RoutingDecision,
)

STRATA = ("single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope")
STRESS_TYPES = ("direct", "paraphrased", "implicit_bridge", "caveat_trap", "cross_pillar", "out_of_scope")
DEFAULT_QUESTIONS = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "phase5_eval_questions.json"


# ---------------------------------------------------------------------------
# Question schema
# ---------------------------------------------------------------------------


class EvalQuestion(BaseModel):
    id: str
    stratum: str
    hop_count: int = Field(ge=0)
    stress_type: str
    question: str = Field(min_length=5)
    concept_groups: List[List[str]] = Field(default_factory=list)
    required_chunk_ids: List[str] = Field(default_factory=list)

    @field_validator("stratum")
    @classmethod
    def _stratum(cls, v: str) -> str:
        if v not in STRATA:
            raise ValueError(f"unknown stratum {v}")
        return v

    @field_validator("stress_type")
    @classmethod
    def _stress(cls, v: str) -> str:
        if v not in STRESS_TYPES:
            raise ValueError(f"unknown stress_type {v}")
        return v

    @property
    def answerable(self) -> bool:
        return self.stratum != "out_of_scope"


def load_questions(path: Optional[Path] = None) -> List[EvalQuestion]:
    data = json.loads(Path(path or DEFAULT_QUESTIONS).read_text(encoding="utf-8"))
    qs = [EvalQuestion.model_validate(q) for q in data["questions"]]
    ids = [q.id for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    for q in qs:
        if q.answerable and (not q.concept_groups or not q.required_chunk_ids):
            raise ValueError(f"{q.id}: answerable questions need concept_groups and required_chunk_ids")
        if not q.answerable and q.concept_groups:
            raise ValueError(f"{q.id}: out_of_scope questions must not have concept_groups")
    return qs


# ---------------------------------------------------------------------------
# Concept matching
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lower-case; drop thousands separators (500,000 -> 500000); hyphen/underscore/dash -> space."""
    t = text.lower().replace("\u2013", " ").replace("\u2014", " ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = re.sub(r"[-_]", " ", t)
    return re.sub(r"\s+", " ", t)


def phrase_present(text: str, phrase: str) -> bool:
    p = normalize_text(phrase).strip()
    return bool(p) and re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", normalize_text(text)) is not None


def groups_satisfied(text: str, groups: Sequence[Sequence[str]]) -> List[bool]:
    return [any(phrase_present(text, p) for p in g) for g in groups]


def strip_citations(answer: str) -> str:
    """Citation ids contain words (e.g. 'peng_2023_yarn') that must never count as answer content."""
    return CITATION_RE.sub(" ", answer)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


class CountingGenerator:
    """Wraps a generator to record projected input/output tokens of every call."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "generator")
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(self, system_instruction: str, prompt: str) -> str:
        out = self.inner.generate(system_instruction, prompt)
        self.calls += 1
        self.input_tokens += estimate_tokens(system_instruction) + estimate_tokens(prompt)
        self.output_tokens += estimate_tokens(out)
        return out


class CountingEmbedder:
    """Wraps the vector store's embedder to count query-embedding calls and tokens."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.dim = inner, getattr(inner, "dim", 768)
        self.query_calls = 0
        self.query_tokens = 0

    def embed_query(self, text: str):
        self.query_calls += 1
        self.query_tokens += estimate_tokens(text)
        return self.inner.embed_query(text)

    def embed_documents(self, texts, content_hashes=None):
        return self.inner.embed_documents(texts, content_hashes)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


@dataclass
class SystemOutput:
    answer: str
    refused: bool
    abstained: bool
    gate_valid: bool
    cited_ids: List[str]
    context_chunk_ids: List[str]
    context_text: str
    pathway: str
    route: Optional[str]
    template: Optional[str]
    fallback_triggered: bool
    vector_seeded: bool
    generator_calls: int
    latency_ms: float
    generation_input_tokens: int
    generation_output_tokens: int
    router_input_tokens: int
    router_output_tokens: int
    embedding_tokens: int


def _vector_decision(question: str) -> RoutingDecision:
    return RoutingDecision(question=question, standalone_query=question, vector_subquery=question,
                           route=RouteType.VECTOR_ONLY, original_route=RouteType.VECTOR_ONLY,
                           query_intent=QueryIntent.SINGLE_FACT_DEFINITION, confidence=1.0, router_model="none")


def _context_of(g: GroundedAnswer) -> Tuple[List[str], str]:
    if g.context is None:
        return [], ""
    ids = [p.chunk.chunk_id for p in g.context.passages]
    ids += [c for f in g.context.facts for c in f.chunk_ids if c not in ids]
    text = "\n".join(f.text for f in g.context.facts) + "\n" + "\n".join(p.chunk.raw_text for p in g.context.passages)
    return ids, text


class RagSystem:
    key: str = ""
    label: str = ""
    one_time_cost_usd: float = 0.0

    def __init__(self, vectors: Any, generator: Any, max_passages: int = 6) -> None:
        self.vectors = vectors
        self.gen = CountingGenerator(generator)
        self.synth = GroundedAnswerSynthesizer(self.gen, vectors, max_passages=max_passages)

    def retrieve(self, question: str) -> Tuple[RetrievalResult, str]:
        raise NotImplementedError

    def run(self, question: str) -> SystemOutput:
        emb = self.vectors.embedder
        g0 = (self.gen.calls, self.gen.input_tokens, self.gen.output_tokens)
        e0 = getattr(emb, "query_tokens", 0) if isinstance(emb, CountingEmbedder) else 0
        t0 = time.perf_counter()
        result, pathway = self.retrieve(question)
        grounded = self.synth.synthesize(question, result)
        ms = (time.perf_counter() - t0) * 1000
        ctx_ids, ctx_text = _context_of(grounded)
        d = result.decision
        router_in = router_out = 0
        if getattr(self, "uses_router", False):
            router_in = estimate_tokens(ROUTER_SYSTEM_PROMPT) + estimate_tokens(question)
            router_out = 120  # typical size of the structured RouterLLMOutput JSON
        return SystemOutput(
            answer=grounded.answer, refused=grounded.refused, abstained=grounded.abstained,
            gate_valid=bool(grounded.final_validation and grounded.final_validation.valid),
            cited_ids=[c.chunk_id for c in grounded.citations], context_chunk_ids=ctx_ids, context_text=ctx_text,
            pathway=pathway, route=d.route.value, template=d.cypher_template_id.value if d.cypher_template_id else None,
            fallback_triggered=d.fallback_triggered, vector_seeded=d.vector_seeded,
            generator_calls=self.gen.calls - g0[0], latency_ms=ms,
            generation_input_tokens=self.gen.input_tokens - g0[1], generation_output_tokens=self.gen.output_tokens - g0[2],
            router_input_tokens=router_in, router_output_tokens=router_out,
            embedding_tokens=(emb.query_tokens - e0) if isinstance(emb, CountingEmbedder) else estimate_tokens(question),
        )


class PlainDenseRAG(RagSystem):
    key, label = "S1", "Plain Dense Vector RAG"

    def __init__(self, vectors: Any, generator: Any, top_k: int = 3) -> None:
        super().__init__(vectors, generator)
        self.top_k = top_k
        self.label = f"{self.label} (top_k={top_k})"

    def retrieve(self, question: str) -> Tuple[RetrievalResult, str]:
        hits = self.vectors.search_dense(question, top_k=self.top_k)
        return RetrievalResult(decision=_vector_decision(question), vector_chunks=hits), "VECTOR_DENSE"


class TunedHybridRAG(RagSystem):
    key, label = "S2", "Tuned Hybrid Vector RAG (RRF + Reranker)"

    def __init__(self, vectors: Any, generator: Any, top_k: int = 3, candidate_k: int = 15) -> None:
        super().__init__(vectors, generator)
        self.top_k, self.candidate_k = top_k, candidate_k
        self.label = f"{self.label} (top_k={top_k})"

    def retrieve(self, question: str) -> Tuple[RetrievalResult, str]:
        hits = self.vectors.search_hybrid_reranked(question, top_k=self.top_k, candidate_k=max(self.candidate_k, self.top_k))
        return RetrievalResult(decision=_vector_decision(question), vector_chunks=hits), "VECTOR_RRF_RERANK"


class ArchGraphRAG(RagSystem):
    key, label = "S3", "ArchGraph-RAG"
    uses_router = True

    def __init__(self, corpus: Any, generator: Any, graph_top_k: int = 3, vector_top_k: int = 3,
                 max_passages: int = 6, telemetry_path: Optional[Path] = None) -> None:
        super().__init__(corpus.vectors, generator, max_passages=max_passages)
        router = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors,
                             telemetry_path=telemetry_path or Path(tempfile.mkdtemp()) / "routing.jsonl")
        self.engine = GraphVectorEngine(router)
        self.graph_top_k, self.vector_top_k = graph_top_k, vector_top_k
        self.label = f"{self.label} (graph_top_k={graph_top_k}, vector_top_k={vector_top_k}, max_passages={max_passages})"

    def retrieve(self, question: str) -> Tuple[RetrievalResult, str]:
        r = self.engine.retrieve(question, graph_top_k=self.graph_top_k, vector_top_k=self.vector_top_k)
        d = r.decision
        if d.route == RouteType.OUT_OF_SCOPE:
            return r, "OUT_OF_SCOPE_REFUSAL"
        t = d.cypher_template_id.value.split("_")[0] if d.cypher_template_id else None
        if t == "T6":
            t = "T6_FALLBACK" + ("_VECTOR_SEEDED" if d.vector_seeded else "")
        base = {"VECTOR_ONLY": "VECTOR", "GRAPH_ONLY": "GRAPH", "HYBRID_PARALLEL": "HYBRID"}[d.route.value]
        if d.fallback_triggered:
            base += "(low-conf)"
        return r, base + (f":{t}" if t else "")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class QuestionGrade:
    qid: str
    system: str
    strict: float
    concept_recall: float
    groups_hit: List[bool]
    evidence_recall: Optional[float]
    citation_support_precision: Optional[float]
    gate_valid: bool
    refused_or_abstained: bool
    pathway: str
    failure: str = ""
    missing_groups: List[Dict[str, Any]] = field(default_factory=list)


def grade(q: EvalQuestion, out: SystemOutput, chunk_text: Callable[[str], str], system: str) -> QuestionGrade:
    declined = out.refused or out.abstained
    answer_text = strip_citations(out.answer)
    if not q.answerable:
        ok = declined
        return QuestionGrade(q.id, system, float(ok), float(ok), [], None, None, out.gate_valid, declined, out.pathway,
                             "" if ok else "answered an out-of-scope question instead of abstaining")

    hits = [False] * len(q.concept_groups) if declined else groups_satisfied(answer_text, q.concept_groups)
    recall = sum(hits) / len(hits)
    ev = sum(c in out.context_chunk_ids for c in q.required_chunk_ids) / len(q.required_chunk_ids)
    support = None
    if out.cited_ids:
        def supports(cid: str) -> bool:
            return cid in q.required_chunk_ids or any(groups_satisfied(chunk_text(cid), q.concept_groups))
        support = sum(supports(c) for c in out.cited_ids) / len(out.cited_ids)
    strict = float(not declined and out.gate_valid and all(hits))

    missing, failure = [], ""
    if not strict:
        for g, h in zip(q.concept_groups, hits):
            if not h:
                where = "in_context_dropped_by_generator" if any(phrase_present(out.context_text, p) for p in g) else "not_retrieved"
                missing.append({"group": g, "diagnosis": where})
        if declined:
            failure = "refused" if out.refused else "abstained (no supported claim survived)"
        elif not out.gate_valid:
            failure = "citation gate failed"
        elif all(m["diagnosis"] == "not_retrieved" for m in missing):
            failure = "retrieval miss: required evidence never reached the context"
        elif all(m["diagnosis"] == "in_context_dropped_by_generator" for m in missing):
            failure = "generation miss: evidence was in context but not in the answer"
        else:
            failure = "mixed: some evidence not retrieved, some dropped by the generator"
    return QuestionGrade(q.id, system, strict, recall, hits, ev, support, out.gate_valid, declined, out.pathway, failure, missing)


def oracle_check(questions: Sequence[EvalQuestion], chunk_text: Callable[[str], str]) -> Dict[str, Any]:
    """Benchmark validity: an answer made of the required chunks' text must satisfy every group."""
    failures = []
    for q in questions:
        if not q.answerable:
            continue
        text = " ".join(chunk_text(c) for c in q.required_chunk_ids)
        miss = [g for g, h in zip(q.concept_groups, groups_satisfied(text, q.concept_groups)) if not h]
        if miss or any(not chunk_text(c) for c in q.required_chunk_ids):
            failures.append({"qid": q.id, "missing_groups": miss})
    answerable = sum(q.answerable for q in questions)
    return {"answerable": answerable, "oracle_satisfiable": answerable - len(failures), "failures": failures}


# ---------------------------------------------------------------------------
# Runner & aggregation
# ---------------------------------------------------------------------------


def _pct(xs: Sequence[float]) -> Optional[float]:
    return round(100.0 * sum(xs) / len(xs), 1) if xs else None


def _percentile(xs: Sequence[float], p: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def query_cost_usd(o: SystemOutput) -> float:
    return (o.router_input_tokens * settings.router_input_usd_per_1m + o.router_output_tokens * settings.router_output_usd_per_1m
            + o.generation_input_tokens * settings.generation_input_usd_per_1m
            + o.generation_output_tokens * settings.generation_output_usd_per_1m
            + o.embedding_tokens * settings.embedding_usd_per_1m) / 1_000_000


def one_time_costs(chunks: Sequence[Any]) -> Dict[str, float]:
    """Projected one-time costs: embedding every chunk (all systems) and LLM graph extraction (S3 only)."""
    embed = sum(c.approx_token_count for c in chunks) * settings.embedding_usd_per_1m / 1_000_000
    extraction = CostBudgetEstimator(chunks, ExtractionCache(":memory:")).estimate().projected_usd
    return {"embedding_usd": round(embed, 6), "graph_extraction_usd": round(extraction, 6)}


def summarize(questions: Sequence[EvalQuestion], grades: Sequence[QuestionGrade], outputs: Sequence[SystemOutput]) -> Dict[str, Any]:
    qmap = {q.id: q for q in questions}
    by = lambda pred: [g for g in grades if pred(qmap[g.qid])]  # noqa: E731
    answerable = by(lambda q: q.answerable)
    summary: Dict[str, Any] = {
        "strict_by_stratum": {s: _pct([g.strict for g in by(lambda q, s=s: q.stratum == s)]) for s in STRATA},
        "concept_recall_by_stratum": {s: _pct([g.concept_recall for g in by(lambda q, s=s: q.stratum == s)]) for s in STRATA if s != "out_of_scope"},
        "strict_by_stress_type": {t: _pct([g.strict for g in by(lambda q, t=t: q.stress_type == t)]) for t in STRESS_TYPES},
        "strict_overall": _pct([g.strict for g in grades]),
        "strict_answerable": _pct([g.strict for g in answerable]),
        "strict_multi_hop": _pct([g.strict for g in by(lambda q: q.stratum in ("two_hop", "three_hop"))]),
        "concept_recall_answerable": _pct([g.concept_recall for g in answerable]),
        "evidence_recall_answerable": _pct([g.evidence_recall for g in answerable if g.evidence_recall is not None]),
        "citation_support_precision": _pct([g.citation_support_precision for g in answerable if g.citation_support_precision is not None]),
        "citation_gate_valid_rate": _pct([float(g.gate_valid) for g in answerable if not g.refused_or_abstained]),
        "false_refusal_rate_answerable": _pct([float(g.refused_or_abstained) for g in answerable]),
        "latency_ms_p50": round(_percentile([o.latency_ms for o in outputs], 0.5), 2),
        "latency_ms_p95": round(_percentile([o.latency_ms for o in outputs], 0.95), 2),
        "projected_cost_per_query_usd": round(statistics.mean(query_cost_usd(o) for o in outputs), 6),
        "generator_calls_mean": round(statistics.mean(o.generator_calls for o in outputs), 3),
    }
    dist: Dict[str, Dict[str, int]] = {}
    for g in grades:
        row = dist.setdefault(g.pathway, {"questions": 0, "passing": 0})
        row["questions"] += 1
        row["passing"] += int(g.strict)
    summary["pathway_distribution"] = dict(sorted(dist.items(), key=lambda kv: -kv[1]["questions"]))
    return summary


def build_systems(corpus: Any, generator_factory: Callable[[], Any] = OfflineGroundedGeneratorLLM,
                  include_sensitivity: bool = True) -> List[RagSystem]:
    if not isinstance(corpus.vectors.embedder, CountingEmbedder):
        corpus.vectors.embedder = CountingEmbedder(corpus.vectors.embedder)
    systems: List[RagSystem] = [
        PlainDenseRAG(corpus.vectors, generator_factory(), top_k=3),
        TunedHybridRAG(corpus.vectors, generator_factory(), top_k=3),
        ArchGraphRAG(corpus, generator_factory(), graph_top_k=3, vector_top_k=3),
    ]
    if include_sensitivity:
        s2k6 = TunedHybridRAG(corpus.vectors, generator_factory(), top_k=6)
        s2k6.key = "S2@k6"
        s3b = ArchGraphRAG(corpus, generator_factory(), graph_top_k=3, vector_top_k=3, max_passages=3)
        s3b.key = "S3@3p"
        systems += [s2k6, s3b]
    return systems


def run_benchmark(corpus: Any, questions: Sequence[EvalQuestion], systems: Optional[List[RagSystem]] = None) -> Dict[str, Any]:
    systems = systems or build_systems(corpus)
    texts: Dict[str, str] = {}

    def chunk_text(cid: str) -> str:
        if cid not in texts:
            rows = corpus.vectors.fetch_chunks_by_ids([cid])
            texts[cid] = rows[0].raw_text if rows else ""
        return texts[cid]

    one_time = one_time_costs(corpus.report.chunks)
    results: Dict[str, Any] = {"systems": {}, "questions": [q.model_dump() for q in questions],
                               "oracle": oracle_check(questions, chunk_text), "one_time_costs": one_time,
                               "corpus": {"papers": len(corpus.report.papers), "chunks": len(corpus.report.chunks),
                                          "graph_entities": corpus.graph.count_nodes("Entity"),
                                          "graph_relationships": corpus.graph.count_relationships()}}
    for s in systems:
        outs, grades = [], []
        for q in questions:
            o = s.run(q.question)
            outs.append(o)
            grades.append(grade(q, o, chunk_text, s.key))
        summ = summarize(questions, grades, outs)
        summ["one_time_cost_usd"] = round(one_time["embedding_usd"] + (one_time["graph_extraction_usd"] if isinstance(s, ArchGraphRAG) else 0.0), 6)
        results["systems"][s.key] = {
            "label": s.label,
            "summary": summ,
            "per_question": [
                {**asdict(g), "answer": o.answer, "route": o.route, "template": o.template,
                 "fallback_triggered": o.fallback_triggered, "vector_seeded": o.vector_seeded,
                 "context_chunk_ids": o.context_chunk_ids, "cited_ids": o.cited_ids,
                 "latency_ms": round(o.latency_ms, 3), "projected_cost_usd": round(query_cost_usd(o), 7)}
                for g, o in zip(grades, outs)
            ],
        }
    return results
