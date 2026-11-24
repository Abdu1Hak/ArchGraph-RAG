"""Unit tests for Phase 5: question-set validity, pathway-agnostic grading, runner determinism."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from src.evaluation.benchmark_runner import (
    EvalQuestion,
    SystemOutput,
    build_systems,
    grade,
    groups_satisfied,
    load_questions,
    oracle_check,
    phrase_present,
    run_benchmark,
    strip_citations,
    summarize,
)
from src.generation.synthesizer import INSUFFICIENT_TEXT, GroundedAnswerSynthesizer, is_abstention
from src.ingestion.offline_demo import build_offline_corpus


@pytest.fixture(scope="module")
def corpus():
    return build_offline_corpus()


@pytest.fixture(scope="module")
def questions():
    return load_questions()


def _text(corpus):
    def chunk_text(cid: str) -> str:
        rows = corpus.vectors.fetch_chunks_by_ids([cid])
        return rows[0].raw_text if rows else ""
    return chunk_text


def _out(answer: str, **kw) -> SystemOutput:
    base = dict(answer=answer, refused=False, abstained=False, gate_valid=True, cited_ids=[], context_chunk_ids=[],
                context_text="", pathway="VECTOR", route="VECTOR_ONLY", template=None, fallback_triggered=False,
                vector_seeded=False, generator_calls=1, latency_ms=1.0, generation_input_tokens=10,
                generation_output_tokens=5, router_input_tokens=0, router_output_tokens=0, embedding_tokens=3)
    base.update(kw)
    return SystemOutput(**base)


# ---------------------------------------------------------------------------
# Question-set validity
# ---------------------------------------------------------------------------


def test_question_set_composition(questions) -> None:
    assert len(questions) == 60
    assert Counter(q.stratum for q in questions) == {"single_hop": 15, "two_hop": 15, "three_hop": 15, "aggregation": 8, "out_of_scope": 7}
    stress = Counter(q.stress_type for q in questions)
    assert stress["out_of_scope"] == 7 and min(stress[t] for t in ("implicit_bridge", "caveat_trap", "cross_pillar")) >= 10
    for q in questions:
        if q.stratum in ("two_hop", "three_hop"):
            assert q.hop_count >= 2 and len(q.concept_groups) >= 2


def test_questions_do_not_echo_router_templates(questions) -> None:
    """Guard against circular prompts: almost none may start with the router's literal cue phrases."""
    cues = ("trace the lineage of", "compare the architecture of", "which techniques mitigate", "what links", "what connects")
    echoing = [q.id for q in questions if q.question.lower().startswith(cues)]
    assert len(echoing) <= 2, echoing


def test_oracle_every_answerable_question_is_answerable_from_its_required_chunks(corpus, questions) -> None:
    oracle = oracle_check(questions, _text(corpus))
    assert oracle["answerable"] == 53 and oracle["oracle_satisfiable"] == 53, oracle["failures"]


def test_out_of_scope_questions_name_no_corpus_entity(corpus, questions) -> None:
    for q in questions:
        if not q.answerable:
            assert not corpus.resolver.resolve_text_mentions_in_chunk(q.question), q.id


def test_loader_rejects_malformed_questions(tmp_path) -> None:
    p = tmp_path / "q.json"
    p.write_text('{"questions": [{"id": "X", "stratum": "two_hop", "hop_count": 2, "stress_type": "direct", '
                 '"question": "Some question?", "concept_groups": [], "required_chunk_ids": []}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(p)


# ---------------------------------------------------------------------------
# Matching & grading
# ---------------------------------------------------------------------------


def test_phrase_matching_normalization_and_boundaries() -> None:
    assert phrase_present("base frequency of 500,000", "500000")
    assert phrase_present("Grouped-Query Attention", "grouped query attention")
    assert phrase_present("uses MLA.", "mla") and not phrase_present("the mlab cluster", "mla")
    assert groups_satisfied("PPO needs a critic", [["critic", "value model"], ["reward"]]) == [True, False]
    assert not phrase_present(strip_citations("unrelated text [peng_2023_yarn::s01::c00]"), "yarn")


def test_grade_out_of_scope_and_answerable_rules(corpus) -> None:
    ct = _text(corpus)
    oos = EvalQuestion(id="O", stratum="out_of_scope", hop_count=0, stress_type="out_of_scope", question="weather in Paris?")
    assert grade(oos, _out(INSUFFICIENT_TEXT, abstained=True), ct, "S").strict == 1.0
    assert grade(oos, _out("Paris is sunny [x::s01::c00].", cited_ids=["x::s01::c00"]), ct, "S").strict == 0.0

    q = EvalQuestion(id="Q", stratum="two_hop", hop_count=2, stress_type="cross_pillar", question="q about ppo and critic?",
                     concept_groups=[["ppo"], ["critic", "value model"]], required_chunk_ids=["deepseek_2025_r1::s01::c00"])
    full = _out("DPO replaces PPO; GRPO drops the critic [deepseek_2025_r1::s01::c00].", cited_ids=["deepseek_2025_r1::s01::c00"],
                context_chunk_ids=["deepseek_2025_r1::s01::c00"])
    g = grade(q, full, ct, "S")
    assert g.strict == 1.0 and g.concept_recall == 1.0 and g.evidence_recall == 1.0 and g.citation_support_precision == 1.0
    assert grade(q, replace(full, gate_valid=False), ct, "S").strict == 0.0
    assert grade(q, replace(full, abstained=True), ct, "S").failure.startswith("abstained")
    half = grade(q, _out("DPO replaces PPO [a::s01::c00].", context_text="the critic is removed"), ct, "S")
    assert half.strict == 0.0 and half.concept_recall == 0.5
    assert half.missing_groups[0]["diagnosis"] == "in_context_dropped_by_generator"


def test_grading_is_pathway_agnostic(corpus) -> None:
    ct = _text(corpus)
    q = EvalQuestion(id="Q", stratum="three_hop", hop_count=3, stress_type="implicit_bridge", question="Which test question?",
                     concept_groups=[["mla"], ["gqa"]], required_chunk_ids=["deepseek_2024_v2::s02::c00"])
    o = _out("MLA extends GQA [deepseek_2024_v2::s02::c00].", cited_ids=["deepseek_2024_v2::s02::c00"])
    scores = {grade(q, replace(o, pathway=p, route=r, template=t), ct, "S").strict
              for p, r, t in (("VECTOR", "VECTOR_ONLY", None), ("GRAPH:T1", "GRAPH_ONLY", "T1_COMPONENT_LINEAGE"),
                              ("HYBRID(low-conf):T6_FALLBACK", "HYBRID_PARALLEL", "T6_DYNAMIC_SUBGRAPH_FALLBACK"))}
    assert scores == {1.0}


def test_summarize_reports_percentiles_and_pathways(corpus) -> None:
    q = EvalQuestion(id="Q", stratum="single_hop", hop_count=1, stress_type="direct", question="Which test question?",
                     concept_groups=[["x"]], required_chunk_ids=["a::s01::c00"])
    grades = [grade(q, _out("x here [a::s01::c00].", latency_ms=ms, pathway="P"), _text(corpus), "S") for ms in (1.0, 2.0, 3.0, 10.0)]
    s = summarize([q], grades, [_out("x", latency_ms=ms) for ms in (1.0, 2.0, 3.0, 10.0)])
    assert s["latency_ms_p50"] == 2.5 and s["latency_ms_p95"] > 8.0
    assert s["pathway_distribution"]["P"] == {"questions": 4, "passing": 4}


# ---------------------------------------------------------------------------
# Synthesizer abstention & end-to-end runner
# ---------------------------------------------------------------------------


class _Say:
    name = "say"

    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        return self.text


def test_explicit_abstention_is_respected(corpus) -> None:
    assert is_abstention(INSUFFICIENT_TEXT) and not is_abstention(f"Claim with words here [a::s01::c00].")
    from src.retrieval.router import GraphVectorEngine, QueryRouter

    r = GraphVectorEngine(QueryRouter(corpus.resolver, corpus.graph, corpus.vectors)).retrieve("Explain the DPO loss equation.")
    gen = _Say(INSUFFICIENT_TEXT)
    a = GroundedAnswerSynthesizer(gen, corpus.vectors).synthesize("q", r)
    assert a.abstained and not a.refused and gen.calls == 1 and a.answer == INSUFFICIENT_TEXT and not a.citations


def test_runner_is_deterministic_and_complete(corpus, questions) -> None:
    subset = [q for q in questions if q.id in ("S03", "T05", "H04", "A01", "O01")]
    r1 = run_benchmark(corpus, subset, build_systems(corpus, include_sensitivity=False))
    r2 = run_benchmark(corpus, subset, build_systems(corpus, include_sensitivity=False))
    assert set(r1["systems"]) == {"S1", "S2", "S3"}
    for k in r1["systems"]:
        a = [(p["qid"], p["strict"], p["answer"]) for p in r1["systems"][k]["per_question"]]
        b = [(p["qid"], p["strict"], p["answer"]) for p in r2["systems"][k]["per_question"]]
        assert a == b and len(a) == 5
    assert r1["systems"]["S3"]["summary"]["one_time_cost_usd"] > r1["systems"]["S2"]["summary"]["one_time_cost_usd"]
    assert r1["oracle"]["oracle_satisfiable"] == r1["oracle"]["answerable"] == 4
