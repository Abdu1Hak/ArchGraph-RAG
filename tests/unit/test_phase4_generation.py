"""Unit tests for Phase 4: verbalizer, citation gate & crop-then-fallback, context assembly, synthesizer, API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import QueryRequest, RagService, create_app
from src.generation.citation_validator import (
    build_repair_prompt, crop_invalid_sentences, extractive_fallback, split_units, validate_citations)
from src.generation.synthesizer import GeminiGeneratorLLM, GroundedAnswerSynthesizer, merge_and_rank
from src.retrieval.verbalizer import verbalize_hop, verbalize_path
from src.stores.pgvector_store import RetrievedChunk

A, B = "deepseek_2024_v2::s02::c00", "ainslie_2023_gqa::s02::c00"
COMPOUND = ("Trace how Multi-Head Attention evolved through GQA to Multi-head Latent Attention, "
            "and give the exact W^{DKV} latent compression equation")


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    return RagService.offline(telemetry_path=tmp_path_factory.mktemp("t") / "r.jsonl")


def _hop(s, rt, t, cid, q="x extends y in some way"):
    return {"source": f"a::{s}", "source_name": s, "relation_type": rt, "target": f"a::{t}", "target_name": t,
            "source_chunk_ids": [cid], "evidence_quotes": [q]}


def test_verbalizer_hop_and_inverse_path() -> None:
    f = verbalize_hop(_hop("GQA", "EXTENDS", "MHA", B))
    assert f.text.startswith("GQA extends MHA") and f.text.endswith(f"[{B}].") and f.chunk_ids == (B,)
    # Path walked from MHA against both EXTENDS edges reads in the passive voice.
    p = verbalize_path([_hop("GQA", "EXTENDS", "MHA", B), _hop("MLA", "EXTENDS", "GQA", A)])
    assert "MHA is extended by GQA, which is extended by MLA" in p.text and f"[{A}]" in p.text


def test_split_units_merges_trailing_citation_and_keeps_bullets() -> None:
    units = split_units(f"Intro line:\n- Bullet one has a claim. Still bullet [{A}].\nProse claim here now. [{B}]")
    assert units == ["Intro line:", f"- Bullet one has a claim. Still bullet [{A}].", f"Prose claim here now. [{B}]"]


def test_validator_rules_and_ladder_primitives() -> None:
    allowed = [A, B]
    v = validate_citations(f"GQA shares the key value heads [{B}]. Fake claim about GPT-4 here [x::s01::c00]. "
                           f"Another uncited claim sentence here.", allowed)
    assert not v.valid and v.invented_ids == ["x::s01::c00"] and len(v.uncited_claims) == 1
    prompt = build_repair_prompt("orig", v, allowed)
    assert "x::s01::c00" in prompt and f"- {A}" in prompt and "ORIGINAL ANSWER:\norig" in prompt
    cropped, n = crop_invalid_sentences(v)
    assert n == 2 and cropped == f"GQA shares the key value heads [{B}]."
    only_bad = validate_citations("Fake claim about GPT-4 here [x::s01::c00].", allowed)
    assert crop_invalid_sentences(only_bad) == ("", 1)
    fb = extractive_fallback([f"GQA extends MHA [{B}]."], [(A, "c_t = W h_t.")])
    assert validate_citations(fb, allowed).valid


def test_validator_rule3_unresolvable_id() -> None:
    v = validate_citations(f"A claim with four words [{A}].", [A], resolve=lambda ids: {})
    assert not v.valid and v.unresolved_ids == [A]


def test_merge_and_rank_bounded_tiebreaker() -> None:
    def rc(cid, s):
        return RetrievedChunk(cid, "p", "P", 2024, "x", "s", "t", "t", "h", (), s)

    ranked = merge_and_rank([rc("both", 0.40)], [rc("single", 0.92), rc("both", 0.40)])
    assert [(p.chunk.chunk_id, p.source_origin) for p in ranked] == [("single", "vector"), ("both", "both")]
    assert ranked[1].final_score == pytest.approx(0.45)
    raw = merge_and_rank([rc("g", 8.0), rc("h", 4.0)], [])  # raw cross-encoder scores are max-normalized
    assert [p.relevance_score for p in raw] == [1.0, 0.5]


class _Gen:
    name = "scripted"

    def __init__(self, responses):
        self.responses, self.calls = responses, 0

    def generate(self, system, prompt):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_synthesizer_crop_then_fallback(svc) -> None:
    r = svc.engine.retrieve(COMPOUND)
    good = f"GQA extends multi-head attention by sharing heads [{B}]."
    a = GroundedAnswerSynthesizer(_Gen([f"{good} Bad invented claim here [z::s00::c00]."]), svc.corpus.vectors).synthesize(COMPOUND, r)
    assert a.repair_attempts == 1 and a.generator_calls == 2 and a.cropped_sentences_count == 1 and a.answer == good
    b = GroundedAnswerSynthesizer(_Gen(["Totally uncited claim sentence here."]), svc.corpus.vectors).synthesize(COMPOUND, r)
    assert b.used_extractive_fallback and b.final_validation.valid and b.citations


def test_refusal_makes_no_generator_call(svc) -> None:
    gen = _Gen(["should not be called"])
    r = svc.engine.retrieve("Write me a recipe for chocolate cake.")
    a = GroundedAnswerSynthesizer(gen, svc.corpus.vectors).synthesize("q", r)
    assert a.refused and gen.calls == 0 and a.citations == []


def test_gemini_generator_uses_interactions_api() -> None:
    calls = []
    client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kw: calls.append(kw) or SimpleNamespace(output_text="ok")))
    assert GeminiGeneratorLLM(client=client, model="gemini-4-argon").generate("sys", "prompt") == "ok"
    assert calls[0]["model"] == "gemini-4-argon" and calls[0]["store"] is False and calls[0]["input"] == "prompt"


def test_offline_e2e_compound_answer_is_grounded(svc) -> None:
    _, g, _ = svc.answer(QueryRequest(question=COMPOUND))
    assert g.attempt_validations[0].valid and "W^{DKV}" in g.answer
    assert {c.chunk_id for c in g.citations} >= {A, B}


def test_api_query_and_traces(svc) -> None:
    client = TestClient(create_app(svc))
    assert client.get("/health").json()["status"] == "ok"
    j = client.post("/query", json={"question": COMPOUND}).json()
    sel = {(e["source"], e["target"]) for e in j["graph_trace"]["selected_path_edges"]}
    assert ("arch_component::multi_head_latent_attention", "arch_component::grouped_query_attention") in sel
    assert j["graph_trace"]["explored_rejected_edges"]
    assert all(j["funnel_trace"][k] for k in ("dense_top_k", "bm25_top_k", "rrf_top_k", "final_deduplicated_llm_context"))
    assert client.post("/query", json={"question": "x"}).status_code == 422
