"""Tests for backend text normalization, synthesizer integration, and panels.js DOM math rendering."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from src.config import settings
from src.generation.synthesizer import (
    GENERATION_SYSTEM_PROMPT,
    GroundedAnswerSynthesizer,
    clean_passage_for_prompt,
    clean_section_path,
)
from src.generation.text_normalizer import (
    clean_generated_answer,
    clean_passage_text,
)
from src.ingestion.offline_demo import build_offline_corpus
from src.retrieval.router import GraphVectorEngine, QueryRouter
from src.stores.pgvector_store import RetrievedChunk

ROOT = Path(__file__).resolve().parents[2]
PANELS_JS = ROOT / "src" / "api" / "static" / "panels.js"

# Exact raw text from live Postgres chunk `xiong_2020_preln::s03::c02`
XIONG_RAW_CHUNK = (
    "and layer normalization are also key components to the Transformer. For any vector _v_ , "
    "the layer normalization is computed as LayerNorm( _v_ ) = _γ_<sup>_v−_</sup> _σ_<sup>_<u>µ</u>_</sup> + _β_ , "
    "in which _µ, σ_ are the mean and standard deviation of the elements in _v_ , i.e., "
    "_µ_ = _d_<sup><u>1</u></sup> \ufffd _dk_ =1<sup>_vk_and</sup><sup>_σ_2=</sup> _d_<sup><u>1</u></sup> "
    "\ufffd _dk_ =1<sup>(</sup><sup>_vk −µ_)2.Scale</sup><sup>_γ_</sup> and bias vector _β_ are parameters.\n\n"
    "Different orders of the sub-layers, residual connection and layer normalization in a Transformer layer "
    "lead to variants of Transformer architectures. One of the original and most popularly used architecture "
    "for the Transformer and BERT (Vaswani et al., 2017; Devlin et al., 2018) follows “selfattention (FFN) "
    "sub-layer _→_ residual connection _→_ layer normalization”, which we call the Transformer with PostLayer "
    "normalization (Post-LN Transformer), as illustrated in Figure 1.\n\n"
    "**Post-LN Transformer** Denote _xl,i_ as the input of the _l_ -th Transformer layer at position _i_ , "
    "where _xl,i_ is a real-valued vector of dimension _d_ , _i_ = 1 _,_ 2 _, ..., n_ , _l_ = 1 _,_ 2 _, ..., L_ . "
    "_n_ is the length of the sequence and _L_ is the number of layers. For completeness, we define _x_ 0 _,i_ "
    "as the input embedding at position _i_ which is usually a combination of word embedding and positional "
    "embedding. The computations inside the _l_ -th layer are composed of several steps, and we use "
    "super-scripts on _x_ to present the input(output) of different steps as in Table 1 (left), where "
    "_W_<sup>1</sup><sup>_,l_</sup> , _W_<sup>2</sup><sup>_,l_</sup> , _b_<sup>1</sup><sup>_,l_</sup> and "
    "_b_<sup>2</sup><sup>_,l_</sup> are parameters of the FFN sub-layer in the _l_ -th layer."
)

USER_BROKEN_EQUATION = "μ = (1)/(dk)∑_{k=1}^{dk}vk and σ2 = (1)/(dk)∑_{k=1}^{dk}(vk - μ)2)"


_NODE_HARNESS_JS = r"""
const fs = require("fs");
const vm = require("vm");

function makeFakeDom() {
  class Node {
    constructor(tag, text = "") {
      this.nodeType = tag ? 1 : 3;
      this.tagName = tag ? tag.toUpperCase() : null;
      this.className = "";
      this.attrs = {};
      this.children = [];
      this._text = text;
      this.hidden = false;
    }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return this.attrs[k]; }
    append(...nodes) {
      for (const n of nodes) {
        if (n === null || n === undefined || n === false) continue;
        if (typeof n === "string") this.children.push(new Node(null, n));
        else if (n.tagName === "#FRAGMENT") this.children.push(...n.children);
        else this.children.push(n);
      }
    }
    set innerHTML(_val) { this.children = []; this._text = ""; }
    set textContent(val) { this.children = [new Node(null, String(val ?? ""))]; this._text = ""; }
    get textContent() {
      if (this.nodeType === 3) return this._text;
      return this.children.map((c) => c.textContent).join("");
    }
    toHTML() {
      if (this.nodeType === 3) return this._text;
      if (this.tagName === "#FRAGMENT") return this.children.map((c) => c.toHTML()).join("");
      const cls = this.className ? ` class="${this.className}"` : "";
      const extra = Object.entries(this.attrs).map(([k, v]) => ` ${k}="${v}"`).join("");
      const tag = this.tagName.toLowerCase();
      return `<${tag}${cls}${extra}>${this.children.map((c) => c.toHTML()).join("")}</${tag}>`;
    }
  }
  const elements = {};
  const document = {
    createElement: (tag) => new Node(tag),
    createDocumentFragment: () => new Node("#FRAGMENT"),
    addEventListener: () => {},
    querySelectorAll: () => [],
  };
  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (!k.startsWith("on") && v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid !== null && kid !== undefined && kid !== false) e.append(kid.nodeType ? kid : String(kid));
    }
    return e;
  }
  function $(id) {
    if (!elements[id]) elements[id] = new Node("div");
    return elements[id];
  }
  return { document, el, $, elements };
}

const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const { document, el, $, elements } = makeFakeDom();
const sandbox = {
  document,
  window: {},
  el,
  $,
  fmt: (n) => Number(n || 0).toLocaleString("en-US"),
  App: { onHighlight: () => {}, citationsById: new Map(), roundTripMs: 12 },
  MODE_LABELS: { live: "Live", gemini: "Gemini", offline: "Offline" },
  modelLabel: (m) => m || "mock-model",
  RegExp, Map, Set, Object, Array, String, Number, Math, Boolean, URLSearchParams,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(payload.panelsPath, "utf8"), sandbox);

const results = {};
if (payload.answers) {
  results.answers = payload.answers.map((ans) => {
    const frag = sandbox.renderMarkdownLite(ans);
    return { html: frag.toHTML(), text: frag.textContent };
  });
}
if (payload.excerpts) {
  results.excerpts = payload.excerpts.map((txt) => {
    const box = el("div", {}, ...sandbox.renderExcerptParagraphs(txt));
    return { html: box.toHTML(), text: box.textContent };
  });
}
if (payload.renderResponse) {
  let citationsTouched = false;
  const orig$ = sandbox.$;
  sandbox.$ = (id) => {
    if (id === "citations") citationsTouched = true;
    return orig$(id);
  };
  sandbox.window.renderAnswer(payload.renderResponse, "Explain LayerNorm");
  const gateInfo = vm.runInContext('({ gateClassType: typeof gateClass, gateLabelType: typeof gateLabel, gateLabelPass: gateLabel("PASS"), gateClassPass: gateClass("PASS") })', sandbox);
  results.ui = {
    meterHtml: elements.meter ? elements.meter.toHTML() : "",
    meterText: elements.meter ? elements.meter.textContent : "",
    answerHtml: elements.answer ? elements.answer.toHTML() : "",
    citationsTouched,
    ...gateInfo,
  };
}
process.stdout.write(JSON.stringify(results));
"""


def _run_node_vm(payload: dict) -> dict:
    assert shutil.which("node"), "Node.js is required to test panels.js DOM rendering"
    body = {"panelsPath": str(PANELS_JS), **payload}
    proc = subprocess.run(
        ["node", "-e", _NODE_HARNESS_JS],
        input=json.dumps(body),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _assert_clean_dom_text(tc: str, raw_src: str = "") -> None:
    assert not re.search(r"</?(?:u|sup|sub|br)\b|<!--", tc, re.I), f"Raw HTML leaked into DOM text: {tc[:200]}"
    assert "\ufffd" not in tc, f"U+FFFD leaked into DOM text: {tc[:200]}"
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", tc), f"Control char leaked into DOM text: {tc[:200]}"
    assert "**" not in tc, f"Stray ** leaked into DOM text: {tc[:200]}"
    assert not re.search(r"[_^]\{[^{}]*\}", tc), f"Raw brace script leaked into DOM text: {tc[:200]}"
    if "```" not in raw_src:
        no_urls = re.sub(r"https?://\S+", "", tc)
        assert not re.search(r"(?<![A-Za-z0-9_])_[^_\n]{1,25}_(?![A-Za-z0-9_])", no_urls), (
            f"Glued underscore leaked into DOM text: {no_urls[:200]}"
        )


# ---------------------------------------------------------------------------
# 1. Exact user strings & core normalizer guarantees
# ---------------------------------------------------------------------------


def test_user_broken_equation_normalizer_and_dom() -> None:
    cleaned_passage = clean_passage_text(USER_BROKEN_EQUATION)
    assert "(1)/(dk)" not in cleaned_passage
    assert "1/d_k" in cleaned_passage
    assert "∑_{k=1}^{d_k} v_k" in cleaned_passage
    assert "σ^2" in cleaned_passage
    assert "(v_k - μ)^2" in cleaned_passage

    cleaned_answer = clean_generated_answer(f"{USER_BROKEN_EQUATION} [xiong_2020_preln::s03::c02]")
    assert r"\frac{1}{d_k}" in cleaned_answer
    assert "∑_{k=1}^{d_k} v_k" in cleaned_answer
    assert "σ^2" in cleaned_answer
    assert "(v_k - μ)^2" in cleaned_answer
    assert "[xiong_2020_preln::s03::c02]" in cleaned_answer

    dom = _run_node_vm({"answers": [USER_BROKEN_EQUATION, cleaned_answer]})
    for item in dom["answers"]:
        html = item["html"]
        assert '<span class="frac">' in html
        assert '<span class="frac-num">1</span>' in html
        assert '<span class="frac-den">d<sub>k</sub></span>' in html
        assert '<span class="math-op"><span class="math-sym">∑</span><span class="math-limits">' in html
        assert "<sup>d<sub>k</sub></sup><sub>k=1</sub>" in html
        assert "v<sub>k</sub>" in html
        assert "σ<sup>2</sup>" in html
        assert "(v<sub>k</sub> - μ)<sup>2</sup>" in html
        _assert_clean_dom_text(item["text"])

    latex_sigma2 = r"$$\sigma2 = (1)/(dk)\sum_{k=1}^{dk}(vk - \mu)2)$$"
    cleaned_latex_sigma2 = clean_generated_answer(latex_sigma2)
    assert r"\sigma^2" in cleaned_latex_sigma2
    assert r"\frac{1}{d_k}" in cleaned_latex_sigma2
    assert r"(v_k - \mu)^2" in cleaned_latex_sigma2
    assert "σ^2" in clean_passage_text(latex_sigma2)
    dom_latex = _run_node_vm({"answers": [latex_sigma2, cleaned_latex_sigma2]})
    for item in dom_latex["answers"]:
        assert "σ<sup>2</sup>" in item["html"]
        assert '<span class="frac">' in item["html"]
        assert "<sup>d<sub>k</sub></sup><sub>k=1</sub>" in item["html"]
        assert "(v<sub>k</sub> - μ)<sup>2</sup>" in item["html"]
        _assert_clean_dom_text(item["text"])


def test_xiong_2020_preln_chunk_clean_and_dom() -> None:
    sec = "**3. Optimization for the Transformer** > **3.1. Transformer with Post-Layer Normalization**"
    assert clean_section_path(sec) == "3. Optimization for the Transformer > 3.1. Transformer with Post-Layer Normalization"

    cleaned = clean_passage_text(XIONG_RAW_CHUNK)
    for bad in ("<u>", "</u>", "<sup>", "</sup>", "<sub>", "</sub>", "<br>", "<!--", "\ufffd", "_vk_and", "2.Scale"):
        assert bad not in cleaned, f"Unexpected artifact {bad!r} in {cleaned}"
    assert "LayerNorm(v) = γ (v - μ)/σ + β" in cleaned
    assert "μ = 1/d ∑_{k=1}^{d} v_k and σ^2 = 1/d ∑_{k=1}^{d} (v_k - μ)^2. Scale γ" in cleaned

    dom = _run_node_vm({"excerpts": [XIONG_RAW_CHUNK, cleaned]})
    for item in dom["excerpts"]:
        _assert_clean_dom_text(item["text"], XIONG_RAW_CHUNK)
        assert "σ<sup>2</sup>" in item["html"]
        assert "v<sub>k</sub>" in item["html"]
        assert '<span class="math-limits"><sup>d</sup><sub>k=1</sub></span>' in item["html"]
        assert "Scale γ" in item["text"]


def test_preserve_snake_case_token_tags_citations_and_ocr_fixes() -> None:
    sample = (
        "<!-- Start of picture text -->\n"
        "FFN( x ) = max(0 , xW 1 + b 1) W 2 + b 2 (2)<br>"
        "<!-- End of picture text -->\n"
        "Preserve _vk_and_σ_2= and _a_ and _N_ while keeping vaswani_2017_transformer, "
        "d_model, d_ff, core_block, partial_block, <eos>, <bos>, <pad>, <resv>, "
        "and [vaswani_2017_transformer::s09::c00] plus\x13\x17 ~~√~~ dk."
    )
    cleaned = clean_passage_text(sample)
    assert "FFN(x) = max(0, xW_1 + b_1) W_2 + b_2 (2)" in cleaned
    assert "v_k and σ^2 =" in cleaned
    assert " a " in cleaned and " N " in cleaned
    for token in (
        "vaswani_2017_transformer",
        "d_model",
        "d_ff",
        "core_block",
        "partial_block",
        "<eos>",
        "<bos>",
        "<pad>",
        "<resv>",
        "[vaswani_2017_transformer::s09::c00]",
    ):
        assert token in cleaned, f"Lost {token!r} in {cleaned!r}"
    for bad in ("<!--", "<br>", "\x13", "\x17", "~~"):
        assert bad not in cleaned

    # Check Node VM preserves multi-word snake_case, token tags, and renders math subscripts cleanly
    dom = _run_node_vm({"answers": [cleaned]})
    res = dom["answers"][0]
    _assert_clean_dom_text(res["text"])
    for token in ("vaswani_2017_transformer", "core_block", "partial_block", "<eos>", "<bos>", "<pad>", "<resv>"):
        assert token in res["text"]
    assert "d<sub>model</sub>" in res["html"]
    assert "d<sub>ff</sub>" in res["html"]
    assert "xW<sub>1</sub> + b<sub>1</sub>" in res["html"]
    assert 'data-chunk="vaswani_2017_transformer::s09::c00"' in res["html"]


def test_ufffd_sum_repair_patterns() -> None:
    cases = [
        "\ufffd dk =1 v_k",
        "\ufffd k=1 x_k",
        "\ufffd i=1 a_i",
        "\ufffd t=1 r_t",
        "\ufffd j=1 w_j",
        "\ufffd nj =1 x_{l,j}",
        "\ufffd Nn =1 sim(q_m, k_n)",
        "\ufffd |oi| r_i",
        "\ufffd G A_i",
        "stray \ufffd bracket",
    ]
    cleaned = [clean_passage_text(c) for c in cases]
    for out in cleaned[:-1]:
        assert "∑" in out and "\ufffd" not in out
    assert "\ufffd" not in cleaned[-1] and "∑" not in cleaned[-1]


# ---------------------------------------------------------------------------
# 2. Domain equations across Attention, RoPE, Chinchilla, DPO, GRPO, GSPO, FlashAttention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expr,expected_fragments",
    [
        (
            "attention",
            r"$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$ and $$c_t^{KV} = W^{DKV} h_t$$",
            ['<span class="frac">', '<span class="math-sqrt">', "d<sub>k</sub>", "W<sup>DKV</sup>", "c<sub>t</sub><sup>KV</sup>"],
        ),
        (
            "rope",
            r"$$f_q(x_m, m) = R^d_{\Theta, m} W_q x_m, \quad \theta_i = 10000^{-2(i-1)/d}$$ and $\frac{\sum_{n=1}^{N} \phi(q_m)^\top \varphi(k_n) v_n}{\sum_{n=1}^{N} \phi(q_m)^\top \varphi(k_n)}$",
            ['<div class="math-block">', "R<sup>d</sup><sub>Θ, m</sub>", '<span class="frac">', '<span class="math-op">'],
        ),
        (
            "chinchilla",
            r"$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$",
            ['<span class="frac">', "N<sup>α</sup>", "D<sup>β</sup>"],
        ),
        (
            "dpo",
            r"$$L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$",
            ["L<sub>DPO</sub>", "𝔼<sub>", '<span class="frac">', "π<sub>θ</sub>", "π<sub>ref</sub>"],
        ),
        (
            "grpo",
            r"$$J_{GRPO}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \mathrm{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right)$$",
            ["J<sub>GRPO</sub>", '<span class="math-op">', "<sup>G</sup><sub>i=1</sub>", '<span class="frac">', "θ<sub>old</sub>"],
        ),
        (
            "gspo",
            r"$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$ and $\sum_{i=1}^{G} \min(s_i(\theta)\hat{A}_i, \mathrm{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i)$",
            ['<span class="frac">', "<sup><span class=\"frac\">", '<span class="math-op">'],
        ),
        (
            "flashattention",
            r"$T_r = \lceil \frac{N}{B_r} \rceil, \quad T_c = \lceil \frac{N}{B_c} \rceil$ and $m^{(j)} = \max(m^{(j-1)}, \mathrm{rowmax}(S^{(j)}))$",
            ["T<sub>r</sub>", "B<sub>r</sub>", '<span class="frac">', "m<sup>(j)</sup>"],
        ),
    ],
)
def test_domain_equations_recursive_dom_rendering(name: str, expr: str, expected_fragments: list[str]) -> None:
    cleaned = clean_generated_answer(expr)
    dom = _run_node_vm({"answers": [cleaned]})
    res = dom["answers"][0]
    _assert_clean_dom_text(res["text"])
    assert "\\" not in res["text"], f"Raw LaTeX backslash leaked in {name}: {res['text']}"
    assert "{" not in res["text"] and "}" not in res["text"], f"Raw brace leaked in {name}: {res['text']}"
    for frag in expected_fragments:
        assert frag in res["html"], f"Missing {frag!r} in {name} HTML: {res['html']}"


# ---------------------------------------------------------------------------
# 3. Synthesizer + UI panels.js contract checks
# ---------------------------------------------------------------------------


def test_synthesizer_and_panels_ui_contracts() -> None:
    # Re-exports exist and prompt has explicit math formatting guidance
    assert clean_passage_for_prompt is clean_passage_text
    assert r"\frac{1}{d_k}" in GENERATION_SYSTEM_PROMPT
    assert r"\sum_{k=1}^{d_k}" in GENERATION_SYSTEM_PROMPT

    corpus = build_offline_corpus(seed_only=True)
    router = QueryRouter(corpus.resolver, corpus.graph, corpus.vectors)
    engine = GraphVectorEngine(router, corpus.graph, corpus.vectors)
    res = engine.retrieve("What is Multi-head Latent Attention W^{DKV}?")
    allowed = [c.chunk_id for c in res.vector_chunks] or ["deepseek_2024_v2::s02::c00"]
    cid = allowed[0]

    class _FakeGen:
        name = "fake-gemini"

        def generate(self, sys_prompt: str, prompt: str) -> str:
            return (
                f"MLA computes $c_t^{{KV}} = W^{{DKV}} h_t$ and variance "
                f"σ2 = (1)/(dk)∑_{{k=1}}^{{dk}}(vk - μ)2) [{cid}]."
            )

    ans = GroundedAnswerSynthesizer(_FakeGen(), corpus.vectors).synthesize(
        "Explain MLA W^{DKV}", res
    )
    assert ans.final_validation and ans.final_validation.valid
    assert "W^{DKV}" in ans.answer
    assert r"\frac{1}{d_k}" in ans.answer
    assert "σ^2" in ans.answer
    assert f"[{cid}]" in ans.answer

    # panels.js meter() removes "✓ 100% Verified" and renderAnswer() does not write $("citations"),
    # while gateClass & gateLabel globals stay intact for funnel.js Block 6.
    dom = _run_node_vm({
        "renderResponse": {
            "answer": ans.answer,
            "mode": "offline",
            "refused": False,
            "latency_ms": 42,
            "routing_latency_ms": 5,
            "retrieval_latency_ms": 20,
            "generation_latency_ms": 17,
            "citation_gate": {"status": "PASS", "cited_ids": [cid], "invented_ids": []},
            "usage_telemetry": {"total_tokens": 100, "generator_model": "gemini-3.5-flash-lite"},
            "citations": [
                {
                    "chunk_id": cid,
                    "paper_id": cid.split("::")[0],
                    "paper_title": "DeepSeek-V2",
                    "section_path": "**2 Architecture**",
                    "source_origin": "vector",
                    "score": 0.99,
                    "text_snippet": XIONG_RAW_CHUNK[:200],
                    "full_text": XIONG_RAW_CHUNK,
                }
            ],
        }
    })
    ui = dom["ui"]
    assert "✓ 100% Verified" not in ui["meterText"]
    assert not ui["citationsTouched"]
    assert ui["gateClassType"] == "function" and ui["gateClassPass"] == "ok"
    assert ui["gateLabelType"] == "function" and ui["gateLabelPass"] == "✓ 100% Verified"
    assert '<span class="frac">' in ui["answerHtml"]


# ---------------------------------------------------------------------------
# 4. Live Postgres 620-chunk Python + Node VM DOM sweep
# ---------------------------------------------------------------------------


def test_live_postgres_620_chunks_formatting_and_dom() -> None:
    try:
        import psycopg

        conn = psycopg.connect(settings.postgres_dsn, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Live Postgres unreachable: {exc}")

    with conn:
        rows = conn.execute("SELECT chunk_id, section_path, raw_text FROM chunks ORDER BY chunk_id").fetchall()

    assert len(rows) >= 600, f"Expected ~620 live chunks, found {len(rows)}"

    cleaned_payload = []
    raw_payload = []
    for cid, sec_path, raw_text in rows:
        csp = clean_section_path(sec_path)
        ctxt = clean_passage_text(raw_text)

        for target in (csp, ctxt):
            assert not re.search(r"</?(?:u|sup|sub|br)\b|<!--", target, re.I), f"{cid}: raw HTML remained"
            assert "\ufffd" not in target, f"{cid}: U+FFFD remained"
            assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", target), f"{cid}: control char remained"
            assert "~~" not in target, f"{cid}: strikethrough remained"

        # No odd/stray ** per line in cleaned text or section path
        assert "**" not in csp, f"{cid}: ** remained in section path"
        for line in ctxt.splitlines():
            assert line.count("**") % 2 == 0, f"{cid}: stray ** in {line!r}"

        # Valid snake_case IDs and token tags inside raw_text must survive in cleaned text
        for tag in ("<eos>", "<bos>", "<pad>", "<resv>"):
            if tag in (raw_text or ""):
                assert tag in ctxt, f"{cid}: lost token tag {tag}"

        for m in re.finditer(r"(?<![A-Za-z0-9_])[a-z]{2,}_[a-z0-9]+(?:_[a-z0-9]+)+(?![A-Za-z0-9_])", raw_text or ""):
            tok = m.group(0)
            if "http" not in tok:
                assert tok in ctxt, f"{cid}: lost snake_case token {tok}"

        cleaned_payload.append(ctxt)
        raw_payload.append(raw_text)

    # Render ALL 620 chunks through panels.js in Node VM (both cleaned and raw!)
    for label, texts in (("cleaned", cleaned_payload), ("raw", raw_payload)):
        dom = _run_node_vm({"excerpts": texts})
        for (cid, _, raw_text), item in zip(rows, dom["excerpts"]):
            _assert_clean_dom_text(item["text"], raw_text)
            for tag in ("<eos>", "<bos>", "<pad>", "<resv>"):
                if tag in (raw_text or ""):
                    assert tag in item["text"], f"{cid} ({label}): DOM lost {tag}"
