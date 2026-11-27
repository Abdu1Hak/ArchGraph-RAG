// Answer card: question bubble, grounded answer with citation pills + popover, telemetry footer, sources.
"use strict";

const CITE_RE = /\[([a-z0-9_]+::s\d+::c\d+)\]/g;
const gateClass = (s) => ({ PASS: "ok", REPAIRED_PASS: "ok", CROPPED: "warn", EXTRACTIVE_FALLBACK: "warn", ABSTAINED: "warn", REFUSED: "bad" }[s] || "warn");
const gateLabel = (s) => ({ PASS: "✓ 100% Verified", REPAIRED_PASS: "✓ Verified after repair", CROPPED: "Cropped", EXTRACTIVE_FALLBACK: "Extractive fallback",
  ABSTAINED: "Abstained", REFUSED: "Refused", UNVALIDATED: "Unvalidated" }[s] || s || "-");

function citePill(id) {
  const paper = id.split("::")[0], sec = id.split("::").slice(1).join("·");
  return el("span", { class: "cite", "data-chunk": id, title: id, onclick: () => App.highlight(id, { source: "answer" }) }, `${paper} · ${sec}`);
}

// Inline text -> nodes: **bold** and [chunk_id] citation pills; everything else is plain text (never HTML).
function inline(text) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*)/g)) {
    if (/^\*\*[^*]+\*\*$/.test(part)) { out.push(el("strong", {}, ...inline(part.slice(2, -2)))); continue; }
    let last = 0;
    for (const m of part.matchAll(CITE_RE)) { out.push(part.slice(last, m.index)); out.push(citePill(m[1])); last = m.index + m[0].length; }
    out.push(part.slice(last));
  }
  return out.filter((x) => x !== "");
}

function renderMarkdownLite(text) {
  const frag = document.createDocumentFragment();
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) { if (!list) { list = el("ul"); frag.append(list); } list.append(el("li", {}, ...inline(bullet[1]))); continue; }
    list = null;
    if (line.trim()) frag.append(el("p", {}, ...inline(line)));
  }
  return frag;
}

function meter(j) {
  const u = j.usage_telemetry || {}, g = j.citation_gate || {};
  const router = (u.router_input_tokens || 0) + (u.router_output_tokens || 0), embed = u.embedding_tokens_est || 0;
  const gen = (u.generator_input_tokens || 0) + (u.generator_output_tokens || 0), total = u.total_tokens || router + embed + gen;
  const pct = (x) => (total ? (100 * x) / total : 0);
  const retrievalOnly = Math.max(0, (j.retrieval_latency_ms || 0) - (j.routing_latency_ms || 0));
  const box = $("meter"); box.innerHTML = "";
  box.append(
    el("div", { class: "tok" },
      el("div", { class: "tok-bar", title: "token share by component" },
        el("span", { class: "seg-router", style: `width:${pct(router)}%` }), el("span", { class: "seg-embed", style: `width:${pct(embed)}%` }),
        el("span", { class: "seg-gen", style: `width:${pct(gen)}%` })),
      el("div", { class: "tok-legend" }, el("i", { class: "seg-router" }), "Router: ", el("b", {}, fmt(router)), " tok",
        el("i", { class: "seg-embed" }), "Embed: ", el("b", {}, fmt(embed)), " tok", el("i", { class: "seg-gen" }), "Generator: ", el("b", {}, fmt(gen)), " tok",
        " · Total: ", el("b", {}, fmt(total)), ` tok · ${u.total_api_requests_used || 0} API reqs` + (u.embedding_cache_hits ? ` (${u.embedding_cache_hits} embed cache hits)` : ""))),
    el("span", { class: "badge", title: `round-trip ${Math.round(App.roundTripMs || 0)} ms` },
      `⏱ ${fmt(Math.round(j.latency_ms))} ms`, el("span", { class: "muted" },
        `routing ${Math.round(j.routing_latency_ms || 0)} · retrieval ${Math.round(retrievalOnly)} · gen ${Math.round(j.generation_latency_ms || 0)} ms`)),
    el("span", { class: "badge " + gateClass(g.status), title: `cited ${(g.cited_ids || []).length}, invented ${(g.invented_ids || []).length}` }, gateLabel(g.status)),
    ...(u.note ? [el("span", { class: "muted" }, u.note)] : []));
}

function showCitation(id) {
  const pop = $("cite-pop");
  const c = ((App.last && App.last.citations) || []).find((x) => x.chunk_id === id);
  if (!id) { pop.hidden = true; return; }
  pop.innerHTML = ""; pop.hidden = false;
  pop.append(el("span", { class: "x", onclick: () => App.highlight(null, { toggle: false }) }, "✕"),
    c ? [el("div", { class: "t" }, c.paper_title), el("div", { class: "s" }, `${c.section_path} · ${c.chunk_id} · ${c.source_origin}${c.score != null ? ` · ${c.score}` : ""}`),
         el("div", {}, `“${c.text_snippet}”`)]
      : [el("div", { class: "t" }, id), el("div", { class: "s" }, "retrieved chunk (not cited in the final answer)")]);
}

window.renderAnswer = function (j, question) {
  $("q-text").textContent = question;
  const u = j.usage_telemetry || {};
  $("ans-meta").textContent = `${MODE_LABELS[j.mode] || j.backend} · ${modelLabel(u.generator_model || "")}`;
  const box = $("answer"); box.innerHTML = "";
  if (j.refused) box.append(el("p", { class: "refused" }, "Refused: out of scope or the subject is not in the indexed corpus."));
  box.append(renderMarkdownLite(j.answer));
  $("cite-pop").hidden = true;
  meter(j);
  const cards = $("citations"); cards.innerHTML = "";
  for (const c of j.citations || []) {
    cards.append(el("div", { class: "source-card", "data-chunk": c.chunk_id, onclick: () => App.highlight(c.chunk_id, { source: "answer" }) },
      el("div", { class: "h" }, el("span", {}, c.chunk_id), el("span", { class: "origin " + c.source_origin }, c.source_origin)),
      el("div", { class: "t" }, `${c.paper_title} · ${c.section_path}`), el("div", {}, c.text_snippet)));
  }
  for (const f of j.verbalized_graph_facts || []) cards.append(el("div", { class: "fact" }, f));
};

App.onHighlight((chunk) => {
  document.querySelectorAll("[data-chunk]").forEach((n) => n.classList.toggle("hl", !!chunk && n.dataset.chunk === chunk));
  showCitation(chunk);
});

document.addEventListener("DOMContentLoaded", () => {
  const q = $("question");
  q.addEventListener("input", autosize);
  q.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); ask(); } });
  $("composer").addEventListener("submit", ask);
  loadConfig().then(() => {
    const params = new URLSearchParams(location.search); // shareable: /?mode=offline&q=...
    if (params.get("mode")) { $("mode").value = params.get("mode"); $("mode").dispatchEvent(new Event("change")); }
    if (params.get("view") === "full") GraphUI.view = "full";
    if (params.get("chunks") === "1") $("chunks").checked = true;
    if (params.get("q")) { q.value = params.get("q"); ask(); }
  }).catch((e) => toast(`Could not load /ui-config or /examples: ${e.message}`, { error: true }));
});
