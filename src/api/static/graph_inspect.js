// Left screen controls: floating inspector, node filter, view toggle, chunk toggle, fit, chunk highlighting.
"use strict";

function chunkChips(ids) {
  if (!ids || !ids.length) return el("div", { class: "empty" }, "no source chunks recorded");
  return el("div", {}, [...new Set(ids)].slice(0, 24).map((c) =>
    el("div", { class: "chip", "data-chunk": c, onclick: () => App.highlight(c, { source: "graph" }) }, el("span", { class: "cid" }, c))));
}

function openInspector(target, ...content) {
  if (GraphUI.selEl) GraphUI.selEl.classList.remove("sel");
  GraphUI.selEl = target; if (target) target.classList.add("sel");
  const box = $("inspector");
  box.innerHTML = ""; box.hidden = false;
  box.append(el("span", { class: "x", onclick: () => { box.hidden = true; if (GraphUI.selEl) GraphUI.selEl.classList.remove("sel"); } }, "✕"), ...content);
}

function inspectNode(n) {
  const m = n.meta || {};
  let chunks = m.source_chunk_ids || [];
  if (!chunks.length) chunks = View.edges.filter((e) => e.source === n.id || e.target === n.id).flatMap((e) => e.source_chunk_ids || []);
  if (n.type === "Chunk") chunks = [n.id];
  openInspector(n.el,
    el("h4", {}, n.name),
    el("div", { class: "kv" },
      el("span", { class: "k" }, "entity_type"), el("span", { class: "v" }, n.type),
      el("span", { class: "k" }, "entity_id"), el("span", { class: "v" }, n.id),
      n.role ? [el("span", { class: "k" }, "role"), el("span", { class: "v" }, n.role)] : null,
      m.category ? [el("span", { class: "k" }, "category"), el("span", { class: "v" }, m.category)] : null,
      m.chunk_count ? [el("span", { class: "k" }, "chunks"), el("span", { class: "v" }, m.chunk_count)] : null,
      m.section_path ? [el("span", { class: "k" }, "section"), el("span", { class: "v" }, m.section_path)] : null),
    (m.aliases || []).length ? el("div", { class: "muted" }, "aliases: " + m.aliases.join(", ")) : null,
    m.description ? el("div", { class: "q" }, m.description) : null,
    el("div", { class: "muted" }, "source_chunk_ids"), chunkChips(chunks));
}

function inspectEdge(e) {
  const a = View.byId.get(e.source), b = View.byId.get(e.target);
  const quotes = (e.evidence_quotes && e.evidence_quotes.length) ? e.evidence_quotes : (e.evidence_quote ? [e.evidence_quote] : []);
  openInspector(e.el,
    el("h4", {}, `${a ? a.name : e.source} → ${b ? b.name : e.target}`),
    el("div", { class: "kv" },
      el("span", { class: "k" }, "relation"), el("span", { class: "v" }, e.relation_type),
      el("span", { class: "k" }, "status"), el("span", { class: "v" }, e.kind === "selected"
        ? `selected · hop ${(e.hop_index ?? 0) + 1} of path ${(e.path_index ?? 0) + 1}${e.score != null ? ` · score ${Number(e.score).toFixed(3)}` : ""}`
        : e.kind === "rejected" ? "explored neighbor, not chosen" : "corpus edge (not on this search)"),
      el("span", { class: "k" }, "source"), el("span", { class: "v" }, e.source),
      el("span", { class: "k" }, "target"), el("span", { class: "v" }, e.target)),
    quotes.length ? quotes.map((q) => el("div", { class: "q" }, `“${q}”`)) : el("div", { class: "empty" }, "no evidence quote on this edge"),
    el("div", { class: "muted" }, "source_chunk_ids"), chunkChips(e.source_chunk_ids));
}

function applyFilter() {
  const q = $("graph-filter").value.trim().toLowerCase();
  const hits = new Set();
  for (const n of View.nodes) {
    const m = n.raw || n.meta || {};
    const match = q && [n.name, n.id, n.type, ...(m.aliases || [])].some((s) => String(s || "").toLowerCase().includes(q));
    if (match) hits.add(n.id);
  }
  for (const n of View.nodes) { n.el.classList.toggle("match", hits.has(n.id)); n.el.classList.toggle("fade", !!q && !hits.has(n.id) && !n.role); }
  for (const e of View.edges) { const keep = !q || hits.has(e.source) || hits.has(e.target) || e.kind === "selected"; e.el.classList.toggle("fade", !keep); if (e.label) e.label.classList.toggle("fade", !keep); }
  return hits;
}

App.onHighlight((chunk, source) => {
  let first = null; // eslint-disable-line no-unused-vars
  for (const e of View.edges) {
    const on = !!chunk && (e.source_chunk_ids || []).includes(chunk);
    e.el.classList.toggle("hl-edge", on);
    if (on) { e.el.classList.remove("flash"); void e.el.getBBox(); e.el.classList.add("flash"); first = first || e; }
  }
  for (const n of View.nodes) {
    const on = !!chunk && (n.id === chunk || ((n.meta && n.meta.source_chunk_ids) || []).includes(chunk));
    n.el.classList.toggle("hl-node", on);
  }
});

document.addEventListener("DOMContentLoaded", () => {
  View.init($("graph"));
  document.querySelectorAll("#graph-view button").forEach((b) => b.addEventListener("click", () => {
    GraphUI.view = b.dataset.view; if (App.last) window.showGraph();
  }));
  $("chunks").addEventListener("change", () => { if (GraphUI.view === "full" && App.last) renderFull(); });
  $("fit").addEventListener("click", () => View.fit());
  $("graph-filter").addEventListener("input", applyFilter);
  $("graph-filter").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { const h = applyFilter(); if (h.size) View.fit(h); } });
  $("graph").addEventListener("click", (ev) => { if (ev.target === $("graph")) $("inspector").hidden = true; });
  let t = null;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (App.last) window.showGraph(); }, 300); });
});
