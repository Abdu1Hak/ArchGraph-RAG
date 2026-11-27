// Right screen: the retrieval pipeline as a node-and-arrow flow diagram built from this query's
// routing decision, graph_trace and funnel_trace. Inactive branches are dimmed; active arrows pulse.
"use strict";

const Flow = { arrows: [], ro: null };

function tierBadge(t) {
  if (!t) return el("span", { class: "tier", title: "Tier 1 alias scan over the question" }, "T1");
  const cls = t.includes("TIER_3") ? "t3" : t.includes("TIER_2") ? "t2" : t.includes("PAPER") ? "tp" : "";
  const txt = t.includes("TIER_3") ? "T3" : t.includes("TIER_2") ? "T2" : t.includes("PAPER") ? "T1·paper" : "T1";
  return el("span", { class: "tier " + cls, title: t }, txt);
}

function entityPills(j) {
  const names = new Map(((j.graph_trace || {}).nodes || []).map((n) => [n.entity_id, n.canonical_name]));
  const tiers = new Map((j.entity_links || []).map((l) => [l.entity_id, l]));
  const ids = j.resolved_entity_ids || [];
  if (!ids.length) return el("div", { class: "empty" }, "no entities resolved");
  return el("div", {}, ids.map((id) => {
    const l = tiers.get(id);
    return el("span", { class: "ent", title: l ? `"${l.phrase}" -> ${id} (${l.tier}, ${l.score})` : `${id} (alias scan)` },
      tierBadge(l && l.tier), names.get(id) || short(id));
  }));
}

function fchip(c, { badge = null, move = null } = {}) {
  return el("div", { class: "chip", "data-chunk": c.chunk_id, title: `${c.chunk_id} · rank ${c.rank} · score ${c.score}`,
    onclick: (ev) => { ev.stopPropagation(); App.highlight(c.chunk_id, { source: "flow" }); } },
    badge, el("span", { class: "cid" }, c.chunk_id), move, el("span", { class: "sc" }, Number(c.score).toFixed(3)));
}
const chips = (list, opt) => (list && list.length ? list.map((c) => fchip(c, opt ? opt(c) : {})) : el("div", { class: "empty" }, "empty"));

function fnode(id, num, title, stat, body, more, { off = false, lane = "", center = false } = {}) {
  return el("div", { id, class: `fnode ${lane}${off ? " off" : ""}${center ? " center" : ""}`,
    onclick: (ev) => { if (!ev.target.closest(".chip")) { ev.currentTarget.classList.toggle("open"); requestAnimationFrame(drawArrows); } } },
    el("div", { class: "fh" }, el("span", { class: "num" }, num), title, el("span", { class: "fs" }, stat)),
    el("div", { class: "fb" }, body), more ? el("div", { class: "more" }, more) : null);
}
const kv = (...pairs) => el("div", { class: "kv" }, pairs.filter(Boolean).map(([k, v]) => [el("span", { class: "k" }, k), el("span", { class: "v" }, v)]));

window.renderFlow = function (j) {
  const f = j.funnel_trace || {}, gt = j.graph_trace || {}, u = j.usage_telemetry || {}, gate = j.citation_gate || {};
  const oos = j.route === "OUT_OF_SCOPE";
  const graphOn = !oos && !!j.cypher_template_id, vecOn = !oos;
  const hops = [...(gt.selected_path_edges || [])].sort((a, b) => (a.path_index ?? 0) - (b.path_index ?? 0) || (a.hop_index ?? 0) - (b.hop_index ?? 0));
  const nameOf = new Map((gt.nodes || []).map((n) => [n.entity_id, n.canonical_name]));
  const hopRows = hops.map((h) => el("div", { class: "hop" }, el("b", {}, `[Hop ${(h.hop_index ?? 0) + 1}: ${h.relation_type}] `),
    `${nameOf.get(h.source) || short(h.source)} → ${nameOf.get(h.target) || short(h.target)}`));
  const rrfRank = f.rrf_rank_by_chunk || {};
  const moveOf = (c) => {
    const from = rrfRank[c.chunk_id];
    if (!from) return { move: el("span", { class: "mv", title: "outside the RRF top-20 pool" }, "new") };
    const d = from - c.rank;
    return { move: el("span", { class: "mv " + (d > 0 ? "up" : d < 0 ? "down" : ""), title: `RRF rank ${from} → reranked rank ${c.rank}` },
      `#${from}→#${c.rank}${d > 0 ? " ↑" : d < 0 ? " ↓" : ""}`) };
  };
  const origin = (c) => ({ badge: el("span", { class: "origin " + (c.source_origin || "vector") }, c.source_origin === "both" ? "both +0.05" : c.source_origin || "vector") });
  const bridge = (j.fallback_reasons || []).filter((r) => /bridged|cascaded|T6|vector-seeded/.test(r));
  const flow = $("flow");
  flow.innerHTML = "";
  flow.append(
    fnode("f-query", 1, "User Query", `${(j.resolved_entity_ids || []).length} entities`,
      [el("div", {}, App.lastQuestion), entityPills(j)], null, { center: true }),
    fnode("f-router", 2, "Agent Router", u.router_model || "",
      [el("span", { class: "route " + j.route }, j.route), " ", el("span", { class: "muted" }, j.query_intent),
       el("div", { class: "conf", title: `confidence ${j.confidence}` }, el("span", { style: `width:${Math.round(j.confidence * 100)}%` })),
       kv(["template", j.cypher_template_id || "none"], ["graph_subquery", j.graph_subquery || "-"], ["vector_subquery", j.vector_subquery || "-"])],
      [kv(["$params", JSON.stringify(j.template_params || {})], ["confidence", Number(j.confidence).toFixed(2)],
          ["routing", `${Number(j.routing_latency_ms || 0).toFixed(1)} ms`], ["fallbacks", (j.fallback_reasons || []).join(" ; ") || "none"])],
      { center: true }),
    el("div", { class: "lanes" },
      el("div", { class: "lane" }, el("div", { class: "lane-title" }, "Graph retrieval (Neo4j / Cypher)"),
        fnode("f-linker", "3a", "Entity Linker", `${(j.entity_links || []).length} query-time links`, entityPills(j),
          (j.entity_links || []).map((l) => el("div", { class: "hop" }, `"${l.phrase}" → ${short(l.entity_id)} · ${l.tier} · ${l.score}`)),
          { off: !graphOn, lane: "graph-lane" }),
        fnode("f-cypher", "3b", "Cypher Template + Endpoint Bridging", `${hops.length} hops`,
          [kv(["template", j.cypher_template_id || "not run"]), hopRows.slice(0, 4)],
          [hopRows.slice(4), kv(["$params", JSON.stringify(j.template_params || {})]), bridge.map((r) => el("div", { class: "muted" }, r))],
          { off: !graphOn, lane: "graph-lane" }),
        fnode("f-gchunks", "3c", "Per-Hop Stratified Graph Chunks", `${(f.reranked_graph_supporting_top_k || []).length}`,
          chips(f.reranked_graph_supporting_top_k), null, { off: !graphOn, lane: "graph-lane" })),
      el("div", { class: "lane" }, el("div", { class: "lane-title" }, "Vector + lexical retrieval (pgvector)"),
        el("div", { class: "pair" },
          fnode("f-dense", "4a", "Dense (Cosine)", `top ${(f.dense_top_k || []).length}`, chips(f.dense_top_k), null, { off: !vecOn }),
          fnode("f-bm25", "4b", "BM25 Lexical", `top ${(f.bm25_top_k || []).length}`, chips(f.bm25_top_k), null, { off: !vecOn })),
        fnode("f-rrf", "4c", "Hybrid RRF Fusion", `top ${(f.rrf_top_k || []).length}`, chips(f.rrf_top_k),
          f.query ? el("div", { class: "muted" }, `query: ${f.query}`) : null, { off: !vecOn }),
        fnode("f-rerank", "4d", "Cross-Encoder Reranked Top-K", "RRF rank → reranked", chips(f.reranked_vector_top_k, moveOf), null, { off: !vecOn }))),
    fnode("f-context", 5, "Score-Deduplicated LLM Context", `${(f.final_deduplicated_llm_context || []).length} chunks`,
      chips(f.final_deduplicated_llm_context, origin), null, { off: oos, center: true }),
    fnode("f-gen", 6, "Gemini Generator + Citation Gate", u.generator_model || "",
      [el("span", { class: "badge " + gateClass(gate.status) }, gateLabel(gate.status)), " ",
       el("span", { class: "muted" }, `${fmt(u.generator_input_tokens)} in / ${fmt(u.generator_output_tokens)} out tok · ${Number(j.generation_latency_ms || 0).toFixed(0)} ms`),
       el("div", {}, (gate.cited_ids || []).map((c) => fchip({ chunk_id: c, rank: "", score: NaN })))],
      [kv(["generator calls", gate.generator_calls ?? j.generator_calls], ["repairs", gate.repair_attempts], ["cropped", gate.cropped_sentences],
          ["invented ids", (gate.invented_ids || []).join(", ") || "none"], ["uncited claims", gate.uncited_claims])],
      { center: true }),
  );
  flow.querySelectorAll("#f-gen .chip .sc").forEach((s) => (s.textContent = "cited"));
  Flow.arrows = [
    ["f-query", "f-router", true], ["f-router", "f-linker", graphOn, "graph"], ["f-linker", "f-cypher", graphOn, "graph"],
    ["f-cypher", "f-gchunks", graphOn, "graph"], ["f-router", "f-dense", vecOn], ["f-router", "f-bm25", vecOn],
    ["f-dense", "f-rrf", vecOn], ["f-bm25", "f-rrf", vecOn], ["f-rrf", "f-rerank", vecOn],
    ["f-gchunks", "f-context", graphOn, "graph"], ["f-rerank", "f-context", vecOn], ["f-context", "f-gen", !oos],
    ...(oos ? [["f-router", "f-gen", true]] : []),
  ];
  $("flow-sub").textContent = oos ? "refused at the router / subject gate" : `${graphOn ? "graph + " : ""}vector channels`;
  if (!Flow.ro) { Flow.ro = new ResizeObserver(() => drawArrows()); Flow.ro.observe(flow); }
  requestAnimationFrame(drawArrows);
  App.onHighlightFlow = true;
};

function drawArrows() {
  const flow = $("flow");
  let svg = flow.querySelector(".flow-arrows");
  if (!svg) { svg = svgEl("svg", { class: "flow-arrows" }); flow.prepend(svg); }
  svg.innerHTML = "";
  const defs = svgEl("defs");
  for (const [id, c] of [["fa-on", "#3b82f6"], ["fa-graph", "#10d9a0"], ["fa-off", "#2f3542"]]) {
    const m = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
    m.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: c })); defs.append(m);
  }
  svg.append(defs);
  const box = flow.getBoundingClientRect();
  for (const [from, to, on, kind] of Flow.arrows) {
    const a = $(from), b = $(to);
    if (!a || !b) continue;
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    const x1 = ra.left + ra.width / 2 - box.left, y1 = ra.bottom - box.top, x2 = rb.left + rb.width / 2 - box.left, y2 = rb.top - box.top - 2;
    const dy = Math.max(18, (y2 - y1) / 2), d = `M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`;
    const cls = `${kind || ""} ${on ? "on" : "off"}`;
    svg.append(svgEl("path", { d, class: cls, "marker-end": `url(#${on ? (kind === "graph" ? "fa-graph" : "fa-on") : "fa-off"})` }));
    if (on) svg.append(svgEl("path", { d, class: "pulse" }));
  }
}
