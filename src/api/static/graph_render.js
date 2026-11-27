// Left screen: "Search Subgraph" (this query's traversal) and "Full 50-Paper Graph" (GET /graph/full),
// with the current search's seeds and winning path drawn on top. Inspector, filter, fit, chunk highlight.
"use strict";

const GraphUI = { view: "search", fullCache: new Map(), selEl: null };
const ekey = (e) => `${e.source}|${e.relation_type}|${e.target}`;

function traceIndex() {
  const gt = (App.last && App.last.graph_trace) || {};
  return {
    gt, roles: new Map((gt.nodes || []).map((n) => [n.entity_id, n.role])),
    meta: new Map((gt.nodes || []).map((n) => [n.entity_id, n])),
    selected: new Map((gt.selected_path_edges || []).map((e) => [ekey(e), e])),
    rejected: new Map((gt.explored_rejected_edges || []).map((e) => [ekey(e), e])),
  };
}

function drawGraph(nodes, edges, { full = false } = {}) {
  const root = View.reset();
  View.nodes = nodes; View.edges = edges; View.byId = new Map(nodes.map((n) => [n.id, n]));
  const pairs = new Map();
  for (const e of edges) { const k = [e.source, e.target].sort().join("|"); const c = pairs.get(k) || 0; pairs.set(k, c + 1); e.bend = c ? (c % 2 ? 1 : -1) * Math.ceil(c / 2) : 0; }
  const gE = svgEl("g"), gL = svgEl("g"), gN = svgEl("g");
  root.append(gE, gL, gN);
  const order = (e) => ({ plumb: 0, bg: 1, rejected: 2, selected: 3 }[e.kind]);
  for (const e of [...edges].sort((a, b) => order(a) - order(b))) {
    const marker = e.kind === "selected" ? "ar-sel" : e.kind === "rejected" ? "ar-rej" : e.kind === "bg" ? "ar-bg" : null;
    e.el = svgEl("path", { class: `edge ${e.kind}`, ...(marker ? { "marker-end": `url(#${marker})` } : {}) });
    const t = svgEl("title"); t.textContent = `${e.relation_type}: ${e.source} -> ${e.target}`; e.el.append(t);
    e.el.addEventListener("click", (ev) => { ev.stopPropagation(); inspectEdge(e); });
    gE.append(e.el);
    if (e.kind === "selected" || (!full && e.kind === "rejected")) {
      e.label = svgEl("text", { class: "elabel" + (e.kind === "rejected" ? " rejected" : ""), "text-anchor": "middle" });
      e.label.textContent = e.kind === "selected" ? `[Hop ${(e.hop_index ?? 0) + 1}: ${e.relation_type}]` : e.relation_type.toLowerCase();
      gL.append(e.label);
    }
  }
  for (const n of [...nodes].sort((a, b) => (a.cls === "bg") - (b.cls === "bg")).reverse()) {
    n.el = svgEl("g", { class: `node ${n.cls}` }); n.el.__node = n;
    n.el.append(svgEl("circle", { r: n.r, fill: n.color }));
    const tt = svgEl("title"); tt.textContent = `${n.name} (${n.type})`; n.el.append(tt);
    if (n.label) { const t = svgEl("text", { x: n.r + 4, y: 4 }); t.textContent = n.label; n.el.append(t); }
    n.el.addEventListener("click", (ev) => { ev.stopPropagation(); inspectNode(n); });
    gN.append(n.el);
  }
  View.onTick = place; place();
}

function place() {
  for (const e of View.edges) {
    const a = View.byId.get(e.source), b = View.byId.get(e.target);
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, nx = -dy / d, ny = dx / d;
    const mx = (a.x + b.x) / 2 + nx * e.bend * 24, my = (a.y + b.y) / 2 + ny * e.bend * 24;
    const sx = a.x + (dx / d) * (a.r + 1), sy = a.y + (dy / d) * (a.r + 1), tx = b.x - (dx / d) * (b.r + 4), ty = b.y - (dy / d) * (b.r + 4);
    e.el.setAttribute("d", `M${sx},${sy} Q${mx},${my} ${tx},${ty}`);
    if (e.label) { e.label.setAttribute("x", (sx + 2 * mx + tx) / 4); e.label.setAttribute("y", (sy + 2 * my + ty) / 4 - 3); }
  }
  for (const n of View.nodes) n.el.setAttribute("transform", `translate(${n.x},${n.y})`);
}

const clip = (s, n = 30) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

function renderSearch() {
  const { gt, meta } = traceIndex();
  const nodes = (gt.nodes || []).map((m, i) => ({ id: m.entity_id, name: m.canonical_name, type: m.entity_type, role: m.role, meta: m,
    cls: m.role, r: m.role === "seed" ? 12 : m.role === "selected_path" ? 9.5 : 6,
    color: m.role === "seed" ? "var(--seed)" : m.role === "selected_path" ? "var(--path)" : "var(--rej)",
    label: (m.role === "seed" ? "SEED · " : "") + clip(m.canonical_name, 34), x: 0, y: 0, vx: 0, vy: 0, i }));
  const edges = [...(gt.selected_path_edges || []).map((e) => ({ ...e, kind: "selected" })),
                 ...(gt.explored_rejected_edges || []).map((e) => ({ ...e, kind: "rejected" }))];
  for (const e of edges) for (const id of [e.source, e.target]) if (!nodes.some((n) => n.id === id)) {
    nodes.push({ id, name: short(id), type: "?", role: "rejected_neighbor", cls: "rejected_neighbor", r: 6, color: "var(--rej)", label: clip(short(id)), x: 0, y: 0, vx: 0, vy: 0 });
  }
  View.reset();
  // Seeds on the left, path nodes by hop distance from a seed, rejected neighbors float around them.
  const depth = new Map(nodes.filter((n) => n.role === "seed").map((n) => [n.id, 0]));
  for (let pass = 0; pass < 6; pass++) for (const e of edges) if (e.kind === "selected") {
    const ds = depth.get(e.source), dt = depth.get(e.target);
    if (ds !== undefined && (dt === undefined || dt > ds + 1)) depth.set(e.target, ds + 1);
    if (dt !== undefined && (ds === undefined || ds > dt + 1)) depth.set(e.source, dt + 1);
  }
  const maxD = Math.max(1, ...depth.values()), seen = new Map();
  nodes.forEach((n, i) => {
    const d = depth.get(n.id);
    if (d !== undefined) { const k = seen.get(d) || 0; seen.set(d, k + 1); n.ax = 90 + (d / maxD) * (View.W - 300); n.x = n.ax; n.y = 80 + ((k * 113) % (View.H - 160)); }
    else { n.ax = null; n.x = View.W / 2 + 260 * Math.cos(i * 2.39); n.y = View.H / 2 + 220 * Math.sin(i * 2.39); }
  });
  View.simulate(nodes, edges, { iters: 320, repel: 9000, length: (e) => (e.kind === "selected" ? 200 : 120),
    spring: (e) => (e.kind === "selected" ? 0.05 : 0.03), anchorX: (n) => n.ax });
  drawGraph(nodes, edges);
  View.fit();
  $("graph-stats").textContent = nodes.length
    ? `${nodes.length} nodes · ${(gt.selected_path_edges || []).length} selected hops · ${(gt.explored_rejected_edges || []).length} rejected edges` +
      (gt.rejected_edges_truncated ? ` (+${gt.rejected_edges_truncated} not drawn)` : "")
    : "No graph traversal for this question (vector-only route or refused)";
  setLegend("search");
}

async function renderFull() {
  const chunks = $("chunks").checked, mode = App.last ? App.last.mode : currentMode();
  const key = `${mode}|${chunks}`;
  $("graph-stats").textContent = "Loading the full knowledge graph...";
  let data = GraphUI.fullCache.get(key);
  if (!data) {
    const params = new URLSearchParams({ include_chunks: chunks });
    if (["live", "offline", "gemini"].includes(mode)) params.set("mode", mode);
    const r = await fetch(`/graph/full?${params}`);
    data = await r.json();
    if (!r.ok) { $("graph-stats").textContent = `GET /graph/full failed: ${data.detail || r.status}`; return; }
    GraphUI.fullCache.set(key, data);
  }
  if (GraphUI.view !== "full") return;
  View.reset(); // measure the SVG before computing starting positions from View.W / View.H
  const { roles, meta, selected, rejected } = traceIndex();
  const deg = new Map();
  for (const e of data.edges) for (const id of [e.source, e.target]) deg.set(id, (deg.get(id) || 0) + 1);
  const types = Object.keys(TYPE_COLORS);
  const nodes = data.nodes.map((n, i) => {
    const role = roles.get(n.entity_id), isChunk = n.entity_type === "Chunk", ti = Math.max(0, types.indexOf(n.entity_type));
    const ang = (ti / types.length) * 2 * Math.PI + (i % 17) * 0.02, rad = isChunk ? 420 : 140 + (i % 9) * 22;
    return { id: n.entity_id, name: n.canonical_name, type: n.entity_type, role, meta: meta.get(n.entity_id) || n, raw: n,
      cls: role || "bg", color: role === "seed" ? "var(--seed)" : role === "selected_path" ? "var(--path)" : TYPE_COLORS[n.entity_type] || "#64748b",
      r: role ? 9 : isChunk ? 2.6 : 3.5 + Math.min(7, Math.sqrt(n.chunk_count || deg.get(n.entity_id) || 1) * 1.3),
      label: role || (!isChunk && (deg.get(n.entity_id) || 0) >= 7) ? clip(n.canonical_name, 26) : null,
      x: View.W / 2 + rad * Math.cos(ang), y: View.H / 2 + rad * Math.sin(ang), vx: 0, vy: 0 };
  });
  const edges = data.edges.map((e) => {
    const k = ekey(e), sel = selected.get(k);
    const kind = sel ? "selected" : rejected.has(k) ? "rejected" : ["MENTIONED_IN", "IN_PAPER"].includes(e.relation_type) ? "plumb" : "bg";
    return { ...e, ...(sel ? { hop_index: sel.hop_index, path_index: sel.path_index } : {}), kind };
  });
  View.reset();
  View.simulate(nodes, edges, { iters: chunks ? 200 : 260, repel: chunks ? 1400 : 2600, gravity: 0.006,
    length: (e) => (e.kind === "plumb" ? 40 : 75), spring: (e) => (e.kind === "plumb" ? 0.02 : 0.04) });
  drawGraph(nodes, edges, { full: true });
  View.fit();
  const s = data.stats;
  $("graph-stats").textContent = `${s.papers} papers · ${s.entities} entities · ${s.chunks} chunks · ${s.relationships} relationships · showing ${s.returned_nodes} nodes / ${s.returned_edges} edges (${data.backend})`;
  setLegend("full");
  applyFilter();
}

function setLegend(view) {
  const box = $("legend"); box.innerHTML = "";
  const dot = (c, t) => el("span", {}, el("i", { style: `background:${c}` }), t);
  const ln = (c, t, dash) => el("span", {}, el("i", { class: "ln", style: `border-color:${c};${dash ? "border-top-style:dashed" : ""}` }), t);
  box.append(dot("var(--seed)", "Seed"), dot("var(--path)", "Winning path"), ln("var(--cyan)", "[Hop n] selected"), ln("var(--rej)", "Rejected neighbor", true));
  if (view === "full") for (const [t, c] of Object.entries(TYPE_COLORS)) if (t !== "Chunk" || $("chunks").checked) box.append(dot(c, t));
}

window.showGraph = function () {
  document.querySelectorAll("#graph-view button").forEach((b) => b.classList.toggle("on", b.dataset.view === GraphUI.view));
  $("chunks-toggle").classList.toggle("disabled", GraphUI.view !== "full");
  $("graph-filter").style.display = GraphUI.view === "full" ? "" : "none";
  $("inspector").hidden = true;
  return GraphUI.view === "full" ? renderFull() : renderSearch();
};
