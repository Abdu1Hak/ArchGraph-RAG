// Graph engine: SVG viewport with pan / zoom / node drag / zoom-to-fit, plus a force layout.
// The layout only positions the nodes and edges the API returned; nothing is invented.
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

const TYPE_COLORS = {
  Paper: "#94a3b8", ModelFamily: "#60a5fa", ArchComponent: "#a78bfa", TrainingMethod: "#f472b6",
  InferenceTechnique: "#fbbf24", Bottleneck: "#f87171", BenchmarkOrDataset: "#34d399", HyperparameterOrMetric: "#fb923c", Chunk: "#475569",
};

const View = {
  svg: null, root: null, k: 1, tx: 0, ty: 0, nodes: [], edges: [], byId: new Map(), onTick: null,

  init(svg) {
    this.svg = svg;
    svg.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const p = this.clientToSvg(ev.clientX, ev.clientY), f = Math.exp(-ev.deltaY * 0.0015);
      const k2 = Math.max(0.08, Math.min(6, this.k * f));
      this.tx = p.x - ((p.x - this.tx) * k2) / this.k; this.ty = p.y - ((p.y - this.ty) * k2) / this.k; this.k = k2;
      this.apply();
    }, { passive: false });
    let pan = null, drag = null;
    svg.addEventListener("pointerdown", (ev) => {
      const g = ev.target.closest && ev.target.closest(".node");
      if (g && g.__node) { drag = g.__node; drag.fixed = true; ev.preventDefault(); return; }
      pan = { x: ev.clientX, y: ev.clientY, tx: this.tx, ty: this.ty }; svg.classList.add("panning");
    });
    window.addEventListener("pointermove", (ev) => {
      if (drag) { const p = this.toWorld(ev.clientX, ev.clientY); drag.x = p.x; drag.y = p.y; this.onTick && this.onTick(); }
      else if (pan) { const s = this.screenScale(); this.tx = pan.tx + (ev.clientX - pan.x) * s; this.ty = pan.ty + (ev.clientY - pan.y) * s; this.apply(); }
    });
    window.addEventListener("pointerup", () => { drag = null; pan = null; svg.classList.remove("panning"); });
  },
  screenScale() { const r = this.svg.getBoundingClientRect(); const vb = this.svg.viewBox.baseVal; return vb && vb.width ? vb.width / r.width : 1; },
  clientToSvg(cx, cy) { const pt = this.svg.createSVGPoint(); pt.x = cx; pt.y = cy; return pt.matrixTransform(this.svg.getScreenCTM().inverse()); },
  toWorld(cx, cy) { const p = this.clientToSvg(cx, cy); return { x: (p.x - this.tx) / this.k, y: (p.y - this.ty) / this.k }; },
  apply() { if (this.root) this.root.setAttribute("transform", `translate(${this.tx},${this.ty}) scale(${this.k})`); },

  reset() {
    const r = this.svg.getBoundingClientRect();
    this.W = Math.max(400, r.width); this.H = Math.max(400, r.height);
    this.svg.setAttribute("viewBox", `0 0 ${this.W} ${this.H}`);
    this.svg.innerHTML = "";
    const defs = svgEl("defs");
    for (const [id, color] of [["glow-green", "#22e05a"], ["glow-cyan", "#22d3ee"]]) {
      const f = svgEl("filter", { id, x: "-60%", y: "-60%", width: "220%", height: "220%" });
      f.append(svgEl("feGaussianBlur", { stdDeviation: 3.2, result: "b" }));
      const m = svgEl("feMerge"); m.append(svgEl("feMergeNode", { in: "b" }), svgEl("feMergeNode", { in: "SourceGraphic" }));
      f.append(m); defs.append(f);
    }
    for (const [id, color] of [["ar-sel", "#22d3ee"], ["ar-rej", "#ff8a7a"], ["ar-bg", "#4b5568"]]) {
      const mk = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
      mk.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: color })); defs.append(mk);
    }
    this.svg.append(defs);
    this.root = svgEl("g"); this.svg.append(this.root);
    this.k = 1; this.tx = 0; this.ty = 0; this.apply();
    return this.root;
  },

  fit(onlyIds = null) {
    const ns = this.nodes.filter((n) => !onlyIds || onlyIds.has(n.id));
    if (!ns.length) return;
    const xs = ns.map((n) => n.x), ys = ns.map((n) => n.y);
    const x0 = Math.min(...xs) - 60, x1 = Math.max(...xs) + 160, y0 = Math.min(...ys) - 40, y1 = Math.max(...ys) + 40;
    this.k = Math.min(2.2, this.W / (x1 - x0), this.H / (y1 - y0));
    this.tx = (this.W - (x1 - x0) * this.k) / 2 - x0 * this.k; this.ty = (this.H - (y1 - y0) * this.k) / 2 - y0 * this.k;
    this.apply();
  },

  // Force layout. opts: {iters, repel, spring(edge), length(edge), anchorX(node) | null}
  simulate(nodes, edges, opts) {
    const idx = new Map(nodes.map((n) => [n.id, n]));
    const E = edges.map((e) => [idx.get(e.source), idx.get(e.target), e]).filter(([a, b]) => a && b && a !== b);
    const iters = opts.iters || 300, cx = this.W / 2, cy = this.H / 2;
    for (let it = 0; it < iters; it++) {
      const cool = 1 - it / iters;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]; const dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01;
          if (d2 > 250000) continue;
          const f = opts.repel / d2, d = Math.sqrt(d2);
          a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
        }
      }
      for (const [a, b, e] of E) {
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01, f = (d - opts.length(e)) * opts.spring(e);
        a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
      }
      for (const n of nodes) {
        if (n.fixed) { n.vx = n.vy = 0; continue; }
        const ax = opts.anchorX ? opts.anchorX(n) : null;
        if (ax !== null && ax !== undefined) n.vx += (ax - n.x) * 0.05;
        n.vx += (cx - n.x) * (opts.gravity || 0.003); n.vy += (cy - n.y) * (opts.gravity || 0.003) * 1.4;
        n.x += Math.max(-25, Math.min(25, n.vx)) * cool; n.y += Math.max(-25, Math.min(25, n.vy)) * cool;
        n.vx *= 0.55; n.vy *= 0.55;
      }
    }
  },
};
