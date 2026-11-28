// Right screen: Dual-mode Retrieval Flow ("This Query's Flow" vs "Full Architecture") built from
// pipeline_trace, graph_trace, funnel_trace and citation_gate, with port-routed SVG arrows.
"use strict";

const Flow = {
  view: "trace",
  arrows: [],
  ro: null,
  wired: false,
};

const _PREFIX_ENTITY_TYPES = {
  paper: "Paper",
  model_family: "ModelFamily",
  arch_component: "ArchComponent",
  training_method: "TrainingMethod",
  inference_technique: "InferenceTechnique",
  bottleneck: "Bottleneck",
  hyperparam_metric: "HyperparameterOrMetric",
  benchmark_dataset: "BenchmarkOrDataset",
};

const _PATTERN_META = {
  1: { code: "HYBRID_DIRECT", label: "Hybrid Direct (T1–T4 + Bridge + PPR)" },
  2: { code: "HYBRID_TWO_TURN", label: "Hybrid + Turn-2 Evidence Gap" },
  3: { code: "VECTOR_SEEDED_T6", label: "Vector-Seeded T6 Entry" },
  4: { code: "AGGREGATION_SHORT_CIRCUIT", label: "T5 Aggregation Short-Circuit" },
  5: { code: "VECTOR_ONLY_FAST", label: "Vector-Only Fast Path" },
  6: { code: "REFUSAL_SHORT_CIRCUIT", label: "Refusal Short-Circuit" },
};

const _fallbackGateClass = (s) => ({
  PASS: "ok", REPAIRED_PASS: "ok", CROPPED: "warn", EXTRACTIVE_FALLBACK: "warn", ABSTAINED: "warn", REFUSED: "bad",
}[s] || "warn");

const _fallbackGateLabel = (s) => ({
  PASS: "✓ 100% Verified",
  REPAIRED_PASS: "✓ Verified after repair",
  CROPPED: "Cropped",
  EXTRACTIVE_FALLBACK: "Extractive fallback",
  ABSTAINED: "Abstained",
  REFUSED: "Refused",
  UNVALIDATED: "Unvalidated",
}[s] || s || "-");

const _svg = (tag, attrs = {}) => {
  if (typeof svgEl === "function") return svgEl(tag, attrs);
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

const _ms = (v) => {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  const n = Number(v);
  return n >= 100 ? `${Math.round(n)} ms` : `${n.toFixed(1)} ms`;
};

function inferEntityType(entityId, explicit) {
  if (explicit) return explicit;
  const prefix = String(entityId || "").split("::", 1)[0];
  return _PREFIX_ENTITY_TYPES[prefix] || "Entity";
}

function tierBadge(t) {
  const s = String(t || "");
  if (!s) return el("span", { class: "tier", title: "Tier 1 alias scan" }, "T1");
  if (s.includes("TIER_3")) return el("span", { class: "tier t3", title: s }, "T3");
  if (s.includes("TIER_2")) return el("span", { class: "tier t2", title: s }, "T2");
  if (s.includes("PAPER")) return el("span", { class: "tier tp", title: s }, "T1·paper");
  return el("span", { class: "tier", title: s }, "T1");
}

function buildNameCollisions(entities) {
  const counts = new Map();
  const seenIds = new Set();
  for (const e of entities || []) {
    if (!e || !e.entity_id || seenIds.has(e.entity_id)) continue;
    seenIds.add(e.entity_id);
    const key = String(e.canonical_name || short(e.entity_id)).trim().toLowerCase();
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return counts;
}

function entityPill(ent, nameCounts, { slot = null, showTier = false } = {}) {
  const eid = ent.entity_id || "";
  const cname = ent.canonical_name || short(eid);
  const etype = inferEntityType(eid, ent.entity_type);
  const key = String(cname).trim().toLowerCase();
  const isDuplicate = (nameCounts.get(key) || 0) > 1;
  const tipParts = [
    slot ? `slot: ${slot}` : null,
    `${cname} [${etype}]`,
    `id: ${eid}`,
    ent.tier ? `tier: ${ent.tier}` : null,
    ent.phrase ? `matched: "${ent.phrase}"` : null,
    ent.score != null ? `score: ${Number(ent.score).toFixed(3)}` : null,
  ].filter(Boolean);

  return el("span", { class: "ent" + (isDuplicate ? " dup" : ""), title: tipParts.join(" · ") },
    slot ? el("span", { class: "slot-lbl" }, slot) : null,
    showTier ? tierBadge(ent.tier) : null,
    el("span", { class: "ent-name" }, cname),
    isDuplicate ? el("span", { class: "etype-badge", title: `Disambiguated type: ${etype}` }, `[${etype}]`) : null
  );
}

function renderEntityList(entities, nameCounts, opts = {}) {
  if (!entities || !entities.length) return el("div", { class: "empty" }, opts.emptyText || "none");
  return el("div", { class: "ent-list" }, entities.map((e) => entityPill(e, nameCounts, opts)));
}

function fchip(c, { badge = null, move = null, hideScore = false } = {}) {
  return el("div", {
    class: "chip",
    "data-chunk": c.chunk_id,
    title: `${c.chunk_id}${c.rank ? ` · rank ${c.rank}` : ""}${Number.isFinite(Number(c.score)) ? ` · score ${Number(c.score).toFixed(3)}` : ""}`,
    onclick: (ev) => { ev.stopPropagation(); App.highlight(c.chunk_id, { source: "flow" }); },
  },
  badge,
  el("span", { class: "cid" }, c.chunk_id),
  move,
  hideScore ? null : el("span", { class: "sc" }, Number.isFinite(Number(c.score)) ? Number(c.score).toFixed(3) : "cited"));
}

const chips = (list, opt, emptyText = "empty") =>
  (list && list.length ? list.map((c) => fchip(c, opt ? opt(c) : {})) : el("div", { class: "empty" }, emptyText));

function fnode(id, num, title, stat, body, more, { off = false, active = false, danger = false, warn = false, lane = "", center = false, extraClass = "" } = {}) {
  const cls = [
    "fnode",
    lane,
    extraClass,
    off ? "off" : "",
    active ? "is-active" : "",
    danger ? "is-danger" : "",
    warn ? "is-warn" : "",
    center ? "center" : "",
  ].filter(Boolean).join(" ");
  return el("div", {
    id,
    class: cls,
    onclick: (ev) => {
      if (ev.target.closest(".chip, button, a")) return;
      ev.currentTarget.classList.toggle("open");
      requestAnimationFrame(drawArrows);
    },
  },
  el("div", { class: "fh" },
    el("span", { class: "num" }, num),
    el("span", { class: "ftitle" }, title),
    el("span", { class: "fs" }, stat)),
  el("div", { class: "fb" }, body),
  more ? el("div", { class: "more" }, more) : null);
}

const kv = (...pairs) => el("div", { class: "kv" }, pairs.filter(Boolean).map(([k, v]) => [el("span", { class: "k" }, k), el("span", { class: "v" }, v)]));

function normalizeTrace(j) {
  const f = j.funnel_trace || {};
  const gt = j.graph_trace || {};
  const u = j.usage_telemetry || {};
  const gate = j.citation_gate || {};
  const pt = j.pipeline_trace || {};
  const fallbacks = j.fallback_reasons || [];
  const nodeMeta = new Map((gt.nodes || []).map((n) => [n.entity_id, n]));
  const linkMeta = new Map((j.entity_links || []).map((l) => [l.entity_id, l]));

  const enrich = (item) => {
    if (!item) return null;
    if (typeof item === "string") {
      const nm = nodeMeta.get(item), lm = linkMeta.get(item);
      return {
        entity_id: item,
        canonical_name: (lm && lm.canonical_name) || (nm && nm.canonical_name) || short(item),
        entity_type: inferEntityType(item, (lm && lm.entity_type) || (nm && nm.entity_type)),
        tier: lm && lm.tier,
        phrase: lm && lm.phrase,
        score: lm && lm.score,
      };
    }
    const eid = item.entity_id || "";
    const nm = nodeMeta.get(eid), lm = linkMeta.get(eid);
    return {
      ...item,
      entity_id: eid,
      canonical_name: item.canonical_name || (lm && lm.canonical_name) || (nm && nm.canonical_name) || short(eid),
      entity_type: inferEntityType(eid, item.entity_type || (lm && lm.entity_type) || (nm && nm.entity_type)),
    };
  };

  const erRaw = pt.entity_resolution || {};
  const rawResolved = erRaw.resolved_entities || j.resolved_entities ||
    (j.resolved_entity_ids || []).map((id) => enrich(id));
  const resolved = rawResolved.map(enrich).filter(Boolean);
  const er = {
    latency_ms: erRaw.latency_ms,
    tier1_count: erRaw.tier1_count ?? resolved.filter((e) => !e.tier || String(e.tier).includes("TIER_1")).length,
    tier2_count: erRaw.tier2_count ?? resolved.filter((e) => String(e.tier || "").includes("TIER_2")).length,
    tier3_count: erRaw.tier3_count ?? resolved.filter((e) => String(e.tier || "").includes("TIER_3")).length,
    paper_expansions: erRaw.paper_expansions ?? resolved.filter((e) => String(e.tier || "").includes("INTRODUCES")).length,
    resolved_entities: resolved,
  };

  const rtRaw = pt.routing || {};
  const oos = Boolean(j.refused || j.route === "OUT_OF_SCOPE" || rtRaw.route === "OUT_OF_SCOPE");
  const tplId = rtRaw.cypher_template_id ?? j.cypher_template_id ?? null;
  const rt = {
    latency_ms: rtRaw.latency_ms ?? j.routing_latency_ms,
    router_model: rtRaw.router_model || u.router_model || "",
    route: rtRaw.route || j.route || "VECTOR_ONLY",
    original_route: rtRaw.original_route || j.route || "VECTOR_ONLY",
    query_intent: rtRaw.query_intent || j.query_intent || "",
    confidence: rtRaw.confidence ?? j.confidence ?? 0,
    template_assigned_by: rtRaw.template_assigned_by || (tplId ? "heuristic_fallback" : "none"),
    cypher_template_id: tplId,
    refused_at: rtRaw.refused_at ?? (oos ? ((j.unverified_subjects || []).length ? "subject_gate" : "router_llm") : null),
    unverified_subjects: rtRaw.unverified_subjects || j.unverified_subjects || [],
    structural_upgrade: Boolean(rtRaw.structural_upgrade),
    structural_signals: rtRaw.structural_signals || [],
    confidence_upgrade: Boolean(rtRaw.confidence_upgrade),
  };

  const vecTop = f.reranked_vector_top_k || [];
  const graphTop = f.reranked_graph_supporting_top_k || [];
  const finalCtx = f.final_deduplicated_llm_context || [];

  const vpRaw = pt.vector_pass1 || {};
  const vp = {
    latency_ms: vpRaw.latency_ms,
    ran: vpRaw.ran ?? !oos,
    query: vpRaw.query || f.query || j.vector_subquery || "",
    top_k_count: vpRaw.top_k_count ?? vecTop.length,
    endpoint_b_entities: (vpRaw.endpoint_b_entities || []).map(enrich).filter(Boolean),
  };

  const vsRaw = pt.vector_seeded_entry || {};
  const vsTriggered = vsRaw.triggered ?? fallbacks.some((r) => /vector-seeded/i.test(r));
  const vs = {
    triggered: Boolean(vsTriggered),
    reason: vsRaw.reason || (vsTriggered ? "0 non-paper query entities -> borrowed seeds from 4d vector hits" : ""),
    borrowed_entities: (vsRaw.borrowed_entities || (vsTriggered ? (j.template_params || {}).seed_entity_ids || [] : [])).map(enrich).filter(Boolean),
  };

  const ctRaw = pt.cypher_template || {};
  const fallbackBound = [];
  if (!ctRaw.bound_entities && j.template_params) {
    for (const [slot, val] of Object.entries(j.template_params)) {
      if (Array.isArray(val)) {
        val.forEach((v, i) => {
          if (typeof v === "string" && v.includes("::")) fallbackBound.push({ slot: `${slot}[${i}]`, ...enrich(v) });
        });
      } else if (typeof val === "string" && val.includes("::")) {
        fallbackBound.push({ slot, ...enrich(val) });
      }
    }
  }
  const ct = {
    ran: ctRaw.ran ?? (!oos && !!tplId),
    latency_ms: ctRaw.latency_ms,
    template_id: ctRaw.template_id || tplId,
    bound_entities: (ctRaw.bound_entities || fallbackBound).map((b) => ({ ...enrich(b), slot: b.slot || "entity" })),
    raw_paths_count: ctRaw.raw_paths_count ?? (gt.selected_path_edges || []).length,
    cascaded_t6: Boolean(ctRaw.cascaded_t6 ?? fallbacks.some((r) => /cascaded?\s*T6/i.test(r))),
    cascade_paths_added: ctRaw.cascade_paths_added ?? 0,
    reason: ctRaw.reason || (!tplId ? (oos ? "Refused before retrieval" : "VECTOR_ONLY route (no Cypher template)") : ""),
  };

  const ebRaw = pt.endpoint_bridging || {};
  const isT5 = ct.template_id === "T5_CORPUS_AGGREGATION";
  const ebDefaultStatus = (!ct.ran || isT5) ? "short_circuited" : (fallbacks.some((r) => /bridged/i.test(r)) ? "ran" : "skipped");
  const eb = {
    status: ebRaw.status || ebDefaultStatus,
    reason: ebRaw.reason || (isT5 ? "Bypassed for T5 global aggregation" : !ct.ran ? "Graph lane inactive" : ""),
    latency_ms: ebRaw.latency_ms,
    endpoint_a: (ebRaw.endpoint_a || []).map(enrich).filter(Boolean),
    endpoint_b: (ebRaw.endpoint_b || []).map(enrich).filter(Boolean),
    paths_added: ebRaw.paths_added ?? 0,
  };

  const pprRaw = pt.hipporag_ppr || {};
  const pprDefaultStatus = (!ct.ran || isT5) ? "short_circuited" : (fallbacks.some((r) => /PPR/i.test(r)) ? "ran" : "skipped");
  const ppr = {
    status: pprRaw.status || pprDefaultStatus,
    reason: pprRaw.reason || (isT5 ? "Bypassed for T5 global aggregation" : !ct.ran ? "Graph lane inactive" : ""),
    latency_ms: pprRaw.latency_ms,
    alpha: pprRaw.alpha ?? 0.85,
    seed_a_count: pprRaw.seed_a_count ?? 0,
    seed_b_count: pprRaw.seed_b_count ?? 0,
    top_heatmap_nodes: (pprRaw.top_heatmap_nodes || []).map(enrich).filter(Boolean),
    paths_added: pprRaw.paths_added ?? 0,
  };

  const flRaw = pt.floor_short_circuit || {};
  const bothBypassed = eb.status !== "ran" && ppr.status !== "ran";
  const fl = {
    triggered: flRaw.triggered ?? (oos || !ct.ran || isT5 || bothBypassed),
    reason: flRaw.reason || (oos ? "OUT_OF_SCOPE refusal" : !ct.ran ? "VECTOR_ONLY route" : isT5 ? "T5_CORPUS_AGGREGATION global scan" : "0 bridging/PPR endpoints"),
  };

  const psRaw = pt.path_selection || {};
  const ps = {
    latency_ms: psRaw.latency_ms,
    candidate_paths: psRaw.candidate_paths ?? ct.raw_paths_count ?? 0,
    diverse_selection_enabled: psRaw.diverse_selection_enabled ?? true,
    selected_paths: psRaw.selected_paths ?? Math.min(3, (gt.selected_path_edges || []).length),
    unique_entities_covered: psRaw.unique_entities_covered ?? new Set((gt.selected_path_edges || []).flatMap((e) => [e.source, e.target])).size,
    top_paths: psRaw.top_paths || [],
  };

  const gcfRaw = pt.graph_chunk_fetch || {};
  const gcf = {
    ran: gcfRaw.ran ?? ct.ran,
    latency_ms: gcfRaw.latency_ms,
    unique_candidate_chunk_ids: gcfRaw.unique_candidate_chunk_ids ?? 0,
    fetched_count: gcfRaw.fetched_count ?? graphTop.length,
    kept_count: gcfRaw.kept_count ?? graphTop.length,
  };

  const egRaw = pt.evidence_gap || {};
  const egTriggered = Boolean(egRaw.triggered ?? fallbacks.some((r) => /evidence-gap/i.test(r)));
  const eg = {
    checked: egRaw.checked ?? !oos,
    latency_ms: egRaw.latency_ms,
    triggered: egTriggered,
    reason: egRaw.reason || "",
    missing_clauses: egRaw.missing_clauses || [],
    missing_entities: egRaw.missing_entities || [],
    missing_symbols: egRaw.missing_symbols || [],
    gap_query: egRaw.gap_query || "",
    pass2_vector_ms: egRaw.pass2_vector_ms,
    pass2_vector_new_chunks: egRaw.pass2_vector_new_chunks ?? 0,
    pass2_ppr_ms: egRaw.pass2_ppr_ms,
    pass2_ppr_paths_added: egRaw.pass2_ppr_paths_added ?? 0,
    pass2_graph_chunks_added: egRaw.pass2_graph_chunks_added ?? 0,
  };

  const cmRaw = pt.context_merge || {};
  const cm = {
    vector_in: cmRaw.vector_in ?? vecTop.length,
    graph_in: cmRaw.graph_in ?? graphTop.length,
    duplicates_merged: cmRaw.duplicates_merged ?? Math.max(0, vecTop.length + graphTop.length - finalCtx.length),
    final_count: cmRaw.final_count ?? finalCtx.length,
  };

  const cgRaw = pt.citation_gate || {};
  const gateStatus = cgRaw.status || gate.status || (oos ? "REFUSED" : j.abstained ? "ABSTAINED" : "PASS");
  const defaultRungs = oos
    ? [
        { id: "generate", label: "Generate", status: "skip", detail: "0 LLM calls" },
        { id: "validate", label: "Validate", status: "skip", detail: "bypassed" },
        { id: "repair", label: "Repair", status: "skip", detail: "bypassed" },
        { id: "crop", label: "Crop", status: "skip", detail: "bypassed" },
        { id: "fallback", label: "Fallback", status: "skip", detail: "bypassed" },
        { id: "deliver", label: "Refused", status: "fail", detail: rt.refused_at || "OUT_OF_SCOPE" },
      ]
    : [
        { id: "generate", label: "Generate", status: "pass", detail: `${gate.generator_calls ?? j.generator_calls ?? 1} call` },
        { id: "validate", label: "Validate", status: (gate.repair_attempts || j.repair_attempts) ? "warn" : "pass", detail: `${(gate.cited_ids || []).length} cited` },
        { id: "repair", label: "Repair", status: (gate.repair_attempts || j.repair_attempts) ? "warn" : "skip", detail: (gate.repair_attempts || j.repair_attempts) ? `${gate.repair_attempts || j.repair_attempts} retry` : "not needed" },
        { id: "crop", label: "Crop", status: (gate.cropped_sentences || j.cropped_sentences_count) ? "warn" : "skip", detail: (gate.cropped_sentences || j.cropped_sentences_count) ? `${gate.cropped_sentences || j.cropped_sentences_count} cropped` : "not needed" },
        { id: "fallback", label: "Fallback", status: j.used_extractive_fallback ? "warn" : "skip", detail: j.used_extractive_fallback ? "extractive" : "not needed" },
        { id: "deliver", label: j.abstained ? "Abstained" : "Deliver", status: j.abstained ? "warn" : "pass", detail: gateStatus },
      ];
  const cg = {
    status: gateStatus,
    short_circuited_refusal: cgRaw.short_circuited_refusal ?? oos,
    refusal_reason: cgRaw.refusal_reason || (oos ? (rt.unverified_subjects.length ? `Blocked unindexed subject(s): ${rt.unverified_subjects.join(", ")}` : "Out-of-scope query refused at Router/Subject Gate") : ""),
    rungs: (cgRaw.rungs && cgRaw.rungs.length) ? cgRaw.rungs : defaultRungs,
  };

  let fp = pt.flow_pattern;
  if (!fp || !fp.id) {
    const pid = oos ? 6 : !ct.ran ? 5 : isT5 ? 4 : vs.triggered ? 3 : eg.triggered ? 2 : 1;
    fp = { id: pid, total: 6, ..._PATTERN_META[pid] };
  }

  const allEntities = [
    ...er.resolved_entities,
    ...vp.endpoint_b_entities,
    ...vs.borrowed_entities,
    ...ct.bound_entities,
    ...eb.endpoint_a,
    ...eb.endpoint_b,
    ...ppr.top_heatmap_nodes,
  ];
  const nameCounts = buildNameCollisions(allEntities);

  return { f, gt, u, gate, fallbacks, oos, fp, er, rt, vp, vs, ct, eb, ppr, fl, ps, gcf, eg, cm, cg, nameCounts, vecTop, graphTop, finalCtx };
}

function renderTemplateParams(j, ct, nameCounts) {
  const parts = [];
  if (ct.bound_entities && ct.bound_entities.length) {
    parts.push(
      el("div", { class: "bound-row" },
        ct.bound_entities.map((be) => entityPill(be, nameCounts, { slot: be.slot })))
    );
  }
  const extraParams = Object.entries(j.template_params || {}).filter(([k, v]) => {
    if (k === "seed_entity_ids" || k.endsWith("_id")) return false;
    return v !== null && v !== undefined && v !== "";
  });
  if (extraParams.length) {
    parts.push(
      el("div", { class: "param-pills" },
        extraParams.map(([k, v]) => el("span", { class: "param-pill" }, el("b", {}, `${k}:`), ` ${Array.isArray(v) ? v.join(", ") : String(v)}`)))
    );
  }
  if (!parts.length && ct.ran) {
    parts.push(el("div", { class: "empty" }, "Global corpus scan (no entity slot bindings)"));
  }
  return parts;
}

function renderPprHeatmap(ppr, nameCounts) {
  const nodes = ppr.top_heatmap_nodes || [];
  if (!nodes.length) {
    return el("div", { class: "empty" }, ppr.status === "ran" ? "No non-seed heatmap nodes returned" : "PPR heatmap inactive");
  }
  const maxS = Math.max(...nodes.map((n) => Number(n.score || 0)), 1e-6);
  return el("div", { class: "ppr-heat" },
    nodes.slice(0, 6).map((n) => {
      const sc = Number(n.score || 0);
      const pct = Math.max(4, Math.min(100, Math.round((sc / maxS) * 100)));
      const cname = n.canonical_name || short(n.entity_id);
      const etype = inferEntityType(n.entity_id, n.entity_type);
      const dup = (nameCounts.get(String(cname).trim().toLowerCase()) || 0) > 1;
      return el("div", { class: "heat-row", title: `${cname} [${etype}] · ${n.entity_id} · PPR=${sc.toFixed(4)}` },
        el("span", { class: "heat-name" }, cname, dup ? el("span", { class: "etype-badge" }, `[${etype}]`) : null),
        el("span", { class: "heat-track" }, el("span", { class: "heat-fill", style: `width:${pct}%` })),
        el("span", { class: "heat-val" }, sc.toFixed(3)));
    })
  );
}

function renderCitationLadder(cg) {
  const icon = { pass: "✓", warn: "!", fail: "✕", skip: "–" };
  return el("div", { class: "gate-ladder", role: "list", "aria-label": "Citation Gate Ladder" },
    (cg.rungs || []).map((r, idx) => [
      idx > 0 ? el("span", { class: "gate-step-arrow", "aria-hidden": "true" }, "→") : null,
      el("div", { class: `gate-rung ${r.status || "skip"}`, role: "listitem", title: r.detail ? `${r.label}: ${r.detail}` : r.label },
        el("div", { class: "gr-top" },
          el("span", { class: "gr-ico" }, icon[r.status] || "·"),
          el("span", { class: "gr-lbl" }, r.label)),
        r.detail ? el("div", { class: "gr-sub" }, r.detail) : null),
    ])
  );
}

function renderTurn2Card(eg, isFullMode) {
  const active = eg.triggered;
  return fnode("f-turn2", "T2", "Turn-2 Gap Re-Flow",
    active ? `2× pass · +${(eg.pass2_vector_new_chunks || 0) + (eg.pass2_graph_chunks_added || 0)} chunks` : "Pass-1 sufficient (not fired)",
    [
      el("div", { class: "turn2-pipe" },
        el("div", { class: "t2-box query" },
          el("div", { class: "t2-lbl" }, "Gap Query"),
          el("div", { class: "t2-val" }, eg.gap_query || (active ? eg.reason || "Focused gap recovery" : "Synthesized only if Pass-1 misses clause/entity/symbol"))),
        el("div", { class: "t2-arrow" }, "→"),
        el("div", { class: "t2-branches" },
          el("div", { class: "t2-box" },
            el("div", { class: "t2-lbl" }, "1× pgvector Search"),
            el("div", { class: "t2-stat" }, active ? `${_ms(eg.pass2_vector_ms)} · +${eg.pass2_vector_new_chunks} chunks` : "standby")),
          el("div", { class: "t2-box" },
            el("div", { class: "t2-lbl" }, "1× PPR Expansion"),
            el("div", { class: "t2-stat" }, active ? `${_ms(eg.pass2_ppr_ms)} · +${eg.pass2_ppr_paths_added} paths` : "standby"))),
        el("div", { class: "t2-arrow" }, "→"),
        el("div", { class: "t2-box merge" },
          el("div", { class: "t2-lbl" }, "Merge Pass-1 + Pass-2"),
          el("div", { class: "t2-stat" }, active ? `+${eg.pass2_graph_chunks_added} graph chunks` : (isFullMode ? "0.9× score scale" : ""))))
    ],
    eg.reason ? kv(["gap trigger", eg.reason], ["pass-2 graph chunks", `+${eg.pass2_graph_chunks_added}`]) : null,
    { off: !active, warn: active, extraClass: "turn2-card" });
}

function wireFlowControls() {
  const btns = document.querySelectorAll("#flow-view button[data-flow-view]");
  if (!btns.length) return;
  btns.forEach((btn) => {
    if (btn.dataset.flowWired === "1") return;
    btn.dataset.flowWired = "1";
    btn.addEventListener("click", () => {
      const next = btn.dataset.flowView === "full" ? "full" : "trace";
      if (Flow.view === next) return;
      Flow.view = next;
      syncFlowButtons();
      if (App.last) window.renderFlow(App.last);
    });
  });
}

function syncFlowButtons() {
  document.querySelectorAll("#flow-view button[data-flow-view]").forEach((b) => {
    const on = b.dataset.flowView === Flow.view;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
}

window.renderFlow = function (j) {
  wireFlowControls();
  syncFlowButtons();
  const isFull = Flow.view === "full";
  const T = normalizeTrace(j || {});
  const { f, gt, u, gate, oos, fp, er, rt, vp, vs, ct, eb, ppr, fl, ps, gcf, eg, cm, cg, nameCounts, vecTop, graphTop, finalCtx } = T;
  const gClass = typeof gateClass === "function" ? gateClass : _fallbackGateClass;
  const gLabel = typeof gateLabel === "function" ? gateLabel : _fallbackGateLabel;

  const badge = $("flow-pattern-badge");
  if (badge) {
    badge.hidden = false;
    badge.className = `pattern-badge p-${fp.id}${oos ? " refusal" : ""}`;
    badge.textContent = `Pattern ${fp.id}/${fp.total || 6} · ${fp.label}`;
  }
  const sub = $("flow-sub");
  if (sub) {
    sub.textContent = isFull
      ? "Full Architecture Map · active path illuminated, untouched branches dimmed"
      : (oos ? `Refused at ${rt.refused_at === "subject_gate" ? "Subject Gate" : "Router LLM"} · 0 DB / 0 Gen calls`
             : `${ct.ran ? "Graph + Vector" : "Vector-Only"}${eg.triggered ? " · Turn-2 Gap Re-Flow" : ""}`);
  }

  const questionText = App.lastQuestion || j.question || f.query || vp.query || j.graph_subquery || j.vector_subquery || "(query)";
  const rrfRank = f.rrf_rank_by_chunk || {};
  const moveOf = (c) => {
    const from = rrfRank[c.chunk_id];
    if (!from) return { move: el("span", { class: "mv", title: "outside the RRF top-20 pool" }, "new") };
    const d = from - c.rank;
    return {
      move: el("span", {
        class: "mv " + (d > 0 ? "up" : d < 0 ? "down" : ""),
        title: `RRF rank ${from} → reranked rank ${c.rank}`,
      }, `#${from}→#${c.rank}${d > 0 ? " ↑" : d < 0 ? " ↓" : ""}`),
    };
  };

  const citedSet = new Set(gate.cited_ids || []);
  const origin = (c) => ({
    badge: [
      el("span", { class: "origin " + (c.source_origin || "vector") }, c.source_origin === "both" ? "both +0.05" : c.source_origin || "vector"),
      citedSet.has(c.chunk_id)
        ? el("span", { class: "cited-tag ok", title: "Cited in final answer" }, "✓ cited")
        : el("span", { class: "cited-tag muted", title: "Passed to LLM context but not cited" }, "context only"),
    ],
  });

  const hops = [...(gt.selected_path_edges || [])].sort((a, b) => (a.path_index ?? 0) - (b.path_index ?? 0) || (a.hop_index ?? 0) - (b.hop_index ?? 0));
  const nameOf = new Map((gt.nodes || []).map((n) => [n.entity_id, n.canonical_name]));
  const hopRows = hops.map((h) => el("div", { class: "hop" },
    el("b", {}, `[Hop ${(h.hop_index ?? 0) + 1}: ${h.relation_type}] `),
    `${nameOf.get(h.source) || short(h.source)} → ${nameOf.get(h.target) || short(h.target)}`));

  // Block 1: User Query (clean question text only, no entity clutter)
  const nQuery = fnode("f-query", 1, "User Query", `${questionText.length} chars`,
    el("div", { class: "q-text-node" }, questionText), null, { center: true, active: true });

  // Block 2: Entity Resolver & Linker (BEFORE Agent Router)
  const dupNames = [...nameCounts.entries()].filter(([, cnt]) => cnt > 1).map(([k]) => k);
  const tierChips = [
    er.tier1_count ? el("span", { class: "mini-stat" }, el("span", { class: "tier" }, "T1"), ` ${er.tier1_count}`) : null,
    er.tier2_count ? el("span", { class: "mini-stat" }, el("span", { class: "tier t2" }, "T2"), ` ${er.tier2_count}`) : null,
    er.tier3_count ? el("span", { class: "mini-stat" }, el("span", { class: "tier t3" }, "T3"), ` ${er.tier3_count}`) : null,
    er.paper_expansions ? el("span", { class: "mini-stat" }, el("span", { class: "tier tp" }, "T1·paper"), ` +${er.paper_expansions} intro`) : null,
  ].filter(Boolean);

  const nResolver = fnode("f-resolver", 2, "Entity Resolver & Linker (Tier 1 / 2 / 3)",
    `${er.resolved_entities.length} entities · ${_ms(er.latency_ms)}`,
    [
      tierChips.length ? el("div", { class: "tier-summary" }, tierChips) : null,
      renderEntityList(er.resolved_entities, nameCounts, { showTier: true, emptyText: "0 entities resolved from question" }),
      dupNames.length ? el("div", { class: "disambig-note" }, `Disambiguated duplicate name (${dupNames.join(", ")}) by entity_type`) : null,
    ],
    er.resolved_entities.length
      ? er.resolved_entities.map((e) => el("div", { class: "hop" },
          `${e.phrase ? `"${e.phrase}" → ` : ""}${e.canonical_name} [${e.entity_type}] · ${e.entity_id} · ${e.tier || "T1"}${e.score != null ? ` (${Number(e.score).toFixed(2)})` : ""}`))
      : null,
    { center: true, active: true });

  // Block 3: Agent Router & Subject Gate
  const tplAssignMap = {
    llm: { cls: "llm", txt: "Assigned by LLM" },
    heuristic_fallback: { cls: "fb", txt: "Fallback" },
    none: { cls: "none", txt: "No Template" },
  };
  const assignMeta = tplAssignMap[rt.template_assigned_by] || tplAssignMap.none;
  const gateBadge = oos
    ? el("span", { class: "sg-pill bad" },
        rt.refused_at === "subject_gate" ? "Refused at Subject Gate" : "Refused at Router LLM")
    : el("span", { class: "sg-pill ok" }, "Subject Gate: Passed");

  const nRouter = fnode("f-router", 3, "Agent Router & Subject Gate",
    `${rt.router_model || "router"} · ${_ms(rt.latency_ms)}`,
    [
      el("div", { class: "router-top" },
        el("span", { class: "route " + rt.route }, rt.route),
        rt.original_route && rt.original_route !== rt.route
          ? el("span", { class: "upgrade-pill", title: "Route upgraded deterministically" }, `${rt.original_route} → ${rt.route}`)
          : null,
        el("span", { class: "intent-tag" }, rt.query_intent),
        el("span", { class: `assign-pill ${assignMeta.cls}` },
          assignMeta.txt, rt.cypher_template_id ? ` · ${rt.cypher_template_id}` : "")),
      el("div", { class: "conf-row" },
        el("div", { class: "conf", title: `confidence ${Number(rt.confidence).toFixed(2)}` },
          el("span", { style: `width:${Math.round(Number(rt.confidence || 0) * 100)}%` })),
        el("span", { class: "conf-num" }, Number(rt.confidence || 0).toFixed(2))),
      el("div", { class: "sg-row" },
        gateBadge,
        (rt.unverified_subjects || []).map((s) => el("span", { class: "blocked-subj" }, `✕ ${s}`)),
        rt.structural_upgrade ? el("span", { class: "upgrade-pill" }, `Structural: ${(rt.structural_signals || []).join(", ") || "upgrade"}`) : null,
        rt.confidence_upgrade ? el("span", { class: "upgrade-pill" }, "Confidence < 0.78") : null),
      (!oos && (j.graph_subquery || j.vector_subquery))
        ? kv(
            j.graph_subquery ? ["graph_subquery", j.graph_subquery] : null,
            j.vector_subquery ? ["vector_subquery", j.vector_subquery] : null)
        : null,
    ],
    kv(
      ["original route", rt.original_route],
      ["template origin", `${assignMeta.txt}${rt.cypher_template_id ? ` (${rt.cypher_template_id})` : ""}`],
      ["routing latency", _ms(rt.latency_ms)],
      ["fallbacks", T.fallbacks.join(" ; ") || "none"]),
    { center: true, active: !oos, danger: oos });

  // Vector + Lexical Lane (Right Column - runs first!)
  const vecActive = vp.ran && !oos;
  const nDense = fnode("f-dense", "4a", "Dense (Cosine)", `top ${(f.dense_top_k || []).length}`,
    chips(f.dense_top_k), null, { off: !vecActive, active: vecActive });
  const nBm25 = fnode("f-bm25", "4b", "BM25 Lexical", `top ${(f.bm25_top_k || []).length}`,
    chips(f.bm25_top_k), null, { off: !vecActive, active: vecActive });
  const nRrf = fnode("f-rrf", "4c", "Hybrid RRF Fusion", `top ${(f.rrf_top_k || []).length} fused`,
    chips(f.rrf_top_k), vp.query ? el("div", { class: "muted" }, `query: ${vp.query}`) : null,
    { off: !vecActive, active: vecActive });
  const nRerank = fnode("f-rerank", "4d", "Cross-Encoder Reranked Top-K",
    `kept ${vecTop.length} · ${_ms(vp.latency_ms)}`,
    [
      vp.endpoint_b_entities.length
        ? el("div", { class: "epb-export" },
            el("span", { class: "epb-lbl" }, "← Exports Endpoint B to Graph:"),
            renderEntityList(vp.endpoint_b_entities, nameCounts))
        : null,
      chips(vecTop, moveOf),
    ],
    kv(
      u.reranker_model ? ["reranker", u.reranker_model] : null,
      u.reranker_input_tokens || u.reranker_output_tokens ? ["tokens", `${fmt((u.reranker_input_tokens || 0) + (u.reranker_output_tokens || 0))}`] : null,
      vp.query ? ["vector query", vp.query] : null),
    { off: !vecActive, active: vecActive });

  // Graph Lane (Left Column)
  const graphActive = ct.ran && !oos;
  const nVecSeed = fnode("f-vecseed", "VS", "Vector-Seeded T6 Entry",
    vs.triggered ? `${vs.borrowed_entities.length} borrowed` : "not triggered",
    [
      el("div", { class: "stage-note" }, vs.reason || "Triggered when 0 query concept entities resolve: borrows top vector chunk entities as T6 seeds"),
      vs.triggered ? renderEntityList(vs.borrowed_entities, nameCounts) : el("div", { class: "empty" }, "Query had resolved entities — bypassed"),
    ],
    null, { off: !vs.triggered, active: vs.triggered, lane: "graph-lane", extraClass: isFull ? "vecseed-offset" : "" });

  const nCypher = fnode("f-cypher", "3a", "Cypher Template Execution",
    ct.ran ? `${ct.template_id || "T?"} · ${ct.raw_paths_count} paths · ${_ms(ct.latency_ms)}` : "bypassed",
    [
      el("div", { class: "tpl-head" },
        el("span", { class: "tpl-pill" }, ct.template_id || "none"),
        el("span", { class: `assign-pill ${assignMeta.cls}` }, assignMeta.txt),
        ct.cascaded_t6 ? el("span", { class: "cascade-pill" }, `T6 Cascade +${ct.cascade_paths_added}`) : null),
      ct.ran ? renderTemplateParams(j, ct, nameCounts) : el("div", { class: "empty" }, ct.reason || "Not executed"),
    ],
    ct.reason ? el("div", { class: "muted" }, ct.reason) : null,
    { off: !graphActive, active: graphActive, lane: "graph-lane" });

  const ebRan = eb.status === "ran" && !oos;
  const nBridge = fnode("f-bridge", "3b", "Goal-Directed Endpoint Bridging",
    ebRan ? `+${eb.paths_added} bridges · ${_ms(eb.latency_ms)}` : "Short-circuited",
    ebRan || isFull
      ? [
          el("div", { class: "bridge-grid" },
            el("div", { class: "bridge-col" },
              el("div", { class: "bridge-lbl" }, "Endpoint A (Query)"),
              renderEntityList(eb.endpoint_a, nameCounts, { emptyText: "none" })),
            el("div", { class: "bridge-mid" }, "↔ T4 ↔"),
            el("div", { class: "bridge-col" },
              el("div", { class: "bridge-lbl" }, "Endpoint B (from 4d)"),
              renderEntityList(eb.endpoint_b, nameCounts, { emptyText: "none" }))),
          !ebRan ? el("div", { class: "sc-reason" }, `Short-circuited: ${eb.reason || "skipped"}`) : null,
        ]
      : el("div", { class: "sc-reason" }, `Short-circuited: ${eb.reason || "skipped"}`),
    null, { off: !ebRan, active: ebRan, lane: "graph-lane", extraClass: !ebRan ? "short-card" : "" });

  const pprRan = ppr.status === "ran" && !oos;
  const nPpr = fnode("f-ppr", "3c", "HippoRAG PPR Heatmap",
    pprRan ? `α=${ppr.alpha} · +${ppr.paths_added} paths · ${_ms(ppr.latency_ms)}` : "Short-circuited",
    pprRan || isFull
      ? [
          el("div", { class: "ppr-meta" },
            el("span", { class: "param-pill" }, `α = ${ppr.alpha}`),
            el("span", { class: "param-pill" }, `Seed A: ${ppr.seed_a_count}`),
            el("span", { class: "param-pill" }, `Seed B (4d): ${ppr.seed_b_count}`),
            el("span", { class: "param-pill" }, `top-5 1–2 hop paths: +${ppr.paths_added}`)),
          pprRan ? renderPprHeatmap(ppr, nameCounts) : el("div", { class: "sc-reason" }, `Short-circuited: ${ppr.reason || "skipped"}`),
        ]
      : el("div", { class: "sc-reason" }, `Short-circuited: ${ppr.reason || "skipped"}`),
    null, { off: !pprRan, active: pprRan, lane: "graph-lane", extraClass: !pprRan ? "short-card" : "" });

  const nFloor = el("div", {
    id: "f-floor",
    class: `floor-banner${fl.triggered ? " active" : " off"}`,
  },
  el("span", { class: "floor-badge" }, "Floor-Case Short-Circuit"),
  el("span", { class: "floor-txt" },
    fl.triggered
      ? `3b Bridging & 3c PPR bypassed — ${fl.reason}`
      : "Standby: fires when Bridging & PPR are both bypassed (T5 / Vector-Only / OOS / 0 endpoints)"));

  const nGChunks = fnode("f-gchunks", "3d", "Diverse Path Selection & Vector DB Chunk Fetch",
    graphActive ? `${ps.selected_paths}/${ps.candidate_paths} paths · kept ${gcf.kept_count} · ${_ms((ps.latency_ms || 0) + (gcf.latency_ms || 0))}` : "bypassed",
    graphActive
      ? [
          el("div", { class: "sel-summary" },
            el("span", { class: "param-pill" }, `${ps.candidate_paths} pool → top ${ps.selected_paths} paths`),
            el("span", { class: "param-pill" }, ps.diverse_selection_enabled ? "Greedy Diversity ON" : "Base Score Only"),
            el("span", { class: "param-pill" }, `${ps.unique_entities_covered} entities covered`),
            el("span", { class: "param-pill" }, `path rank ${_ms(ps.latency_ms)}`)),
          (ps.top_paths && ps.top_paths.length)
            ? el("div", { class: "top-paths-row" },
                ps.top_paths.slice(0, 3).map((tp) =>
                  el("span", { class: "param-pill" },
                    `#${tp.rank} · ${tp.hops}h · sc ${Number(tp.score || 0).toFixed(2)}${tp.diversity_gain ? ` (+${Number(tp.diversity_gain).toFixed(2)} div)` : ""}`)))
            : null,
          hopRows.length ? el("div", { class: "hop-list" }, hopRows.slice(0, 5)) : null,
          el("div", { class: "pgfetch-bar" },
            el("span", { class: "pgfetch-lbl" }, "pgvector fetch_chunks_by_ids + per-hop rerank"),
            el("span", { class: "pgfetch-stat" },
              `${gcf.unique_candidate_chunk_ids} IDs → ${gcf.fetched_count} fetched → ${gcf.kept_count} kept · ${_ms(gcf.latency_ms)}`)),
          chips(graphTop),
        ]
      : el("div", { class: "empty" }, "Not executed"),
    hopRows.length > 5 ? hopRows.slice(5) : null,
    { off: !graphActive, active: graphActive, lane: "graph-lane" });

  // Hidden Layer 4.5: Evidence-Gap Coverage Gate (+ optional Turn-2 Re-Flow)
  const missingPills = [
    ...eg.missing_clauses.map((c) => el("span", { class: "gap-miss" }, `clause: ${c}`)),
    ...eg.missing_entities.map((e) => el("span", { class: "gap-miss" }, `entity: ${e}`)),
    ...eg.missing_symbols.map((s) => el("span", { class: "gap-miss" }, `symbol: ${s}`)),
  ];
  const nGap = fnode("f-gap", "4.5", "Evidence-Gap Coverage Gate",
    oos ? "bypassed" : `${eg.triggered ? "Turn-2 Triggered" : "Pass-1 Sufficient"} · ${_ms(eg.latency_ms)}`,
    oos
      ? el("div", { class: "empty" }, "Skipped due to refusal")
      : [
          eg.triggered
            ? el("div", { class: "gap-status warn" }, `↻ Pass-1 Incomplete — Triggered Turn-2 Re-Flow (${eg.reason || "gap detected"})`)
            : el("div", { class: "gap-status ok" }, "✓ Pass-1 Sufficient — All clauses, entities & symbols covered → Block 5"),
          missingPills.length ? el("div", { class: "gap-misses" }, missingPills) : null,
        ],
    eg.gap_query ? kv(["gap_query", eg.gap_query], ["reason", eg.reason || "-"]) : null,
    { off: oos, active: !oos, warn: eg.triggered, center: !(eg.triggered || isFull) });

  const showTurn2 = isFull || eg.triggered;
  const nTurn2 = showTurn2 ? renderTurn2Card(eg, isFull) : null;

  // Block 5: Context Merge
  const nContext = fnode("f-context", 5, "Score-Deduplicated LLM Context",
    oos ? "0 chunks" : `${cm.final_count} unique${cm.duplicates_merged ? ` (${cm.duplicates_merged} merged)` : ""}`,
    oos
      ? el("div", { class: "empty" }, "Bypassed on refusal")
      : [
          el("div", { class: "merge-bar" },
            el("span", { class: "param-pill" }, `Vector in: ${cm.vector_in}`),
            el("span", { class: "param-pill" }, `Graph in: ${cm.graph_in}`),
            el("span", { class: "param-pill" }, `Merged dupes: ${cm.duplicates_merged}`),
            el("span", { class: "param-pill" }, `Final context: ${cm.final_count}`)),
          chips(finalCtx, origin),
        ],
    null, { off: oos, active: !oos, center: true });

  // Block 6: Gemini Generator + Visual Citation Gate Ladder
  const citeStat = citedSet.size > finalCtx.length
    ? `${citedSet.size} cited (${finalCtx.length} passages)`
    : `cited ${citedSet.size}/${finalCtx.length}`;
  const nGen = fnode("f-gen", 6, "Gemini Generator + Citation Gate Ladder",
    oos ? `Refused · 0 calls` : `${citeStat} · ${u.generator_model || ""}`,
    [
      cg.short_circuited_refusal
        ? el("div", { class: "refusal-banner" },
            el("strong", {}, "SHORT-CIRCUIT REFUSAL: "),
            cg.refusal_reason || "Out of scope",
            el("span", { class: "refusal-zeros" }, "Floor-Case Short-Circuit · 0 Neo4j · 0 pgvector · 0 Gen calls"))
        : null,
      el("div", { class: "gen-meta-row" },
        el("span", { class: "badge " + gClass(cg.status) }, gLabel(cg.status)),
        el("span", { class: "muted" },
          `${fmt(u.generator_input_tokens)} in / ${fmt(u.generator_output_tokens)} out tok · ${_ms(j.generation_latency_ms || 0)}`)),
      renderCitationLadder(cg),
      (gate.cited_ids || []).length
        ? el("div", { class: "gen-cites" }, (gate.cited_ids || []).map((c) => fchip({ chunk_id: c, rank: "", score: NaN }, { hideScore: true })))
        : null,
    ],
    kv(
      ["generator calls", gate.generator_calls ?? j.generator_calls ?? 0],
      ["repairs", gate.repair_attempts ?? j.repair_attempts ?? 0],
      ["cropped", gate.cropped_sentences ?? j.cropped_sentences_count ?? 0],
      ["invented ids", (gate.invented_ids || []).join(", ") || "none"],
      ["uncited claims", gate.uncited_claims ?? 0]),
    { center: true, active: true, danger: oos });

  const flow = $("flow");
  if (!flow) return;
  flow.innerHTML = "";
  flow.classList.toggle("full-mode", isFull);
  flow.classList.toggle("trace-mode", !isFull);
  flow.classList.toggle("refusal-trace", !isFull && oos);

  // Assemble clean topology for "trace" vs "full"
  const children = [nQuery, nResolver, nRouter];
  const arrows = [
    { from: "f-query", to: "f-resolver", on: true, kind: "vec" },
    { from: "f-resolver", to: "f-router", on: true, kind: "vec" },
  ];

  const showGraphLane = isFull || graphActive;
  const showVecLane = isFull || vecActive;

  if (showGraphLane || showVecLane) {
    const showFloorBesideVec = !isFull && !showGraphLane && fl.triggered;
    const lanesAttrs = { class: "lanes" + (!showGraphLane && !showFloorBesideVec ? " single-lane" : "") };
    const laneEls = [];

    if (showGraphLane) {
      const gKids = [el("div", { class: "lane-title" }, "Graph Traversal & PPR (Neo4j)")];
      if (isFull || vs.triggered) gKids.push(nVecSeed);
      gKids.push(nCypher, nBridge, nPpr);
      if (isFull || fl.triggered) gKids.push(nFloor);
      gKids.push(nGChunks);
      laneEls.push(el("div", { class: "lane graph-col" }, gKids));
    } else if (showFloorBesideVec) {
      laneEls.push(
        el("div", { class: "lane floor-col" },
          el("div", { class: "lane-title" }, "Graph Lane (Short-Circuited)"),
          nFloor)
      );
    }

    if (showVecLane) {
      laneEls.push(
        el("div", { class: "lane vec-col" },
          el("div", { class: "lane-title" }, "Vector + Lexical (pgvector · runs first)"),
          el("div", { class: "pair" }, nDense, nBm25),
          nRrf,
          nRerank)
      );
    }
    children.push(el("div", lanesAttrs, laneEls));

    // Arrows into Vector Lane
    if (showVecLane) {
      arrows.push(
        { from: "f-router", to: "f-dense", on: vecActive, kind: "vec", x1Frac: (showGraphLane || showFloorBesideVec) ? 0.68 : 0.35 },
        { from: "f-router", to: "f-bm25", on: vecActive, kind: "vec", x1Frac: (showGraphLane || showFloorBesideVec) ? 0.80 : 0.65 },
        { from: "f-dense", to: "f-rrf", on: vecActive, kind: "vec", x2Frac: 0.32 },
        { from: "f-bm25", to: "f-rrf", on: vecActive, kind: "vec", x2Frac: 0.68 },
        { from: "f-rrf", to: "f-rerank", on: vecActive, kind: "vec" },
      );
    }
    if (showFloorBesideVec) {
      arrows.push({ from: "f-router", to: "f-floor", on: false, kind: "graph", x1Frac: 0.25 });
    }

    // Arrows inside Graph Lane + Cross-Lane Arrows from 4d
    if (showGraphLane) {
      const hasVecSeedEl = isFull || vs.triggered;
      if (hasVecSeedEl) {
        arrows.push(
          { from: "f-rerank", to: "f-vecseed", on: vs.triggered && !oos, kind: "cross", port: "cross-left", y1Frac: 0.24, label: vs.triggered ? "Borrowed Seeds" : (isFull ? "0-entity T6" : null) },
          { from: "f-vecseed", to: "f-cypher", on: vs.triggered && graphActive, kind: "graph", x1Frac: 0.55, x2Frac: 0.62 }
        );
      }
      if (!vs.triggered || isFull) {
        arrows.push({
          from: "f-router",
          to: "f-cypher",
          on: graphActive && !vs.triggered,
          kind: "graph",
          port: hasVecSeedEl ? "bypass-left" : "vertical",
          x1Frac: 0.22,
          x2Frac: hasVecSeedEl ? 0.14 : 0.5,
        });
      }

      if (ebRan && pprRan) {
        arrows.push(
          { from: "f-cypher", to: "f-bridge", on: graphActive, kind: "graph" },
          { from: "f-bridge", to: "f-ppr", on: graphActive, kind: "graph" },
          isFull
            ? { from: "f-ppr", to: "f-gchunks", on: graphActive, kind: "graph", port: "graph-bypass", railOffset: 10 }
            : { from: "f-ppr", to: "f-gchunks", on: graphActive, kind: "graph" },
        );
      } else if (!ebRan && pprRan) {
        arrows.push(
          { from: "f-cypher", to: "f-bridge", on: false, kind: "graph" },
          { from: "f-bridge", to: "f-ppr", on: false, kind: "graph" },
          { from: "f-cypher", to: "f-ppr", on: graphActive, kind: "graph", port: "graph-bypass", label: "3b skipped" },
          isFull
            ? { from: "f-ppr", to: "f-gchunks", on: graphActive, kind: "graph", port: "graph-bypass", railOffset: 10 }
            : { from: "f-ppr", to: "f-gchunks", on: graphActive, kind: "graph" },
        );
      } else if (ebRan && !pprRan) {
        arrows.push(
          { from: "f-cypher", to: "f-bridge", on: graphActive, kind: "graph" },
          { from: "f-bridge", to: "f-ppr", on: false, kind: "graph" },
          { from: "f-ppr", to: "f-gchunks", on: false, kind: "graph" },
          { from: "f-bridge", to: "f-gchunks", on: graphActive, kind: "graph", port: "graph-bypass", label: "3c skipped" },
        );
      } else {
        arrows.push(
          { from: "f-cypher", to: "f-bridge", on: false, kind: "graph" },
          { from: "f-bridge", to: "f-ppr", on: false, kind: "graph" },
          { from: "f-ppr", to: "f-floor", on: false, kind: "graph" },
          { from: "f-floor", to: "f-gchunks", on: false, kind: "graph" },
          { from: "f-cypher", to: "f-gchunks", on: graphActive, kind: "graph", port: "graph-bypass", label: "Floor skip" },
        );
      }

      if (isFull && (ebRan || pprRan)) {
        arrows.push(
          { from: "f-ppr", to: "f-floor", on: false, kind: "graph" },
          { from: "f-floor", to: "f-gchunks", on: false, kind: "graph" },
        );
      }

      if (showVecLane) {
        arrows.push(
          { from: "f-rerank", to: "f-bridge", on: ebRan && vecActive, kind: "cross", port: "cross-left", y1Frac: 0.50, label: (ebRan || isFull) ? "Endpoint B" : null },
          { from: "f-rerank", to: "f-ppr", on: pprRan && vecActive, kind: "cross", port: "cross-left", y1Frac: 0.76, label: (pprRan || isFull) ? "Seed B" : null },
        );
      }
    }

    // Layer 4.5 + Turn-2 + Block 5 + Block 6
    if (showTurn2) {
      children.push(el("div", { class: "gap-row" }, nGap, nTurn2));
      arrows.push(
        { from: "f-gap", to: "f-turn2", on: eg.triggered && !oos, kind: "turn2", port: "cross-right", label: eg.triggered ? "Gap Detected" : (isFull ? "Gap?" : null) },
        { from: "f-turn2", to: "f-context", on: eg.triggered && !oos, kind: "turn2", x2Frac: 0.68 },
      );
    } else {
      children.push(nGap);
    }

    if (showGraphLane) {
      arrows.push({ from: "f-gchunks", to: "f-gap", on: graphActive && !oos, kind: "graph", x2Frac: 0.28 });
    }
    if (showVecLane) {
      arrows.push({ from: "f-rerank", to: "f-gap", on: vecActive, kind: "vec", x2Frac: 0.74 });
    }

    arrows.push(
      { from: "f-gap", to: "f-context", on: !oos && !eg.triggered, kind: "vec", x2Frac: showTurn2 ? 0.32 : 0.50 },
      { from: "f-context", to: "f-gen", on: !oos, kind: "vec" },
    );

    children.push(nContext, nGen);

    if (isFull) {
      arrows.push({
        from: "f-router",
        to: "f-gen",
        on: oos,
        kind: "refusal",
        port: "outer-right",
        label: oos ? "REFUSAL SHORT-CIRCUIT (0 DB / 0 Gen)" : "Out-of-Scope Refusal",
      });
    }
  } else {
    // Pattern 6 Refusal in "trace" mode: direct prominent red short-circuit arrow 3 -> 6
    children.push(nGen);
    arrows.push({
      from: "f-router",
      to: "f-gen",
      on: true,
      kind: "refusal",
      port: "vertical",
      label: "REFUSAL SHORT-CIRCUIT · 0 Neo4j · 0 pgvector · 0 Gen calls",
    });
  }

  flow.append(...children);
  Flow.arrows = arrows;

  if (!Flow.ro && typeof ResizeObserver === "function") {
    Flow.ro = new ResizeObserver(() => requestAnimationFrame(drawArrows));
    Flow.ro.observe(flow);
  }
  requestAnimationFrame(drawArrows);
};

function addSvgLabel(svg, x, y, text, cls, rotate = false) {
  if (!text) return;
  const g = _svg("g", { class: `arrow-lbl ${cls || ""}`, ...(rotate ? { transform: `rotate(-90 ${x} ${y})` } : {}) });
  const w = Math.max(50, Math.round(text.length * 6.0 + 14));
  g.append(_svg("rect", { x: x - w / 2, y: y - 10, width: w, height: 20, rx: 10 }));
  const t = _svg("text", { x, y: y + 3.5, "text-anchor": "middle" });
  t.textContent = text;
  g.append(t);
  svg.append(g);
}

function drawArrows() {
  const flow = $("flow");
  if (!flow) return;
  let svg = flow.querySelector(".flow-arrows");
  if (!svg) {
    svg = _svg("svg", { class: "flow-arrows" });
    flow.prepend(svg);
  }
  svg.innerHTML = "";

  const defs = _svg("defs");
  const markers = [
    ["fa-vec", "#60a5fa"],
    ["fa-graph", "#34d399"],
    ["fa-cross", "#22d3ee"],
    ["fa-turn2", "#f5c451"],
    ["fa-refusal", "#f87171"],
    ["fa-off", "#334155"],
  ];
  for (const [id, c] of markers) {
    const m = _svg("marker", { id, viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 6.5, markerHeight: 6.5, orient: "auto" });
    m.append(_svg("path", { d: "M0,0 L10,5 L0,10 z", fill: c }));
    defs.append(m);
  }
  svg.append(defs);

  const box = flow.getBoundingClientRect();
  if (!box.width || !box.height) return;
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  svg.setAttribute("width", String(box.width));
  svg.setAttribute("height", String(box.height));

  const cards = [...flow.querySelectorAll(".fnode, .floor-banner")];
  const maxCardRight = cards.length ? Math.max(...cards.map((n) => n.getBoundingClientRect().right - box.left)) : box.width - 24;
  const minCardLeft = cards.length ? Math.min(...cards.map((n) => n.getBoundingClientRect().left - box.left)) : 24;
  const outerRightX = maxCardRight + 24;
  const outerLeftX = Math.max(14, minCardLeft - 20);

  const labelsToDraw = [];

  for (const spec of Flow.arrows) {
    const a = $(spec.from), b = $(spec.to);
    if (!a || !b) continue;
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    const on = Boolean(spec.on);
    const kind = spec.kind || "vec";
    const markerId = on ? ({ graph: "fa-graph", cross: "fa-cross", turn2: "fa-turn2", refusal: "fa-refusal" }[kind] || "fa-vec") : "fa-off";

    let d = "";
    let lx = 0, ly = 0, rotate = false;
    const sideBySide = ra.left > rb.right - 12;

    if (spec.port === "outer-right") {
      const x1 = ra.right - box.left + 2, y1 = ra.top + ra.height * 0.5 - box.top;
      const x2 = rb.right - box.left + 3, y2 = rb.top + rb.height * 0.5 - box.top;
      const ax = Math.max(outerRightX, x1 + 24, x2 + 24);
      const r = 14;
      d = `M${x1},${y1} L${ax - r},${y1} Q${ax},${y1} ${ax},${y1 + r} L${ax},${y2 - r} Q${ax},${y2} ${ax - r},${y2} L${x2},${y2}`;
      lx = ax;
      ly = (y1 + y2) / 2;
      rotate = true;
    } else if (spec.port === "graph-bypass") {
      const x1 = ra.left - box.left - 2, y1 = ra.top + ra.height * 0.5 - box.top;
      const x2 = rb.left - box.left - 3, y2 = rb.top + rb.height * 0.5 - box.top;
      const ax = Math.max(8, Math.min(outerLeftX, x1 - 20, x2 - 20) - (spec.railOffset || 0));
      const r = 12;
      d = `M${x1},${y1} L${ax + r},${y1} Q${ax},${y1} ${ax},${y1 + r} L${ax},${y2 - r} Q${ax},${y2} ${ax + r},${y2} L${x2},${y2}`;
      lx = ax;
      ly = (y1 + y2) / 2;
      rotate = true;
    } else if (spec.port === "bypass-left") {
      const vsEl = $("f-vecseed");
      const x1 = ra.left + ra.width * (spec.x1Frac ?? 0.22) - box.left;
      const y1 = ra.bottom - box.top;
      const y2 = rb.top - box.top - 2;
      const rbLeft = rb.left - box.left;
      const vsLeft = vsEl ? vsEl.getBoundingClientRect().left - box.left : rbLeft + 80;
      const vsTop = vsEl ? vsEl.getBoundingClientRect().top - box.top : y2;
      if (vsEl && vsLeft - rbLeft >= 28) {
        const clearX = Math.max(rbLeft + 14, vsLeft - 16);
        const yTurn = Math.min(y1 + 14, Math.max(y1 + 6, vsTop - 14));
        if (Math.abs(x1 - clearX) < 6) {
          d = `M${x1},${y1} L${clearX},${y2}`;
        } else {
          const dir = clearX <= x1 ? -1 : 1;
          const r = Math.min(12, Math.max(2, Math.abs(x1 - clearX) * 0.4), Math.max(2, (yTurn - y1) * 0.5));
          d = `M${x1},${y1} L${x1},${yTurn - r} Q${x1},${yTurn} ${x1 + dir * r},${yTurn} L${clearX - dir * r},${yTurn} Q${clearX},${yTurn} ${clearX},${yTurn + r} L${clearX},${y2}`;
        }
        lx = clearX;
        ly = (yTurn + y2) / 2;
      } else {
        const ax = Math.max(10, Math.min(outerLeftX, vsLeft - 18, rbLeft - 18));
        const yTurn = Math.min(y1 + 14, Math.max(y1 + 6, vsTop - 14));
        const yEnter = rb.top + rb.height * 0.35 - box.top;
        const r = 10;
        d = `M${x1},${y1} L${x1},${yTurn - r} Q${x1},${yTurn} ${x1 - r},${yTurn} L${ax + r},${yTurn} Q${ax},${yTurn} ${ax},${yTurn + r} L${ax},${yEnter - r} Q${ax},${yEnter} ${ax + r},${yEnter} L${rbLeft - 3},${yEnter}`;
        lx = ax;
        ly = (yTurn + yEnter) / 2;
      }
    } else if (spec.port === "cross-left" && sideBySide) {
      const x1 = ra.left - box.left - 2;
      const y1 = ra.top + ra.height * (spec.y1Frac ?? 0.5) - box.top;
      const x2 = rb.right - box.left + 3;
      const y2 = rb.top + rb.height * 0.5 - box.top;
      const dx = Math.max(18, (x1 - x2) * 0.55);
      d = `M${x1},${y1} C${x1 - dx},${y1} ${x2 + dx},${y2} ${x2},${y2}`;
      lx = (x1 + x2) / 2;
      ly = (y1 + y2) / 2 - 12;
    } else if (spec.port === "cross-left") {
      const x1 = ra.left - box.left - 2, y1 = ra.top + ra.height * (spec.y1Frac ?? 0.5) - box.top;
      const x2 = rb.left - box.left - 3, y2 = rb.top + rb.height * 0.5 - box.top;
      const ax = Math.max(10, Math.min(x1, x2) - 18);
      d = `M${x1},${y1} C${ax},${y1} ${ax},${y2} ${x2},${y2}`;
      lx = ax;
      ly = (y1 + y2) / 2;
      rotate = true;
    } else if (spec.port === "cross-right" && rb.left > ra.right - 12) {
      const x1 = ra.right - box.left + 2, y1 = ra.top + ra.height * 0.5 - box.top;
      const x2 = rb.left - box.left - 3, y2 = rb.top + rb.height * 0.5 - box.top;
      const dx = Math.max(14, (x2 - x1) * 0.5);
      d = `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
      lx = (x1 + x2) / 2;
      ly = (y1 + y2) / 2 - 14;
    } else {
      const x1 = ra.left + ra.width * (spec.x1Frac ?? 0.5) - box.left;
      const y1 = ra.bottom - box.top;
      const x2 = rb.left + rb.width * (spec.x2Frac ?? 0.5) - box.left;
      const y2 = rb.top - box.top - 2;
      const dy = Math.max(16, (y2 - y1) * 0.48);
      d = `M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`;
      lx = (x1 + x2) / 2;
      ly = (y1 + y2) / 2;
    }

    const cls = `${kind} ${on ? "on" : "off"}`;
    svg.append(_svg("path", { d, class: cls, "marker-end": `url(#${markerId})` }));
    if (on) svg.append(_svg("path", { d, class: `pulse ${kind}` }));
    if (spec.label && on) labelsToDraw.push({ x: lx, y: ly, text: spec.label, cls: kind, rotate });
  }

  for (const lbl of labelsToDraw) addSvgLabel(svg, lbl.x, lbl.y, lbl.text, lbl.cls, lbl.rotate);
}

document.addEventListener("DOMContentLoaded", () => {
  wireFlowControls();
  window.addEventListener("resize", () => requestAnimationFrame(drawArrows));
});
