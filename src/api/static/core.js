// Core: DOM helpers, shared state, chunk-highlight bus, config/examples, multi-turn composer & submission.
// Every value rendered by the workbench comes from the API response of the question just asked.
"use strict";

const App = {
  last: null,
  lastQuestion: null,
  highlighted: null,
  listeners: [],
  config: null,
  turns: [],
  activeTurnId: null,
  citationsById: new Map(),
  modeOverride: null,
};

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid !== null && kid !== undefined && kid !== false) e.append(kid.nodeType ? kid : String(kid));
  return e;
}
const $ = (id) => document.getElementById(id);
const short = (id) => (String(id || "").includes("::") ? String(id).split("::").slice(1).join("::") : String(id || ""));
const fmt = (n) => Number(n || 0).toLocaleString("en-US");

const MODEL_LABELS = { "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite", "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite" };
const MODE_LABELS = { live: "Live (Neo4j + pgvector + Gemini)", gemini: "Gemini + In-Memory", offline: "Offline Instant" };
const modelLabel = (m) => MODEL_LABELS[m] || m.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// Highlight bus: one chunk_id is highlighted across the answer, the graph and the retrieval flow.
App.onHighlight = (fn) => App.listeners.push(fn);
App.highlight = (chunkId, { toggle = true, source = "", container = null } = {}) => {
  App.highlighted = toggle && App.highlighted === chunkId ? null : chunkId;
  App.listeners.forEach((fn) => fn(App.highlighted, source, container));
};

function toast(msg, { error = false, actions = [] } = {}) {
  const box = $("status");
  if (!box) return;
  box.innerHTML = "";
  if (!msg) return;
  box.append(el("div", { class: "toast" + (error ? " error" : "") }, msg,
    actions.length ? el("div", {}, actions.map(([label, fn]) => el("button", { class: "btn", type: "button", onclick: fn }, label))) : null));
}

const currentMode = () => App.modeOverride || (App.config && App.config.default_mode) || "live";

async function loadConfig() {
  const [cfg, examples] = await Promise.all([fetch("/ui-config").then((r) => r.json()), fetch("/examples").then((r) => r.json())]);
  App.config = cfg;
  const model = $("model");
  if (model) {
    model.innerHTML = "";
    for (const m of cfg.generation_models || []) model.append(el("option", { value: m }, modelLabel(m)));
    if (cfg.default_generation_model) model.value = cfg.default_generation_model;
  }
  const cards = $("cards");
  if (cards) {
    cards.innerHTML = "";
    for (const ex of examples) {
      cards.append(el("button", { class: "card-q", type: "button", onclick: () => { const qEl = $("question"); if (qEl) qEl.value = ex.question; ask(); } },
        el("span", { class: "tag" }, ex.tag || "Example"), el("div", { class: "lbl" }, ex.label), el("div", { class: "txt" }, ex.question)));
    }
  }
}

function autosize() {
  const t = $("question");
  if (!t) return;
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 180) + "px";
}

function archivePreviousTurn() {
  if (!App.last || !App.lastQuestion) return;
  const turn = { id: App.activeTurnId || `t-${Date.now()}`, question: App.lastQuestion, response: App.last };
  if (!App.turns.some((t) => t.id === turn.id)) App.turns.push(turn);
  const hist = $("history");
  if (!hist) return;

  const u = turn.response.usage_telemetry || {};
  const pop = el("div", { class: "cite-pop", hidden: true });
  const ansHtml = window.buildAnswerNodes ? window.buildAnswerNodes(turn.response, { source: "history", container: pop }) : [];
  const card = el("article", { class: "turn archived", "data-turn-id": turn.id },
    el("div", { class: "q-row" }, el("div", { class: "q-bubble" }, turn.question)),
    el("div", { class: "answer-card" },
      el("div", { class: "answer-head" },
        el("span", { class: "logo-dot" }),
        el("span", {}, "ArchGraph-RAG"),
        el("span", { class: "meta" }, `${MODE_LABELS[turn.response.mode] || turn.response.backend} · ${modelLabel(u.generator_model || "")}`)),
      el("div", { class: "answer" }, ansHtml),
      pop)
  );
  hist.append(card);
}

async function ask(evt) {
  if (evt) evt.preventDefault();
  const qEl = $("question");
  const question = qEl ? qEl.value.trim() : "";
  if (question.length < 3) return;
  const mode = currentMode();
  const body = { question, mode };
  const modelEl = $("model");
  if (mode !== "offline" && modelEl && modelEl.value) body.generation_model = modelEl.value;

  const btn = $("send");
  if (btn) { btn.disabled = true; btn.classList.add("busy"); }

  if (App.setPage) App.setPage("ask");
  archivePreviousTurn();
  App.last = null; App.highlighted = null;
  document.body.classList.remove("is-landing", "is-refused");
  document.body.classList.add("is-active", "is-loading");

  const qText = $("q-text");
  if (qText) qText.textContent = question;
  const ansMeta = $("ans-meta");
  if (ansMeta) ansMeta.textContent = `${modelLabel((modelEl && modelEl.value) || "")}`;
  const ansBox = $("answer");
  if (ansBox) {
    ansBox.innerHTML = "";
    ansBox.append(el("div", { class: "answer-loading" }, el("span", { class: "dot-pulse" }), "Traversing knowledge graph & verifying citations..."));
  }
  const citePop = $("cite-pop");
  if (citePop) citePop.hidden = true;
  const meterEl = $("meter");
  if (meterEl) meterEl.innerHTML = "";
  if (qEl) { qEl.value = ""; autosize(); }
  const activeTurnEl = $("active-turn");
  if (activeTurnEl) activeTurnEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  toast("");

  const t0 = performance.now();
  try {
    const r = await fetch("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) {
      const detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail, null, 2);
      const err = new Error(detail); err.status = r.status; throw err;
    }
    const turnId = `t-${App.turns.length + 1}-${Date.now()}`;
    App.last = j; App.lastQuestion = question; App.activeTurnId = turnId;
    App.highlighted = null; App.roundTripMs = performance.now() - t0;
    App.turns.push({ id: turnId, question, response: j });
    document.body.classList.remove("is-loading");
    const framesEl = $("frames");
    if (framesEl) framesEl.hidden = false;
    if (window.renderAnswer) window.renderAnswer(j, question);
    if (window.renderFlow) window.renderFlow(j);
    requestAnimationFrame(() => { if (window.showGraph) window.showGraph({ force: true }); });
    toast("");
  } catch (e) {
    document.body.classList.remove("is-loading");
    if (ansBox) {
      ansBox.innerHTML = "";
      ansBox.append(el("p", { class: "refused" }, `Request failed${e.status ? ` (HTTP ${e.status})` : ""}: ${e.message}`));
    }
    toast(`Request failed${e.status ? ` (HTTP ${e.status})` : ""}:\n${e.message}`, { error: true });
  } finally {
    if (btn) { btn.disabled = false; btn.classList.remove("busy"); }
  }
}
