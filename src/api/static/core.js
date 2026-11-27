// Core: DOM helpers, shared state, chunk-highlight bus, config/examples, composer and query submission.
// Every value rendered by the workbench comes from the API response of the question just asked.
"use strict";

const App = { last: null, highlighted: null, listeners: [], config: null };

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

const MODEL_LABELS = { "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite", "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
  "gemini-3.8-flash": "Gemini 3.8 Flash", "gemma-4-31b-it": "Gemma 4 31B IT" };
const MODE_LABELS = { live: "Live (Neo4j + pgvector + Gemini)", gemini: "Gemini + In-Memory", offline: "Offline Instant" };
const modelLabel = (m) => MODEL_LABELS[m] || m.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// Highlight bus: one chunk_id is highlighted across the answer, the graph and the retrieval flow.
App.onHighlight = (fn) => App.listeners.push(fn);
App.highlight = (chunkId, { toggle = true, source = "" } = {}) => {
  App.highlighted = toggle && App.highlighted === chunkId ? null : chunkId;
  App.listeners.forEach((fn) => fn(App.highlighted, source));
};

function toast(msg, { error = false, actions = [] } = {}) {
  const box = $("status");
  box.innerHTML = "";
  if (!msg) return;
  box.append(el("div", { class: "toast" + (error ? " error" : "") }, msg,
    actions.length ? el("div", {}, actions.map(([label, fn]) => el("button", { class: "btn", type: "button", onclick: fn }, label))) : null));
}

const currentMode = () => $("mode").value;

async function loadConfig() {
  const [cfg, examples] = await Promise.all([fetch("/ui-config").then((r) => r.json()), fetch("/examples").then((r) => r.json())]);
  App.config = cfg;
  const mode = $("mode");
  for (const m of cfg.modes) mode.append(el("option", { value: m.id }, MODE_LABELS[m.id] || m.label));
  mode.value = cfg.modes.some((m) => m.id === cfg.default_mode) ? cfg.default_mode : "offline";
  const model = $("model");
  for (const m of cfg.generation_models) model.append(el("option", { value: m }, `${modelLabel(m)}`));
  model.value = cfg.default_generation_model;
  const sync = () => { model.disabled = mode.value === "offline"; model.title = model.disabled ? "Offline mode uses the extractive generator" : "Generation model"; };
  mode.addEventListener("change", sync); sync();
  const cards = $("cards");
  for (const ex of examples) {
    cards.append(el("button", { class: "card-q", type: "button", onclick: () => { $("question").value = ex.question; ask(); } },
      el("span", { class: "tag" }, ex.tag || "Example"), el("div", { class: "lbl" }, ex.label), el("div", { class: "txt" }, ex.question)));
  }
}

function autosize() { const t = $("question"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 180) + "px"; }

async function ask(evt) {
  if (evt) evt.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 3) return;
  const mode = currentMode();
  const body = { question, mode };
  if (mode !== "offline") body.generation_model = $("model").value;
  const btn = $("send");
  btn.disabled = true; btn.classList.add("busy");
  toast(`Running in ${MODE_LABELS[mode]} mode...`);
  const t0 = performance.now();
  try {
    const r = await fetch("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) {
      const detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail, null, 2);
      const err = new Error(detail); err.status = r.status; throw err;
    }
    App.last = j; App.lastQuestion = question; App.highlighted = null; App.roundTripMs = performance.now() - t0;
    if (App.setPage) App.setPage("ask");
    document.body.classList.remove("is-landing"); document.body.classList.add("is-active");
    window.renderAnswer(j, question);
    window.renderFlow(j);
    requestAnimationFrame(() => window.showGraph()); // measure the graph screen after layout
    toast("");
    $("question").value = ""; autosize();
  } catch (e) {
    const actions = mode !== "offline" ? [["Switch to Offline Instant", () => { $("mode").value = "offline"; $("mode").dispatchEvent(new Event("change")); toast(""); ask(); }]] : [];
    toast(`Request failed${e.status ? ` (HTTP ${e.status})` : ""}:\n${e.message}`, { error: true, actions });
  } finally {
    btn.disabled = false; btn.classList.remove("busy");
  }
}
