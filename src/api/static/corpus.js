// Collapsible ChatGPT-style sidebar + dedicated Research Corpus page (35-paper 2017-2026 spine).
"use strict";

(() => {
  const SB_STORAGE_KEY = "archgraph.sidebar";
  const state = { page: "ask", filterPillar: "all", query: "" };

  function setSidebarOpen(open, { persist = true } = {}) {
    document.body.classList.toggle("sidebar-open", !!open);
    const openBtn = $("sb-open");
    if (openBtn) openBtn.setAttribute("aria-expanded", open ? "true" : "false");
    if (persist) {
      try { localStorage.setItem(SB_STORAGE_KEY, open ? "1" : "0"); } catch (_) {}
    }
    if (App.last && state.page === "ask" && window.showGraph) {
      setTimeout(() => window.showGraph(), 230);
    }
  }

  function renderCorpusLists() {
    const data = App.corpusData;
    const lists = $("corpus-lists");
    const matchEl = $("corpus-match");
    if (!data || !lists) return;

    const q = state.query.trim().toLowerCase();
    lists.innerHTML = "";
    let shown = 0;

    for (const pillar of data.pillars || []) {
      if (state.filterPillar !== "all" && pillar.id !== state.filterPillar) continue;
      const matched = (pillar.papers || []).filter((p) => {
        if (!q) return true;
        return (
          p.title.toLowerCase().includes(q) ||
          (p.org || "").toLowerCase().includes(q) ||
          String(p.year).includes(q) ||
          p.arxiv_id.toLowerCase().includes(q)
        );
      });
      if (!matched.length) continue;
      shown += matched.length;

      lists.append(
        el("section", { class: "pillar-card", "data-pillar": pillar.id },
          el("div", { class: "pillar-head" },
            el("h2", {}, pillar.name),
            el("span", { class: "pillar-count" }, `${matched.length} ${matched.length === 1 ? "paper" : "papers"}`)
          ),
          pillar.summary ? el("p", { class: "pillar-summary" }, pillar.summary) : null,
          el("ul", { class: "paper-list" },
            matched.map((p) =>
              el("li", { class: "paper-row" },
                el("div", { class: "paper-main" },
                  el("a", { class: "paper-link", href: p.abs_url, target: "_blank", rel: "noopener noreferrer" }, p.title)
                ),
                el("div", { class: "paper-meta" },
                  el("span", { class: "paper-year" }, String(p.year)),
                  p.org ? el("span", { class: "paper-org" }, p.org) : null,
                  el("a", { class: "paper-arxiv", href: p.abs_url, target: "_blank", rel: "noopener noreferrer", title: `Open arXiv abstract ${p.arxiv_id}` }, `arXiv:${p.arxiv_id}`),
                  el("a", { class: "paper-pdf", href: p.pdf_url, target: "_blank", rel: "noopener noreferrer", title: `Open PDF for ${p.arxiv_id}` }, "PDF \u2197")
                )
              )
            )
          )
        )
      );
    }

    if (!shown) {
      lists.append(el("div", { class: "corpus-empty" }, "No papers match that filter. Try clearing the search box or pillar filter."));
    }
    if (matchEl) {
      matchEl.textContent = shown === data.total_papers ? `${data.total_papers} papers` : `${shown} / ${data.total_papers} papers`;
    }
  }

  function renderCorpusPills() {
    const data = App.corpusData;
    const pills = $("corpus-pills");
    if (!data || !pills) return;
    pills.innerHTML = "";

    const makePill = (id, label) => {
      const btn = el("button", {
        type: "button",
        class: "corpus-pill" + (state.filterPillar === id ? " on" : ""),
        onclick: () => { state.filterPillar = id; renderCorpusPills(); renderCorpusLists(); },
      }, label);
      return btn;
    };

    pills.append(makePill("all", `All (${data.total_papers})`));
    for (const pillar of data.pillars || []) {
      const shortName = pillar.name.replace(/^\d+\.\s*/, "").split("&")[0].trim();
      pills.append(makePill(pillar.id, `${shortName} (${pillar.count})`));
    }
  }

  async function loadCorpus() {
    if (App.corpusData) return App.corpusData;
    const res = await fetch("/corpus");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    App.corpusData = data;

    const titleEl = $("corpus-title");
    const subEl = $("corpus-sub");
    const badgeEl = $("sb-corpus-count");
    if (titleEl && data.title) titleEl.textContent = data.title;
    if (subEl) subEl.textContent = data.subtitle || `${data.total_papers} papers across ${data.year_range}`;
    if (badgeEl) badgeEl.textContent = String(data.total_papers);

    renderCorpusPills();
    renderCorpusLists();
    return data;
  }

  App.setPage = function (page) {
    const next = page === "corpus" ? "corpus" : "ask";
    state.page = next;
    document.body.classList.toggle("is-corpus", next === "corpus");

    const corpusSection = $("corpus-view");
    if (corpusSection) corpusSection.hidden = next !== "corpus";

    document.querySelectorAll(".nav-item[data-page]").forEach((btn) => {
      const active = btn.dataset.page === next;
      btn.classList.toggle("active", active);
      if (active) btn.setAttribute("aria-current", "page");
      else btn.removeAttribute("aria-current");
    });

    if (next === "corpus") {
      loadCorpus().catch((e) => {
        const lists = $("corpus-lists");
        if (lists) lists.innerHTML = "";
        if (lists) lists.append(el("div", { class: "corpus-empty" }, `Could not load research corpus: ${e.message}`));
      });
    } else if (App.last && window.showGraph) {
      requestAnimationFrame(() => window.showGraph());
    }

    if (window.innerWidth < 900) setSidebarOpen(false, { persist: false });
  };

  document.addEventListener("DOMContentLoaded", () => {
    let initialOpen = window.innerWidth >= 900;
    try {
      const saved = localStorage.getItem(SB_STORAGE_KEY);
      if (saved === "1") initialOpen = true;
      else if (saved === "0") initialOpen = false;
    } catch (_) {}
    setSidebarOpen(initialOpen, { persist: false });

    const openBtn = $("sb-open");
    const closeBtn = $("sb-close");
    if (openBtn) openBtn.addEventListener("click", () => setSidebarOpen(true));
    if (closeBtn) closeBtn.addEventListener("click", () => setSidebarOpen(false));

    document.querySelectorAll(".nav-item[data-page]").forEach((btn) => {
      btn.addEventListener("click", () => App.setPage(btn.dataset.page));
    });

    const newBtn = $("nav-new");
    if (newBtn) {
      newBtn.addEventListener("click", () => {
        document.body.classList.remove("is-active");
        document.body.classList.add("is-landing");
        App.last = null;
        App.highlighted = null;
        toast("");
        App.setPage("ask");
        const q = $("question");
        if (q) { q.value = ""; autosize(); q.focus(); }
      });
    }

    const search = $("corpus-search");
    if (search) {
      search.addEventListener("input", () => {
        state.query = search.value;
        renderCorpusLists();
      });
    }

    loadCorpus().catch(() => {});

    const params = new URLSearchParams(location.search);
    if (params.get("page") === "corpus") App.setPage("corpus");
  });
})();
