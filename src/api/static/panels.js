// Answer card: recursive math/markdown renderer, grounded answer with citation pills, verbatim PDF excerpt card, telemetry footer.
"use strict";

const CITE_RE = /\[([A-Za-z0-9_.\-]+::[A-Za-z0-9_.:\-]+)\]/g;
const gateClass = (s) => ({ PASS: "ok", REPAIRED_PASS: "ok", CROPPED: "warn", EXTRACTIVE_FALLBACK: "warn", ABSTAINED: "warn", REFUSED: "bad" }[s] || "warn");
const gateLabel = (s) => ({ PASS: "✓ 100% Verified", REPAIRED_PASS: "✓ Verified after repair", CROPPED: "Cropped", EXTRACTIVE_FALLBACK: "Extractive fallback",
  ABSTAINED: "Abstained", REFUSED: "Refused", UNVALIDATED: "Unvalidated" }[s] || s || "-");

const LATEX_SYMBOLS = {
  "\\rightarrow": "→", "\\to": "→", "\\leftarrow": "←", "\\leftrightarrow": "↔",
  "\\Rightarrow": "⇒", "\\Leftarrow": "⇐", "\\Leftrightarrow": "⇔", "\\mapsto": "↦", "\\implies": "⇒",
  "\\times": "×", "\\cdot": "·", "\\div": "÷", "\\pm": "±", "\\mp": "∓",
  "\\top": "⊤", "\\intercal": "⊤", "\\circ": "∘", "\\odot": "⊙", "\\otimes": "⊗", "\\oplus": "⊕",
  "\\le": "≤", "\\leq": "≤", "\\ge": "≥", "\\geq": "≥", "\\ne": "≠", "\\neq": "≠",
  "\\ll": "≪", "\\gg": "≫", "\\approx": "≈", "\\equiv": "≡", "\\sim": "~", "\\simeq": "≃", "\\propto": "∝",
  "\\infty": "∞", "\\sum": "∑", "\\prod": "∏", "\\int": "∫", "\\partial": "∂", "\\nabla": "∇",
  "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ", "\\epsilon": "ε", "\\varepsilon": "ε",
  "\\zeta": "ζ", "\\eta": "η", "\\theta": "θ", "\\vartheta": "ϑ", "\\iota": "ι", "\\kappa": "κ",
  "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ", "\\pi": "π", "\\rho": "ρ",
  "\\sigma": "σ", "\\tau": "τ", "\\upsilon": "υ", "\\phi": "φ", "\\varphi": "φ", "\\chi": "χ",
  "\\psi": "ψ", "\\omega": "ω",
  "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ", "\\Xi": "Ξ",
  "\\Pi": "Π", "\\Sigma": "Σ", "\\Upsilon": "Υ", "\\Phi": "Φ", "\\Psi": "Ψ", "\\Omega": "Ω",
  "\\in": "∈", "\\notin": "∉", "\\subset": "⊂", "\\subseteq": "⊆", "\\cup": "∪", "\\cap": "∩",
  "\\forall": "∀", "\\exists": "∃", "\\ell": "ℓ", "\\star": "⋆", "\\ast": "*", "\\dagger": "†",
  "\\dots": "…", "\\ldots": "…", "\\cdots": "⋯", "\\mid": "|", "\\parallel": "‖", "\\|": "‖",
  "\\{": "{", "\\}": "}", "\\%": "%", "\\&": "&", "\\_": "_",
  "\\quad": " ", "\\qquad": "  ", "\\,": " ", "\\;": " ", "\\:": " ", "\\!": "",
  "\\exp": "exp", "\\log": "log", "\\ln": "ln", "\\max": "max", "\\min": "min",
  "\\argmax": "argmax", "\\argmin": "argmin", "\\clip": "clip", "\\tanh": "tanh",
  "\\softmax": "softmax", "\\ReLU": "ReLU", "\\SiLU": "SiLU", "\\Attn": "Attn",
  "\\Attention": "Attention", "\\MultiHead": "MultiHead", "\\Concat": "Concat",
  "\\RMSNorm": "RMSNorm", "\\LayerNorm": "LayerNorm", "\\mean": "mean", "\\std": "std", "\\Huber": "Huber",
};

const BLACKBOARD_MAP = { E: "𝔼", R: "ℝ", N: "ℕ", Z: "ℤ", Q: "ℚ", C: "ℂ", P: "ℙ" };
const MATH_TEXT_WRAPPERS = new Set(["text", "mathrm", "mathbf", "mathit", "operatorname", "boldsymbol", "mathcal", "mathsf", "mathtt"]);
const MATH_ACCENT_COMBINING = { hat: "\u0302", tilde: "\u0303", bar: "\u0304", overline: "\u0304", vec: "\u20d7" };
const MATH_BASES = new Set(["xW", "QK", "RMS", "LN", "Attn", "KL", "head", "ReLU", "SiLU"]);
const MATH_SUBS = new Set([
  "model", "ff", "attn", "embd", "layer", "vocab", "ctx", "ref", "old", "vllm",
  "min", "max", "opt", "res", "pre", "post", "DKV", "KV", "KL", "UK", "UV",
  "GAE", "DPO", "GRPO", "GSPO", "PPO", "low", "high", "mean", "std", "beh", "tar", "mem", "cl",
]);

const VALID_SNAKE_CASE_RE = /(?<![A-Za-z0-9_])(?:[a-z]{2,}(?:_[a-z0-9]+)+|d_(?:model|ff|attn|embd|k|v|q|h|c|cl)|n_(?:layer|vocab|ctx)|r_(?:min|max)|(?:prev|new)_[KV]|P_[qkvo]|[A-Za-z0-9]+(?:_[a-z0-9]+){2,})(?![A-Za-z0-9_])/g;
const PROSE_WORDS_RE = /\b(?:the|and|for|with|where|from|that|this|these|those|which|when|while|are|was|were|has|have|had|can|will|not|into|each|denotes|elements|scale|scaled|reward|advantage|response|responses|question|policy|sample|sampled|computed|between|number|layers|tokens|total|average|only|note|then|also|both|more|less|using|used|over|under|such|since|thus|here|before|after|because|although|however|otherwise|additive|simplified|third|commit|wait|follows|paired|normalized|calculating|calculated|normalizing)\b/i;

function normalizeMathUnicode(raw) {
  return String(raw || "")
    .replace(/[\u{1D400}-\u{1D7FF}\u210E\u2113\u00B5]/gu, (ch) => ch.normalize("NFKC"))
    .replace(/\u2212/g, "-")
    .replace(/[\uFE00-\uFE0F\u200B\uFEFF]/g, "");
}

function cleanSectionPath(raw) {
  const items = [];
  let s = String(raw || "").replace(VALID_SNAKE_CASE_RE, (m) => {
    const idx = items.length;
    items.push(m);
    return `\uE000${idx}\uE001`;
  });
  s = normalizeMathUnicode(s)
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "")
    .replace(/<u>([\s\S]*?)<\/u>/gi, "$1")
    .replace(/<\/?(?:u|sup|sub)>/gi, "")
    .replace(/\ufffd/g, "")
    .replace(/\*\*([^*\n]*?)\*\*/g, "$1")
    .replace(/\*\*/g, "")
    .replace(/(?<!\w)_([^_\n]+?)_(?!\w)/g, "$1")
    .replace(/(?<!\w)\*([^*\n]+?)\*(?!\w)/g, "$1")
    .replace(/(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)/g, "");
  s = s.replace(/\uE000(\d+)\uE001/g, (_, idx) => items[Number(idx)]);
  return s.replace(/(?:\s*>\s*)+$/, "").replace(/[ \t]{2,}/g, " ").trim();
}

function makeProtector() {
  const items = [];
  return {
    protect(text, re, cleanInner = false) {
      return text.replace(re, (m) => {
        const val = cleanInner ? m.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]/g, "") : m;
        const idx = items.length;
        items.push(val);
        return `\uE000${idx}\uE001`;
      });
    },
    restore(text) {
      let out = text;
      for (let pass = 0; pass < 4 && out.includes("\uE000"); pass++) {
        out = out.replace(/\uE000(\d+)\uE001/g, (_, idx) => items[Number(idx)]);
      }
      return out;
    },
  };
}

function replaceSupTags(text) {
  return text.replace(/<sup>([\s\S]*?)<\/sup>/gi, (full, inner, offset) => {
    const before = text.slice(Math.max(0, offset - 6), offset);
    const after = text.slice(offset + full.length, offset + full.length + 6);
    const compact = inner.replace(/[*_\s]+/g, "");
    if (!compact) return before && after && /\w/.test(before.slice(-1)) && /\w/.test(after[0]) ? " " : "";

    let hasValidBase = Boolean(
      before &&
      !/\s/.test(before.slice(-1)) &&
      !/(?:=1|=0|<\/sup>|[\u2211\u220f(\[,;:=+\-*/])$/.test(before) &&
      /[A-Za-z0-9)\]_*\u0370-\u03ff]$/.test(before)
    );
    if (/[.,]$/.test(before) && /^\d+$/.test(compact)) hasValidBase = true;

    const isProse =
      !hasValidBase ||
      compact.length > 10 ||
      PROSE_WORDS_RE.test(inner) ||
      inner.includes("=") ||
      inner.includes(". ") ||
      /[A-Za-z]\.[A-Za-z]/.test(inner) ||
      /[<>]/.test(inner) ||
      /^[()[\],.;:\u00b7\u00d7+\-\u2212]$/.test(compact) ||
      (compact.includes("_") && compact.length > 4 && !/^_?[A-Za-z]{1,4}_?(?:model|ff|attn|DKV|KV)?$/.test(compact));

    if (isProse) {
      const unwrapped = inner.trim().replace(/([.,;:])(?=[A-Za-z])/g, "$1 ");
      const padL = before && !/[ \n\t([{]$/.test(before) && !before.endsWith("<sup>") ? " " : "";
      const padR = after && !/^[ \n\t.,;:!?)\]}<]/.test(after) ? " " : "";
      return `${padL}${unwrapped}${padR}`;
    }

    const cleanSup = inner.replace(/_/g, "").replace(/\*\*/g, "").trim();
    if (!cleanSup) return "";
    if (cleanSup === "√") return "√";
    const outSup = cleanSup.length === 1 && /^[A-Za-z0-9]$/.test(cleanSup) ? `^${cleanSup}` : `^{${cleanSup}}`;
    const padR = after && /^[A-Za-z]/.test(after[0]) && !cleanSup.endsWith("(") ? " " : "";
    return `${outSup}${padR}`;
  });
}

function cleanDisplayText(raw) {
  let s = String(raw || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!s) return "";

  s = s.replace(/<_?\s*([A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\s*_?>/g, "<$1>");

  const prot = makeProtector();
  s = prot.protect(s, /```[\s\S]*?```|`[^`\n]+`/g, true);
  s = prot.protect(s, /https?:\/\/[^\s<>"'`\])]+/g, true);
  s = prot.protect(s, CITE_RE);
  s = prot.protect(s, /<\/?(?:eos|bos|pad|resv|unk|sep|cls|mask|start_of_turn|end_of_turn|think|answer)>/gi);
  s = prot.protect(s, /<[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>/g);
  s = prot.protect(s, /"[^"\n]{1,80}(?:->|\u2212>)[^"\n]{0,80}"/g);

  s = s.replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "");
  s = normalizeMathUnicode(s);

  // Strikethrough/overbar √ and PDF fraction / <u> repair.
  s = s.replace(/~~\s*(?:_?\s*)?(?:<u>\s*)?[\ufffd\u221a](?:\s*<\/u>)?(?:\s*_?)?\s*~~/gi, "√")
    .replace(/~~([^~\n]{1,32})~~/g, "$1")
    .replace(/~~/g, "");
  s = s.replace(/<sup>\s*_?([^<>\n]{1,18}?)[-\u2212]_?\s*<\/sup>\s*_?([A-Za-z\u0370-\u03ff]{1,6})_?\s*<sup>\s*_?<u>([^<>\n]{1,18}?)<\/u>_?\s*<\/sup>/gi,
    (_, numL, den, numR) => `(${numL.replace(/_/g, "").trim()} - ${numR.replace(/_/g, "").trim()})/${den.replace(/_/g, "")}`);
  s = s.replace(/<sup>\s*_?\u2202_?\s*<\/sup>\s*<sup>\s*<u>([^<>\n]{1,12})<\/u>\s*<\/sup>\s*_?\u2202([A-Za-z]{1,6})_?\s*<sup>\s*<u>\(<\/u>\s*<\/sup>\s*<sup>\s*_?([A-Za-z0-9]+)_?\s*<\/sup>\s*<sup>\s*<u>\)<\/u>\s*<\/sup>/gi,
    (_, fn, v, arg) => `∂${fn}(${arg})/∂${v}`);
  s = s.replace(/\u221a\s*_?([kn])_?\s*<u>\s*1\s*(?:[\ufffd\u2211])?\s*<\/u>/gi, (_, k) => `√(1/${k}) ∑`);
  s = s.replace(/(?<![A-Za-z0-9_])_?(d|n|k|B|N|\u03b2|r|s|L|\u03c3)_?\s*<sup>\s*_?<u>([A-Za-z0-9\u0370-\u03ff\u0394\u2206*\u2032+\-]+)<\/u>_?\s*<\/sup>(?:\s*_?([rc]|model|ff|attn)_?(?=\s|[.,;:)\]\u2309]|$))?/gi,
    (_, den, num, sub) => `${num.replace(/_/g, "").trim()}/${sub ? `${den}_${sub}` : den}`);
  s = s.replace(/(^|[\s(=+\-*\u00b7])\s*<sup>\s*_?<u>([A-Za-z0-9\u0370-\u03ff\u0394\u2206]{1,10})<\/u>_?\s*<\/sup>\s*_?([A-Za-z0-9\u0370-\u03ff\u0394\u2206\u2032*]{1,8})_?(?:\s*<sup>\s*_?([\u2032'*])_?\s*<\/sup>)?/gi,
    (_, lead, num, den, prime) => `${lead}${num.replace(/_/g, "").trim()}/${den.replace(/_/g, "").trim()}${prime || ""}`);
  s = s.replace(/(?:\u221a\s*<u>\s*1\s*<\/u>|<u>\s*1\s*<\/u>\s*\u221a)\s*_?([A-Za-z0-9_\u0370-\u03ff]{1,8})_?/gi,
    (_, d) => `1/√${d.replace(/_/g, "")}`);
  s = s.replace(/<u>([\s\S]*?)<\/u>/gi, "$1").replace(/<\/?u>/gi, "");

  // Repair U+FFFD summation glyphs, then strip any remaining U+FFFD.
  s = s.replace(/\ufffd{2,}/g, "");
  s = s.replace(/(?:<sup>\s*)?\ufffd(?:\s*<\/sup>)?(?=\s*(?:_?[A-Za-z][ijknt]_?\s*=\s*1\b|(?:<sup>[^<\n]{1,10}<\/sup>\s*)*_?[ijknt]_?\s*=\s*1\b|\|o_?i\||G(?:\s|$|\s*_?[ij])|l\s*=\s*0\b|_?l_?\s*(?:<sup>\s*_?[-]_?\s*<\/sup>\s*=0|\s*-\s*1)|_?j_?\s*\u2208|_?y_?(?:\s|$|<sup>)|_?i_?(?:\s|$|<sup>)|_?\u03b1_?i|\u2202|Runs\b))/gi, "∑");
  s = s.replace(/\ufffd/g, "")
    .replace(/\u2211\s*<sup>\s*_?([A-Za-z])_?\s*<\/sup>\s*_?([ijknt])_?\s*=\s*1(?![0-9])/gi, "∑_{$2=1}^{$1}")
    .replace(/\u2211\s*_?([dknNtG])([ijknt])_?\s*=\s*1(?![0-9])/g, "∑_{$2=1}^{$1}")
    .replace(/\u2211\s*_?([ijkntl])_?\s*=\s*([01])(?![0-9])/g, "∑_{$1=$2}");

  s = replaceSupTags(s);
  s = s.replace(/<sub>([\s\S]*?)<\/sub>/gi, (_, inner) => (inner.trim() ? `_{${inner.replace(/_/g, "").trim()}}` : ""))
    .replace(/<\/?(?:sup|sub|u)>/gi, "");

  // Glued underscore italics unwrapping while strictly protecting valid snake_case & braced LaTeX.
  s = s.replace(/(?<=\d)_\._(?=\d)/g, ".");
  for (let p = 0; p < 3; p++) {
    const prev = s;
    s = s.replace(/([0-9A-Za-z\u0370-\u03ff)])_([dn])_(model|ff|attn|layer|vocab|ctx|embd)(?=[A-Za-z0-9+\-*/()|\s.,;:]|$)/g, "$1 $2_$3")
      .replace(/(?<![A-Za-z0-9_])_([dn])_(model|ff|attn|layer|vocab|ctx|embd)(?=[0-9A-Za-z+\-*/()])/g, "$1_$2 ");
    if (s === prev) break;
  }
  s = s.replace(/(?<![A-Za-z0-9_])_([dn])_\s*(model|ff|attn|layer|vocab|ctx|embd|[kvqKVQ])\b/g, "$1_$2")
    .replace(/(?<![A-Za-z0-9_])_d(ff|model|attn|k|v|q)_(?![A-Za-z0-9_])/g, "d_$1")
    .replace(/\*\*_([^_\n*]{1,24})_\*\*/g, "$1")
    .replace(/\*\*([A-Za-z0-9\u0370-\u03ff]{1,3})\*\*/g, "$1")
    .replace(/(?<![A-Za-z0-9_\u2211\u220f])_(\{[^{}\n_]{0,40}\}|[{}])_(?![A-Za-z0-9_])/g, "$1");

  s = prot.protect(s, VALID_SNAKE_CASE_RE);
  s = prot.protect(s, /[_^](?:\{[^{}\n]{1,48}\}|\\[A-Za-z]+)/g);

  for (let p = 0; p < 4; p++) {
    const prev = s;
    s = s.replace(/(?<![A-Za-z0-9_])_([A-Za-z0-9\u0370-\u03ff\u2192+\-*/|]{1,18})_(?=[A-Za-z0-9\u0370-\u03ff=+\-*/(])/g, "$1 ")
      .replace(/([A-Za-z0-9\u0370-\u03ff)\]*])_([A-Za-z0-9\u0370-\u03ff\u2192]{1,12})_(?=[A-Za-z0-9\u0370-\u03ff=+\-*/(])/g, "$1 $2 ")
      .replace(/(?<=[A-Za-z0-9)\]*])_([A-Za-z0-9\u0370-\u03ff\u2192+\-*/|]{1,18})_(?![A-Za-z0-9_])/g, " $1");
    if (s === prev) break;
  }

  s = prot.protect(s, /(?<![A-Za-z0-9_])[A-Za-z\u0370-\u03ff]{1,3}_[A-Za-z0-9\u0370-\u03ff]{1,3}(?![A-Za-z0-9_])/g);
  for (let p = 0; p < 3; p++) {
    const prev = s;
    s = s.replace(/(?<![A-Za-z0-9_])_([^\s_\n][^_\n]{0,60}?[^\s_\n]|[^\s_\n])_(?![A-Za-z0-9_])/g, "$1")
      .replace(/(?<![A-Za-z0-9_\u2211\u220f])_(\{[^{}\n]{1,60}\})_(?![A-Za-z0-9_])/g, "$1");
    if (s === prev) break;
  }
  s = s.replace(/(?<![A-Za-z0-9_{])_(?=[A-Za-z0-9\u0370-\u03ff(<\[{\u2308.,;:])/g, "")
    .replace(/(?<=[A-Za-z0-9\u0370-\u03ff)>.,;:\]\u2309}])_(?![A-Za-z0-9_{])/g, "");

  // Strip stray/unmatched ** per line.
  s = s.split("\n").map((line) => {
    let l = line.replace(/\*\*\s*\*\*/g, "").replace(/^\s*\*\*\s*(?=[:;.,])/, "");
    if ((l.match(/\*\*/g) || []).length % 2 === 1) l = l.replace(/\*\*/g, "");
    return l;
  }).join("\n");

  s = prot.restore(s);

  const prot2 = makeProtector();
  s = prot2.protect(s, /```[\s\S]*?```|`[^`\n]+`/g);
  s = prot2.protect(s, /https?:\/\/[^\s<>"'`\])]+/g);
  s = prot2.protect(s, CITE_RE);
  s = prot2.protect(s, /<\/?(?:eos|bos|pad|resv|unk|sep|cls|mask|start_of_turn|end_of_turn|think|answer)>/gi);
  s = prot2.protect(s, /<[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>/g);
  s = prot2.protect(s, /"[^"\n]{1,80}(?:->|\u2212>)[^"\n]{0,80}"/g);
  s = prot2.protect(s, VALID_SNAKE_CASE_RE);

  // OCR spacing and common math shorthand cleanup.
  s = s.replace(/([0-9)\]])\.(?=[A-Z][a-z])/g, "$1. ")
    .replace(/\(\s+([^()\n]{1,80}?)\s+\)/g, "($1)")
    .replace(/\(\s+([A-Za-z0-9\u0370-\u03ff|])/g, "($1")
    .replace(/([A-Za-z0-9\u0370-\u03ff|])\s+\)/g, "$1)")
    .replace(/([A-Za-z0-9)\]*_\u0370-\u03ff])\s+,/g, "$1,")
    .replace(/([A-Za-z0-9)\]\u0370-\u03ff])\s+\.(?=\s|$)/g, "$1.")
    .replace(/(\bxW|(?<![A-Za-z0-9_])W)\s+([12]|Q|K|V|O|q|k|v|o)\b/g, "$1_$2")
    .replace(/(?<![A-Za-z0-9_])b\s+([12])\b/g, "b_$1")
    .replace(/(?<![A-Za-z0-9_])d\s+(model|ff|k|v|q|h|c)\b/g, "d_$1")
    .replace(/(?<![A-Za-z0-9_])\u03c3\s*_?\s*2(?![0-9.])/g, "σ^2")
    .replace(/\\sigma\s*_?\s*2(?![0-9.])/g, "\\sigma^2")
    .replace(/(\([^()\n]{1,64}\))2(?![0-9A-Za-z])/g, "$1^2")
    .replace(/(\^[0-9A-Za-z]+)=/g, "$1 =")
    .replace(/(?<![A-Za-z0-9_])dk(?![A-Za-z0-9_])/g, "d_k")
    .replace(/(?<![A-Za-z0-9_])dv(?![A-Za-z0-9_])/g, "d_v")
    .replace(/(?<![A-Za-z0-9_])dq(?![A-Za-z0-9_])/g, "d_q")
    .replace(/(?<![A-Za-z0-9_])vk(?![A-Za-z0-9_])/g, "v_k")
    .replace(/\u221a\s+([A-Za-z0-9_]+)/g, "√$1")
    .replace(/(?<=[A-Za-z0-9}\)])(?=\u2211|\\sum)/g, " ")
    .replace(/(?<=\s)-(?=[\u03bc\u03c3\u03b1\u03b2\u03b3\u03b4\u03b5\u03b8\u03bb\u03c0])/g, "- ")
    .replace(/((?:\u2211|\\sum)(?:_\{[^{}]+\}|\^\{[^{}]+\}|_[A-Za-z0-9]|\^[A-Za-z0-9]){1,2})(?=[A-Za-z\u0370-\u03ff(])/g, "$1 ");

  s = prot2.restore(s);
  return s.replace(/[ \t]+\n/g, "\n").replace(/\n[ \t]+/g, "\n").replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

// ---------------------------------------------------------------------------
// Recursive balanced-brace math AST -> safe DOM elements (strictly createElement)
// ---------------------------------------------------------------------------

function readBalanced(src, openIdx, openCh = "{", closeCh = "}") {
  if (openIdx >= src.length || src[openIdx] !== openCh) return null;
  let depth = 0;
  for (let i = openIdx; i < src.length; i++) {
    const ch = src[i];
    if (openCh === "{" && ch === "\\" && i + 1 < src.length) { i++; continue; }
    if (ch === openCh) depth++;
    else if (ch === closeCh) {
      depth--;
      if (depth === 0) return { content: src.slice(openIdx + 1, i), end: i + 1 };
    }
  }
  return null;
}

function skipSpaces(src, pos) {
  while (pos < src.length && (src[pos] === " " || src[pos] === "\t")) pos++;
  return pos;
}

function readScriptArg(src, pos, mathMode) {
  const sp = skipSpaces(src, pos);
  if (sp < src.length && src[sp] === "{") {
    const grp = readBalanced(src, sp, "{", "}");
    if (grp) return { raw: grp.content, next: grp.end, braced: true };
  }
  if (pos >= src.length) return null;
  if (src[pos] === "\\") {
    const m = src.slice(pos).match(/^\\(?:[A-Za-z]+|[{}_%&#|])/);
    if (m) {
      let end = pos + m[0].length;
      const name = m[0].slice(1);
      if (MATH_TEXT_WRAPPERS.has(name) || name === "mathbb" || name === "sqrt") {
        const grp = readBalanced(src, skipSpaces(src, end), "{", "}");
        if (grp) return { raw: src.slice(pos, grp.end), next: grp.end, braced: false };
      }
      return { raw: m[0], next: end, braced: false };
    }
  }
  const rest = src.slice(pos);
  if (mathMode) {
    const m = rest.match(/^(?:[-+]?[0-9]+(?:\.[0-9]+)?|[A-Za-z0-9\u0370-\u03ff*⊤⊺′]+)/);
    if (m) return { raw: m[0], next: pos + m[0].length, braced: false };
  }
  return null;
}

function readBigOpLimits(src, pos) {
  let lower = null;
  let upper = null;
  let cur = pos;
  for (let pass = 0; pass < 2; pass++) {
    const sp = skipSpaces(src, cur);
    if (sp >= src.length) break;
    const op = src[sp];
    if (op === "_" && lower === null) {
      const arg = readScriptArg(src, sp + 1, true);
      if (!arg) break;
      lower = arg.raw;
      cur = arg.next;
    } else if (op === "^" && upper === null) {
      const arg = readScriptArg(src, sp + 1, true);
      if (!arg) break;
      upper = arg.raw;
      cur = arg.next;
    } else {
      break;
    }
  }
  return { lower, upper, next: cur };
}

function pushText(out, str) {
  if (!str) return;
  if (out.length && typeof out[out.length - 1] === "string") out[out.length - 1] += str;
  else out.push(str);
}

function trailingBaseInfo(out) {
  if (!out.length) return { ok: false, base: "", snakePrefix: false };
  const last = out[out.length - 1];
  if (typeof last !== "string") {
    // Allow chained subscript + superscript (`c_t^{KV}` -> <sub>t</sub> followed by `^`).
    return { ok: last && (last.tagName === "SUB" || last.tagName === "SUP"), base: "", snakePrefix: false };
  }
  const m = last.match(/([A-Za-z0-9\u0370-\u03ff\u2102\u210d\u2115\u2119\u211a\u211d\u2124\u{1D53C})\u2032*\u2217\]]+)$/u);
  if (!m) return { ok: false, base: "", snakePrefix: false };
  const base = m[1];
  const prefixBefore = last.slice(0, last.length - base.length);
  const snakePrefix = /_[A-Za-z0-9\u0370-\u03ff]*$/.test(prefixBefore);
  return { ok: true, base, snakePrefix };
}

function canRenderProseScript(out, op, src, nextPos) {
  const { ok, base, snakePrefix } = trailingBaseInfo(out);
  if (!ok) return null;
  // Explicit braced `_{...}` / `^{...}` ALWAYS renders as sub/sup (0 raw braces left).
  const sp = skipSpaces(src, nextPos);
  if (sp < src.length && src[sp] === "{") {
    const grp = readBalanced(src, sp, "{", "}");
    if (grp) return { raw: grp.content, next: grp.end, braced: true };
  }
  if (op === "^") {
    const rest = src.slice(nextPos);
    if (rest.startsWith("\\")) return readScriptArg(src, nextPos, true);
    const m = rest.match(/^(?:[-+]?[0-9]{1,4}|[A-Za-z\u0370-\u03ff*⊤⊺′]{1,4})(?![A-Za-z0-9_])/);
    if (m) return { raw: m[0], next: nextPos + m[0].length, braced: false };
    return null;
  }
  // Strict snake_case guard for unbraced `_` in prose (`core_block`, `vaswani_2017_transformer` untouched).
  if (snakePrefix || !base) return null;
  const m = src.slice(nextPos).match(/^([A-Za-z0-9\u0370-\u03ff*+−\-]{1,12})(?![A-Za-z0-9_])/);
  if (!m) return null;
  const sub = m[1];
  if (src[nextPos + sub.length] === "_") return null; // part of a_b_c snake_case
  const isSingleBase = /^[A-Za-z\u0370-\u03ff]$/.test(base);
  if ((isSingleBase || MATH_BASES.has(base)) && (MATH_SUBS.has(sub) || /^[A-Za-z0-9\u0370-\u03ff*+−\-]{1,2}$/.test(sub))) {
    return { raw: sub, next: nextPos + sub.length, braced: false };
  }
  return null;
}

function renderMathAst(raw, mathMode = false) {
  const src = normalizeMathUnicode(String(raw || "").replace(/\*\*/g, ""));
  const out = [];
  let i = 0;

  while (i < src.length) {
    const ch = src[i];

    // 1. LaTeX commands (`\frac`, `\sqrt`, `\sum`, `\left`, `\text`, Greek, symbols)
    if (ch === "\\") {
      const cmdMatch = src.slice(i).match(/^\\([A-Za-z]+|[{}_%&#|\\])/);
      if (cmdMatch) {
        const fullCmd = cmdMatch[0];
        const name = cmdMatch[1];
        let afterCmd = i + fullCmd.length;

        if (name === "left" || name === "right") {
          const sp = skipSpaces(src, afterCmd);
          if (src[sp] === "\\" && (src[sp + 1] === "{" || src[sp + 1] === "}" || src[sp + 1] === "|")) {
            pushText(out, src[sp + 1] === "|" ? "‖" : src[sp + 1]);
            i = sp + 2;
          } else if (sp < src.length && "()[]|.".includes(src[sp])) {
            if (src[sp] !== ".") pushText(out, src[sp]);
            i = sp + 1;
          } else {
            i = sp;
          }
          continue;
        }

        if (name === "frac" || name === "dfrac" || name === "tfrac") {
          const num = readBalanced(src, skipSpaces(src, afterCmd), "{", "}");
          const den = num ? readBalanced(src, skipSpaces(src, num.end), "{", "}") : null;
          if (num && den) {
            out.push(el("span", { class: "frac" },
              el("span", { class: "frac-num" }, ...renderMathAst(num.content, true)),
              el("span", { class: "frac-den" }, ...renderMathAst(den.content, true))));
            i = den.end;
            continue;
          }
        }

        if (name === "sqrt") {
          let cur = skipSpaces(src, afterCmd);
          let rootIdx = null;
          if (src[cur] === "[") {
            const br = readBalanced(src, cur, "[", "]");
            if (br) { rootIdx = br.content; cur = skipSpaces(src, br.end); }
          }
          const body = readBalanced(src, cur, "{", "}");
          if (body) {
            if (rootIdx !== null) out.push(el("sup", {}, ...renderMathAst(rootIdx, true)));
            out.push(el("span", { class: "math-sqrt" }, "√",
              el("span", { class: "math-sqrt-body" }, ...renderMathAst(body.content, true))));
            i = body.end;
            continue;
          }
        }

        if (name === "sum" || name === "prod" || name === "int") {
          const sym = name === "sum" ? "∑" : name === "prod" ? "∏" : "∫";
          const lims = readBigOpLimits(src, afterCmd);
          if (lims.lower !== null || lims.upper !== null) {
            const limKids = [];
            if (lims.upper !== null) limKids.push(el("sup", {}, ...renderMathAst(lims.upper, true)));
            if (lims.lower !== null) limKids.push(el("sub", {}, ...renderMathAst(lims.lower, true)));
            out.push(el("span", { class: "math-op" },
              el("span", { class: "math-sym" }, sym),
              el("span", { class: "math-limits" }, ...limKids)));
            i = lims.next;
            continue;
          }
          pushText(out, sym);
          i = afterCmd;
          continue;
        }

        if (name === "mathbb") {
          const grp = readBalanced(src, skipSpaces(src, afterCmd), "{", "}");
          if (grp) {
            const trimmed = grp.content.trim();
            pushText(out, BLACKBOARD_MAP[trimmed] || trimmed);
            i = grp.end;
            continue;
          }
        }

        if (MATH_TEXT_WRAPPERS.has(name)) {
          const grp = readBalanced(src, skipSpaces(src, afterCmd), "{", "}");
          if (grp) {
            const innerNodes = renderMathAst(grp.content, name !== "text");
            for (const n of innerNodes) {
              if (typeof n === "string") pushText(out, n);
              else out.push(n);
            }
            i = grp.end;
            continue;
          }
        }

        if (MATH_ACCENT_COMBINING[name]) {
          const grp = readBalanced(src, skipSpaces(src, afterCmd), "{", "}");
          if (grp) {
            const comb = MATH_ACCENT_COMBINING[name];
            const inner = grp.content.trim();
            if (inner.length === 1) {
              pushText(out, inner + comb);
            } else {
              for (const n of renderMathAst(inner, true)) {
                if (typeof n === "string") pushText(out, n);
                else out.push(n);
              }
              pushText(out, comb);
            }
            i = grp.end;
            continue;
          }
        }

        if (Object.prototype.hasOwnProperty.call(LATEX_SYMBOLS, fullCmd)) {
          if (name === "sigma" && src[afterCmd] === "2" && !(afterCmd + 1 < src.length && /[0-9.]/.test(src[afterCmd + 1]))) {
            pushText(out, "σ");
            out.push(el("sup", {}, "2"));
            i = afterCmd + 1;
            continue;
          }
          pushText(out, LATEX_SYMBOLS[fullCmd]);
          i = afterCmd;
          continue;
        }
        pushText(out, name);
        i = afterCmd;
        continue;
      }
    }

    // 2. Unicode big operators (`∑`, `∏`, `∫`) with optional `_{...}^{...}` limits
    if (ch === "∑" || ch === "∏" || ch === "∫") {
      const lims = readBigOpLimits(src, i + 1);
      if (lims.lower !== null || lims.upper !== null) {
        const limKids = [];
        if (lims.upper !== null) limKids.push(el("sup", {}, ...renderMathAst(lims.upper, true)));
        if (lims.lower !== null) limKids.push(el("sub", {}, ...renderMathAst(lims.lower, true)));
        out.push(el("span", { class: "math-op" },
          el("span", { class: "math-sym" }, ch),
          el("span", { class: "math-limits" }, ...limKids)));
        i = lims.next;
        continue;
      }
      pushText(out, ch);
      i++;
      continue;
    }

    // 3. Unicode square root with balanced `(...)` (`√(a + x)` -> `.math-sqrt`)
    if (ch === "√" && src[i + 1] === "(") {
      const grp = readBalanced(src, i + 1, "(", ")");
      if (grp) {
        out.push(el("span", { class: "math-sqrt" }, "√",
          el("span", { class: "math-sqrt-body" }, ...renderMathAst(grp.content, true))));
        i = grp.end;
        continue;
      }
    }

    // 4. Balanced parenthesized fraction `(num)/(den)`
    if (ch === "(") {
      const numGrp = readBalanced(src, i, "(", ")");
      if (numGrp && src[numGrp.end] === "/" && src[numGrp.end + 1] === "(") {
        const denGrp = readBalanced(src, numGrp.end + 1, "(", ")");
        if (denGrp && numGrp.content.trim() && denGrp.content.trim() && !numGrp.content.includes("http")) {
          out.push(el("span", { class: "frac" },
            el("span", { class: "frac-num" }, ...renderMathAst(numGrp.content, true)),
            el("span", { class: "frac-den" }, ...renderMathAst(denGrp.content, true))));
          i = denGrp.end;
          continue;
        }
      }
    }

    // 5. Subscripts & Superscripts (`_` / `^`)
    if (ch === "_" || ch === "^") {
      let arg = null;
      if (mathMode) {
        const { ok, base, snakePrefix } = trailingBaseInfo(out);
        const sp = skipSpaces(src, i + 1);
        if (ok && sp < src.length && src[sp] === "{") {
          arg = readScriptArg(src, i + 1, true);
        } else if (ok && !(ch === "_" && (snakePrefix || (base.length >= 2 && !MATH_BASES.has(base) && !/^[0-9)]+$/.test(base))))) {
          const cand = readScriptArg(src, i + 1, true);
          if (cand && !(ch === "_" && base.length >= 2 && cand.raw.length >= 4 && !MATH_SUBS.has(cand.raw))) {
            arg = cand;
          }
        }
      } else {
        arg = canRenderProseScript(out, ch, src, i + 1);
      }
      if (arg) {
        out.push(el(ch === "^" ? "sup" : "sub", {}, ...renderMathAst(arg.raw, true)));
        i = arg.next;
        continue;
      }
    }

    pushText(out, ch);
    i++;
  }

  return out;
}

function renderMathTokens(text, allowLooseSub = false) {
  return renderMathAst(text, allowLooseSub);
}

function citePill(id, opts = {}) {
  const paper = id.split("::")[0], sec = id.split("::").slice(1).join("·");
  return el("span", {
    class: "cite", "data-chunk": id, title: id,
    onclick: () => App.highlight(id, { source: opts.source || "answer", container: opts.container || null }),
  }, `${paper} · ${sec}`);
}

const INLINE_TOKEN_RE = new RegExp([
  "(\\[[A-Za-z0-9_.\\-]+::[A-Za-z0-9_.:\\-]+\\])",
  "(\\${2}[\\s\\S]+?\\${2})",
  "(\\\\\\[[\\s\\S]+?\\\\\\])",
  "(\\$[^$\\n]+?\\$)",
  "(\\\\\\([^\\n]+?\\\\\\))",
  "(<(?:sup|sub)>[^<\\n]{1,80}?<\\/(?:sup|sub)>)",
  "(`[^`\\n]+`)",
  "(\\*\\*[^*\\n]+?\\*\\*)",
  "(?<![A-Za-z0-9\\\\^*])(\\*(?!\\s|\\*)[^*\\n]{1,160}?(?<!\\s|\\*)\\*)(?![A-Za-z0-9^*])",
  "(?<![A-Za-z0-9_])(_(?!\\s)[^_\\n]{1,80}?(?<!\\s)_)(?![A-Za-z0-9_])",
].join("|"), "g");

function isCurrencySpan(expr) {
  return /^\s*\d[\d,.]*(?:\s*(?:[kmbKMB]|million|billion|trillion))?(?:\s|$)/.test(expr) && !/[=\\^_{}∑∏∫√]/.test(expr);
}

// Inline text -> safe DOM nodes (citations, formatted math, <sup>/<sub>, bold/italic, code); never HTML injection.
function inline(text, opts = {}) {
  const src = String(text || "");
  const out = [];
  let last = 0;
  for (const m of src.matchAll(INLINE_TOKEN_RE)) {
    const tok = m[0];
    if (m.index > last) out.push(...renderMathTokens(src.slice(last, m.index).replace(/\*\*/g, ""), false));
    if (tok.startsWith("[") && tok.endsWith("]")) {
      const cid = tok.slice(1, -1);
      out.push(opts.citations === false ? tok : citePill(cid, opts));
    } else if (tok.startsWith("$$") || tok.startsWith("\\[")) {
      const expr = tok.slice(2, -2).trim();
      out.push(el("div", { class: "math-block" }, ...renderMathTokens(expr, true)));
    } else if (tok.startsWith("$") || tok.startsWith("\\(")) {
      const expr = tok.startsWith("\\(") ? tok.slice(2, -2).trim() : tok.slice(1, -1).trim();
      if (tok.startsWith("$") && (isCurrencySpan(expr) || (tok.length > 2 && /^\s/.test(tok.slice(1)) && /\s$/.test(tok.slice(0, -1))))) {
        out.push(...renderMathTokens(tok.replace(/\*\*/g, ""), false));
      } else {
        out.push(el("span", { class: "math-inline" }, ...renderMathTokens(expr, true)));
      }
    } else if (tok.startsWith("<sup") || tok.startsWith("<sub")) {
      const tag = tok.slice(1, 4).toLowerCase();
      const inner = tok.slice(5, -6);
      out.push(el(tag, {}, ...inline(inner, { ...opts, citations: false })));
    } else if (tok.startsWith("`")) {
      out.push(el("code", {}, tok.slice(1, -1)));
    } else if (tok.startsWith("**")) {
      out.push(el("strong", {}, ...inline(tok.slice(2, -2), opts)));
    } else if (tok.startsWith("*")) {
      out.push(el("em", {}, ...inline(tok.slice(1, -1), opts)));
    } else if (tok.startsWith("_")) {
      out.push(el("em", {}, ...inline(tok.slice(1, -1), { ...opts, citations: false })));
    }
    last = m.index + tok.length;
  }
  if (last < src.length) out.push(...renderMathTokens(src.slice(last).replace(/\*\*/g, ""), false));
  return out.filter((x) => x !== "");
}

function extractBlockConstructs(text) {
  const blocks = [];
  let s = String(text || "");
  s = s.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => {
    const idx = blocks.length;
    blocks.push({ kind: "code", body: code.replace(/\n$/, "") });
    return `\n\uE002${idx}\uE003\n`;
  });
  s = s.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g, (_, dBody, bBody) => {
    const idx = blocks.length;
    blocks.push({ kind: "math", body: (dBody ?? bBody).trim() });
    return `\n\uE002${idx}\uE003\n`;
  });
  return { text: s, blocks };
}

function renderMarkdownLite(text, opts = {}) {
  const frag = document.createDocumentFragment();
  const { text: staged, blocks } = extractBlockConstructs(cleanDisplayText(text));
  let list = null;
  let listTag = null;
  for (const raw of staged.split("\n")) {
    const line = raw.trimEnd();
    const blk = line.trim().match(/^\uE002(\d+)\uE003$/);
    if (blk) {
      list = null; listTag = null;
      const b = blocks[Number(blk[1])];
      if (b.kind === "code") frag.append(el("pre", {}, el("code", {}, b.body)));
      else frag.append(el("div", { class: "math-block" }, ...renderMathTokens(b.body, true)));
      continue;
    }
    const heading = line.match(/^\s*#{1,4}\s+(.*)$/);
    if (heading) { list = null; listTag = null; frag.append(el("h4", { class: "ans-h" }, ...inline(heading[1], opts))); continue; }
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const ordered = !bullet && line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || ordered) {
      const tag = bullet ? "ul" : "ol";
      if (!list || listTag !== tag) { list = el(tag); listTag = tag; frag.append(list); }
      list.append(el("li", {}, ...inline((bullet || ordered)[1], opts)));
      continue;
    }
    list = null; listTag = null;
    if (line.trim()) frag.append(el("p", {}, ...inline(line, opts)));
  }
  return frag;
}

function renderExcerptParagraphs(text) {
  const { text: staged, blocks } = extractBlockConstructs(cleanDisplayText(text));
  const paras = staged.split(/\n\s*\n+/).map((p) => p.trim()).filter(Boolean);
  if (!paras.length) return [el("div", { class: "empty" }, "No text recorded.")];
  return paras.map((p) => {
    const blk = p.match(/^\uE002(\d+)\uE003$/);
    if (blk) {
      const b = blocks[Number(blk[1])];
      if (b.kind === "code") return el("pre", {}, el("code", {}, b.body));
      return el("div", { class: "math-block" }, ...renderMathTokens(b.body, true));
    }
    const lines = p.split("\n").map((l) => l.trim()).filter(Boolean);
    const kids = [];
    lines.forEach((line, idx) => {
      if (idx > 0) kids.push(el("br"));
      const m = line.match(/^\uE002(\d+)\uE003$/);
      if (m) {
        const b = blocks[Number(m[1])];
        if (b.kind === "code") kids.push(el("code", {}, b.body));
        else kids.push(el("span", { class: "math-inline" }, ...renderMathTokens(b.body, true)));
      } else {
        kids.push(...inline(line, { citations: false }));
      }
    });
    return el("p", {}, ...kids);
  });
}

function meter(j) {
  const u = j.usage_telemetry || {};
  const router = (u.router_input_tokens || 0) + (u.router_output_tokens || 0), embed = u.embedding_tokens_est || 0;
  const rerank = (u.reranker_input_tokens || 0) + (u.reranker_output_tokens || 0);
  const gen = (u.generator_input_tokens || 0) + (u.generator_output_tokens || 0), total = u.total_tokens || router + embed + rerank + gen;
  const pct = (x) => (total ? (100 * x) / total : 0);
  const retrievalOnly = Math.max(0, (j.retrieval_latency_ms || 0) - (j.routing_latency_ms || 0));
  const box = $("meter"); box.innerHTML = "";
  box.append(
    el("div", { class: "tok" },
      el("div", { class: "tok-bar", title: "token share by component" },
        el("span", { class: "seg-router", style: `width:${pct(router)}%` }),
        el("span", { class: "seg-embed", style: `width:${pct(embed)}%` }),
        rerank > 0 ? el("span", { class: "seg-rerank", style: `width:${pct(rerank)}%` }) : null,
        el("span", { class: "seg-gen", style: `width:${pct(gen)}%` })),
      el("div", { class: "tok-legend" }, el("i", { class: "seg-router" }), "Router: ", el("b", {}, fmt(router)), " tok",
        el("i", { class: "seg-embed" }), "Embed: ", el("b", {}, fmt(embed)), " tok",
        rerank > 0 ? [el("i", { class: "seg-rerank" }), "Reranker: ", el("b", {}, fmt(rerank)), " tok"] : null,
        el("i", { class: "seg-gen" }), "Generator: ", el("b", {}, fmt(gen)), " tok",
        " · Total: ", el("b", {}, fmt(total)), ` tok · ${u.total_api_requests_used || 0} API reqs` + (u.embedding_cache_hits ? ` (${u.embedding_cache_hits} embed cache hits)` : ""))),
    el("span", { class: "badge", title: `round-trip ${Math.round(App.roundTripMs || 0)} ms` },
      `⏱ ${fmt(Math.round(j.latency_ms))} ms`, el("span", { class: "muted" },
        `routing ${Math.round(j.routing_latency_ms || 0)} · retrieval ${Math.round(retrievalOnly)} · gen ${Math.round(j.generation_latency_ms || 0)} ms`)),
    ...(u.note ? [el("span", { class: "muted" }, u.note)] : []));
}

function buildCitationCard(id, c, closeFn) {
  const bodyText = (c && (c.full_text || c.text_snippet)) || "";
  const titleText = c ? (c.pub_year ? `${c.paper_title} (${c.pub_year})` : c.paper_title) : id;
  return el("div", { class: "pdf-excerpt" },
    el("div", { class: "pdf-head" },
      el("div", { class: "pdf-title" }, titleText),
      c && c.pdf_url
        ? el("a", { class: "paper-pdf", href: c.pdf_url, target: "_blank", rel: "noopener noreferrer", title: `Open arXiv PDF ${c.arxiv_id || ""}` }, "Open PDF ↗")
        : null,
      el("span", { class: "x", title: "Close excerpt", onclick: closeFn }, "✕")
    ),
    c
      ? el("div", { class: "pdf-meta" },
          el("span", { class: "pdf-sec" }, `Section: ${cleanSectionPath(c.section_path)}`),
          el("span", { class: "pdf-cid" }, c.chunk_id),
          el("span", { class: "origin " + c.source_origin }, c.source_origin),
          c.score != null ? el("span", { class: "muted" }, `score ${c.score}`) : null)
      : el("div", { class: "pdf-meta" }, el("span", {}, "Retrieved chunk (not directly cited in the final answer)")),
    c ? el("blockquote", { class: "pdf-body" }, ...renderExcerptParagraphs(bodyText)) : null
  );
}

function showCitation(id, targetBox = null) {
  const activePop = $("cite-pop");
  if (!id) {
    document.querySelectorAll(".cite-pop").forEach((p) => { p.hidden = true; p.innerHTML = ""; });
    return;
  }
  const c = ((App.last && App.last.citations) || []).find((x) => x.chunk_id === id) || App.citationsById.get(id);
  const pop = targetBox || activePop;
  document.querySelectorAll(".cite-pop").forEach((p) => { if (p !== pop) { p.hidden = true; p.innerHTML = ""; } });
  if (!pop) return;
  pop.innerHTML = "";
  pop.hidden = false;
  pop.append(buildCitationCard(id, c, () => App.highlight(null, { toggle: false })));
}

window.buildAnswerNodes = function (j, opts = {}) {
  const nodes = [];
  if (j.refused) nodes.push(el("p", { class: "refused" }, "Refused: out of scope or the subject is not in the indexed corpus."));
  nodes.push(renderMarkdownLite(j.answer, opts));
  return nodes;
};

window.renderAnswer = function (j, question) {
  for (const c of j.citations || []) App.citationsById.set(c.chunk_id, c);
  $("q-text").textContent = question;
  const u = j.usage_telemetry || {};
  $("ans-meta").textContent = `${MODE_LABELS[j.mode] || j.backend} · ${modelLabel(u.generator_model || "")}`;
  const box = $("answer"); box.innerHTML = "";
  box.append(...window.buildAnswerNodes(j));
  $("cite-pop").hidden = true;
  meter(j);
};

App.onHighlight((chunk, source, container) => {
  document.querySelectorAll("[data-chunk]").forEach((n) => n.classList.toggle("hl", !!chunk && n.dataset.chunk === chunk));
  showCitation(chunk, container || null);
});

document.addEventListener("DOMContentLoaded", () => {
  const q = $("question");
  q.addEventListener("input", autosize);
  q.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); ask(); } });
  $("composer").addEventListener("submit", ask);
  loadConfig().then(() => {
    const params = new URLSearchParams(location.search); // shareable: /?q=... or optional ?mode=...
    if (params.get("mode")) App.modeOverride = params.get("mode");
    if (params.get("view") === "full") GraphUI.view = "full";
    if (params.get("chunks") === "1") $("chunks").checked = true;
    if (params.get("q")) { q.value = params.get("q"); ask(); }
  }).catch((e) => toast(`Could not load /ui-config or /examples: ${e.message}`, { error: true }));
});
