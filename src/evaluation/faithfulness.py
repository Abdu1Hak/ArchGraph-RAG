"""Claim-Level Faithfulness & Entailment Verification (Phase 5 Evaluation).

Splits a generated answer into atomic factual claims, maps inline `[chunk_id]`
citations (including paragraph-level forward attribution matching
`src.generation.citation_validator`), and verifies each claim against the cited
chunk text(s) across:
1. Citation resolution (uncited claims or missing cited chunks fail).
2. Numbers, percentages, scales, equations, and math symbols (e.g., `93.3%`,
   `500,000`, `W^{DKV}`, `1/sqrt(d_k)`).
3. Key technical entities, acronyms, and versioned model/benchmark names.
4. Negation / contradiction polarity against the best-matching chunk clause.
5. Minimum lexical concept grounding (with an optional pluggable NLI
   `entailment_scorer` callback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.generation.citation_validator import CITATION_RE, is_claim, split_units
from src.generation.synthesizer import is_abstention
from src.retrieval.subject_gate import extract_subject_names, normalize_phrase

# ---------------------------------------------------------------------------
# Regexes & lexical tables
# ---------------------------------------------------------------------------

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
_DISPLAY_MATH_MULTILINE_RE = re.compile(r"\$\$\s*\n(.*?)\n\s*\$\$", re.S)
_VERBALIZER_PREFIX_RE = re.compile(r"^(?:Multi-hop path \(\d+\s+hops?\):|Corpus aggregate:)\s*", re.I)
_VERBALIZER_AGG_COUNT_RE = re.compile(r"^(Corpus aggregate: .*?\bby )\d+(\s+sources?\b)", re.I)

_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_CLAUSE_SPLIT_RE = re.compile(
    r"[;.\n]+|\s+(?:rather\s+than|instead\s+of|without)\s+|"
    r",\s*(?=(?:unlike|whereas|while|however|but|although|though|yet|"
    r"and\s+(?:adds|requires|does|do|did|is|are|was|were|has|have|can|will|it|they))\b)",
    re.I,
)

# Numbers with optional scale / unit suffixes, avoiding matches glued to letters/identifiers (e.g. V3, FP8, GPT-4).
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.@])(?<![A-Za-z]-)\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:%|percent|x|times|k|m|b|t|million|billion|trillion))?(?![A-Za-z0-9_])",
    re.I,
)
_RANGE_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(\d[\d,]*(?:\.\d+)?)\s*[-\u2013]\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|percent|x|times|k|m|b|t|million|billion|trillion)\b",
    re.I,
)
_UNIT_MULT: Dict[str, float] = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_UNIT_NORM: Dict[str, str] = {
    "%": "%",
    "percent": "%",
    "x": "x",
    "times": "x",
    "k": "k",
    "m": "m",
    "million": "m",
    "b": "b",
    "billion": "b",
    "t": "t",
    "trillion": "t",
}

# Symbolic / mathematical expressions: superscripts, LaTeX commands, short subscripted variables, O(...), pass@1.
_SYMBOL_RE = re.compile(
    r"(?:"
    r"(?:\d+/)?sqrt\([^)\s]+\)"
    r"|O\([^)\n]{1,40}\)"
    r"|[A-Za-z0-9\\{}()_]+(?:\^\{?[A-Za-z0-9_\\+\-*/.,]+\}?)+"
    r"|\\[A-Za-z]+(?:\{[^}\n]{1,40}\})*"
    r"|\b[A-Za-z]{1,3}_(?:\{[A-Za-z0-9,]+\}|[A-Za-z0-9]{1,3})(?:\^\{?[A-Za-z0-9,]+\}?)?\b"
    r"|\b[A-Za-z]+@\d+\b"
    r")"
)

_ACRONYM_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "mla": ("multi head latent attention",),
    "gqa": ("grouped query attention",),
    "mqa": ("multi query attention",),
    "mha": ("multi head attention",),
    "rope": ("rotary position embedding", "rotary positional embedding"),
    "dpo": ("direct preference optimization",),
    "ppo": ("proximal policy optimization",),
    "grpo": ("group relative policy optimization",),
    "mtp": ("multi token prediction",),
    "kv": ("key value",),
    "rl": ("reinforcement learning",),
    "rlhf": ("reinforcement learning from human feedback",),
    "sft": ("supervised fine tuning",),
    "cot": ("chain of thought",),
    "llm": ("large language model", "language model"),
    "llms": ("large language models", "language models"),
}

_STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "be", "been",
    "being", "by", "with", "that", "this", "these", "those", "it", "its", "they", "their", "them", "we", "our",
    "as", "at", "from", "into", "via", "per", "than", "then", "also", "only", "both", "each", "all", "some",
    "most", "many", "such", "more", "less", "very", "much", "well", "while", "whereas", "although", "though",
    "because", "since", "when", "where", "which", "who", "what", "how", "if", "can", "could", "may", "might",
    "will", "would", "should", "must", "do", "does", "did", "has", "have", "had", "not", "no", "nor", "neither",
    "without", "never", "none", "rather", "instead", "unlike", "compared", "however", "therefore", "moreover",
    "furthermore", "additionally", "meanwhile", "overall", "specifically", "notably", "first", "second", "finally",
    "use", "uses", "used", "using", "make", "makes", "made", "take", "takes", "took", "give", "gives", "gave",
    "keep", "keeps", "kept", "need", "needs", "needed", "require", "requires", "required", "include", "includes",
    "state", "states", "stated", "show", "shows", "shown", "report", "reports", "reported", "achieve", "achieves",
    "achieved", "reach", "reaches", "reached", "source", "paper", "study", "model", "models", "method", "approach",
    "technique", "system", "architecture", "result", "results", "data", "during", "across", "between", "over",
    "under", "above", "below", "about", "after", "before", "still", "even", "just", "same", "other", "another",
}

_NON_ENTITY_PROPER: Set[str] = {
    "the", "a", "an", "in", "on", "for", "with", "by", "to", "of", "and", "as", "is", "are", "was", "were",
    "its", "it", "this", "that", "these", "those", "because", "while", "whereas", "however", "although",
    "though", "since", "unlike", "compared", "existing", "previous", "prior", "standard", "both", "each",
    "all", "some", "many", "most", "such", "other", "another", "furthermore", "additionally", "moreover",
    "therefore", "meanwhile", "overall", "specifically", "notably", "importantly", "similarly", "conversely",
    "instead", "rather", "neither", "nor", "no", "not", "none", "under", "during", "before", "after", "when",
    "where", "if", "at", "from", "they", "their", "we", "our", "here", "there", "first", "second", "third",
    "finally", "one", "two", "three", "also", "only", "even", "still", "yes", "given", "despite", "beyond",
    "through", "across", "between", "among", "within", "without", "multi", "single", "corpus", "section",
    "table", "figure", "equation", "abstract", "introduction", "conclusion", "method", "methods", "results",
    "result", "model", "models", "paper", "papers", "author", "authors", "researchers", "system", "systems",
    "approach", "technique", "techniques", "training", "scaling", "using", "according", "based",
}
_COMMON_STARTER_SUFFIX_RE = re.compile(
    r"(?:ly|ing|ed|tion|sion|ment|ments|ness|ity|ities|ance|ence|ency|ancy|ive|ous|ful|less|able|ible)$",
    re.I,
)

_GENERIC_ACRONYMS: Set[str] = {"US", "UK", "OK", "II", "III", "IV", "VI", "EQ", "FIG"}

_NEGATION_CUE_RE = re.compile(
    r"\b(?:not|no|never|without|neither|nor|none|nothing|nowhere|cannot|unable|"
    r"unchanged|lacks?|lacking|absence\s+of|free\s+of|"
    r"(?:can|does|do|did|is|are|was|were|will|would|could|should)'?n'?t)\b"
    r"|\b(?:can't|doesn't|don't|didn't|isn't|aren't|wasn't|weren't|won't|couldn't|wouldn't|shouldn't)\b",
    re.I,
)
_NEGATIVE_PREFIX_CLAUSE_RE = re.compile(r"^\s*(?:rather\s+than|instead\s+of|without|unlike)\b", re.I)
_UNLIKE_COMMA_SPLIT_RE = re.compile(r"\b(unlike\b[^,;.\n]{1,80}),\s*", re.I)
_AFFIRMATIVE_IDIOM_RE = re.compile(r"\bnot\s+only\b", re.I)

EntailmentScorer = Callable[[str, str], Any]


# ---------------------------------------------------------------------------
# Audit & Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClaimAudit:
    index: int
    text: str
    cited_ids: List[str]
    resolved_chunk_ids: List[str]
    missing_chunk_ids: List[str]
    supported: bool
    numeric_mismatch: bool = False
    negation_mismatch: bool = False
    entity_mismatch: bool = False
    lexical_overlap: float = 0.0
    entailment_score: Optional[float] = None
    unmatched_numbers: List[str] = field(default_factory=list)
    unmatched_symbols: List[str] = field(default_factory=list)
    unmatched_entities: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class FaithfulnessReport:
    faithfulness_score: Optional[float]
    supported_claims: int
    total_claims: int
    unsupported_claims: int
    numeric_mismatch_claims: int
    claims: List[ClaimAudit] = field(default_factory=list)
    negation_mismatch_claims: int = 0
    entity_mismatch_claims: int = 0
    uncited_claims: int = 0
    refused_or_abstained: bool = False


# ---------------------------------------------------------------------------
# Text normalization & feature extraction helpers
# ---------------------------------------------------------------------------


def _stem(token: str) -> str:
    if len(token) <= 3:
        return token
    for suf in ("ization", "ation", "ments", "ment", "ness", "ities", "ity", "ingly", "edly", "ing", "ied", "ies", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            if suf in ("ied", "ies"):
                token = token[:-3] + "y"
            else:
                token = token[: -len(suf)]
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def _extract_chunk_texts(chunk_obj: Any) -> Tuple[str, str]:
    """Return `(raw_text, context_text)` for a chunk string, mapping, or dataclass."""
    if chunk_obj is None:
        return "", ""
    if isinstance(chunk_obj, str):
        return chunk_obj, chunk_obj
    if isinstance(chunk_obj, Mapping):
        raw = str(chunk_obj.get("raw_text") or chunk_obj.get("text") or chunk_obj.get("context_prefixed_text") or "")
        ctx = str(chunk_obj.get("context_prefixed_text") or raw)
        title = str(chunk_obj.get("paper_title") or "")
        section = str(chunk_obj.get("section_path") or "")
        pid = str(chunk_obj.get("paper_id") or "")
    else:
        raw = str(
            getattr(chunk_obj, "raw_text", None)
            or getattr(chunk_obj, "text", None)
            or getattr(chunk_obj, "context_prefixed_text", None)
            or ""
        )
        ctx = str(getattr(chunk_obj, "context_prefixed_text", None) or raw)
        title = str(getattr(chunk_obj, "paper_title", None) or "")
        section = str(getattr(chunk_obj, "section_path", None) or "")
        pid = str(getattr(chunk_obj, "paper_id", None) or "")
    pid_hint = re.sub(r"([a-z])(\d)", r"\1 \2", pid.replace("_", " ")) if pid else ""
    extras = [x for x in (title, section, pid_hint) if x and x not in ctx]
    if extras:
        ctx = f"{' | '.join(extras)}\n{ctx}".strip()
    return raw, ctx


def _is_factual_claim(unit: str) -> bool:
    """True when `unit` asserts a factual statement rather than headings/boilerplate."""
    body = CITATION_RE.sub("", unit).strip()
    if not body or body.endswith(":") or body.startswith("#") or set(body) <= set("-=*|$` "):
        return False
    if is_claim(unit):
        return True
    alnum = _WORD_TOKEN_RE.findall(body)
    alpha = _ALPHA_WORD_RE.findall(body)
    has_quant_or_sym = bool(_NUM_RE.search(body) or _SYMBOL_RE.search(body))
    if len(alnum) >= 3 and (has_quant_or_sym or (CITATION_RE.search(unit) and len(alpha) >= 2)):
        return True
    return False


def _strip_claim_framing(unit: str) -> str:
    body = CITATION_RE.sub("", unit).strip()
    body = _BULLET_PREFIX_RE.sub("", body).strip()
    body = _VERBALIZER_AGG_COUNT_RE.sub(r"\1\2", body)
    body = _VERBALIZER_PREFIX_RE.sub("", body).strip()
    return body


def _strip_numeric_framing(body: str) -> str:
    return _VERBALIZER_AGG_COUNT_RE.sub(r"\1\2", body)


def _parse_number(raw: str) -> Optional[Tuple[float, str]]:
    m = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)\s*([a-z%]*)", raw.strip().lower())
    if not m:
        return None
    val_str, unit_raw = m.group(1).replace(",", ""), m.group(2)
    try:
        val = float(val_str)
    except ValueError:
        return None
    unit = _UNIT_NORM.get(unit_raw, unit_raw)
    return val, unit


def _extract_numbers(text: str) -> List[Tuple[str, float, str]]:
    """Extract `(raw_text, numeric_value, normalized_unit)` tokens from text."""
    # Capture range endpoints sharing a trailing unit ("2-4x", "60-80%") first.
    range_first_units: Dict[int, str] = {}
    for m in _RANGE_NUM_RE.finditer(text):
        range_first_units[m.start(1)] = _UNIT_NORM.get(m.group(3).lower(), m.group(3).lower())

    items: List[Tuple[str, float, str]] = []
    for m in _NUM_RE.finditer(text):
        parsed = _parse_number(m.group(0))
        if parsed is None:
            continue
        val, unit = parsed
        if not unit and m.start() in range_first_units:
            unit = range_first_units[m.start()]
        items.append((m.group(0).strip(), val, unit))
    return items


def _numbers_match(claim_num: Tuple[float, str], chunk_nums: Sequence[Tuple[str, float, str]]) -> bool:
    c_val, c_unit = claim_num
    for _, k_val, k_unit in chunk_nums:
        if c_unit == k_unit and math.isclose(c_val, k_val, rel_tol=1e-6, abs_tol=1e-9):
            return True
        if c_unit in _UNIT_MULT and k_unit in _UNIT_MULT:
            if math.isclose(c_val * _UNIT_MULT[c_unit], k_val * _UNIT_MULT[k_unit], rel_tol=1e-6, abs_tol=1e-9):
                return True
    return False


def _norm_symbol(sym: str) -> str:
    s = sym.strip().strip("$").lower()
    s = re.sub(r"\\(?:mathrm|text|mathbf|mathcal)\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(
        r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|psi|omega)\b",
        r"\1",
        s,
    )
    s = s.replace(r"\cdot", "")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    return s


def _extract_symbols(text: str) -> List[str]:
    found: List[str] = []
    for m in _SYMBOL_RE.finditer(text):
        raw = m.group(0).strip(".,;:")
        if len(raw) >= 2 and raw not in found:
            found.append(raw)
    return found


def _symbol_present(sym: str, norm_chunk: str) -> bool:
    ns = _norm_symbol(sym)
    if not ns:
        return True
    if ns in norm_chunk:
        return True
    if ns.startswith("1/sqrt(") and ns[2:] in norm_chunk and "/" in norm_chunk:
        return True
    return False


def _initials_match(acronym: str, norm_ctx: str) -> bool:
    """True when consecutive content words in `norm_ctx` have initials spelling `acronym`."""
    acr = acronym.lower()
    if not acr.isalpha() or not (2 <= len(acr) <= 6):
        return False
    words = re.findall(r"[a-z]+", norm_ctx)
    n = len(acr)
    for i in range(len(words) - n + 1):
        if "".join(words[i + j][0] for j in range(n)) == acr:
            return True
    return False


def _phrase_in_text(phrase: str, norm_ctx: str) -> bool:
    np = normalize_phrase(phrase)
    if not np:
        return True
    candidates = [np]
    compact = re.sub(r"\s+(?=\d+\b)", "", np)
    if compact != np and len(np.split()) == 2 and len(np.split()[0]) <= 3:
        candidates.append(compact)
    if np.endswith("s") and len(np) > 3:
        candidates.append(np[:-1])
    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(cand) + r"(?![a-z0-9])", norm_ctx) is not None
        for cand in candidates
    )


def _entity_grounded(entity: str, norm_ctx: str, chunk_stems: Optional[Set[str]] = None) -> bool:
    if _phrase_in_text(entity, norm_ctx):
        return True
    low = normalize_phrase(entity)
    if low in _ACRONYM_EXPANSIONS and any(_phrase_in_text(exp, norm_ctx) for exp in _ACRONYM_EXPANSIONS[low]):
        return True
    for acr, exps in _ACRONYM_EXPANSIONS.items():
        if any(low == exp for exp in exps) and _phrase_in_text(acr, norm_ctx):
            return True
    if entity.isupper() and entity.isalpha() and _initials_match(entity, norm_ctx):
        return True
    # Single capitalized concept nouns (non-versioned, non-CamelCase, non-acronym) can match via stem ("Uptraining" <-> "uptrained").
    if (
        chunk_stems is not None
        and " " not in low
        and "-" not in entity
        and not any(ch.isdigit() for ch in entity)
        and not entity.isupper()
        and sum(1 for ch in entity if ch.isupper()) <= 1
    ):
        stems = _content_stems(entity)
        if stems and all(s in chunk_stems for s in stems):
            return True
    # Multi-word descriptive concept phrases (e.g. "Sinusoidal Positional Encoding", "KV Cache Memory Bandwidth Bottleneck")
    # that are not versioned/CamelCase model names: match if acronym initials match or strong majority of stems appear.
    words = low.split()
    if (
        chunk_stems is not None
        and len(words) >= 2
        and not any(ch.isdigit() for ch in entity)
    ):
        initials = "".join(w[0] for w in words if w)
        if 2 <= len(initials) <= 6 and _phrase_in_text(initials, norm_ctx):
            return True
        stems = _content_stems(entity)
        if stems and (sum(1 for s in stems if s in chunk_stems) / len(stems)) >= 0.75:
            return True
    return False


def _extract_entities(body: str) -> List[str]:
    """Extract technical entities, acronyms, versioned names, and proper nouns from `body`."""
    entities: List[str] = []
    covered_spans: List[Tuple[int, int]] = []

    def add(ent: str, span: Optional[Tuple[int, int]] = None) -> None:
        ent = ent.strip(".,;:()'\"")
        if ent and ent not in entities:
            entities.append(ent)
        if span is not None:
            covered_spans.append(span)

    def in_covered(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in covered_spans)

    # 1. Versioned models & external systems via repo subject gate (e.g., GPT-4, Mixtral, DeepSeek-V3).
    for name in extract_subject_names(body):
        add(name)

    # 2. Alphanumeric identifiers with letters+digits (DeepSeek-V2, H800, FP8, GQA-8, LLaMA-13B).
    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b", body):
        tok = m.group(0)
        if any(ch.isalpha() for ch in tok) and any(ch.isdigit() for ch in tok):
            if not re.fullmatch(r"\d+(?:st|nd|rd|th)", tok, re.I):
                add(tok, m.span())

    # 3. ALL-CAPS acronyms (MLA, GQA, PPO, GRPO, MMLU, BLEU, KV).
    for m in re.finditer(r"\b[A-Z]{2,8}s?\b", body):
        tok = m.group(0)
        base = tok[:-1] if tok.endswith("s") and len(tok) > 2 else tok
        if base in _GENERIC_ACRONYMS:
            continue
        add(tok, m.span())

    # 4. Mixed-case / CamelCase technical terms (PagedAttention, LLaMA, SwiGLU, RMSNorm).
    for m in re.finditer(r"\b[A-Za-z]*[A-Z][A-Za-z]*[A-Z][A-Za-z0-9-]*\b", body):
        tok = m.group(0)
        if not tok.isupper() and any(ch.islower() for ch in tok):
            add(tok, m.span())

    # 5. Multi-word Title-Case / hyphenated phrases (e.g. "Rotary Position Embedding", "Group Relative Policy Optimization").
    for m in re.finditer(r"\b([A-Z][A-Za-z-]+(?:\s+(?:[A-Z][A-Za-z-]+|of|and|in|on|for|to|from)){1,6})\b", body):
        phrase = m.group(1)
        cap_words = [w for w in phrase.split() if w[0].isupper() and w.lower() not in _NON_ENTITY_PROPER]
        if len(cap_words) >= 2:
            # Trim leading non-entity determiners/starters if any (e.g. "Both DeepSeek").
            parts = phrase.split()
            while parts and parts[0].lower() in _NON_ENTITY_PROPER:
                parts.pop(0)
            if len([w for w in parts if w[0].isupper()]) >= 2:
                add(" ".join(parts), m.span())

    # 6. Single proper nouns not already covered by a multi-word phrase or CamelCase/versioned entity.
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:-[A-Z][a-z]+)*)\b", body):
        if in_covered(m.start(), m.end()):
            continue
        word = m.group(1)
        low = word.lower()
        if low in _NON_ENTITY_PROPER:
            continue
        prefix = body[: m.start()].rstrip()
        at_start = not prefix or prefix[-1] in (".", "!", "?", ":", '"', "\u201c")
        if not at_start or not _COMMON_STARTER_SUFFIX_RE.search(word):
            add(word, m.span())

    return entities


def _content_stems(text: str) -> List[str]:
    norm = normalize_phrase(text)
    toks = _WORD_TOKEN_RE.findall(norm)
    out: List[str] = []
    for t in toks:
        if len(t) < 2 or t.isdigit() or t in _STOPWORDS:
            continue
        out.append(_stem(t))
    return out


def _expanded_chunk_stems(raw_ctx: str) -> Set[str]:
    stems = set(_content_stems(raw_ctx))
    norm = normalize_phrase(raw_ctx)
    for acr, exps in _ACRONYM_EXPANSIONS.items():
        if _phrase_in_text(acr, norm):
            for exp in exps:
                stems.update(_content_stems(exp))
        elif any(_phrase_in_text(exp, norm) for exp in exps):
            stems.add(_stem(acr))
    return stems


def _split_polarity_clauses(text: str) -> List[Tuple[str, bool]]:
    """Split text into `(clause_text, is_negative)` pairs preserving `rather than / without` polarity."""
    out: List[Tuple[str, bool]] = []
    text = _UNLIKE_COMMA_SPLIT_RE.sub(r"\1; ", text)
    parts = re.split(
        r"(?=[;.\n])|(?=\b(?:rather\s+than|instead\s+of|without|unlike)\b)|"
        r",?\s*\b(?:and|or)\s+(?=(?:adds?|requires?|uses?|does|do|did|is|are|was|were|has|have|can|will|no|not|never|it|they)\b)|"
        r",\s*(?=(?:whereas|while|however|but|although|though|yet)\b)",
        text,
        flags=re.I,
    )
    for part in parts:
        c = part.strip(" ;.,\n")
        if len(c) < 8:
            continue
        cleaned = _AFFIRMATIVE_IDIOM_RE.sub(" ", c)
        cues = _NEGATION_CUE_RE.findall(cleaned)
        neg = bool(cues) or bool(_NEGATIVE_PREFIX_CLAUSE_RE.match(c))
        out.append((c, neg))
    return out


def _has_negation_mismatch(claim_body: str, raw_chunk_text: str) -> bool:
    norm_claim = re.sub(r"\s+([.,;!?])", r"\1", claim_body).strip(" .")
    norm_chunk = re.sub(r"\s+([.,;!?])", r"\1", raw_chunk_text)
    if normalize_phrase(norm_claim) in normalize_phrase(norm_chunk):
        return False
    claim_clauses = _split_polarity_clauses(claim_body)
    chunk_clauses = _split_polarity_clauses(raw_chunk_text)
    if not claim_clauses or not chunk_clauses:
        return False
    chunk_clause_stems = [(set(_content_stems(k_text)), k_neg) for k_text, k_neg in chunk_clauses]
    chunk_clause_stems = [(ks, kn) for ks, kn in chunk_clause_stems if ks]
    for c_text, c_neg in claim_clauses:
        c_stems = set(_content_stems(c_text))
        if len(c_stems) < 2:
            continue
        for k_stems, k_neg in chunk_clause_stems:
            if k_neg == c_neg:
                continue
            shared = c_stems & k_stems
            if len(shared) < 2:
                continue
            if (len(shared) / len(c_stems)) < 0.40 and (len(shared) / len(k_stems)) < 0.60:
                continue
            # Only flag contradiction if no matching-polarity clause already covers those exact shared stems.
            if not any(shared <= p_stems for p_stems, p_neg in chunk_clause_stems if p_neg == c_neg):
                return True
    return False


def _score_entailment(scorer: Any, claim_text: str, evidence_text: str) -> Optional[float]:
    if scorer is None:
        return None
    if callable(scorer):
        raw = scorer(claim_text, evidence_text)
    elif hasattr(scorer, "score"):
        vals = scorer.score(claim_text, [evidence_text])
        raw = vals[0] if vals else 0.0
    else:
        raise TypeError("entailment_scorer must be callable(claim, evidence) or implement .score(claim, [evidence])")

    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, Mapping):
        if "entailment" in raw:
            ent = float(raw["entailment"])
            contra = float(raw.get("contradiction", 0.0))
            return 0.0 if contra > ent else ent
        label = str(raw.get("label", "")).lower()
        score = float(raw.get("score", 0.5))
        if "contradict" in label or "neutral" in label or "non" in label:
            return min(score, 0.25) if "contradict" in label else min(score, 0.45)
        return score
    return float(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_faithfulness(
    answer: str,
    chunks_by_id: Mapping[str, Any],
    *,
    refused: bool = False,
    abstained: bool = False,
    entailment_scorer: Optional[Any] = None,
    min_lexical_overlap: float = 0.35,
    min_entailment_score: float = 0.50,
) -> FaithfulnessReport:
    """Verify each factual claim in `answer` against its cited chunks in `chunks_by_id`.

    Returns `faithfulness_score = None` for refusals, abstentions, or 0 factual claims.
    """
    if refused or abstained or not answer or not answer.strip() or is_abstention(answer):
        return FaithfulnessReport(
            faithfulness_score=None,
            supported_claims=0,
            total_claims=0,
            unsupported_claims=0,
            numeric_mismatch_claims=0,
            claims=[],
            refused_or_abstained=True,
        )

    normalized_answer = _DISPLAY_MATH_MULTILINE_RE.sub(
        lambda m: "$$ " + " ".join(m.group(1).splitlines()) + " $$",
        answer,
    )
    units = split_units(normalized_answer)

    resolved_lookup: Dict[str, Tuple[str, str]] = {}
    for cid, obj in chunks_by_id.items():
        raw, ctx = _extract_chunk_texts(obj)
        if raw.strip() or ctx.strip():
            resolved_lookup[cid] = (raw, ctx)

    unit_records: List[Dict[str, Any]] = []
    any_missing_citation = False
    for idx, u in enumerate(units):
        cited = list(dict.fromkeys(CITATION_RE.findall(u)))
        missing = [c for c in cited if c not in resolved_lookup]
        if missing:
            any_missing_citation = True
        unit_records.append(
            {
                "index": idx,
                "text": u,
                "is_claim": _is_factual_claim(u),
                "cited_ids": [c for c in cited if c not in missing],
                "missing_ids": missing,
                "raw_cited_ids": cited,
            }
        )

    # Paragraph-level forward citation attribution matching src/generation/citation_validator.py.
    if not any_missing_citation:
        for i in range(len(unit_records) - 1):
            cur = unit_records[i]
            if cur["is_claim"] and not cur["cited_ids"] and not cur["missing_ids"]:
                for nxt in unit_records[i + 1 :]:
                    if nxt["missing_ids"]:
                        break
                    if nxt["cited_ids"]:
                        cur["cited_ids"] = list(nxt["cited_ids"])
                        cur["raw_cited_ids"] = list(nxt["cited_ids"])
                        break

    audits: List[ClaimAudit] = []
    for rec in unit_records:
        if not rec["is_claim"]:
            continue

        cited_ids: List[str] = rec["raw_cited_ids"]
        resolved_ids: List[str] = rec["cited_ids"]
        missing_ids: List[str] = rec["missing_ids"]
        claim_text: str = rec["text"]

        if not cited_ids:
            audits.append(
                ClaimAudit(
                    index=rec["index"],
                    text=claim_text,
                    cited_ids=[],
                    resolved_chunk_ids=[],
                    missing_chunk_ids=[],
                    supported=False,
                    reasons=["uncited_claim"],
                )
            )
            continue

        if missing_ids or not resolved_ids:
            audits.append(
                ClaimAudit(
                    index=rec["index"],
                    text=claim_text,
                    cited_ids=cited_ids,
                    resolved_chunk_ids=resolved_ids,
                    missing_chunk_ids=missing_ids,
                    supported=False,
                    reasons=["missing_cited_chunk"],
                )
            )
            continue

        combined_raw = "\n".join(resolved_lookup[cid][0] for cid in resolved_ids)
        combined_ctx = "\n".join(resolved_lookup[cid][1] for cid in resolved_ids)
        norm_ctx = normalize_phrase(combined_ctx)
        norm_sym_chunk = _norm_symbol(combined_raw)

        claim_body = _strip_claim_framing(claim_text)
        claim_num_text = _strip_numeric_framing(claim_body)

        # 1. Numbers & quantities check against raw chunk text (plus title-backed model versions).
        chunk_nums = _extract_numbers(combined_raw)
        claim_nums = _extract_numbers(claim_num_text)
        chunk_stems = _expanded_chunk_stems(combined_ctx)
        # Numbers appearing inside a grounded entity/version name (e.g. "Llama 3" in paper_title) are valid.
        grounded_entity_nums: Set[str] = set()
        for ent in extract_subject_names(claim_body):
            if _entity_grounded(ent, norm_ctx, chunk_stems):
                for raw_n, _, _ in _extract_numbers(ent):
                    grounded_entity_nums.add(raw_n)

        unmatched_nums = [
            raw_n
            for raw_n, val, unit in claim_nums
            if raw_n not in grounded_entity_nums and not _numbers_match((val, unit), chunk_nums)
        ]

        # 2. Math symbols / equations check.
        unmatched_syms = [s for s in _extract_symbols(claim_body) if not _symbol_present(s, norm_sym_chunk)]

        # 3. Technical entities / acronyms check.
        unmatched_ents = [e for e in _extract_entities(claim_body) if not _entity_grounded(e, norm_ctx, chunk_stems)]

        # 4. Negation / contradiction polarity check.
        neg_mismatch = _has_negation_mismatch(claim_body, combined_raw)

        # 5. Content-word lexical coverage (plus optional NLI entailment scorer).
        claim_stems = _content_stems(claim_body)
        if claim_stems:
            matched_stems = sum(1 for s in claim_stems if s in chunk_stems)
            lex_overlap = matched_stems / len(claim_stems)
        else:
            lex_overlap = 1.0

        ent_score = _score_entailment(entailment_scorer, claim_body, combined_raw)

        numeric_mismatch = bool(unmatched_nums) or any(any(ch.isdigit() for ch in s) for s in unmatched_syms)
        entity_mismatch = bool(unmatched_ents) or bool(unmatched_syms)

        reasons: List[str] = []
        if numeric_mismatch:
            reasons.append(f"numeric_mismatch({', '.join(unmatched_nums or unmatched_syms)})")
        if unmatched_syms and not numeric_mismatch:
            reasons.append(f"symbol_mismatch({', '.join(unmatched_syms)})")
        if unmatched_ents:
            reasons.append(f"entity_mismatch({', '.join(unmatched_ents)})")
        if neg_mismatch:
            reasons.append("negation_mismatch")

        if entailment_scorer is not None and ent_score is not None:
            if ent_score < min_entailment_score:
                reasons.append(f"entailment_below_threshold({ent_score:.2f})")
        elif lex_overlap < min_lexical_overlap:
            reasons.append(f"low_lexical_grounding({lex_overlap:.2f})")

        supported = not reasons
        audits.append(
            ClaimAudit(
                index=rec["index"],
                text=claim_text,
                cited_ids=cited_ids,
                resolved_chunk_ids=resolved_ids,
                missing_chunk_ids=[],
                supported=supported,
                numeric_mismatch=numeric_mismatch,
                negation_mismatch=neg_mismatch,
                entity_mismatch=entity_mismatch,
                lexical_overlap=round(lex_overlap, 4),
                entailment_score=round(ent_score, 4) if ent_score is not None else None,
                unmatched_numbers=unmatched_nums,
                unmatched_symbols=unmatched_syms,
                unmatched_entities=unmatched_ents,
                reasons=reasons,
            )
        )

    total = len(audits)
    if total == 0:
        return FaithfulnessReport(
            faithfulness_score=None,
            supported_claims=0,
            total_claims=0,
            unsupported_claims=0,
            numeric_mismatch_claims=0,
            claims=[],
        )

    supported_count = sum(1 for a in audits if a.supported)
    return FaithfulnessReport(
        faithfulness_score=round(supported_count / total, 4),
        supported_claims=supported_count,
        total_claims=total,
        unsupported_claims=total - supported_count,
        numeric_mismatch_claims=sum(1 for a in audits if a.numeric_mismatch),
        claims=audits,
        negation_mismatch_claims=sum(1 for a in audits if a.negation_mismatch),
        entity_mismatch_claims=sum(1 for a in audits if a.entity_mismatch),
        uncited_claims=sum(1 for a in audits if "uncited_claim" in a.reasons),
        refused_or_abstained=False,
    )
