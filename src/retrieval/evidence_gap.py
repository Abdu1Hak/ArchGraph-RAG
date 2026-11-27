"""Bounded 2-Turn Evidence-Gap Refinement for Multi-Hop Retrieval.

Inspects Pass-1 retrieved text (`vector_chunks + supporting_chunks`) against the
question's clauses, explicit/discovered entities, symbolic math terms (`W^{DKV}`),
comparisons, and quantitative asks (`how much`, `how many`, `how long`, etc.).

Hard-bounded at `max_turns = 2` total (at most ONE Pass-2 refinement turn).
Skips `OUT_OF_SCOPE` routes and queries already satisfied by Pass-1 evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.retrieval.cypher_templates import TemplateResult
from src.stores.pgvector_store import RetrievedChunk, symbol_tokens, tokenize

HARD_MAX_TURNS = 2

_CLAUSE_SENT_RE = re.compile(r"[.?!;]+")
_CLAUSE_CONJ_RE = re.compile(
    r",?\s+(?:and|or|while|whereas|then)\s+(?=(?:"
    r"what|which|how|why|when|where|whose|who|does|did|do|is|are|was|were|can|could|would|should|"
    r"has|have|give|show|list|name|explain|describe|compare|state|quote|report|"
    r"by\s+how|in\s+which|at\s+what)\b)",
    re.I,
)
_CLAUSE_COMMA_WH_RE = re.compile(
    r",\s+(?=(?:what|which|how|why|when|where|whose|who|does|did|do|is|are|was|were|can|"
    r"by\s+how|in\s+which|at\s+what)\b)",
    re.I,
)
_PAREN_NOTE_RE = re.compile(r"\([^)]*\b(?:do not confuse|note that|not to be confused)\b[^)]*\)", re.I)
_SYMBOL_MATH_RE = re.compile(r"[^\s?!,;()]*[\^_{}\\][^\s?!,;()]*")
_QUANT_ASK_RE = re.compile(
    r"\b(?:how\s+(?:many|much|long|far|often|fast|large|small|old)|"
    r"by\s+how\s+much|what\s+(?:fraction|percent(?:age)?|share|accuracy|score|ratio|size|length|number))\b",
    re.I,
)
_HEAD_WH_SPLIT_RE = re.compile(r"\s+(?:for|that\s+(?:were|was|are|is)|which\s+(?:were|was|are|is))\s+", re.I)
_BRIDGE_CUE_RE = re.compile(
    r"\b(?:predecessor|precursor|earlier|checkpoint|validated|distilled|complementary|"
    r"generalization|ancestor|built\s+on|inherited)\b",
    re.I,
)
_EQUATION_ASK_RE = re.compile(r"\b(?:equation|formula)\b", re.I)
_HAS_EQUATION_RE = re.compile(r"[=]|\\frac|\\sqrt|\$\$")
_MODEL_VERSION_RE = re.compile(
    r"\b(?:llama|gpt|deepseek|qwen|gemma|claude|gemini|mistral|gqa)[ -]*"
    r"(?:v?\d+(?:\.\d+)?|r\d+(?:[- ]zero)?)|\b(?:fp|bf|int)\d+\b|\b(?:19|20)\d\d\b|\b34b\b|\b70b\b|\b405b\b",
    re.I,
)
_NUMBER_WORD_RE = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
_HAS_DIGIT_RE = re.compile(r"\d")

_GENERIC_QUESTION_WORDS: frozenset[str] = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "does", "did", "do", "is", "are", "was", "were", "be", "been", "being",
    "can", "could", "would", "should", "may", "might", "must", "will", "have", "has", "had",
    "give", "show", "tell", "list", "name", "names", "naming", "state", "report", "reports",
    "explain", "describe", "described", "compare", "comparison", "quote", "verbatim", "exact",
    "trace", "lineage", "evolution", "evolve", "evolv", "history",
    "provide", "mention", "mentions", "mentioned", "say", "says", "call", "called",
    "need", "needs", "needed", "require", "requires", "required", "use", "uses", "used", "using",
    "make", "makes", "made", "take", "takes", "took", "get", "gets", "got", "put", "puts",
    "keep", "keeps", "kept", "let", "lets", "become", "becomes", "turn", "turns", "turned",
    "pair", "pairs", "paired", "knob", "knobs", "strike", "strikes", "go", "goes", "wrong",
    "happen", "happens", "mean", "means", "still", "also", "only", "just", "even", "actually",
    "roughly", "about", "around", "much", "many", "more", "most", "far", "very", "well", "such",
    "else", "other", "another", "first", "second", "third", "original", "classic", "new", "extra",
    "own", "same", "each", "both", "all", "every", "any", "some", "several", "two", "one",
    "single", "distinct", "between", "among", "across", "within", "without", "through", "into",
    "onto", "from", "than", "like", "here", "there", "these", "those", "this", "that", "its",
    "their", "they", "them", "itself", "instead", "rather", "once", "since", "while", "whereas",
    "whether", "before", "after", "during", "until", "against", "relative", "aside", "besides",
    "corpus", "indexed", "paper", "papers", "study", "work", "research", "model", "models",
    "family", "families", "series", "generation", "version", "herd", "flagship",
    "method", "methods", "technique", "techniques", "approach", "approaches", "scheme",
    "mechanism", "mechanisms", "component", "components", "variant", "variants",
    "design", "designs", "recipe", "strategy", "algorithm", "system", "systems",
    "step", "steps", "part", "parts", "thing", "things", "way", "ways", "kind", "type",
    "role", "feature", "detail", "details", "problem", "issue", "number", "amount", "share",
    "fraction", "percent", "percentage", "value", "values", "level", "size", "data",
    "publicly", "available", "foundation", "based", "existing", "pure", "main", "core",
    "meta", "google", "deepmind", "2017", "2021", "2022", "2023", "2024", "2025",
    "s", "t",
})

_SYNONYM_MAP: Dict[str, str] = {
    "reduc": "_pred_reduce",
    "cut": "_pred_reduce",
    "mitigat": "_pred_reduce",
    "reliev": "_pred_reduce",
    "alleviat": "_pred_reduce",
    "shrink": "_pred_reduce",
    "decreas": "_pred_reduce",
    "sav": "_pred_reduce",
    "replac": "_pred_remove",
    "swap": "_pred_remove",
    "eliminat": "_pred_remove",
    "remov": "_pred_remove",
    "drop": "_pred_remove",
    "foreg": "_pred_remove",
    "forego": "_pred_remove",
    "discard": "_pred_remove",
    "rid": "_pred_remove",
    "avoid": "_pred_remove",
    "extend": "_pred_extend",
    "grow": "_pred_extend",
    "stretch": "_pred_extend",
    "expand": "_pred_extend",
    "alter": "_pred_change",
    "chang": "_pred_change",
    "modif": "_pred_change",
    "modify": "_pred_change",
    "drift": "_pred_move",
    "mov": "_pred_move",
    "move": "_pred_move",
    "conversion": "convert",
    "convers": "convert",
    "serv": "_pred_serve",
    "inferenc": "_pred_serve",
    "deploy": "_pred_serve",
    "fast": "_pred_speed",
    "faster": "_pred_speed",
    "speed": "_pred_speed",
    "throughput": "_pred_speed",
    "beat": "_pred_beat",
    "outperform": "_pred_beat",
    "exceed": "_pred_beat",
    "ahead": "_pred_beat",
    "predecessor": "_pred_prior",
    "precursor": "_pred_prior",
    "earlier": "_pred_prior",
    "previous": "_pred_prior",
    "older": "_pred_prior",
    "ancestor": "_pred_prior",
    "comput": "_pred_cost",
    "cost": "_pred_cost",
    "complex": "_pred_cost",
    "complexity": "_pred_cost",
    "budget": "_pred_cost",
    "long": "_pred_length",
    "length": "_pred_length",
}


def _stem_token(token: str) -> str:
    for suf in ("ations", "ation", "ition", "ments", "ment", "ities", "ity", "ings", "ing", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            token = token[: -len(suf)]
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return _SYNONYM_MAP.get(token, token)


def _canonical_terms(text: str, drop_generic: bool = False) -> List[str]:
    out: List[str] = []
    for tok in tokenize(text):
        if drop_generic and tok in _GENERIC_QUESTION_WORDS:
            continue
        st = _stem_token(tok)
        if drop_generic and st in _GENERIC_QUESTION_WORDS:
            continue
        if st not in out:
            out.append(st)
    return out


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_\u2013\u2014]", " ", text.lower())).strip()


def _has_quantitative_value(raw_text: str) -> bool:
    stripped = _MODEL_VERSION_RE.sub(" ", raw_text)
    return bool(_HAS_DIGIT_RE.search(stripped) or _NUMBER_WORD_RE.search(stripped))


def split_question_clauses(question: str) -> List[str]:
    """Split a compound or multi-part question into meaningful sub-clauses."""
    cleaned = _PAREN_NOTE_RE.sub(" ", question).strip()
    clauses: List[str] = []
    for sent in _CLAUSE_SENT_RE.split(cleaned):
        sent = sent.strip(" ,")
        if not sent:
            continue
        for part in _CLAUSE_CONJ_RE.split(sent):
            for sub in _CLAUSE_COMMA_WH_RE.split(part):
                sub = sub.strip(" ,")
                if sub and len(tokenize(sub)) >= 2:
                    clauses.append(sub)
    return clauses or ([cleaned] if cleaned else [])


@dataclass(frozen=True)
class EvidenceGapAssessment:
    """Result of inspecting Pass-1 retrieved context for missing evidence."""

    has_gap: bool
    gap_query: Optional[str] = None
    gap_seed_entities: Tuple[str, ...] = ()
    missing_clauses: Tuple[str, ...] = ()
    missing_entities: Tuple[str, ...] = ()
    missing_symbols: Tuple[str, ...] = ()
    reason: str = ""


class EvidenceGapRefiner:
    """Detects evidence gaps in Pass-1 context and generates 1 focused Pass-2 refinement query."""

    def __init__(self, max_turns: int = HARD_MAX_TURNS) -> None:
        # Hard-bounded at 2 total turns (at most 1 Pass-2 refinement turn).
        self.max_turns = min(max(1, int(max_turns)), HARD_MAX_TURNS)

    @staticmethod
    def _entity_label(eid: str, entity_names: Optional[Mapping[str, str]]) -> str:
        if entity_names and eid in entity_names:
            return entity_names[eid]
        return eid.split("::", 1)[-1].replace("_", " ")

    @staticmethod
    def _entity_aliases(eid: str, resolver: Any, entity_names: Optional[Mapping[str, str]]) -> List[str]:
        names: List[str] = []
        if resolver is not None and hasattr(resolver, "nodes") and eid in resolver.nodes:
            node = resolver.nodes[eid]
            names.append(node.canonical_name)
            names.extend(node.aliases)
        elif entity_names and eid in entity_names:
            names.append(entity_names[eid])
        else:
            names.append(eid.split("::", 1)[-1].replace("_", " "))
        return list(dict.fromkeys(n for n in names if n))

    @staticmethod
    def _entity_in_text(names: Sequence[str], lowered_text: str) -> bool:
        norm_text = _normalize_phrase(lowered_text)
        for name in names:
            norm = _normalize_phrase(name)
            if not norm:
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(norm) + r"(?![a-z0-9])", norm_text):
                return True
        return False

    def assess(
        self,
        question: str,
        decision: Any,
        pass1_chunks: Sequence[RetrievedChunk],
        template_result: Optional[TemplateResult] = None,
        entity_names: Optional[Mapping[str, str]] = None,
        resolver: Any = None,
        current_turn: int = 1,
    ) -> EvidenceGapAssessment:
        """Inspect Pass-1 chunks vs question; return a focused Pass-2 query if a gap exists."""
        if current_turn >= self.max_turns or self.max_turns < 2:
            return EvidenceGapAssessment(has_gap=False, reason="max_turns_reached")

        route = getattr(decision, "route", None)
        route_val = getattr(route, "value", str(route))
        if route_val == "OUT_OF_SCOPE":
            return EvidenceGapAssessment(has_gap=False, reason="out_of_scope")

        q = (question or "").strip()
        if not q:
            return EvidenceGapAssessment(has_gap=False, reason="empty_question")

        unique_chunks: List[RetrievedChunk] = list({c.chunk_id: c for c in pass1_chunks}.values())
        if not unique_chunks:
            seeds = tuple(e for e in (getattr(decision, "resolved_entity_ids", None) or ()) if not str(e).startswith("paper::"))
            return EvidenceGapAssessment(
                has_gap=True,
                gap_query=getattr(decision, "vector_subquery", None) or q,
                gap_seed_entities=seeds,
                reason="no_pass1_chunks",
            )

        # Build per-chunk index over Pass-1 text.
        chunk_records: List[Dict[str, Any]] = []
        all_raw_terms: Set[str] = set()
        all_entity_ids: Set[str] = set()
        entity_chunk_counts: Counter[str] = Counter()
        all_full_text_parts: List[str] = []

        for ch in unique_chunks:
            raw_terms = set(_canonical_terms(ch.raw_text, drop_generic=False))
            ent_terms: Set[str] = set()
            for eid in ch.entity_ids:
                all_entity_ids.add(eid)
                if not str(eid).startswith("paper::"):
                    entity_chunk_counts[eid] += 1
                for nm in self._entity_aliases(eid, resolver, entity_names):
                    ent_terms.update(_canonical_terms(nm, drop_generic=False))
            paper_title_terms = set(_canonical_terms(ch.paper_title, drop_generic=False))
            chunk_records.append({
                "chunk": ch,
                "raw_terms": raw_terms,
                "match_terms": raw_terms | ent_terms | paper_title_terms,
                "raw_lower": ch.raw_text.lower(),
                "full_lower": ch.context_prefixed_text.lower(),
                "has_quant": _has_quantitative_value(ch.raw_text),
                "has_equation": bool(_HAS_EQUATION_RE.search(ch.raw_text)),
            })
            all_raw_terms.update(raw_terms)
            all_full_text_parts.append(ch.context_prefixed_text)

        all_full_text = "\n".join(all_full_text_parts)
        all_full_lower = all_full_text.lower()
        all_full_compact = re.sub(r"[{}\s]", "", all_full_lower)

        # 1. Symbolic / equation checks.
        missing_symbols: List[str] = []
        for raw_sym in _SYMBOL_MATH_RE.findall(q):
            sym = raw_sym.rstrip(".,;:?!").lower()
            if len(sym) < 3 or sym.startswith("http"):
                continue
            sym_compact = re.sub(r"[{}\s]", "", sym)
            if sym not in all_full_lower and sym_compact not in all_full_compact:
                missing_symbols.append(raw_sym.rstrip(".,;:?!"))
        for sym in symbol_tokens(q):
            if ("^" in sym or "_" in sym or "\\" in sym) and sym not in all_full_lower:
                if sym not in [s.lower() for s in missing_symbols]:
                    missing_symbols.append(sym)
        if _EQUATION_ASK_RE.search(q) and not any(rec["has_equation"] for rec in chunk_records):
            if "equation" not in missing_symbols:
                missing_symbols.append("equation")

        # 2. Explicit question entities check (only entities literally named in the question).
        resolved_ids = list(getattr(decision, "resolved_entity_ids", None) or [])
        concept_resolved = [e for e in resolved_ids if not str(e).startswith("paper::")]
        target_resolved = concept_resolved or resolved_ids
        explicit_in_q = [
            eid for eid in target_resolved
            if self._entity_in_text(self._entity_aliases(eid, resolver, entity_names), q)
        ]

        missing_entities: List[str] = []
        gap_seed_entities: List[str] = []
        for eid in explicit_in_q:
            aliases = self._entity_aliases(eid, resolver, entity_names)
            present_in_ids = eid in all_entity_ids
            present_in_text = self._entity_in_text(aliases, all_full_text)
            if not present_in_ids and not present_in_text:
                missing_entities.append(eid)
                if not str(eid).startswith("paper::") and eid not in gap_seed_entities:
                    gap_seed_entities.append(eid)

        # 3. Discovered bridge entities unexplored (e.g. DeepSeek-V2 mentioned once in passing by DeepSeek-V3).
        discovered_bridge_entities: List[str] = []
        if _BRIDGE_CUE_RE.search(q):
            graph_entities: Set[str] = set()
            if template_result is not None:
                for path in template_result.paths:
                    for hop in path:
                        graph_entities.add(hop["source"])
                        graph_entities.add(hop["target"])
            top_chunk_eids = [e for e in chunk_records[0]["chunk"].entity_ids if str(e).startswith("model_family::")]
            other_graph_eids = [e for e in sorted(graph_entities) if str(e).startswith("model_family::") and e not in top_chunk_eids]
            candidate_bridges = [
                eid
                for eid in (top_chunk_eids + other_graph_eids)
                if eid not in resolved_ids
            ]
            q_has_attr = bool(_QUANT_ASK_RE.search(q) or re.search(r"\b(?:parameters|active|total|accuracy|context|tokens)\b", q, re.I))
            if q_has_attr:
                for eid in candidate_bridges:
                    aliases = self._entity_aliases(eid, resolver, entity_names)
                    mentioning_chunks = [
                        rec for rec in chunk_records
                        if eid in rec["chunk"].entity_ids or self._entity_in_text(aliases, rec["raw_lower"])
                    ]
                    title_matching_chunks = [
                        rec for rec in chunk_records
                        if self._entity_in_text(aliases, rec["chunk"].paper_title)
                    ]
                    if len(mentioning_chunks) == 1 and not title_matching_chunks:
                        q_domain = set(_canonical_terms(q, drop_generic=True))
                        if not (len(q_domain & mentioning_chunks[0]["raw_terms"]) >= 2 and mentioning_chunks[0]["has_quant"]):
                            discovered_bridge_entities.append(eid)
                            if eid not in gap_seed_entities:
                                gap_seed_entities.append(eid)

        # 4. Clause-level coverage check.
        clauses = split_question_clauses(q)
        missing_clauses: List[str] = []
        multi_clause = len(clauses) >= 2
        q_anchor_terms: Set[str] = set()
        for eid in explicit_in_q:
            if not str(eid).startswith("paper::"):
                for nm in self._entity_aliases(eid, resolver, entity_names):
                    q_anchor_terms.update(_canonical_terms(nm, drop_generic=True))

        all_entity_name_terms: Set[str] = {
            t
            for eid in all_entity_ids
            for nm in self._entity_aliases(eid, resolver, entity_names)
            for t in _canonical_terms(nm, drop_generic=True)
        }

        for cl in clauses:
            cl_terms = _canonical_terms(cl, drop_generic=True)
            if not cl_terms:
                continue
            cl_set = set(cl_terms)
            is_quant = bool(_QUANT_ASK_RE.search(cl))

            cl_entities = [
                eid for eid in explicit_in_q
                if self._entity_in_text(self._entity_aliases(eid, resolver, entity_names), cl)
            ]

            best_chunk_overlap = max((len(cl_set & rec["match_terms"]) for rec in chunk_records), default=0)
            best_raw_overlap = max((len(cl_set & rec["raw_terms"]) for rec in chunk_records), default=0)
            global_raw_overlap = len(cl_set & (all_raw_terms | all_entity_name_terms))
            cov_ratio = global_raw_overlap / len(cl_set)

            clause_gap = False
            if cl_entities:
                ent_chunks = [
                    rec for rec in chunk_records
                    if any(
                        eid in rec["chunk"].entity_ids
                        or self._entity_in_text(self._entity_aliases(eid, resolver, entity_names), rec["full_lower"])
                        for eid in cl_entities
                    )
                ]
                if not ent_chunks:
                    clause_gap = True
                else:
                    non_anchor = [t for t in cl_terms if t not in q_anchor_terms] or cl_terms
                    best_ent_cov = max((len(set(non_anchor) & rec["raw_terms"]) / len(non_anchor) for rec in ent_chunks), default=0.0)
                    if best_ent_cov < 0.5 and cov_ratio < 0.5:
                        clause_gap = True
                    elif is_quant and not any(rec["has_quant"] and (set(non_anchor) & rec["raw_terms"]) for rec in ent_chunks):
                        clause_gap = True
            elif multi_clause:
                if len(cl_terms) >= 2 and cov_ratio < 0.48 and best_raw_overlap < math.ceil(len(cl_terms) * 0.5):
                    clause_gap = True
                elif len(cl_terms) >= 3 and best_chunk_overlap < math.ceil(len(cl_terms) * 0.5) and cov_ratio < 0.68:
                    clause_gap = True
                elif is_quant:
                    pred_terms = [t for t in cl_terms if t not in q_anchor_terms] or cl_terms
                    if not any(rec["has_quant"] and len(set(pred_terms) & rec["raw_terms"]) >= 1 for rec in chunk_records):
                        clause_gap = True
            else:
                if len(cl_terms) >= 2 and cov_ratio < 0.4 and best_chunk_overlap <= 1:
                    clause_gap = True
                elif is_quant:
                    # Focus on the WH-question head (before trailing 'for ...' / 'that ...' identification clause).
                    head_part = _HEAD_WH_SPLIT_RE.split(cl, maxsplit=1)[0]
                    head_terms = _canonical_terms(head_part, drop_generic=True)
                    pred_terms = [t for t in (head_terms or cl_terms) if t not in q_anchor_terms]
                    if pred_terms and not any(
                        rec["has_quant"] and len(set(pred_terms) & rec["raw_terms"]) >= 1
                        for rec in chunk_records
                    ):
                        clause_gap = True

            if clause_gap:
                missing_clauses.append(cl)

        has_gap = bool(missing_symbols or missing_entities or discovered_bridge_entities or missing_clauses)
        if not has_gap:
            return EvidenceGapAssessment(has_gap=False, reason="pass1_sufficient")

        # Construct a focused Pass-2 query combining missing elements and primary anchor entity.
        anchor_eids: List[str] = list(gap_seed_entities)
        for eid in explicit_in_q:
            if not str(eid).startswith("paper::") and eid not in anchor_eids:
                anchor_eids.append(eid)
        if not anchor_eids and entity_chunk_counts:
            # Pick the most frequent Pass-1 entity (primary subject across Pass-1 chunks).
            top_eid, _ = max(entity_chunk_counts.items(), key=lambda kv: (kv[1], kv[0]))
            anchor_eids.append(top_eid)
            gap_seed_entities.append(top_eid)

        anchor_names: List[str] = []
        for eid in anchor_eids[:2]:
            lbl = self._entity_label(eid, entity_names)
            if lbl not in anchor_names:
                anchor_names.append(lbl)

        query_parts: List[str] = []
        if missing_clauses:
            focused_clauses: List[str] = []
            for cl in missing_clauses[:2]:
                cl_focus = [t for t in tokenize(cl) if t not in _GENERIC_QUESTION_WORDS]
                focused_clauses.append(" ".join(cl_focus) if cl_focus else cl)
            if anchor_names and not any(
                self._entity_in_text([nm], " ".join(focused_clauses)) for nm in anchor_names
            ):
                query_parts.append(anchor_names[0])
            query_parts.extend(focused_clauses)
        elif discovered_bridge_entities:
            if anchor_names:
                query_parts.append(anchor_names[0])
            q_focus = [t for t in tokenize(q) if t not in _GENERIC_QUESTION_WORDS]
            query_parts.append(" ".join(q_focus[:6]))
        elif missing_entities:
            query_parts.append(" ".join(self._entity_label(e, entity_names) for e in missing_entities))
            q_focus = [t for t in tokenize(q) if t not in _GENERIC_QUESTION_WORDS]
            if q_focus:
                query_parts.append(" ".join(q_focus[:5]))
        if missing_symbols:
            query_parts.append(" ".join(missing_symbols))

        gap_query = re.sub(r"\s+", " ", " ".join(p for p in query_parts if p)).strip()
        if not gap_query:
            gap_query = getattr(decision, "vector_subquery", None) or q

        for eid in explicit_in_q:
            if not str(eid).startswith("paper::") and eid not in gap_seed_entities:
                gap_seed_entities.append(eid)

        reasons: List[str] = []
        if missing_symbols:
            reasons.append(f"missing_symbols={missing_symbols}")
        if missing_entities:
            reasons.append(f"missing_entities={missing_entities}")
        if discovered_bridge_entities:
            reasons.append(f"unexplored_bridge={discovered_bridge_entities}")
        if missing_clauses:
            reasons.append(f"missing_clauses={len(missing_clauses)}")

        return EvidenceGapAssessment(
            has_gap=True,
            gap_query=gap_query,
            gap_seed_entities=tuple(dict.fromkeys(gap_seed_entities)),
            missing_clauses=tuple(missing_clauses),
            missing_entities=tuple(missing_entities),
            missing_symbols=tuple(missing_symbols),
            reason="; ".join(reasons),
        )
