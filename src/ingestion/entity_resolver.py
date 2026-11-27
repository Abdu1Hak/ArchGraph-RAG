"""3-Tier Entity Resolution Engine with Hard Version-Conflict Protection (Phase 1).

Why Entity Resolution Is the Make-or-Break Step of Graph RAG:
-------------------------------------------------------------
Across 50 LLM papers, authors refer to the same architectural concept in multiple ways:
- "RoPE", "Rotary Position Embedding", "Rotary Positional Embeddings (Su et al., 2021)"
- "GQA", "Grouped-Query Attention", "grouped query attention"
- "SwiGLU", "Swish-Gated Linear Unit", "SwiGLU FFN"

If these become 3 separate nodes in Neo4j, a 2-hop query like:
  (Llama-3)-[:ADOPTS_COMPONENT]->(RoPE)<-[:EXTENDS]-(YaRN)
breaks completely if Llama-3 linked to "RoPE" and YaRN linked to "Rotary Position Embeddings".

Conversely, naive embedding similarity (>0.88) creates a catastrophic failure mode in ML papers:
- `cosine_sim(embed("LLaMA-1"), embed("LLaMA-2"))` is ~0.95!
- `cosine_sim(embed("FlashAttention"), embed("FlashAttention-2"))` is ~0.94!
- `cosine_sim(embed("DeepSeek-V2"), embed("DeepSeek-V3"))` is ~0.96!

Our 3-Tier Funnel Solves Both Directions:
-----------------------------------------
- Step 0 (Surface Normalization & Hard Guards):
  Strips inline citations `(Su et al., 2021)`, validates parenthetical acronyms
  (`Grouped-Query Attention (GQA)` -> name + alias `GQA`, but `Method A (ours)` -> no alias),
  strips dense checkpoint scales from ModelFamily names (`Llama 2 (70B)` -> `Llama 2`), and
  extracts a structured identity signature (`v:2`, `v:r1-zero`, `moe:8x7b`, `bits:4`, `fp:8`,
  `topk:1`). **Hard Rule:** two entities with conflicting signatures can NEVER merge, at any tier.
- Tier 1 (O(1) Typed Alias Dictionary Lookup):
  Matches the normalized (or space-free compact) surface form against aliases of the SAME
  `EntityType` only. A `Paper` named "DeepSeek-V2" never collapses into the `ModelFamily`.
- Tier 2 (Acronym Initials & High-Precision Lexical Match):
  Initialism match ("GQA" <-> "Grouped Query Attention") or SequenceMatcher >= 0.91.
- Tier 3 (Dense Embedding Cosine Similarity + Gray-Zone Tie-Breaker):
  `sim >= 0.88` auto-merges, `[0.82, 0.88)` asks an optional tie-breaker, else mint a new node.
  Tier 2 and Tier 3 additionally refuse "contrastive siblings" (`Method A` vs `Method B`,
  `Multi-Head Attention` vs `Multi-Query Attention`), which embeddings rate as near-identical.

Entity IDs are order-independent for acronym-bearing names: `Native Sparse Attention (NSA)` and
a bare `NSA` both anchor on the acronym slug (`arch_component::nsa`), whichever arrives first.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import yaml

from src.schema.ontology import (
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    ResolvedEntityNode,
    ResolvedGraphEdge,
)


_CITATION_NOISE_RE = re.compile(
    r"\s*\((?:[A-Z][A-Za-z\-]+\s+et\s+al\.?,?\s*\d{4}|\d{4})\)\s*$"
)
_TRAILING_PAREN_ACRONYM_RE = re.compile(
    r"^(?P<full>.+?)\s*\((?P<acro>[A-Za-z0-9\-]{2,12})\)$"
)
# Parentheticals that look like short tokens but are never acronyms of the preceding name.
_PAREN_TRAP_WORDS = {
    "ours", "our", "baseline", "baselines", "proposed", "default", "original", "new",
    "reproduced", "reimpl", "ablation", "optional", "base", "large", "small", "full",
}
_STOPWORDS = {"of", "the", "and", "with", "for", "in", "to", "a", "an", "on", "via"}

# Dense checkpoint scale tokens (7B, 70B, 1.5B), bare or parenthesized. The look-behind keeps
# MoE grids (8x7B) intact because the '7B' there is preceded by the word character 'x'.
_DENSE_SCALE_RE = re.compile(
    r"\(\s*(?P<paren>\d+(?:\.\d+)?[Bb])\s*\)|(?<![\w.])(?P<bare>\d+(?:\.\d+)?[Bb])\b"
)
_DENSE_SCALE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?[Bb]$")

# Signature extractors, applied in order on lower-cased text; each match is consumed so that
# later (more generic) patterns cannot re-read its digits.
_SIG_MOE_GRID_RE = re.compile(r"(?<![\w.])(\d+)x(\d+(?:\.\d+)?)b\b")
_SIG_INT_BITS_RE = re.compile(r"\bint(\d+)\b")
_SIG_N_BITS_RE = re.compile(r"(?<![\w.])(\d+)[\s\-]?bits?\b")
_SIG_FP_RE = re.compile(r"\bfp(\d+)\b")
_SIG_BF_RE = re.compile(r"\bbf(\d+)\b")
_SIG_TOPK_RE = re.compile(r"\btop[\s\-]?(\d+|k)\b")
_SIG_DENSE_SCALE_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?b\b")
_SIG_R_LINE_RE = re.compile(r"\br(\d+)(?:[\s\-_]+(zero))?\b")
_SIG_V_PREFIX_RE = re.compile(r"\bv(\d+(?:\.\d+)?)\b")
_SIG_GLUED_RE = re.compile(r"(?<=[a-z])(\d+(?:\.\d+)?)(?![^\s\-_])")
_SIG_SEPARATED_RE = re.compile(r"(?<![^\s\-_])(\d+(?:\.\d+)?)(?![^\s\-_])")

_MENTION_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def normalize_surface_key(text: str) -> str:
    """Normalize an entity name into a canonical lookup key."""
    cleaned = _CITATION_NOISE_RE.sub("", text.strip())
    cleaned = cleaned.lower()
    # Replace hyphens/slashes with spaces, remove punctuation except alphanumeric
    cleaned = re.sub(r"[\-/]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s\.]", "", cleaned)
    tokens = cleaned.split()
    # Singularize common trailing plural words ("embeddings" -> "embedding", "models" -> "model")
    if tokens and len(tokens[-1]) > 4 and tokens[-1].endswith("s") and not tokens[-1].endswith("ss"):
        tokens[-1] = tokens[-1][:-1]
    return " ".join(tokens).strip(".")


def compact_surface_key(text: str) -> str:
    """Space-free variant of `normalize_surface_key` ('gpt 3' == 'gpt3', 'qwen 2.5' == 'qwen2.5')."""
    return normalize_surface_key(text).replace(" ", "")


def strip_dense_scale_tokens(name: str) -> Tuple[str, List[str]]:
    """Strip dense checkpoint scales: 'Llama 2 (70B)' -> ('Llama 2', ['70B']).

    MoE expert grids ('Mixtral 8x7B') are part of the model identity and are left untouched.
    If stripping would leave nothing meaningful (e.g. the name *is* '70B'), the name is returned as is.
    """
    scales: List[str] = []

    def _take(m: re.Match) -> str:
        scales.append((m.group("paren") or m.group("bare")).upper())
        return " "

    stripped = _DENSE_SCALE_RE.sub(_take, name)
    stripped = re.sub(r"\(\s*\)", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" -_,")
    if not scales or len(stripped) < 2:
        return name, []
    return stripped, scales


def is_acronym_match(short_str: str, long_str: str) -> bool:
    """Check if `short_str` (e.g., 'GQA' or 'RoPE') is a plausible initialism of `long_str`.

    Tries hyphen-split and space-only splits ('Multi-head Latent Attention' -> MHLA or MLA),
    with and without stopwords ('Mixture of Experts' -> ME or MoE), plus a two-letter first
    syllable form ('Rotary Position Embedding' -> RoPE).
    """
    if re.search(r"\s", short_str.strip()):
        return False
    s = re.sub(r"[^a-zA-Z0-9]", "", short_str).lower()
    if len(s) < 2:
        return False
    return s in acronym_variants(long_str)


def acronym_variants(long_str: str) -> Set[str]:
    """Every initialism `is_acronym_match` accepts for `long_str`."""
    variants: Set[str] = set()
    for splitter in (r"[\s\-]+", r"\s+"):
        raw_words = [w for w in re.split(splitter, long_str.strip()) if w]
        for drop_stopwords in (True, False):
            words = [
                re.sub(r"[^a-z0-9]", "", w.lower())
                for w in raw_words
                if not (drop_stopwords and w.lower() in _STOPWORDS)
            ]
            words = [w for w in words if w]
            if len(words) < 2:
                continue
            initials = "".join(w[0] for w in words)
            variants.add(initials)
            if len(words) >= 3:
                variants.add(words[0][:2] + initials[1:])
    return variants


def split_name_and_parenthetical_acronym(
    raw_name: str,
    known_alias_keys: Optional[Set[str]] = None,
) -> Tuple[str, List[str]]:
    """Split 'Rotary Position Embedding (RoPE)' into ('Rotary Position Embedding', ['RoPE']).

    A trailing parenthetical is accepted as an acronym only if it is an initialism of the name
    or a known seeded alias. Trap words like '(ours)', '(baseline)' and years are dropped;
    dense scale tokens like '(70B)' are left in place for ModelFamily scale stripping; any other
    parenthetical stays part of the name.
    """
    cleaned = _CITATION_NOISE_RE.sub("", raw_name.strip())
    m = _TRAILING_PAREN_ACRONYM_RE.match(cleaned)
    if not m:
        return cleaned, []
    full_part = m.group("full").strip()
    acro_part = m.group("acro").strip()
    if acro_part.lower() in _PAREN_TRAP_WORDS or re.fullmatch(r"\d{4}", acro_part):
        return full_part, []
    if _DENSE_SCALE_TOKEN_RE.match(acro_part) or acro_part.isdigit():
        return cleaned, []
    if is_acronym_match(acro_part, full_part):
        return full_part, [acro_part]
    if known_alias_keys and normalize_surface_key(acro_part) in known_alias_keys:
        return full_part, [acro_part]
    return cleaned, []


def extract_version_signature(text: str) -> Set[str]:
    """Extract structured identity tokens that distinguish otherwise similar names.

    Examples:
    - 'GPT-3', 'GPT3' -> {'v:3'};  'GPT-3.5' -> {'v:3.5'};  'Qwen2.5', 'Qwen-2.5' -> {'v:2.5'}
    - 'DeepSeek-V3' -> {'v:3'};  'DeepSeek-R1' -> {'v:r1'};  'DeepSeek-R1-Zero' -> {'v:r1-zero'}
    - 'Mixtral 8x7B' -> {'moe:8x7b'};  'Llama 3 8B' -> {'v:3'} (dense scale is not identity)
    - 'INT4', '4-bit' -> {'bits:4'};  'FP8' -> {'fp:8'};  'BF16' -> {'bf:16'}
    - 'Top-1' -> {'topk:1'};  'Top-K' -> {'topk:k'}
    """
    s = _CITATION_NOISE_RE.sub("", text.strip()).lower()
    s = re.sub(r"\((?:19|20)\d{2}\)", " ", s).strip(" .,;:")
    signature: Set[str] = set()

    def _consume(pattern: re.Pattern, fmt: Callable[[re.Match], str]) -> None:
        nonlocal s

        def _repl(m: re.Match) -> str:
            signature.add(fmt(m))
            return " "

        s = pattern.sub(_repl, s)

    _consume(_SIG_MOE_GRID_RE, lambda m: f"moe:{m.group(1)}x{m.group(2)}b")
    _consume(_SIG_INT_BITS_RE, lambda m: f"bits:{m.group(1)}")
    _consume(_SIG_N_BITS_RE, lambda m: f"bits:{m.group(1)}")
    _consume(_SIG_FP_RE, lambda m: f"fp:{m.group(1)}")
    _consume(_SIG_BF_RE, lambda m: f"bf:{m.group(1)}")
    _consume(_SIG_TOPK_RE, lambda m: f"topk:{m.group(1)}")
    s = _SIG_DENSE_SCALE_RE.sub(" ", s)  # checkpoint metadata, not identity
    _consume(_SIG_R_LINE_RE, lambda m: f"v:r{m.group(1)}" + ("-zero" if m.group(2) else ""))
    _consume(_SIG_V_PREFIX_RE, lambda m: f"v:{m.group(1)}")
    _consume(_SIG_GLUED_RE, lambda m: f"v:{m.group(1)}")
    _consume(_SIG_SEPARATED_RE, lambda m: f"v:{m.group(1)}")
    return signature


def has_version_conflict(name_a: str, name_b: str) -> bool:
    """Return True if `name_a` and `name_b` carry incompatible identity signatures.

    Implicit-v1 rule: a bare name (empty signature) matches only the exact signature {'v:1'}
    ('FlashAttention' == 'FlashAttention-1'); it never matches 'topk:1', 'bits:4', 'v:2', etc.
    """
    sig_a = extract_version_signature(name_a)
    sig_b = extract_version_signature(name_b)
    if sig_a == sig_b:
        return False
    if not sig_a or not sig_b:
        return (sig_a or sig_b) != {"v:1"}
    return True


def _signature_conflict(sig_a: Iterable[str], sig_b: Iterable[str]) -> bool:
    """`has_version_conflict` on precomputed signatures."""
    sig_a, sig_b = set(sig_a), set(sig_b)
    if sig_a == sig_b:
        return False
    if not sig_a or not sig_b:
        return (sig_a or sig_b) != {"v:1"}
    return True


def is_contrastive_sibling(name_a: str, name_b: str) -> bool:
    """True if the names share context but differ in exactly one dissimilar token.

    'Method A' vs 'Method B', 'Multi-Head Attention' vs 'Multi-Query Attention' and
    'Pre-LN' vs 'Post-LN' are siblings (distinct concepts that embeddings rate ~1.0).
    'Rotary Position Embedding' vs 'Rotary Positional Embedding' is not (the tokens are similar).
    """
    ta = normalize_surface_key(name_a).split()
    tb = normalize_surface_key(name_b).split()
    if not ta or not tb or ta == tb:
        return False
    limit = min(len(ta), len(tb))
    prefix = 0
    while prefix < limit and ta[prefix] == tb[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and ta[-1 - suffix] == tb[-1 - suffix]:
        suffix += 1
    if prefix + suffix == 0:
        return False
    diff_a = ta[prefix : len(ta) - suffix]
    diff_b = tb[prefix : len(tb) - suffix]
    if len(diff_a) != 1 or len(diff_b) != 1:
        return False
    return SequenceMatcher(None, diff_a[0], diff_b[0]).ratio() < 0.8


def _singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def mention_tokens(text: str) -> List[str]:
    """Tokenize text for mention matching: lower-case, punctuation-free, per-token singular.

    'We apply RoPE.' -> ['we', 'apply', 'rope'];  'GPT-3.5' -> ['gpt', '3.5'];
    'Rotary Position Embeddings' -> ['rotary', 'position', 'embedding'].
    """
    return [_singularize_token(t) for t in _MENTION_TOKEN_RE.findall(text.lower())]


def slugify_entity_id(entity_type: EntityType, canonical_name: str) -> str:
    """Mint a deterministic canonical entity_id like 'arch_component::grouped_query_attention'."""
    prefix_map = {
        EntityType.PAPER: "paper",
        EntityType.MODEL_FAMILY: "model_family",
        EntityType.ARCH_COMPONENT: "arch_component",
        EntityType.TRAINING_METHOD: "training_method",
        EntityType.INFERENCE_TECHNIQUE: "inference_technique",
        EntityType.BOTTLENECK: "bottleneck",
        EntityType.HYPERPARAMETER_OR_METRIC: "hyperparam_metric",
        EntityType.BENCHMARK_OR_DATASET: "benchmark_dataset",
    }
    prefix = prefix_map[entity_type]
    slug = re.sub(r"[^a-z0-9]+", "_", canonical_name.lower()).strip("_")
    return f"{prefix}::{slug}"


# Query-time linking: words that never carry entity identity in a user question.
_QUERY_STOPWORDS = {
    "how", "what", "which", "why", "when", "where", "who", "whom", "whose", "does", "did", "do", "is", "are", "was",
    "were", "be", "been", "being", "has", "have", "had", "can", "could", "would", "should", "will", "may", "might",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with", "into", "onto", "and", "or", "but",
    "then", "than", "that", "this", "these", "those", "it", "its", "their", "there", "them", "they", "each", "every",
    "between", "versus", "vs", "compare", "compared", "explain", "describe", "trace", "evolve", "evolved", "evolution",
    "lineage", "change", "changed", "become", "became", "give", "show", "list", "name", "about", "use", "used", "uses",
    "using", "paper", "papers", "via", "also", "both", "all", "any", "some", "same", "other", "s", "please", "tell",
    "me", "we", "our", "you", "your", "i", "so", "as", "if", "not", "no", "yes", "much", "many", "more", "most",
}
# Generic head words dropped from a phrase before lexical comparison ('post-norm layer' ~ 'post-norm').
_QUERY_GENERIC_TOKENS = {
    "layer", "model", "method", "technique", "mechanism", "component", "approach", "architecture", "design",
    "placement", "variant", "version", "scheme", "module", "block", "idea", "concept", "strategy",
}
_QUERY_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.']*")
TIER2_QUERY_THRESHOLD = 0.88


@dataclass
class QueryEntityLink:
    """One query-time link from a user phrase to a graph node (shown in routing telemetry and the UI)."""

    phrase: str
    entity_id: str
    canonical_name: str
    tier: str  # TIER_1_EXACT | TIER_1_PAPER_INTRODUCES | TIER_2_LEXICAL | TIER_3_SEMANTIC (+ _PAPER_INTRODUCES)
    score: float
    matched_surface: str = ""


def extract_query_phrases(question: str, max_phrases: int = 8, max_tokens: int = 6) -> List[str]:
    """Candidate entity phrases: maximal runs of non-stopword tokens ('post-norm layer', 'pre-norm layer')."""
    runs: List[List[str]] = [[]]
    for raw in _QUERY_WORD_RE.findall(question):
        word = re.sub(r"'s$", "", raw.strip(".-'"))
        if not word or word.lower() in _QUERY_STOPWORDS:
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append(word)
    out: List[str] = []
    for run in runs:
        content = [w for w in run if w.lower() not in _QUERY_GENERIC_TOKENS]
        if not content:
            continue
        phrase = " ".join(run[:max_tokens])
        if phrase.lower() not in {p.lower() for p in out}:
            out.append(phrase)
    return out[:max_phrases]


def _content_tokens(text: str) -> List[str]:
    return [t for t in mention_tokens(text) if t not in _QUERY_STOPWORDS]


# Opposed modifiers: a phrase carrying one side never links to a node that only carries the other side
# ('pre-norm' -> 'Post-LN', 'multi-query' -> 'Multi-Head Attention'). Embeddings rate these pairs ~0.95.
_CONTRAST_PAIRS = (("pre", "post"), ("head", "query"), ("encoder", "decoder"), ("absolute", "relative"),
                   ("dense", "sparse"), ("single", "multi"), ("online", "offline"))


def _contrast_conflict(phrase_tokens: Set[str], node_surfaces: Iterable[str]) -> bool:
    return _contrast_conflict_tokens(phrase_tokens, {t for s in node_surfaces for t in mention_tokens(s)})


def _contrast_conflict_tokens(phrase_tokens: Set[str], node_tokens: Iterable[str]) -> bool:
    node_tokens = set(node_tokens)
    for a, b in _CONTRAST_PAIRS:
        for x, y in ((a, b), (b, a)):
            if x in phrase_tokens and y in node_tokens and x not in node_tokens:
                return True
    return False


@dataclass
class ResolutionDecisionLog:
    """Audit trail record for every entity resolution decision (useful for debugging & Phase 5)."""

    raw_name: str
    entity_type: str
    resolved_entity_id: str
    resolved_canonical_name: str
    tier_used: str  # "TIER_1_ALIAS", "TIER_2_ACRONYM_LEXICAL", "TIER_3_EMBEDDING", "TIER_3_LLM_JUDGE", "NEW_NODE"
    similarity_score: float


class EntityResolver:
    """Stateful 3-Tier Entity Resolution engine backed by seed aliases + vector similarity."""

    def __init__(
        self,
        seed_aliases_path: Optional[Path] = None,
        auto_merge_threshold: float = 0.88,
        gray_zone_low_threshold: float = 0.82,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        llm_tiebreaker_fn: Optional[Callable[[str, str, EntityType], bool]] = None,
    ) -> None:
        self.auto_merge_threshold = auto_merge_threshold
        self.gray_zone_low_threshold = gray_zone_low_threshold
        self.embed_fn = embed_fn
        self.llm_tiebreaker_fn = llm_tiebreaker_fn

        # Primary store of canonical nodes keyed by entity_id
        self.nodes: Dict[str, ResolvedEntityNode] = {}
        # Typed O(1) lookup: (entity_type, normalized_surface_key) -> entity_id
        self.alias_index: Dict[Tuple[EntityType, str], str] = {}
        # Typed O(1) lookup on space-free keys: (entity_type, 'gpt3') -> entity_id
        self.compact_alias_index: Dict[Tuple[EntityType, str], str] = {}
        # Informational cross-type key -> first entity_id. NOT used for resolution: identity is
        # (entity_type, name), so a Paper and the ModelFamily it introduces stay distinct.
        self.global_alias_index: Dict[str, str] = {}
        # Normalized keys of every seeded surface (used to accept seeded parenthetical acronyms)
        self.seed_alias_keys: Set[str] = set()
        # Entity IDs that came from curated seeds or the manifest (never renamed/upgraded)
        self.seeded_ids: Set[str] = set()
        # Cached unit-norm embeddings per entity_id for Tier 3
        self.node_embeddings: Dict[str, np.ndarray] = {}
        # Audit log of all resolution decisions
        self.decision_logs: List[ResolutionDecisionLog] = []
        # Lazily built mention index: token tuple -> entity_ids
        self._mention_index: Optional[Dict[Tuple[str, ...], Set[str]]] = None
        self._mention_max_len = 0
        # Query-time linking: paper_entity_id -> concept ids it INTRODUCES (recorded during resolution)
        self.paper_introduces: Dict[str, List[str]] = {}
        # Query-time Tier 3 cache: entity_id -> (node representation text, unit vector)
        self.query_node_embeddings: Dict[str, Tuple[str, np.ndarray]] = {}
        self.allow_live_node_embed: bool = True
        self._paper_mention_index: Optional[Dict[Tuple[str, ...], Set[str]]] = None
        self._paper_mention_max_len = 0
        self._query_feature_cache: Dict[str, Tuple[tuple, Dict[str, object]]] = {}

        if seed_aliases_path and seed_aliases_path.exists():
            self.load_seed_aliases(seed_aliases_path)

    # ------------------------------------------------------------------
    # Registration & indexing
    # ------------------------------------------------------------------

    def load_seed_aliases(self, yaml_path: Path) -> None:
        """Load curated canonical entities and aliases from `seed_aliases.yaml`."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item in data.get("entities", []):
            node = ResolvedEntityNode(
                entity_id=item["entity_id"],
                entity_type=EntityType(item["entity_type"]),
                canonical_name=item["canonical_name"],
                aliases=list(item.get("aliases", [])),
                category=item.get("category"),
                confidence=1.0,
            )
            self._register_seed(node)

    def register_known_paper(self, paper_id: str, title: str, extra_aliases: Iterable[str] = ()) -> ResolvedEntityNode:
        """Register a manifest paper as a seeded `Paper` node with id `paper::{paper_id}`.

        Aliases: the full title, the pre-colon short title ('DeepSeek-V2: A Strong ...' ->
        'DeepSeek-V2'), the paper_id, and any extras. Extracted `Paper` entities that name the
        paper by any of these surfaces resolve to the manifest node instead of minting a new one.
        """
        entity_id = f"paper::{paper_id}"
        aliases: List[str] = [paper_id]
        if ":" in title:
            short = title.split(":", 1)[0].strip()
            if len(short) >= 2:
                aliases.append(short)
        aliases.extend(a for a in extra_aliases if a)
        if entity_id in self.nodes:
            for a in aliases:
                self._add_alias_to_node(entity_id, a)
            return self.nodes[entity_id]
        node = ResolvedEntityNode(
            entity_id=entity_id,
            entity_type=EntityType.PAPER,
            canonical_name=title,
            aliases=aliases,
            paper_ids=[paper_id],
            confidence=1.0,
        )
        self._register_seed(node)
        return node

    def _register_seed(self, node: ResolvedEntityNode) -> None:
        self.seeded_ids.add(node.entity_id)
        for surface in [node.canonical_name] + list(node.aliases):
            key = normalize_surface_key(surface)
            if key:
                self.seed_alias_keys.add(key)
        self._register_node(node)

    def _index_surface(self, node: ResolvedEntityNode, surface: str) -> None:
        """Index one surface form for its node. First registration of a key wins (no stealing)."""
        key = normalize_surface_key(surface)
        if not key:
            return
        self.alias_index.setdefault((node.entity_type, key), node.entity_id)
        self.compact_alias_index.setdefault((node.entity_type, key.replace(" ", "")), node.entity_id)
        self.global_alias_index.setdefault(key, node.entity_id)
        self._mention_index = None
        self._paper_mention_index = None

    def _register_node(self, node: ResolvedEntityNode) -> None:
        """Register a canonical node and index all its surface forms."""
        self.nodes[node.entity_id] = node
        for surface in [node.canonical_name] + list(node.aliases):
            self._index_surface(node, surface)

    def _add_alias_to_node(self, entity_id: str, new_alias: str) -> None:
        """Append a newly discovered surface form to an existing canonical node's alias list."""
        node = self.nodes[entity_id]
        clean_alias = new_alias.strip()
        if not clean_alias:
            return
        existing_lower = {a.lower() for a in node.aliases} | {node.canonical_name.lower()}
        if clean_alias.lower() not in existing_lower:
            node.aliases.append(clean_alias)
        self._index_surface(node, clean_alias)

    def _unique_entity_id(self, base_id: str) -> str:
        """Avoid silently overwriting a different node whose slug collides (e.g. guard-rejected)."""
        if base_id not in self.nodes:
            return base_id
        n = 2
        while f"{base_id}_{n}" in self.nodes:
            n += 1
        return f"{base_id}_{n}"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _get_or_compute_embedding(self, entity_id: str, text_to_embed: str) -> Optional[np.ndarray]:
        """Return unit-normalized embedding vector for an entity if `embed_fn` is configured."""
        if entity_id in self.node_embeddings:
            return self.node_embeddings[entity_id]
        if self.embed_fn is None:
            return None
        vecs = self.embed_fn([text_to_embed])
        if not vecs or not vecs[0]:
            return None
        arr = np.asarray(vecs[0], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        self.node_embeddings[entity_id] = arr
        return arr

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _prepare_surfaces(self, extracted: ExtractedEntity) -> Tuple[str, List[str], List[str], List[str]]:
        """Return (primary_name, parenthetical_acronyms, candidate_surfaces, parameter_scales)."""
        etype = extracted.entity_type
        primary_name, paren_acronyms = split_name_and_parenthetical_acronym(
            extracted.canonical_name, self.seed_alias_keys
        )
        extra_aliases = [_CITATION_NOISE_RE.sub("", a.strip()) for a in extracted.aliases if a and a.strip()]
        scales: List[str] = []
        if etype == EntityType.MODEL_FAMILY:
            primary_name, scales = strip_dense_scale_tokens(primary_name)
            stripped_aliases = []
            for alias in extra_aliases:
                a, s = strip_dense_scale_tokens(alias)
                scales.extend(s)
                stripped_aliases.append(a)
            extra_aliases = stripped_aliases
        candidates: List[str] = []
        for surface in [primary_name] + paren_acronyms + extra_aliases:
            if surface and surface.lower() not in {c.lower() for c in candidates}:
                candidates.append(surface)
        return primary_name, paren_acronyms, candidates, list(dict.fromkeys(scales))

    def _tier1_lookup(self, etype: EntityType, primary_name: str, candidates: List[str]) -> Optional[str]:
        for surf in candidates:
            key = normalize_surface_key(surf)
            if not key:
                continue
            for matched_id in (
                self.alias_index.get((etype, key)),
                self.compact_alias_index.get((etype, key.replace(" ", ""))),
            ):
                if not matched_id:
                    continue
                target = self.nodes[matched_id]
                # Paper titles are long and noisy ('... 2bit Quantization'); an exact typed key
                # match on a manifest paper is trusted without the version guard.
                if etype == EntityType.PAPER or not has_version_conflict(primary_name, target.canonical_name):
                    return matched_id
        return None

    def _blocked(self, primary_name: str, node: ResolvedEntityNode) -> bool:
        """Hard guards shared by Tier 2 and Tier 3."""
        if has_version_conflict(primary_name, node.canonical_name):
            return True
        return any(is_contrastive_sibling(primary_name, s) for s in [node.canonical_name] + list(node.aliases))

    def _merge_into(
        self,
        entity_id: str,
        extracted: ExtractedEntity,
        primary_name: str,
        paren_acronyms: List[str],
        candidates: List[str],
        scales: List[str],
        source_chunk_id: str,
        paper_id: str,
        tier: str,
        score: float,
    ) -> ResolvedEntityNode:
        target = self.nodes[entity_id]
        # Canonical-name upgrade: a minted bare-acronym node ('NSA') adopts the descriptive
        # full name once it is seen ('Native Sparse Attention (NSA)'); the ID stays anchored.
        if (
            entity_id not in self.seeded_ids
            and paren_acronyms
            and normalize_surface_key(target.canonical_name) == normalize_surface_key(paren_acronyms[0])
            and primary_name.lower() != target.canonical_name.lower()
        ):
            old_name = target.canonical_name
            target.canonical_name = primary_name
            target.aliases = [a for a in target.aliases if a.lower() != primary_name.lower()]
            self._add_alias_to_node(entity_id, old_name)
            self._index_surface(target, primary_name)
            self.node_embeddings.pop(entity_id, None)
        for s in candidates:
            self._add_alias_to_node(entity_id, s)
        for scale in scales:
            if scale not in target.parameter_scales:
                target.parameter_scales.append(scale)
        self._record_provenance(target, source_chunk_id, paper_id, extracted.confidence, extracted.description)
        self.decision_logs.append(
            ResolutionDecisionLog(
                raw_name=extracted.canonical_name,
                entity_type=extracted.entity_type.value,
                resolved_entity_id=entity_id,
                resolved_canonical_name=target.canonical_name,
                tier_used=tier,
                similarity_score=round(score, 4),
            )
        )
        return target

    def resolve_entity(
        self,
        extracted: ExtractedEntity,
        source_chunk_id: str,
        paper_id: str,
    ) -> ResolvedEntityNode:
        """Resolve an `ExtractedEntity` into a canonical `ResolvedEntityNode` via the 3-tier funnel.

        Resolution is strictly typed: only nodes of the same `EntityType` are candidates.
        """
        etype = extracted.entity_type
        primary_name, paren_acronyms, candidates, scales = self._prepare_surfaces(extracted)
        merge_args = (extracted, primary_name, paren_acronyms, candidates, scales, source_chunk_id, paper_id)

        # TIER 1: Exact typed alias lookup (normalized or compact key) with version guard
        matched_id = self._tier1_lookup(etype, primary_name, candidates)
        if matched_id:
            return self._merge_into(matched_id, *merge_args, tier="TIER_1_ALIAS", score=1.0)

        # TIER 2: Acronym initials & high-precision lexical match (same type, guarded)
        same_type_nodes = [n for n in self.nodes.values() if n.entity_type == etype]
        best_lexical_id: Optional[str] = None
        best_lexical_score = 0.0
        for node in same_type_nodes:
            if has_version_conflict(primary_name, node.canonical_name):
                continue
            node_surfaces = [node.canonical_name] + list(node.aliases)
            sibling = any(is_contrastive_sibling(primary_name, s) for s in node_surfaces)
            for cand_s in candidates:
                for node_s in node_surfaces:
                    if is_acronym_match(cand_s, node_s) or is_acronym_match(node_s, cand_s):
                        best_lexical_id, best_lexical_score = node.entity_id, 0.99
                        break
                    if sibling:
                        continue
                    ratio = SequenceMatcher(
                        None, normalize_surface_key(cand_s), normalize_surface_key(node_s)
                    ).ratio()
                    if ratio > best_lexical_score:
                        best_lexical_id, best_lexical_score = node.entity_id, ratio
                if best_lexical_score >= 0.99:
                    break
            if best_lexical_score >= 0.99:
                break

        if best_lexical_id and best_lexical_score >= 0.91:
            return self._merge_into(
                best_lexical_id, *merge_args, tier="TIER_2_ACRONYM_LEXICAL", score=best_lexical_score
            )

        # TIER 3: Embedding cosine similarity + gray-zone tie-breaker (same type, guarded)
        if self.embed_fn is not None and same_type_nodes:
            query_repr = f"{etype.value}: {primary_name} ({', '.join(extracted.aliases)})"
            vecs = self.embed_fn([query_repr])
            if vecs and vecs[0]:
                q_vec = np.asarray(vecs[0], dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                best_emb_id: Optional[str] = None
                best_emb_sim = -1.0
                for node in same_type_nodes:
                    if self._blocked(primary_name, node):
                        continue
                    node_repr = f"{node.entity_type.value}: {node.canonical_name} ({', '.join(node.aliases)})"
                    n_vec = self._get_or_compute_embedding(node.entity_id, node_repr)
                    if n_vec is not None:
                        sim = float(np.dot(q_vec, n_vec))
                        if sim > best_emb_sim:
                            best_emb_sim, best_emb_id = sim, node.entity_id

                if best_emb_id is not None:
                    if best_emb_sim >= self.auto_merge_threshold:
                        return self._merge_into(best_emb_id, *merge_args, tier="TIER_3_EMBEDDING", score=best_emb_sim)
                    if best_emb_sim >= self.gray_zone_low_threshold and self.llm_tiebreaker_fn is not None:
                        target = self.nodes[best_emb_id]
                        if self.llm_tiebreaker_fn(primary_name, target.canonical_name, etype):
                            return self._merge_into(
                                best_emb_id, *merge_args, tier="TIER_3_LLM_JUDGE", score=best_emb_sim
                            )

        # FALLBACK: Mint a new canonical node. Acronym-bearing names anchor their ID on the
        # acronym so 'NSA' and 'Native Sparse Attention (NSA)' mint the same ID in either order.
        id_basis = paren_acronyms[0] if paren_acronyms else primary_name
        new_id = self._unique_entity_id(slugify_entity_id(etype, id_basis))
        new_node = ResolvedEntityNode(
            entity_id=new_id,
            entity_type=etype,
            canonical_name=primary_name,
            aliases=[a for a in candidates if a.lower() != primary_name.lower()],
            category=extracted.category.value if extracted.category else None,
            description=extracted.description,
            source_chunk_ids=[source_chunk_id] if source_chunk_id else [],
            paper_ids=[paper_id] if paper_id else [],
            confidence=extracted.confidence,
            parameter_scales=scales,
        )
        self._register_node(new_node)
        self.decision_logs.append(
            ResolutionDecisionLog(
                raw_name=extracted.canonical_name,
                entity_type=etype.value,
                resolved_entity_id=new_id,
                resolved_canonical_name=primary_name,
                tier_used="NEW_NODE",
                similarity_score=0.0,
            )
        )
        return new_node

    def resolve_relationship(
        self,
        rel: ExtractedRelationship,
        source_chunk_id: str,
        paper_id: str,
    ) -> Optional[ResolvedGraphEdge]:
        """Resolve both endpoints of an `ExtractedRelationship` to canonical `entity_id`s."""
        src_entity = self.resolve_entity(
            ExtractedEntity(entity_type=rel.source_type, canonical_name=rel.source_name, confidence=rel.confidence),
            source_chunk_id=source_chunk_id,
            paper_id=paper_id,
        )
        tgt_entity = self.resolve_entity(
            ExtractedEntity(entity_type=rel.target_type, canonical_name=rel.target_name, confidence=rel.confidence),
            source_chunk_id=source_chunk_id,
            paper_id=paper_id,
        )
        if src_entity.entity_id == tgt_entity.entity_id:
            return None
        if (rel.relation_type.value == "INTRODUCES" and src_entity.entity_type == EntityType.PAPER
                and tgt_entity.entity_type in (EntityType.MODEL_FAMILY, EntityType.ARCH_COMPONENT)):
            intro = self.paper_introduces.setdefault(src_entity.entity_id, [])
            if tgt_entity.entity_id not in intro:
                intro.append(tgt_entity.entity_id)

        return ResolvedGraphEdge(
            source_entity_id=src_entity.entity_id,
            target_entity_id=tgt_entity.entity_id,
            relation_type=rel.relation_type,
            source_chunk_id=source_chunk_id,
            paper_id=paper_id,
            evidence_quote=rel.evidence_quote,
            model_or_metric_context=rel.model_or_metric_context,
            confidence=rel.confidence,
        )

    # ------------------------------------------------------------------
    # Chunk mention detection
    # ------------------------------------------------------------------

    def _build_mention_index(self) -> Dict[Tuple[str, ...], Set[str]]:
        index: Dict[Tuple[str, ...], Set[str]] = {}
        max_len = 0
        for node in self.nodes.values():
            if node.entity_type == EntityType.PAPER:
                continue  # chunks link to their paper via IN_PAPER; short titles would over-tag
            for surface in [node.canonical_name] + list(node.aliases):
                toks = tuple(mention_tokens(_CITATION_NOISE_RE.sub("", surface)))
                if not toks or len("".join(toks)) < 3 or (len(toks) == 1 and toks[0].replace(".", "").isdigit()):
                    continue
                index.setdefault(toks, set()).add(node.entity_id)
                max_len = max(max_len, len(toks))
        self._mention_max_len = max_len
        return index

    def resolve_text_mentions_in_chunk(self, raw_text: str) -> List[str]:
        """Return sorted entity_ids whose aliases occur in the chunk text.

        Uses longest-token-span, non-overlapping selection: 'Llama 2' consumes its tokens so the
        shorter alias 'LLaMA' (LLaMA-1) cannot fire inside it; 'FlashAttention-2' likewise
        suppresses 'FlashAttention'.
        """
        if self._mention_index is None:
            self._mention_index = self._build_mention_index()
        index = self._mention_index
        tokens = mention_tokens(raw_text)

        spans: List[Tuple[int, int, Set[str]]] = []
        for start in range(len(tokens)):
            for length in range(1, min(self._mention_max_len, len(tokens) - start) + 1):
                ids = index.get(tuple(tokens[start : start + length]))
                if ids:
                    spans.append((start, start + length, ids))

        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        taken = [False] * len(tokens)
        matched: Set[str] = set()
        for start, end, ids in spans:
            if any(taken[start:end]):
                continue
            for i in range(start, end):
                taken[i] = True
            matched.update(ids)
        return sorted(matched)

    def resolve_Text_mentions_in_chunk(self, raw_text: str) -> List[str]:
        """Backward-compatible alias of `resolve_text_mentions_in_chunk`."""
        return self.resolve_text_mentions_in_chunk(raw_text)

    # ------------------------------------------------------------------
    # Query-time entity linking (user questions -> graph nodes)
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_surface_ok(key: str) -> bool:
        """Paper surfaces are linkable only if multi-word or >= 4 characters ('GQA' alone is too ambiguous)."""
        return len(key.split()) >= 2 or len(key.replace(" ", "")) >= 4

    def _build_paper_mention_index(self) -> Dict[Tuple[str, ...], Set[str]]:
        concept_keys = {k for (etype, k) in self.alias_index if etype != EntityType.PAPER}
        index: Dict[Tuple[str, ...], Set[str]] = {}
        max_len = 0
        for node in self.nodes.values():
            if node.entity_type != EntityType.PAPER:
                continue
            for surface in [node.canonical_name] + list(node.aliases):
                key = normalize_surface_key(surface)
                # A short title that is also a concept name ('GQA', 'DeepSeek-V2', 'LLaMA') names the concept.
                if not key or key in concept_keys or "_" in surface or not self._paper_surface_ok(key):
                    continue
                toks = tuple(mention_tokens(surface))
                if toks:
                    index.setdefault(toks, set()).add(node.entity_id)
                    max_len = max(max_len, len(toks))
        self._paper_mention_max_len = max_len
        return index

    def find_paper_mentions(self, text: str) -> List[str]:
        """Paper entity_ids whose title or distinctive short title occurs in `text` (longest span wins)."""
        if self._paper_mention_index is None:
            self._paper_mention_index = self._build_paper_mention_index()
        index, tokens = self._paper_mention_index, mention_tokens(text)
        spans: List[Tuple[int, int, Set[str]]] = []
        for start in range(len(tokens)):
            for length in range(1, min(self._paper_mention_max_len, len(tokens) - start) + 1):
                ids = index.get(tuple(tokens[start : start + length]))
                if ids:
                    spans.append((start, start + length, ids))
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        taken = [False] * len(tokens)
        found: List[str] = []
        for start, end, ids in spans:
            if any(taken[start:end]):
                continue
            for i in range(start, end):
                taken[i] = True
            found.extend(sorted(i for i in ids if i not in found))
        return found

    def _paper_links(self, paper_id: str, phrase: str, tier: str, score: float, surface: str,
                     introduces_fn: Optional[Callable[[str], List[str]]],
                     accept: Callable[[str], bool]) -> List[QueryEntityLink]:
        """The paper node plus every ModelFamily / ArchComponent the paper INTRODUCES."""
        links = [QueryEntityLink(phrase, paper_id, self.nodes[paper_id].canonical_name, tier, round(score, 4), surface)]
        introduced = introduces_fn(paper_id) if introduces_fn is not None else self.paper_introduces.get(paper_id, [])
        for eid in introduced:
            if eid != paper_id and accept(eid) and eid not in {l.entity_id for l in links}:
                name = self.nodes[eid].canonical_name if eid in self.nodes else eid
                links.append(QueryEntityLink(phrase, eid, name, f"{tier}_PAPER_INTRODUCES", round(score, 4), surface))
        return links

    def _query_features(self, node: ResolvedEntityNode) -> Dict[str, object]:
        """Per-node Tier-2 features, cached until the node's names/description change."""
        stamp = (node.canonical_name, tuple(node.aliases), node.description, node.category)
        cached = self._query_feature_cache.get(node.entity_id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        surfaces = []
        for s in [node.canonical_name] + list(node.aliases):
            ns = normalize_surface_key(s)
            if not ns or "_" in s:
                continue  # paper_id aliases are not natural-language surfaces
            toks = [t for t in _content_tokens(s) if t not in _QUERY_GENERIC_TOKENS]
            acr = frozenset(acronym_variants(s)) if " " in ns and node.entity_type != EntityType.PAPER else frozenset()
            surfaces.append((s, ns, ns.replace(" ", ""), frozenset(toks), len(toks), acr))
        feats: Dict[str, object] = {
            "sig": frozenset(extract_version_signature(node.canonical_name)),
            "surfaces": surfaces,
            "tokens": frozenset(t for s in [node.canonical_name] + list(node.aliases) for t in mention_tokens(s)),
            "context": frozenset(_content_tokens(f"{node.description} {node.category or ''}")),
        }
        self._query_feature_cache[node.entity_id] = (stamp, feats)
        return feats

    def _tier2_query_rank(self, phrase: str, allow_papers: bool) -> List[Tuple[float, str, str]]:
        """(score, entity_id, matched_surface) sorted best-first; lexical + token/description overlap."""
        core_toks = [t for t in _content_tokens(phrase) if t not in _QUERY_GENERIC_TOKENS] or _content_tokens(phrase)
        if not core_toks:
            return []
        core = " ".join(core_toks)
        core_compact = core.replace(" ", "")
        core_set = set(core_toks)
        phrase_sig = frozenset(extract_version_signature(phrase))
        sm_spaced, sm_compact = SequenceMatcher(None), SequenceMatcher(None)
        sm_spaced.set_seq2(core)
        sm_compact.set_seq2(core_compact)
        floor = TIER2_QUERY_THRESHOLD - 0.04
        core_acronym = core_compact if len(core_toks) == 1 and len(core_compact) >= 3 else ""
        ranked: List[Tuple[float, str, str]] = []
        for node in self.nodes.values():
            if node.entity_type == EntityType.PAPER and not allow_papers:
                continue
            f = self._query_features(node)
            if _signature_conflict(phrase_sig, f["sig"]):
                continue  # 'GPT-4' never fuzzy-matches 'GPT-3'
            if _contrast_conflict_tokens(core_set, f["tokens"]):
                continue  # 'pre norm' never matches a node named only 'post norm'
            best, best_surface = 0.0, ""
            for s, ns, cs, surf_toks, n_toks, acr in f["surfaces"]:
                score = 0.0
                # Only a ratio that could still clear the threshold (after the +0.04 context bonus) is computed.
                sm_spaced.set_seq1(ns)
                if sm_spaced.real_quick_ratio() >= floor and sm_spaced.quick_ratio() >= floor:
                    score = sm_spaced.ratio()
                sm_compact.set_seq1(cs)
                if sm_compact.real_quick_ratio() >= floor and sm_compact.quick_ratio() >= floor:
                    score = max(score, sm_compact.ratio())
                if core_compact != cs and (core_compact in cs or cs in core_compact):
                    score = min(score, 0.8)  # 'training' is not 'uptraining'; 'attention' is not 'flashattention'
                if core_acronym and core_acronym in acr:
                    score = max(score, 0.95)
                if n_toks >= 2 and surf_toks <= core_set and n_toks * 2 >= len(core_set):
                    score = max(score, 0.9 + 0.05 * n_toks / len(core_set))
                if score > best:
                    best, best_surface = score, s
            if best < floor or is_contrastive_sibling(core, best_surface):
                continue
            best += 0.04 * len(core_set & f["context"]) / len(core_set)  # description/category overlap: tie-breaker
            ranked.append((round(min(best, 1.0), 4), node.entity_id, best_surface))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return ranked

    @staticmethod
    def query_node_text(node: ResolvedEntityNode) -> str:
        return f"{node.canonical_name}: {node.description} ({', '.join(node.aliases)})"

    def load_cached_query_node_embeddings(self, embedder: Any) -> int:
        """Load pre-embedded node vectors from `embedder` SQLite cache (0 API calls)."""
        lookup = getattr(embedder, "lookup_cached_documents", None)
        if lookup is None:
            return 0
        nodes = list(self.nodes.values())
        texts = [self.query_node_text(n) for n in nodes]
        cached = lookup(texts)
        loaded = 0
        for idx, vec in cached.items():
            node = nodes[idx]
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            self.query_node_embeddings[node.entity_id] = (texts[idx], arr / norm if norm > 0 else arr)
            loaded += 1
        return loaded

    def prewarm_query_node_embeddings(self, embed_fn: Callable[[List[str]], List[Any]], batch_size: int = 128) -> int:
        """Embed any missing entity node representations up front so live queries never stall."""
        missing = [
            n for n in self.nodes.values()
            if self.query_node_embeddings.get(n.entity_id, ("",))[0] != self.query_node_text(n)
        ]
        if not missing:
            return 0
        embedded = 0
        for start in range(0, len(missing), max(1, int(batch_size))):
            batch = missing[start : start + max(1, int(batch_size))]
            texts = [self.query_node_text(n) for n in batch]
            for n, text, vec in zip(batch, texts, embed_fn(texts)):
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                self.query_node_embeddings[n.entity_id] = (text, arr / norm if norm > 0 else arr)
                embedded += 1
        return embedded

    def _tier3_query_rank(self, phrase: str, embed_fn: Callable[[List[str]], List[List[float]]],
                          allow_papers: bool) -> List[Tuple[float, str]]:
        nodes = [n for n in self.nodes.values() if allow_papers or n.entity_type != EntityType.PAPER]
        missing = [n for n in nodes if self.query_node_embeddings.get(n.entity_id, ("",))[0] != self.query_node_text(n)]
        if missing and self.allow_live_node_embed:
            texts = [self.query_node_text(n) for n in missing]
            for n, text, vec in zip(missing, texts, embed_fn(texts)):
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                self.query_node_embeddings[n.entity_id] = (text, arr / norm if norm > 0 else arr)
        available = [n for n in nodes if n.entity_id in self.query_node_embeddings]
        if not available:
            return []
        q = np.asarray(embed_fn([phrase])[0], dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        q = q / qn
        ranked = [(float(np.dot(q, self.query_node_embeddings[n.entity_id][1])), n.entity_id) for n in available]
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return ranked

    def link_query_entity_detailed(
        self,
        phrase: str,
        allow_papers: bool = True,
        min_semantic_sim: float = 0.78,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        candidate_filter: Optional[Callable[[str], bool]] = None,
        introduces_fn: Optional[Callable[[str], List[str]]] = None,
    ) -> List[QueryEntityLink]:
        """Link a free-text question phrase to graph nodes: Tier 1 exact -> Tier 2 lexical -> Tier 3 semantic.

        Unlike `resolve_entity` (ingestion) this is untyped: the phrase can name any EntityType, including a
        Paper by its title, in which case the concepts that paper INTRODUCES are linked too. Every tier keeps
        the hard version guard, and Tier 3 also refuses contrastive siblings. Returns [] when nothing passes.
        """
        phrase = re.sub(r"\s+", " ", phrase or "").strip(" ?.,;:!\"'")
        key = normalize_surface_key(phrase)
        if len(key.replace(" ", "")) < 2:
            return []
        accept = candidate_filter or (lambda _eid: True)

        # TIER 1: exact normalized / compact key across all entity types
        concepts: List[str] = []
        papers: List[str] = []
        for etype in EntityType:
            for eid in (self.alias_index.get((etype, key)), self.compact_alias_index.get((etype, key.replace(" ", "")))):
                if not eid or eid in concepts or eid in papers or not accept(eid):
                    continue
                if etype == EntityType.PAPER:
                    if allow_papers and self._paper_surface_ok(key):
                        papers.append(eid)
                elif not has_version_conflict(phrase, self.nodes[eid].canonical_name):
                    concepts.append(eid)
        if concepts:
            return [QueryEntityLink(phrase, e, self.nodes[e].canonical_name, "TIER_1_EXACT", 1.0, phrase) for e in concepts]
        if papers:
            return self._paper_links(papers[0], phrase, "TIER_1_EXACT", 1.0, phrase, introduces_fn, accept)

        # TIER 2: fuzzy lexical + token / description overlap (version guard + sibling guard inside)
        for score, eid, surface in self._tier2_query_rank(phrase, allow_papers):
            if score < TIER2_QUERY_THRESHOLD:
                break
            if accept(eid):
                if self.nodes[eid].entity_type == EntityType.PAPER:
                    return self._paper_links(eid, phrase, "TIER_2_LEXICAL", score, surface, introduces_fn, accept)
                return [QueryEntityLink(phrase, eid, self.nodes[eid].canonical_name, "TIER_2_LEXICAL", score, surface)]

        # TIER 3: embedding cosine similarity against cached node representations. Single-word phrases are
        # excluded: real embedders score 'trained' ~ 'Uptraining' (0.85) and 'DeepSeek' ~ 'DeepSeekMath' (0.89).
        fn = embed_fn or self.embed_fn
        core_toks = [t for t in _content_tokens(phrase) if t not in _QUERY_GENERIC_TOKENS]
        if fn is not None and len(core_toks) >= 2:
            for sim, eid in self._tier3_query_rank(phrase, fn, allow_papers):
                if sim < min_semantic_sim:
                    break  # below the semantic floor: reject rather than guess
                node = self.nodes[eid]
                if has_version_conflict(phrase, node.canonical_name):
                    continue
                if _contrast_conflict(set(_content_tokens(phrase)), [node.canonical_name] + list(node.aliases)):
                    continue
                if accept(eid):
                    text = self.query_node_text(node)
                    if node.entity_type == EntityType.PAPER:
                        return self._paper_links(eid, phrase, "TIER_3_SEMANTIC", sim, text, introduces_fn, accept)
                    return [QueryEntityLink(phrase, eid, node.canonical_name, "TIER_3_SEMANTIC", round(sim, 4), text)]
        return []

    def link_query_entity(self, phrase: str, allow_papers: bool = True, min_semantic_sim: float = 0.78,
                          **kwargs) -> List[str]:
        """entity_ids for a question phrase (see `link_query_entity_detailed`)."""
        return [l.entity_id for l in self.link_query_entity_detailed(phrase, allow_papers, min_semantic_sim, **kwargs)]

    @staticmethod
    def _record_provenance(
        node: ResolvedEntityNode,
        source_chunk_id: str,
        paper_id: str,
        confidence: float,
        description: str,
    ) -> None:
        if source_chunk_id and source_chunk_id not in node.source_chunk_ids:
            node.source_chunk_ids.append(source_chunk_id)
        if paper_id and paper_id not in node.paper_ids:
            node.paper_ids.append(paper_id)
        if confidence > node.confidence:
            node.confidence = confidence
        if description and not node.description:
            node.description = description
