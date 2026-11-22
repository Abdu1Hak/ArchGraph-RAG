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
- Step 0 (Surface Normalization & Hard Version Guard):
  Strips inline citations `(Su et al., 2021)`, extracts parenthetical acronyms
  `Grouped-Query Attention (GQA)` -> `name="Grouped-Query Attention", alias="GQA"`,
  and extracts version/scale tokens (`v1`, `v2`, `v3`, `1`, `2`, `3`, `r1`, `7b`, `70b`).
  **Hard Rule:** Two entities with conflicting explicit version signatures can NEVER merge.
- Tier 1 (O(1) Canonical & Alias Dictionary Lookup):
  Matches normalized surface form or extracted acronym against `seed_aliases.yaml` and
  all previously resolved entities in the graph.
- Tier 2 (Acronym Initials & High-Precision Lexical Match):
  Checks if one string is the exact initialism of the other (e.g., "GQA" <-> "Grouped Query Attention")
  or has SequenceMatcher / token similarity >= 0.91 within the same `EntityType`.
- Tier 3 (Dense Embedding Cosine Similarity + Gray-Zone Tie-Breaker):
  Computes cosine similarity using `gemini-embedding-001` embeddings:
  - `sim >= 0.88` (and no version conflict) -> Auto-merge into canonical node & append alias.
  - `0.82 <= sim < 0.88` -> Optional pairwise tie-breaker callback (or conservative split).
  - `sim < 0.82` -> Mint a new canonical node and index its aliases + embedding.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Set, Tuple
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
_VERSION_TOKEN_RE = re.compile(
    r"(?:^|[\s\-_])(?P<ver>v?\d+(?:\.\d+)?|r\d+|\d+x\d+b|\d+b)(?:$|[\s\-_])",
    re.IGNORECASE,
)


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
    return " ".join(tokens)


def split_name_and_parenthetical_acronym(raw_name: str) -> Tuple[str, List[str]]:
    """Split 'Rotary Position Embedding (RoPE)' into ('Rotary Position Embedding', ['RoPE'])."""
    cleaned = _CITATION_NOISE_RE.sub("", raw_name.strip())
    m = _TRAILING_PAREN_ACRONYM_RE.match(cleaned)
    if m:
        full_part = m.group("full").strip()
        acro_part = m.group("acro").strip()
        # Check that parenthetical isn't a year like (2023)
        if not acro_part.isdigit():
            return full_part, [acro_part]
    return cleaned, []


def extract_version_signature(text: str) -> Set[str]:
    """Extract normalized version or scale tokens (e.g., {'2'}, {'v3'}, {'r1'}, {'70b'}).

    Note: Bare base names like 'FlashAttention' or 'LLaMA' next to 'FlashAttention-2' or 'LLaMA-2'
    are handled by checking whether one has a version > 1 while the other has no version or version 1.
    """
    norm = text.lower().strip()
    matches = _VERSION_TOKEN_RE.findall(norm)
    signatures: Set[str] = set()
    for token in matches:
        t = token.lower().lstrip("v")
        signatures.add(t)
    return signatures


def has_version_conflict(name_a: str, name_b: str) -> bool:
    """Return True if `name_a` and `name_b` have incompatible version/generation numbers.

    Examples:
    - 'LLaMA-1' vs 'LLaMA-2' -> True (conflict: {'1'} vs {'2'})
    - 'FlashAttention' vs 'FlashAttention-2' -> True (implicit v1 vs explicit {'2'})
    - 'FlashAttention' vs 'FlashAttention-1' -> False (implicit v1 matches explicit {'1'})
    - 'DeepSeek-V2' vs 'DeepSeek-V3' -> True (conflict: {'2'} vs {'3'})
    - 'DeepSeek-V3' vs 'DeepSeek-R1' -> True (conflict: {'3'} vs {'r1'})
    """
    sig_a = extract_version_signature(name_a)
    sig_b = extract_version_signature(name_b)

    if sig_a and sig_b:
        return sig_a != sig_b

    # If one has a version token and the other doesn't, check if they share the same base stem
    if bool(sig_a) != bool(sig_b):
        non_empty = sig_a or sig_b
        # If the explicit version is just "1", allow matching with bare base name (e.g. FlashAttention == FlashAttention-1)
        if non_empty == {"1"}:
            return False
        # Otherwise, e.g. "FlashAttention" vs "FlashAttention-2" -> CONFLICT!
        return True

    return False


def is_acronym_match(short_str: str, long_str: str) -> bool:
    """Check if `short_str` (e.g., 'GQA' or 'RoPE') is a plausible initialism of `long_str`."""
    s = re.sub(r"[^a-zA-Z0-9]", "", short_str).lower()
    words = [
        w.lower()
        for w in re.split(r"[\s\-]+", long_str.strip())
        if w and w.lower() not in {"of", "the", "and", "with", "for", "in", "to"}
    ]
    if len(s) < 2 or len(words) < 2:
        return False
    initials = "".join(w[0] for w in words)
    if s == initials:
        return True
    # Also support prefix-syllable acronyms like RoPE -> Rotary Position Embedding ("ro" + "p" + "e")
    if len(words) >= 3 and s == (words[0][:2] + "".join(w[0] for w in words[1:])):
        return True
    return False


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
        # Fast O(1) lookup: (entity_type, normalized_surface_key) -> entity_id
        self.alias_index: Dict[Tuple[EntityType, str], str] = {}
        # Global cross-type alias lookup for unambiguous canonical names/acronyms
        self.global_alias_index: Dict[str, str] = {}
        # Cached unit-norm embeddings per entity_id for Tier 3
        self.node_embeddings: Dict[str, np.ndarray] = {}
        # Audit log of all resolution decisions
        self.decision_logs: List[ResolutionDecisionLog] = []

        if seed_aliases_path and seed_aliases_path.exists():
            self.load_seed_aliases(seed_aliases_path)

    def load_seed_aliases(self, yaml_path: Path) -> None:
        """Load curated canonical entities and aliases from `seed_aliases.yaml`."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item in data.get("entities", []):
            etype = EntityType(item["entity_type"])
            entity_id = item["entity_id"]
            canonical_name = item["canonical_name"]
            aliases = list(item.get("aliases", []))

            node = ResolvedEntityNode(
                entity_id=entity_id,
                entity_type=etype,
                canonical_name=canonical_name,
                aliases=aliases,
                category=item.get("category"),
                confidence=1.0,
            )
            self._register_node(node)

    def _register_node(self, node: ResolvedEntityNode) -> None:
        """Register or update a canonical node and index all its surface forms."""
        self.nodes[node.entity_id] = node
        all_names = [node.canonical_name] + list(node.aliases)
        for surface in all_names:
            norm_key = normalize_surface_key(surface)
            if norm_key:
                self.alias_index[(node.entity_type, norm_key)] = node.entity_id
                self.global_alias_index[norm_key] = node.entity_id

    def _add_alias_to_node(self, entity_id: str, new_alias: str) -> None:
        """Append a newly discovered surface form to an existing canonical node's alias list."""
        node = self.nodes[entity_id]
        clean_alias = new_alias.strip()
        existing_lower = {a.lower() for a in node.aliases} | {node.canonical_name.lower()}
        if clean_alias and clean_alias.lower() not in existing_lower:
            node.aliases.append(clean_alias)
        norm_key = normalize_surface_key(clean_alias)
        if norm_key:
            self.alias_index[(node.entity_type, norm_key)] = entity_id
            self.global_alias_index[norm_key] = entity_id

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

    def resolve_entity(
        self,
        extracted: ExtractedEntity,
        source_chunk_id: str,
        paper_id: str,
    ) -> ResolvedEntityNode:
        """Resolve an `ExtractedEntity` into a canonical `ResolvedEntityNode` via the 3-tier funnel."""
        primary_name, paren_acronyms = split_name_and_parenthetical_acronym(
            extracted.canonical_name
        )
        candidate_surfaces = [primary_name] + paren_acronyms + list(extracted.aliases)
        etype = extracted.entity_type

        # ---------------------------------------------------------------------
        # TIER 1: Exact Normalized Canonical / Alias Lookup (with Version Guard)
        # ---------------------------------------------------------------------
        for surf in candidate_surfaces:
            norm_key = normalize_surface_key(surf)
            if not norm_key:
                continue
            matched_id = self.alias_index.get((etype, norm_key)) or self.global_alias_index.get(norm_key)
            if matched_id:
                target_node = self.nodes[matched_id]
                if not has_version_conflict(primary_name, target_node.canonical_name):
                    for s in candidate_surfaces:
                        self._add_alias_to_node(matched_id, s)
                    self._record_provenance(
                        target_node, source_chunk_id, paper_id, extracted.confidence, extracted.description
                    )
                    self.decision_logs.append(
                        ResolutionDecisionLog(
                            raw_name=extracted.canonical_name,
                            entity_type=etype.value,
                            resolved_entity_id=matched_id,
                            resolved_canonical_name=target_node.canonical_name,
                            tier_used="TIER_1_ALIAS",
                            similarity_score=1.0,
                        )
                    )
                    return target_node

        # ---------------------------------------------------------------------
        # TIER 2: Acronym Initials Expansion & High-Precision Lexical Match
        # ---------------------------------------------------------------------
        best_lexical_id: Optional[str] = None
        best_lexical_score: float = 0.0

        same_type_nodes = [n for n in self.nodes.values() if n.entity_type == etype]
        for node in same_type_nodes:
            if has_version_conflict(primary_name, node.canonical_name):
                continue

            node_surfaces = [node.canonical_name] + list(node.aliases)
            for cand_s in candidate_surfaces:
                for node_s in node_surfaces:
                    if is_acronym_match(cand_s, node_s) or is_acronym_match(node_s, cand_s):
                        best_lexical_id = node.entity_id
                        best_lexical_score = 0.99
                        break
                    ratio = SequenceMatcher(
                        None,
                        normalize_surface_key(cand_s),
                        normalize_surface_key(node_s),
                    ).ratio()
                    if ratio > best_lexical_score:
                        best_lexical_score = ratio
                        best_lexical_id = node.entity_id
                if best_lexical_score >= 0.99:
                    break

        if best_lexical_id and best_lexical_score >= 0.91:
            target_node = self.nodes[best_lexical_id]
            for s in candidate_surfaces:
                self._add_alias_to_node(best_lexical_id, s)
            self._record_provenance(
                target_node, source_chunk_id, paper_id, extracted.confidence, extracted.description
            )
            self.decision_logs.append(
                ResolutionDecisionLog(
                    raw_name=extracted.canonical_name,
                    entity_type=etype.value,
                    resolved_entity_id=best_lexical_id,
                    resolved_canonical_name=target_node.canonical_name,
                    tier_used="TIER_2_ACRONYM_LEXICAL",
                    similarity_score=round(best_lexical_score, 4),
                )
            )
            return target_node

        # ---------------------------------------------------------------------
        # TIER 3: Embedding Cosine Similarity + Gray-Zone LLM Tie-Breaker
        # ---------------------------------------------------------------------
        if self.embed_fn is not None and same_type_nodes:
            query_repr = f"{etype.value}: {primary_name} ({', '.join(extracted.aliases)})"
            vecs = self.embed_fn([query_repr])
            if vecs and vecs[0]:
                q_vec = np.asarray(vecs[0], dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                best_emb_id: Optional[str] = None
                best_emb_sim: float = -1.0

                for node in same_type_nodes:
                    if has_version_conflict(primary_name, node.canonical_name):
                        continue
                    node_repr = f"{node.entity_type.value}: {node.canonical_name} ({', '.join(node.aliases)})"
                    n_vec = self._get_or_compute_embedding(node.entity_id, node_repr)
                    if n_vec is not None:
                        sim = float(np.dot(q_vec, n_vec))
                        if sim > best_emb_sim:
                            best_emb_sim = sim
                            best_emb_id = node.entity_id

                if best_emb_id is not None:
                    if best_emb_sim >= self.auto_merge_threshold:
                        target_node = self.nodes[best_emb_id]
                        for s in candidate_surfaces:
                            self._add_alias_to_node(best_emb_id, s)
                        self._record_provenance(
                            target_node, source_chunk_id, paper_id, extracted.confidence, extracted.description
                        )
                        self.decision_logs.append(
                            ResolutionDecisionLog(
                                raw_name=extracted.canonical_name,
                                entity_type=etype.value,
                                resolved_entity_id=best_emb_id,
                                resolved_canonical_name=target_node.canonical_name,
                                tier_used="TIER_3_EMBEDDING",
                                similarity_score=round(best_emb_sim, 4),
                            )
                        )
                        return target_node

                    if (
                        best_emb_sim >= self.gray_zone_low_threshold
                        and self.llm_tiebreaker_fn is not None
                    ):
                        target_node = self.nodes[best_emb_id]
                        if self.llm_tiebreaker_fn(
                            primary_name, target_node.canonical_name, etype
                        ):
                            for s in candidate_surfaces:
                                self._add_alias_to_node(best_emb_id, s)
                            self._record_provenance(
                                target_node, source_chunk_id, paper_id, extracted.confidence, extracted.description
                            )
                            self.decision_logs.append(
                                ResolutionDecisionLog(
                                    raw_name=extracted.canonical_name,
                                    entity_type=etype.value,
                                    resolved_entity_id=best_emb_id,
                                    resolved_canonical_name=target_node.canonical_name,
                                    tier_used="TIER_3_LLM_JUDGE",
                                    similarity_score=round(best_emb_sim, 4),
                                )
                            )
                            return target_node

        # ---------------------------------------------------------------------
        # FALLBACK: Mint a New Canonical Node
        # ---------------------------------------------------------------------
        new_id = slugify_entity_id(etype, primary_name)
        all_aliases = [
            a for a in (paren_acronyms + list(extracted.aliases))
            if a.lower() != primary_name.lower()
        ]
        new_node = ResolvedEntityNode(
            entity_id=new_id,
            entity_type=etype,
            canonical_name=primary_name,
            aliases=all_aliases,
            category=extracted.category.value if extracted.category else None,
            description=extracted.description,
            source_chunk_ids=[source_chunk_id],
            paper_ids=[paper_id],
            confidence=extracted.confidence,
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
            ExtractedEntity(
                entity_type=rel.source_type,
                canonical_name=rel.source_name,
                confidence=rel.confidence,
            ),
            source_chunk_id=source_chunk_id,
            paper_id=paper_id,
        )
        tgt_entity = self.resolve_entity(
            ExtractedEntity(
                entity_type=rel.target_type,
                canonical_name=rel.target_name,
                confidence=rel.confidence,
            ),
            source_chunk_id=source_chunk_id,
            paper_id=paper_id,
        )
        if src_entity.entity_id == tgt_entity.entity_id:
            return None

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

    def resolve_Text_mentions_in_chunk(self, raw_text: str) -> List[str]:
        """Scan a chunk's text for known canonical entities & aliases to populate `entity_ids[]` in pgvector."""
        norm_chunk = f" {normalize_surface_key(raw_text)} "
        matched_ids: Set[str] = set()
        for norm_alias, entity_id in self.global_alias_index.items():
            if len(norm_alias) >= 3 and f" {norm_alias} " in norm_chunk:
                matched_ids.add(entity_id)
        return sorted(matched_ids)

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
