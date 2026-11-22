"""Unit tests for Phase 1A: Constrained Ontology, Structure-Aware Chunker & 3-Tier Entity Resolution."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT
from src.ingestion.chunker import StructureAwareChunker, compute_content_hash
from src.ingestion.entity_resolver import (
    EntityResolver,
    has_version_conflict,
    is_acronym_match,
)
from src.schema.ontology import (
    ChunkExtractionPayload,
    ComponentCategory,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
    filter_extractions_by_confidence,
)


SEED_ALIASES_YAML = PROJECT_ROOT / "data" / "manifest" / "seed_aliases.yaml"


# ============================================================================
# 1. Constrained Ontology Tests
# ============================================================================


def test_ontology_rejects_unconstrained_entity_and_relation_types() -> None:
    """Verify open-ended labels outside the 8 entity types and 12 relation types fail validation."""
    with pytest.raises(ValidationError):
        ExtractedEntity(
            entity_type="RandomConcept",  # type: ignore[arg-type]
            canonical_name="Something",
            confidence=0.9,
        )

    with pytest.raises(ValidationError):
        ExtractedRelationship(
            source_name="Llama-3",
            source_type=EntityType.MODEL_FAMILY,
            relation_type="USES_COMPONENT",  # type: ignore[arg-type]
            target_name="Grouped-Query Attention",
            target_type=EntityType.ARCH_COMPONENT,
            evidence_quote="Llama 3 uses grouped-query attention with 8 key-value heads.",
            confidence=0.95,
        )


def test_ontology_prevents_self_loops_and_filters_low_confidence() -> None:
    """Verify self-loops raise ValidationError and low-confidence triples are filtered out."""
    with pytest.raises(ValidationError):
        ExtractedRelationship(
            source_name="RoPE",
            source_type=EntityType.ARCH_COMPONENT,
            relation_type=RelationType.EXTENDS,
            target_name="rope",
            target_type=EntityType.ARCH_COMPONENT,
            evidence_quote="RoPE extends RoPE in a self-referential loop.",
            confidence=0.9,
        )

    payload = ChunkExtractionPayload(
        entities=[
            ExtractedEntity(
                entity_type=EntityType.ARCH_COMPONENT,
                canonical_name="SwiGLU",
                category=ComponentCategory.ACTIVATION_FFN,
                confidence=0.92,
            ),
            ExtractedEntity(
                entity_type=EntityType.ARCH_COMPONENT,
                canonical_name="Speculative Noise Gate",
                confidence=0.40,
            ),
        ],
        relationships=[],
    )
    filtered = filter_extractions_by_confidence(payload, min_confidence=0.65)
    assert len(filtered.entities) == 1
    assert filtered.entities[0].canonical_name == "SwiGLU"


# ============================================================================
# 2. Structure-Aware Chunker Tests
# ============================================================================


SAMPLE_PAPER_MARKDOWN = """# DeepSeek-V2: Economical and Efficient MoE Language Model

DeepSeek-V2 introduces Multi-head Latent Attention (MLA) and DeepSeekMoE.

## 2. Architecture

### 2.1 Multi-head Latent Attention (MLA)

Standard Multi-Head Attention (MHA) suffers from a heavy KV cache bottleneck during inference.
To reduce KV cache size beyond Grouped-Query Attention (GQA), MLA compresses keys and values into a latent vector:

$$
c_t^{KV} = W^{DKV} h_t, \\quad [k_t^C; v_t^C] = W^{UKV} c_t^{KV}
$$

| Attention Mechanism | KV Cache per Token | Performance vs MHA |
| --- | --- | --- |
| Multi-Head Attention (MHA) | $2 n_h d_h l$ | Baseline |
| Grouped-Query Attention (GQA) | $2 n_g d_h l$ | Slight degradation |
| Multi-head Latent Attention (MLA) | $(d_c + d_h^R) l$ | Stronger than MHA |

### 2.2 Decoupled Rotary Position Embedding

Because low-rank KV compression is incompatible with standard RoPE matrix absorption, we propose Decoupled RoPE.
"""


def test_structure_aware_chunker_preserves_equations_tables_and_deterministic_ids() -> None:
    """Verify chunker generates deterministic shared chunk_ids and keeps equations/tables intact."""
    chunker = StructureAwareChunker(target_chars=600, max_chars=1200, overlap_blocks=0)
    chunks_run1 = chunker.chunk_paper(
        paper_id="deepseek_2024_v2",
        paper_title="DeepSeek-V2",
        pub_year=2024,
        pillar="architecture",
        markdown_text=SAMPLE_PAPER_MARKDOWN,
    )
    chunks_run2 = chunker.chunk_paper(
        paper_id="deepseek_2024_v2",
        paper_title="DeepSeek-V2",
        pub_year=2024,
        pillar="architecture",
        markdown_text=SAMPLE_PAPER_MARKDOWN,
    )

    assert len(chunks_run1) >= 3
    # Deterministic IDs and content hashes across runs
    assert [c.chunk_id for c in chunks_run1] == [c.chunk_id for c in chunks_run2]
    assert [c.content_hash for c in chunks_run1] == [c.content_hash for c in chunks_run2]

    # Verify equation block was not split in half
    eq_chunks = [c for c in chunks_run1 if "W^{DKV}" in c.raw_text]
    assert len(eq_chunks) == 1
    assert eq_chunks[0].contains_equation is True
    assert "$$" in eq_chunks[0].raw_text

    # Verify markdown table was kept atomic in a single chunk
    table_chunks = [c for c in chunks_run1 if "| Attention Mechanism |" in c.raw_text]
    assert len(table_chunks) == 1
    assert table_chunks[0].contains_table is True
    assert "| Multi-head Latent Attention (MLA) |" in table_chunks[0].raw_text

    # Verify section hierarchy breadcrumb is embedded in context_prefixed_text
    assert "Section: DeepSeek-V2: Economical and Efficient MoE Language Model > 2. Architecture > 2.1 Multi-head Latent Attention (MLA)" in eq_chunks[0].context_prefixed_text


# ============================================================================
# 3. 3-Tier Entity Resolution & Version-Conflict Guard Tests
# ============================================================================


def test_entity_resolver_collapses_surface_variants_and_acronyms() -> None:
    """Verify RoPE and GQA surface variants collapse into single canonical nodes with merged alias lists."""
    resolver = EntityResolver(seed_aliases_path=SEED_ALIASES_YAML)

    variants = [
        "RoPE",
        "Rotary Position Embedding",
        "Rotary Positional Embeddings (Su et al., 2021)",
        "Rotary Position Embedding (RoPE)",
    ]
    resolved_ids = set()
    for idx, surface in enumerate(variants):
        node = resolver.resolve_entity(
            ExtractedEntity(
                entity_type=EntityType.ARCH_COMPONENT,
                canonical_name=surface,
                confidence=0.95,
            ),
            source_chunk_id=f"paper_x::s01::c0{idx}",
            paper_id="paper_x",
        )
        resolved_ids.add(node.entity_id)

    assert resolved_ids == {"arch_component::rope"}
    rope_node = resolver.nodes["arch_component::rope"]
    assert len(rope_node.source_chunk_ids) == 4


def test_entity_resolver_tier2_unseeded_acronym_collapse() -> None:
    """Verify an unseeded entity 'Native Sparse Attention (NSA)' and later 'NSA' resolve to one node."""
    resolver = EntityResolver(seed_aliases_path=SEED_ALIASES_YAML)

    first = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.ARCH_COMPONENT,
            canonical_name="Native Sparse Attention (NSA)",
            confidence=0.93,
        ),
        source_chunk_id="nsa_2025::s01::c00",
        paper_id="nsa_2025",
    )
    second = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.ARCH_COMPONENT,
            canonical_name="NSA",
            confidence=0.91,
        ),
        source_chunk_id="nsa_2025::s02::c01",
        paper_id="nsa_2025",
    )
    third = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.ARCH_COMPONENT,
            canonical_name="Native Sparse Attention",
            confidence=0.96,
        ),
        source_chunk_id="nsa_2025::s03::c00",
        paper_id="nsa_2025",
    )

    assert first.entity_id == "arch_component::native_sparse_attention"
    assert second.entity_id == first.entity_id
    assert third.entity_id == first.entity_id
    assert "NSA" in resolver.nodes[first.entity_id].aliases


def test_version_conflict_guard_prevents_merging_distinct_model_and_component_versions() -> None:
    """Verify LLaMA-1/2/3 and FlashAttention-1/2/3 never collapse even if embeddings are 0.99 identical."""
    assert has_version_conflict("LLaMA-1", "LLaMA-2") is True
    assert has_version_conflict("Llama 2", "Llama 3") is True
    assert has_version_conflict("FlashAttention", "FlashAttention-2") is True
    assert has_version_conflict("FlashAttention", "FlashAttention-1") is False
    assert has_version_conflict("DeepSeek-V2", "DeepSeek-V3") is True
    assert has_version_conflict("DeepSeek-V3", "DeepSeek-R1") is True

    # Even with a mock embedding function returning identical vectors [1.0, 0.0, 0.0],
    # version-conflicting entities must NOT merge!
    resolver = EntityResolver(
        seed_aliases_path= None,
        embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )

    n1 = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.INFERENCE_TECHNIQUE,
            canonical_name="FlashAttention",
            confidence=0.95,
        ),
        source_chunk_id="fa1::s01::c00",
        paper_id="fa1",
    )
    n2 = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.INFERENCE_TECHNIQUE,
            canonical_name="FlashAttention-2",
            confidence=0.95,
        ),
        source_chunk_id="fa2::s01::c00",
        paper_id="fa2",
    )
    assert n1.entity_id != n2.entity_id


def test_tier3_embedding_and_gray_zone_tiebreaker() -> None:
    """Verify Tier 3 auto-merges >= 0.88 and calls LLM tie-breaker in [0.82, 0.88)."""
    # Vector A = [1.0, 0.0], Vector B = [0.90, 0.43589] -> cosine sim = 0.90 (>= 0.88 auto-merge)
    # Vector C = [0.85, 0.52678] -> cosine sim = 0.85 (in [0.82, 0.88) gray zone)
    mock_vectors = {
        "Bottleneck: KV Cache Memory Footprint ()": [1.0, 0.0],
        "Bottleneck: Key-Value Cache RAM Overhead ()": [0.90, 0.43589],
        "Bottleneck: Autoregressive Decode Memory Bandwidth Limit ()": [0.85, 0.52678],
    }

    tiebreaker_calls = []

    def mock_tiebreaker(cand: str, target: str, etype: EntityType) -> bool:
        tiebreaker_calls.append((cand, target, etype))
        return True

    resolver = EntityResolver(
        seed_aliases_path=None,
        auto_merge_threshold=0.88,
        gray_zone_low_threshold=0.82,
        embed_fn=lambda texts: [mock_vectors.get(t, [0.0, 1.0]) for t in texts],
        llm_tiebreaker_fn=mock_tiebreaker,
    )

    base = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.BOTTLENECK,
            canonical_name="KV Cache Memory Footprint",
            confidence=0.9,
        ),
        source_chunk_id="c1",
        paper_id="p1",
    )
    auto_merged = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.BOTTLENECK,
            canonical_name="Key-Value Cache RAM Overhead",
            confidence=0.9,
        ),
        source_chunk_id="c2",
        paper_id="p1",
    )
    gray_zone_merged = resolver.resolve_entity(
        ExtractedEntity(
            entity_type=EntityType.BOTTLENECK,
            canonical_name="Autoregressive Decode Memory Bandwidth Limit",
            confidence=0.9,
        ),
        source_chunk_id="c3",
        paper_id="p2",
    )

    assert auto_merged.entity_id == base.entity_id
    assert gray_zone_merged.entity_id == base.entity_id
    assert len(tiebreaker_calls) == 1
    assert resolver.decision_logs[-2].tier_used == "TIER_3_EMBEDDING"
    assert resolver.decision_logs[-1].tier_used == "TIER_3_LLM_JUDGE"
