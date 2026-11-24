"""Regression tests for the Phase 1A fixes: signatures, typed resolution, acronyms, mentions, chunker."""

from __future__ import annotations

import pytest

from src.config import PROJECT_ROOT
from src.ingestion.chunker import StructureAwareChunker, contains_equation
from src.ingestion.entity_resolver import (
    EntityResolver,
    extract_version_signature,
    has_version_conflict,
    is_contrastive_sibling,
    split_name_and_parenthetical_acronym,
    strip_dense_scale_tokens,
)
from src.schema.ontology import EntityType, ExtractedEntity

SEEDS = PROJECT_ROOT / "data" / "manifest" / "seed_aliases.yaml"


@pytest.mark.parametrize("text, expected", [
    ("GPT-3", {"v:3"}), ("GPT3", {"v:3"}), ("GPT-3.5", {"v:3.5"}), ("Qwen2", {"v:2"}),
    ("Qwen2.5", {"v:2.5"}), ("Qwen-2.5", {"v:2.5"}), ("FlashAttention3", {"v:3"}),
    ("FlashAttention-3", {"v:3"}), ("Mamba2", {"v:2"}), ("DeepSeek-V2", {"v:2"}),
    ("DeepSeek-V3", {"v:3"}), ("DeepSeek-R1", {"v:r1"}), ("DeepSeek-R1-Zero", {"v:r1-zero"}),
    ("INT4", {"bits:4"}), ("4-bit", {"bits:4"}), ("INT8", {"bits:8"}), ("8-bit", {"bits:8"}),
    ("FP8", {"fp:8"}), ("FP16", {"fp:16"}), ("BF16", {"bf:16"}),
    ("Top-1", {"topk:1"}), ("Top-2", {"topk:2"}), ("Top-K", {"topk:k"}),
    ("Mixtral 8x7B", {"moe:8x7b"}), ("Llama 3 8B", {"v:3"}), ("O(N^2) Attention Complexity", set()),
])
def test_extract_version_signature(text: str, expected: set) -> None:
    assert extract_version_signature(text) == expected


def test_implicit_v1_rule_only_matches_v1() -> None:
    assert not has_version_conflict("FlashAttention", "FlashAttention-1")
    assert not has_version_conflict("LLaMA", "LLaMA-1")
    for other in ("Top-1 Routing", "INT4 Routing", "FP8 Routing", "Routing 2", "Routing-3"):
        assert has_version_conflict("Routing", other), other


def test_dense_scale_stripping_preserves_moe_grid() -> None:
    assert strip_dense_scale_tokens("Llama 2 (70B)") == ("Llama 2", ["70B"])
    assert strip_dense_scale_tokens("Llama 3 8B") == ("Llama 3", ["8B"])
    assert strip_dense_scale_tokens("Mixtral 8x7B") == ("Mixtral 8x7B", [])


def test_parenthetical_validation() -> None:
    assert split_name_and_parenthetical_acronym("Method A (ours)") == ("Method A", [])
    assert split_name_and_parenthetical_acronym("Baseline Model (baseline)") == ("Baseline Model", [])
    assert split_name_and_parenthetical_acronym("Model X (70B)") == ("Model X (70B)", [])
    assert split_name_and_parenthetical_acronym("Native Sparse Attention (NSA)") == ("Native Sparse Attention", ["NSA"])
    assert split_name_and_parenthetical_acronym("PagedAttention (vLLM)")[1] == []


def test_contrastive_siblings() -> None:
    assert is_contrastive_sibling("Method A", "Method B")
    assert is_contrastive_sibling("Multi-Head Attention", "Multi-Query Attention")
    assert not is_contrastive_sibling("Rotary Position Embedding", "Rotary Positional Embedding")
    assert not is_contrastive_sibling("KV Cache Memory Footprint", "Key-Value Cache RAM Overhead")


def _resolve(r: EntityResolver, etype: EntityType, name: str) -> str:
    return r.resolve_entity(ExtractedEntity(entity_type=etype, canonical_name=name, confidence=0.9), "c", "p").entity_id


def test_resolution_is_type_safe() -> None:
    r = EntityResolver(seed_aliases_path=SEEDS)
    assert _resolve(r, EntityType.PAPER, "DeepSeek-V2") == "paper::deepseek_v2"
    assert _resolve(r, EntityType.MODEL_FAMILY, "DeepSeek-V2") == "model_family::deepseek_v2"


def test_acronym_merge_is_order_independent() -> None:
    ids, names = set(), set()
    for order in (("NSA", "Native Sparse Attention (NSA)"), ("Native Sparse Attention (NSA)", "NSA")):
        r = EntityResolver(seed_aliases_path=SEEDS)
        got = {_resolve(r, EntityType.ARCH_COMPONENT, s) for s in order}
        assert len(got) == 1
        ids |= got
        node = r.nodes[got.pop()]
        names.add(node.canonical_name)
        assert "NSA" in node.aliases
    assert ids == {"arch_component::nsa"} and names == {"Native Sparse Attention"}


def test_model_family_scale_goes_to_metadata_not_alias_index() -> None:
    r = EntityResolver(seed_aliases_path=SEEDS)
    eid = _resolve(r, EntityType.MODEL_FAMILY, "Llama 2 (70B)")
    assert eid == "model_family::llama_2" and r.nodes[eid].parameter_scales == ["70B"]
    assert (EntityType.MODEL_FAMILY, "70b") not in r.alias_index


def test_seed_splits() -> None:
    r = EntityResolver(seed_aliases_path=SEEDS)
    assert _resolve(r, EntityType.MODEL_FAMILY, "R1-Zero") == "model_family::deepseek_r1_zero"
    assert _resolve(r, EntityType.MODEL_FAMILY, "DeepSeek R1") == "model_family::deepseek_r1"
    assert _resolve(r, EntityType.BOTTLENECK, "HBM Read/Write Materialization Overhead") == "bottleneck::hbm_io_materialization_overhead"
    assert _resolve(r, EntityType.MODEL_FAMILY, "Llama 3.1") != "model_family::llama_3"


def test_mentions_longest_span_and_alias() -> None:
    r = EntityResolver(seed_aliases_path=SEEDS)
    found = r.resolve_text_mentions_in_chunk("Llama 2 uses GQA, unlike LLaMA. FlashAttention-2 beats FlashAttention.")
    assert {"model_family::llama_2", "model_family::llama_1", "arch_component::grouped_query_attention",
            "inference_technique::flash_attention_2", "inference_technique::flash_attention"} <= set(found)
    assert r.resolve_text_mentions_in_chunk("Llama 2 only.") == ["model_family::llama_2"]
    assert r.resolve_Text_mentions_in_chunk("We apply RoPE.") == ["arch_component::rope"]


@pytest.mark.parametrize("text, expected", [
    ("Training cost was $2M on cloud GPUs.", False),
    ("API pricing is $0.14 per million input tokens versus $0.55 for the baseline.", False),
    ("A $100,000 budget.", False),
    ("costs $5 and $10", False),
    ("Complexity is $O(N^2)$ in sequence length.", True),
    ("Head dim $d_k$ is 128.", True),
    ("$c_t^{KV} = W h_t$", True),
    ("$$ x = y $$", True),
    ("\\begin{equation} a = b \\end{equation}", True),
])
def test_contains_equation(text: str, expected: bool) -> None:
    assert contains_equation(text) is expected


def test_chunker_ignores_headings_in_code_fences_and_supports_h6() -> None:
    md = "# Top\n\nIntro.\n\n```python\n# not a heading\n\nx = 1\n```\n\n###### Deep\n\nBody."
    chunks = StructureAwareChunker().chunk_paper("p", "P", 2024, "architecture", md)
    assert [c.section_path for c in chunks] == ["Top", "Top > Deep"]
    assert "# not a heading" in chunks[0].raw_text and "x = 1" in chunks[0].raw_text
    assert StructureAwareChunker().overlap_blocks == 0
