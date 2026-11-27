"""Graph Path Verbalizer (Phase 4): TemplateResult paths/aggregates -> cited natural-language facts.

Every fact carries inline `[chunk_id]` tags taken from the edge's `source_chunk_ids`, so the
generator can copy graph facts into its answer with citations the validator will accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.cypher_templates import CypherTemplateId, TemplateResult

REL_PHRASES: Dict[str, str] = {
    "INTRODUCES": "introduces",
    "ADOPTS_COMPONENT": "adopts",
    "REPLACES_IN": "replaces",
    "EXTENDS": "extends",
    "MITIGATES": "mitigates",
    "CAUSES_BOTTLENECK": "causes the bottleneck",
    "OUTPERFORMS_ON": "outperforms on",
    "EVALUATED_ON": "is evaluated on",
    "MODIFIES_HYPERPARAMETER": "modifies the hyperparameter",
    "REQUIRES": "requires",
    "CITES_INSPIRED_BY": "is inspired by",
    "MENTIONED_IN": "is mentioned in",
}


PASSIVE: Dict[str, str] = {
    "introduces": "is introduced by", "adopts": "is adopted by", "replaces": "is replaced by", "extends": "is extended by",
    "mitigates": "is mitigated by", "requires": "is required by", "is inspired by": "inspired",
}


@dataclass(frozen=True)
class VerbalizedFact:
    text: str
    chunk_ids: Tuple[str, ...]
    kind: str  # "hop" | "path" | "aggregate"
    edges: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)


def cite(chunk_ids) -> str:
    return " ".join(f"[{c}]" for c in dict.fromkeys(chunk_ids))


def _name(h: Dict[str, Any], side: str) -> str:
    return h.get(f"{side}_name") or h[side].split("::", 1)[-1].replace("_", " ")


def verbalize_hop(h: Dict[str, Any]) -> VerbalizedFact:
    rel = REL_PHRASES.get(h["relation_type"], h["relation_type"].replace("_", " ").lower())
    sentence = f"{_name(h, 'source')} {rel} {_name(h, 'target')}"
    if h["relation_type"] == "REPLACES_IN" and h.get("context"):
        sentence += f" in {h['context']}"
    quote = next(iter(h.get("evidence_quotes") or []), "").strip().rstrip(".")
    if quote:
        sentence += f', as the source states: "{quote}"'
    ids = tuple(dict.fromkeys(h.get("source_chunk_ids") or []))
    return VerbalizedFact(f"{sentence} {cite(ids)}.".replace("  ", " "), ids, "hop",
                          ((h["source"], h["relation_type"], h["target"]),))


def _walk_order(path: List[Dict[str, Any]]) -> List[str]:
    """Node sequence of a path, starting from the endpoint of hop 0 not shared with hop 1."""
    start = path[0]["source"]
    if len(path) > 1 and start in (path[1]["source"], path[1]["target"]):
        start = path[0]["target"]
    nodes = [start]
    for h in path:
        nodes.append(h["target"] if h["source"] == nodes[-1] else h["source"])
    return nodes


def verbalize_path(path: List[Dict[str, Any]]) -> VerbalizedFact:
    """'A extends B, which extends C [c1] [c2].' (direction-aware: '..., which is extended by ...')."""
    nodes = _walk_order(path)
    names: Dict[str, str] = {}
    for h in path:
        names[h["source"]], names[h["target"]] = _name(h, "source"), _name(h, "target")
    parts = [names[nodes[0]]]
    for i, h in enumerate(path):
        rel = REL_PHRASES.get(h["relation_type"], h["relation_type"].lower())
        fwd = h["source"] == nodes[i]
        # Walking against an edge reads in the passive: "X extends Y" seen from Y is "Y is extended by X".
        clause = f"{rel} {names[nodes[i + 1]]}" if fwd else f"{PASSIVE.get(rel, f'is related via {rel} to')} {names[nodes[i + 1]]}"
        parts.append(clause if i == 0 else f"which {clause}")
    ids = tuple(dict.fromkeys(c for h in path for c in h.get("source_chunk_ids") or []))
    text = f"Multi-hop path ({len(path)} hops): {parts[0]} " + ", ".join(parts[1:]) + f" {cite(ids)}."
    return VerbalizedFact(text, ids, "path", tuple((h["source"], h["relation_type"], h["target"]) for h in path))


def verbalize_aggregates(result: TemplateResult) -> List[VerbalizedFact]:
    rel = REL_PHRASES.get(result.params.get("relation_type", ""), "is linked to")
    passive = {"adopts": "adopted", "mitigates": "mitigated", "extends": "extended", "introduces": "introduced"}.get(rel, rel)
    facts = []
    for agg in result.aggregates:
        if "target" not in agg:
            continue
        hops = [p[0] for p in result.paths if p[0]["target"] == agg["target"]]
        ids = tuple(dict.fromkeys(c for h in hops for c in h["source_chunk_ids"]))
        sources = ", ".join(sorted({_name(h, "source") for h in hops}))
        noun = "source" if agg["count"] == 1 else "sources"
        facts.append(VerbalizedFact(f"Corpus aggregate: {agg['target_name'] or agg['target']} is {passive} by {agg['count']} {noun} "
                                    f"({sources}) {cite(ids)}.", ids, "aggregate",
                                    tuple((h["source"], h["relation_type"], h["target"]) for h in hops)))
    return facts


def verbalize_template_result(result: Optional[TemplateResult], max_paths: int = 6) -> List[VerbalizedFact]:
    """Aggregates (T5) as summary bullets; otherwise multi-hop path sentences + one sentence per unique hop."""
    if result is None or not result.paths:
        return []
    if result.template_id == CypherTemplateId.T5_CORPUS_AGGREGATION:
        return verbalize_aggregates(result)
    facts: List[VerbalizedFact] = []
    seen_paths = set()
    for path in result.paths[:max_paths]:
        key = tuple((h["source"], h["relation_type"], h["target"]) for h in path)
        if len(path) > 1 and key not in seen_paths:
            seen_paths.add(key)
            facts.append(verbalize_path(path))
    seen_hops = set()
    for path in result.paths[:max_paths]:
        for h in path:
            k = (h["source"], h["relation_type"], h["target"])
            if k not in seen_hops and h.get("source_chunk_ids"):
                seen_hops.add(k)
                facts.append(verbalize_hop(h))
    return facts
