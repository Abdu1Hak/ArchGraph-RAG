"""Parameterized Cypher Template Catalog T1-T6 (Phase 3), runnable on Neo4j and the in-memory graph.

Every template is a fixed Cypher string: user-derived values travel ONLY as `$params`.
Relationship-type lists inside patterns are constants of the template, never user input;
T5's relation filter uses `type(r) = $relation_type` so it stays a bound parameter.

T1 COMPONENT_LINEAGE        1..3 hops over EXTENDS / REPLACES_IN from $root_entity_id
T2 MODEL_ARCHITECTURE_DIFF  components / methods / hyperparameters of $model_a_id vs $model_b_id
T3 BOTTLENECK_SOLVERS       1..2 hops over MITIGATES / CAUSES_BOTTLENECK / ADOPTS_COMPONENT from $bottleneck_id
T4 BRIDGE_PATH_2_TO_3_HOP   shortest 1..3-hop path(s) between $entity_a_id and $entity_b_id
T5 CORPUS_AGGREGATION       count targets of $relation_type, filtered by $category / $min_year
T6 DYNAMIC_SUBGRAPH_FALLBACK 1..2 hops from $seed_entity_ids over all semantic edges, then each
                            hop (verbalized triple + evidence) is scored against the query with a
                            CrossEncoderReranker and the top hops are returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.schema.ontology import RelationType
from src.stores.pgvector_store import CrossEncoderReranker, TokenInteractionReranker

PLUMBING_RELS = ("IN_PAPER", "MENTIONED_IN")
T1_RELS = ("EXTENDS", "REPLACES_IN")
T2_RELS = ("ADOPTS_COMPONENT", "MODIFIES_HYPERPARAMETER", "REPLACES_IN", "EVALUATED_ON", "OUTPERFORMS_ON", "REQUIRES")
T3_RELS = ("MITIGATES", "CAUSES_BOTTLENECK", "ADOPTS_COMPONENT")
_ENTITY_ID_RE = re.compile(r"^[a-z_]+::[a-z0-9_]{1,160}$")


class CypherTemplateId(str, Enum):
    T1_COMPONENT_LINEAGE = "T1_COMPONENT_LINEAGE"
    T2_MODEL_ARCHITECTURE_DIFF = "T2_MODEL_ARCHITECTURE_DIFF"
    T3_BOTTLENECK_SOLVERS = "T3_BOTTLENECK_SOLVERS"
    T4_BRIDGE_PATH_2_TO_3_HOP = "T4_BRIDGE_PATH_2_TO_3_HOP"
    T5_CORPUS_AGGREGATION = "T5_CORPUS_AGGREGATION"
    T6_DYNAMIC_SUBGRAPH_FALLBACK = "T6_DYNAMIC_SUBGRAPH_FALLBACK"


_HOPS = (
    "[r IN relationships(p) | {source: startNode(r).entity_id, source_name: startNode(r).canonical_name, "
    "relation_type: type(r), target: endNode(r).entity_id, target_name: endNode(r).canonical_name, "
    "source_chunk_ids: coalesce(r.source_chunk_ids, []), evidence_quotes: coalesce(r.evidence_quotes, [])}] AS hops"
)

CYPHER: Dict[CypherTemplateId, str] = {
    CypherTemplateId.T1_COMPONENT_LINEAGE: (
        "MATCH p = (root:Entity {entity_id: $root_entity_id})-[:EXTENDS|REPLACES_IN*1..3]-(x:Entity) "
        f"RETURN {_HOPS} ORDER BY length(p), x.entity_id LIMIT $limit"
    ),
    CypherTemplateId.T2_MODEL_ARCHITECTURE_DIFF: (
        "MATCH p = (m:Entity)-[r]->(x:Entity) WHERE m.entity_id IN $model_ids "
        f"AND type(r) IN {list(T2_RELS)!r} "
        f"RETURN {_HOPS} ORDER BY m.entity_id, type(r), x.entity_id LIMIT $limit"
    ).replace("'", '"'),
    CypherTemplateId.T3_BOTTLENECK_SOLVERS: (
        "MATCH p = (b:Entity {entity_id: $bottleneck_id})-[:MITIGATES|CAUSES_BOTTLENECK|ADOPTS_COMPONENT*1..2]-(x:Entity) "
        f"RETURN {_HOPS} ORDER BY length(p), x.entity_id LIMIT $limit"
    ),
    CypherTemplateId.T4_BRIDGE_PATH_2_TO_3_HOP: (
        "MATCH (a:Entity {entity_id: $entity_a_id}), (b:Entity {entity_id: $entity_b_id}) "
        "MATCH p = allShortestPaths((a)-[*1..3]-(b)) "
        "WHERE none(r IN relationships(p) WHERE type(r) IN [\"IN_PAPER\", \"MENTIONED_IN\"]) "
        f"RETURN {_HOPS} LIMIT $limit"
    ),
    CypherTemplateId.T5_CORPUS_AGGREGATION: (
        "MATCH p = (s:Entity)-[r]->(t:Entity) "
        "WHERE type(r) = $relation_type AND ($category IS NULL OR t.category = $category) "
        "AND ($min_year IS NULL OR EXISTS { MATCH (pp:Paper) WHERE pp.paper_id IN r.paper_ids AND pp.year >= $min_year }) "
        f"RETURN {_HOPS} ORDER BY t.entity_id, s.entity_id LIMIT $limit"
    ),
    CypherTemplateId.T6_DYNAMIC_SUBGRAPH_FALLBACK: (
        "MATCH p = (s:Entity)-[*1..2]-(x:Entity) WHERE s.entity_id IN $seed_entity_ids "
        "AND none(r IN relationships(p) WHERE type(r) IN [\"IN_PAPER\", \"MENTIONED_IN\"]) "
        f"RETURN {_HOPS} LIMIT $limit"
    ),
}

REQUIRED: Dict[CypherTemplateId, Tuple[str, ...]] = {
    CypherTemplateId.T1_COMPONENT_LINEAGE: ("root_entity_id",),
    CypherTemplateId.T2_MODEL_ARCHITECTURE_DIFF: ("model_a_id",),
    CypherTemplateId.T3_BOTTLENECK_SOLVERS: ("bottleneck_id",),
    CypherTemplateId.T4_BRIDGE_PATH_2_TO_3_HOP: ("entity_a_id", "entity_b_id"),
    CypherTemplateId.T5_CORPUS_AGGREGATION: ("relation_type",),
    CypherTemplateId.T6_DYNAMIC_SUBGRAPH_FALLBACK: ("seed_entity_ids",),
}


@dataclass
class TemplateResult:
    template_id: CypherTemplateId
    params: Dict[str, Any]
    paths: List[List[Dict[str, Any]]] = field(default_factory=list)
    aggregates: List[Dict[str, Any]] = field(default_factory=list)
    hop_scores: List[float] = field(default_factory=list)  # T6: cross-encoder score per returned path

    @property
    def unique_hops(self) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for path in self.paths:
            for h in path:
                key = (h["source"], h["relation_type"], h["target"])
                if key not in seen:
                    seen.add(key)
                    out.append(h)
        return out

    @property
    def source_chunk_ids(self) -> List[str]:
        return list(dict.fromkeys(cid for h in self.unique_hops for cid in h["source_chunk_ids"]))


def verbalize_hop(h: Dict[str, Any]) -> str:
    rel = h["relation_type"].replace("_", " ").lower()
    quotes = " ".join(h.get("evidence_quotes") or [])
    return f"{h.get('source_name') or h['source']} {rel} {h.get('target_name') or h['target']}. {quotes}".strip()


def validate_params(template_id: CypherTemplateId, params: Dict[str, Any]) -> Dict[str, Any]:
    """Type-check and normalize params. Values are bound as $params either way; this just fails fast."""
    p = dict(params)
    missing = [k for k in REQUIRED[template_id] if not p.get(k)]
    if missing:
        raise ValueError(f"{template_id.value} missing params: {missing}")
    for k in ("root_entity_id", "model_a_id", "model_b_id", "bottleneck_id", "entity_a_id", "entity_b_id"):
        if p.get(k) is not None and (not isinstance(p[k], str) or len(p[k]) > 200):
            raise ValueError(f"{k} must be a string <= 200 chars")
    if template_id == CypherTemplateId.T2_MODEL_ARCHITECTURE_DIFF:
        p["model_ids"] = [m for m in (p.get("model_a_id"), p.get("model_b_id")) if m]
    if template_id == CypherTemplateId.T5_CORPUS_AGGREGATION:
        p["relation_type"] = RelationType(p["relation_type"]).value
        p.setdefault("category", None)
        p["min_year"] = int(p["min_year"]) if p.get("min_year") is not None else None
    if template_id == CypherTemplateId.T6_DYNAMIC_SUBGRAPH_FALLBACK:
        seeds = p["seed_entity_ids"]
        if isinstance(seeds, str) or not all(isinstance(s, str) for s in seeds):
            raise ValueError("seed_entity_ids must be a list of strings")
        p["seed_entity_ids"] = list(dict.fromkeys(seeds))[:20]
    p["limit"] = max(1, min(int(p.get("limit", 50)), 200))
    return p


# ---------------------------------------------------------------------------
# In-memory execution (same semantics as the Cypher above)
# ---------------------------------------------------------------------------


def _hop(store: Any, key: Tuple[str, str, str], props: Dict[str, Any]) -> Dict[str, Any]:
    s, rt, t = key
    return {"source": s, "source_name": store.nodes.get(s, {}).get("canonical_name"), "relation_type": rt,
            "target": t, "target_name": store.nodes.get(t, {}).get("canonical_name"),
            "source_chunk_ids": list(props.get("source_chunk_ids", [])),
            "evidence_quotes": list(props.get("evidence_quotes", []))}


def _adjacency(store: Any, rel_types: Optional[Iterable[str]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    allowed = set(rel_types) if rel_types else None
    adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, props in store.rels.items():
        s, rt, t = key
        if rt in PLUMBING_RELS or (allowed is not None and rt not in allowed):
            continue
        hop = _hop(store, key, props)
        adj.setdefault(s, []).append((t, hop))
        adj.setdefault(t, []).append((s, hop))
    return adj


def _walk(adj: Dict[str, List[Tuple[str, Dict[str, Any]]]], start: str, max_hops: int) -> List[List[Dict[str, Any]]]:
    """All simple undirected paths of length 1..max_hops from start (hops keep true edge direction)."""
    out: List[List[Dict[str, Any]]] = []

    def rec(node: str, visited: Set[str], acc: List[Dict[str, Any]]) -> None:
        if len(acc) == max_hops:
            return
        for nxt, hop in sorted(adj.get(node, []), key=lambda x: (x[1]["relation_type"], x[0])):
            if nxt in visited:
                continue
            acc.append(hop)
            out.append(list(acc))
            rec(nxt, visited | {nxt}, acc)
            acc.pop()

    rec(start, {start}, [])
    return sorted(out, key=len)


def _shortest_paths(adj: Dict[str, List[Tuple[str, Dict[str, Any]]]], a: str, b: str, max_hops: int) -> List[List[Dict[str, Any]]]:
    for length in range(1, max_hops + 1):
        found = [p for p in _walk(adj, a, length) if len(p) == length and _endpoint(a, p) == b]
        if found:
            return found
    return []


def _endpoint(start: str, path: List[Dict[str, Any]]) -> str:
    node = start
    for h in path:
        node = h["target"] if h["source"] == node else h["source"]
    return node


def _run_in_memory(store: Any, tid: CypherTemplateId, p: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    if tid == CypherTemplateId.T1_COMPONENT_LINEAGE:
        return _walk(_adjacency(store, T1_RELS), p["root_entity_id"], 3)
    if tid == CypherTemplateId.T2_MODEL_ARCHITECTURE_DIFF:
        return [[_hop(store, k, v)] for k, v in store.rels.items() if k[0] in p["model_ids"] and k[1] in T2_RELS]
    if tid == CypherTemplateId.T3_BOTTLENECK_SOLVERS:
        return _walk(_adjacency(store, T3_RELS), p["bottleneck_id"], 2)
    if tid == CypherTemplateId.T4_BRIDGE_PATH_2_TO_3_HOP:
        return _shortest_paths(_adjacency(store, None), p["entity_a_id"], p["entity_b_id"], 3)
    if tid == CypherTemplateId.T5_CORPUS_AGGREGATION:
        out = []
        for (s, rt, t), v in store.rels.items():
            if rt != p["relation_type"] or (p["category"] is not None and store.nodes.get(t, {}).get("category") != p["category"]):
                continue
            if p["min_year"] is not None and not any(
                    (store.nodes.get(f"paper::{pid}", {}).get("year") or 0) >= p["min_year"] for pid in v.get("paper_ids", [])):
                continue
            out.append([_hop(store, (s, rt, t), v)])
        return out
    paths: List[List[Dict[str, Any]]] = []
    adj = _adjacency(store, None)
    for seed in p["seed_entity_ids"]:
        paths.extend(_walk(adj, seed, 2))
    return paths


# ---------------------------------------------------------------------------
# Public executor
# ---------------------------------------------------------------------------


class CypherTemplateCatalog:
    """Execute T1-T6 against `Neo4jGraphStore` (Cypher + $params) or `InMemoryGraphStore`."""

    def __init__(self, reranker: Optional[CrossEncoderReranker] = None, t6_top_k: int = 8) -> None:
        self.reranker = reranker or TokenInteractionReranker()
        self.t6_top_k = t6_top_k

    @staticmethod
    def cypher(template_id: CypherTemplateId) -> str:
        return CYPHER[template_id]

    def run(self, store: Any, template_id: CypherTemplateId, params: Dict[str, Any], query: str = "") -> TemplateResult:
        tid = CypherTemplateId(template_id)
        p = validate_params(tid, params)
        if hasattr(store, "rels"):
            paths = _run_in_memory(store, tid, p)[: p["limit"]]
        else:
            cypher_params = {k: v for k, v in p.items() if k not in ("model_a_id", "model_b_id")}
            paths = [row["hops"] for row in store._read(CYPHER[tid], cypher_params)]
        result = TemplateResult(tid, p, paths)
        if tid == CypherTemplateId.T2_MODEL_ARCHITECTURE_DIFF:
            result.aggregates = self._diff(p, paths)
        if tid == CypherTemplateId.T5_CORPUS_AGGREGATION:
            result.aggregates = self._aggregate(paths)
        if tid == CypherTemplateId.T6_DYNAMIC_SUBGRAPH_FALLBACK:
            self._rerank_hops(result, query)
        return result

    @staticmethod
    def _diff(p: Dict[str, Any], paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        by_model: Dict[str, Set[str]] = {m: set() for m in p["model_ids"]}
        for path in paths:
            by_model.setdefault(path[0]["source"], set()).add(path[0]["target"])
        if len(p["model_ids"]) < 2:
            return [{"model": m, "connected": sorted(v)} for m, v in by_model.items()]
        a, b = (by_model.get(m, set()) for m in p["model_ids"][:2])
        return [{"shared": sorted(a & b), "only_a": sorted(a - b), "only_b": sorted(b - a)}]

    @staticmethod
    def _aggregate(paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = {}
        for path in paths:
            h = path[0]
            row = agg.setdefault(h["target"], {"target": h["target"], "target_name": h["target_name"], "sources": []})
            if h["source"] not in row["sources"]:
                row["sources"].append(h["source"])
        rows = [{**r, "count": len(r["sources"])} for r in agg.values()]
        return sorted(rows, key=lambda r: (-r["count"], r["target"]))

    def _rerank_hops(self, result: TemplateResult, query: str) -> None:
        hops = result.unique_hops
        if not hops:
            return
        scores = self.reranker.score(query or "", [verbalize_hop(h) for h in hops])
        order = sorted(range(len(hops)), key=lambda i: (-scores[i], hops[i]["relation_type"], hops[i]["target"]))[: self.t6_top_k]
        result.paths = [[hops[i]] for i in order]
        result.hop_scores = [float(scores[i]) for i in order]
