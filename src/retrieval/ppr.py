"""HippoRAG Personalized PageRank (PPR) Retrieval over the Knowledge Graph.

Constructs and caches a weighted directed graph (`nx.DiGraph`) directly from
`graph.export_graph(include_chunks=False)` so the exact same implementation runs
uniformly against both `InMemoryGraphStore` and `Neo4jGraphStore` without raw Cypher.

Edge weighting policy:
- Plumbing edges (`IN_PAPER`, `MENTIONED_IN`) are excluded.
- `INTRODUCES` edges are down-weighted (`0.35`) so paper hub nodes do not drown
  out architectural relationships.
- Semantic architecture edges (`REPLACES_IN`, `EXTENDS`, `MITIGATES`,
  `CAUSES_BOTTLENECK`, `ADOPTS_COMPONENT`, `REQUIRES`, `OUTPERFORMS_ON`,
  `MODIFIES_HYPERPARAMETER`, `EVALUATED_ON`, `CITES_INSPIRED_BY`) carry weight `1.0`.
- Reverse transitions are added at `0.75 * forward_weight` for bidirectional flow.

Personalized PageRank (`alpha=0.85`) is seeded from Endpoint A (question entities,
weight `1.0`) and Endpoint B (top vector-hit entities, weight `0.6`, excluding
`paper::*` unless no concept seeds exist), then surfaces top PPR-ranked edges and
1-to-2 hop bridge paths connecting top-PPR nodes with full chunk provenance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

PLUMBING_RELATIONS: frozenset[str] = frozenset({"IN_PAPER", "MENTIONED_IN"})
SEMANTIC_RELATIONS: frozenset[str] = frozenset({
    "REPLACES_IN",
    "EXTENDS",
    "MITIGATES",
    "CAUSES_BOTTLENECK",
    "ADOPTS_COMPONENT",
    "REQUIRES",
    "OUTPERFORMS_ON",
    "MODIFIES_HYPERPARAMETER",
    "EVALUATED_ON",
    "CITES_INSPIRED_BY",
})
INTRODUCES_WEIGHT = 0.35
SEMANTIC_EDGE_WEIGHT = 1.0
REVERSE_EDGE_FACTOR = 0.75
DEFAULT_ALPHA = 0.85
ENDPOINT_A_WEIGHT = 1.0
ENDPOINT_B_WEIGHT = 0.6


def _run_pagerank(
    digraph: nx.DiGraph,
    alpha: float,
    personalization: Dict[str, float],
) -> Dict[str, float]:
    """Run NetworkX PageRank, falling back to NetworkX's pure-Python/NumPy solvers if SciPy is absent."""
    try:
        return nx.pagerank(digraph, alpha=alpha, personalization=personalization, weight="weight")
    except ModuleNotFoundError:
        from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_numpy, _pagerank_python

        try:
            return _pagerank_python(
                digraph,
                alpha=alpha,
                personalization=personalization,
                weight="weight",
                max_iter=200,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            return _pagerank_numpy(digraph, alpha=alpha, personalization=personalization, weight="weight")


class PersonalizedPageRankRetriever:
    """HippoRAG-style Personalized PageRank retriever over entity graphs."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        introduces_weight: float = INTRODUCES_WEIGHT,
        semantic_weight: float = SEMANTIC_EDGE_WEIGHT,
        reverse_factor: float = REVERSE_EDGE_FACTOR,
        endpoint_a_weight: float = ENDPOINT_A_WEIGHT,
        endpoint_b_weight: float = ENDPOINT_B_WEIGHT,
    ) -> None:
        self.alpha = float(alpha)
        self.introduces_weight = float(introduces_weight)
        self.semantic_weight = float(semantic_weight)
        self.reverse_factor = float(reverse_factor)
        self.endpoint_a_weight = float(endpoint_a_weight)
        self.endpoint_b_weight = float(endpoint_b_weight)

        self._cached_graph: Any = None
        self._cached_signature: Optional[Tuple[int, int]] = None
        self._cached_digraph: Optional[nx.DiGraph] = None
        self._cached_node_names: Dict[str, str] = {}
        self._cached_records: List[Dict[str, Any]] = []
        self._cached_adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self.cache_builds: int = 0

    def _edge_weight(self, relation_type: str) -> float:
        if relation_type == "INTRODUCES":
            return self.introduces_weight
        if relation_type in SEMANTIC_RELATIONS:
            return self.semantic_weight
        return self.semantic_weight

    @staticmethod
    def _graph_signature(graph: Any) -> Optional[Tuple[int, int]]:
        if hasattr(graph, "count_nodes") and hasattr(graph, "count_relationships"):
            try:
                nodes = int(graph.count_nodes("Entity"))
            except TypeError:
                nodes = int(graph.count_nodes())
            rels = int(graph.count_relationships())
            return (nodes, rels)
        return None

    def invalidate_cache(self) -> None:
        """Clear cached NetworkX graph state."""
        self._cached_graph = None
        self._cached_signature = None
        self._cached_digraph = None
        self._cached_node_names = {}
        self._cached_records = []
        self._cached_adj = {}

    def build_graph(self, graph: Any) -> nx.DiGraph:
        """Build or return the cached weighted `nx.DiGraph` for `graph`."""
        sig = self._graph_signature(graph)
        if (
            graph is self._cached_graph
            and self._cached_digraph is not None
            and sig is not None
            and sig == self._cached_signature
        ):
            return self._cached_digraph

        exported = graph.export_graph(include_chunks=False)
        nodes = exported.get("nodes", [])
        edges = exported.get("edges", [])
        export_sig = sig if sig is not None else (len(nodes), len(edges))

        if (
            graph is self._cached_graph
            and self._cached_digraph is not None
            and export_sig == self._cached_signature
        ):
            return self._cached_digraph

        digraph = nx.DiGraph()
        node_names: Dict[str, str] = {}
        for node in nodes:
            eid = node["entity_id"]
            cname = node.get("canonical_name") or eid
            node_names[eid] = cname
            digraph.add_node(
                eid,
                canonical_name=cname,
                entity_type=node.get("entity_type"),
                category=node.get("category"),
            )

        records: List[Dict[str, Any]] = []
        adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for edge in edges:
            rt = edge.get("relation_type", "")
            if rt in PLUMBING_RELATIONS:
                continue
            src = edge["source"]
            tgt = edge["target"]
            if src not in digraph:
                node_names[src] = src
                digraph.add_node(src, canonical_name=src)
            if tgt not in digraph:
                node_names[tgt] = tgt
                digraph.add_node(tgt, canonical_name=tgt)

            fwd_w = self._edge_weight(rt)
            rev_w = fwd_w * self.reverse_factor

            prev_fwd = digraph[src][tgt]["weight"] if digraph.has_edge(src, tgt) else 0.0
            if fwd_w > prev_fwd:
                digraph.add_edge(src, tgt, weight=fwd_w, relation_type=rt)
            prev_rev = digraph[tgt][src]["weight"] if digraph.has_edge(tgt, src) else 0.0
            if rev_w > prev_rev:
                digraph.add_edge(tgt, src, weight=rev_w, relation_type=rt)

            rec = {
                "source": src,
                "source_name": node_names.get(src, src),
                "relation_type": rt,
                "target": tgt,
                "target_name": node_names.get(tgt, tgt),
                "source_chunk_ids": list(edge.get("source_chunk_ids") or []),
                "evidence_quotes": list(edge.get("evidence_quotes") or []),
                "weight": fwd_w,
            }
            records.append(rec)
            adj.setdefault(src, []).append((tgt, rec))
            adj.setdefault(tgt, []).append((src, rec))

        self._cached_graph = graph
        self._cached_signature = export_sig
        self._cached_digraph = digraph
        self._cached_node_names = node_names
        self._cached_records = records
        self._cached_adj = adj
        self.cache_builds += 1
        return digraph

    def build_seed_weights(
        self,
        endpoint_a: Sequence[str] = (),
        endpoint_b: Sequence[str] = (),
        valid_nodes: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """Construct personalization weights from Endpoint A (1.0) and Endpoint B (0.6).

        Excludes `paper::*` nodes whenever any non-paper concept seed exists in either endpoint.
        """
        def _valid(eid: str) -> bool:
            return valid_nodes is None or eid in valid_nodes

        a_valid = [e for e in endpoint_a if _valid(e)]
        b_valid = [e for e in endpoint_b if _valid(e)]
        a_concepts = [e for e in a_valid if not str(e).startswith("paper::")]
        b_concepts = [e for e in b_valid if not str(e).startswith("paper::")]

        if a_concepts or b_concepts:
            use_a, use_b = a_concepts, b_concepts
        else:
            use_a, use_b = a_valid, b_valid

        weights: Dict[str, float] = {}
        for eid in use_a:
            weights[eid] = self.endpoint_a_weight
        for eid in use_b:
            weights.setdefault(eid, self.endpoint_b_weight)
        return weights

    def score_nodes(
        self,
        graph: Any,
        endpoint_a: Sequence[str] = (),
        endpoint_b: Sequence[str] = (),
        seed_weights: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        """Run Personalized PageRank and return node -> PPR score."""
        digraph = self.build_graph(graph)
        if len(digraph) == 0:
            return {}
        valid_nodes = set(digraph.nodes)
        if seed_weights is None:
            personalization = self.build_seed_weights(endpoint_a, endpoint_b, valid_nodes=valid_nodes)
        else:
            personalization = {
                eid: float(w)
                for eid, w in seed_weights.items()
                if eid in valid_nodes and float(w) > 0.0
            }
        if not personalization:
            return {}
        return _run_pagerank(digraph, alpha=self.alpha, personalization=personalization)

    @staticmethod
    def _hop_from_record(rec: Dict[str, Any], ppr_score: float) -> Dict[str, Any]:
        return {
            "source": rec["source"],
            "source_name": rec["source_name"],
            "relation_type": rec["relation_type"],
            "target": rec["target"],
            "target_name": rec["target_name"],
            "source_chunk_ids": list(rec["source_chunk_ids"]),
            "evidence_quotes": list(rec["evidence_quotes"]),
            "ppr_score": round(float(ppr_score), 6),
        }

    def retrieve_edges(
        self,
        graph: Any,
        endpoint_a: Sequence[str] = (),
        endpoint_b: Sequence[str] = (),
        top_k: int = 8,
        top_nodes: int = 12,
        seed_weights: Optional[Mapping[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top PPR-ranked ontology edges connecting high-PPR nodes."""
        scores = self.score_nodes(graph, endpoint_a=endpoint_a, endpoint_b=endpoint_b, seed_weights=seed_weights)
        if not scores:
            return []
        ranked_nodes = [n for n, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0.0]
        active_nodes = set(ranked_nodes[:top_nodes])
        for eid in list(endpoint_a) + list(endpoint_b):
            if scores.get(eid, 0.0) > 0.0:
                active_nodes.add(eid)

        scored_hops: List[Tuple[float, Dict[str, Any]]] = []
        for rec in self._cached_records:
            u, v = rec["source"], rec["target"]
            if u not in active_nodes or v not in active_nodes:
                continue
            su, sv = scores.get(u, 0.0), scores.get(v, 0.0)
            if su <= 0.0 or sv <= 0.0:
                continue
            edge_score = rec["weight"] * math.sqrt(su * sv)
            scored_hops.append((edge_score, self._hop_from_record(rec, edge_score)))

        scored_hops.sort(
            key=lambda item: (
                -item[0],
                item[1]["source"],
                item[1]["relation_type"],
                item[1]["target"],
            )
        )
        return [hop for _, hop in scored_hops[:top_k]]

    def retrieve_paths(
        self,
        graph: Any,
        endpoint_a: Sequence[str] = (),
        endpoint_b: Sequence[str] = (),
        top_k: int = 5,
        top_nodes: int = 12,
        max_hops: int = 2,
        seed_weights: Optional[Mapping[str, float]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Return top PPR-ranked 1-to-2 hop bridge paths connecting top-PPR nodes.

        Each returned path is a list of hop dicts compatible with `TemplateResult.paths`,
        carrying `source_chunk_ids`, `evidence_quotes`, and `ppr_score`.
        """
        if top_k <= 0:
            return []
        scores = self.score_nodes(graph, endpoint_a=endpoint_a, endpoint_b=endpoint_b, seed_weights=seed_weights)
        if not scores:
            return []

        valid_nodes = set(scores)
        effective_seeds = (
            set(self.build_seed_weights(endpoint_a, endpoint_b, valid_nodes=valid_nodes))
            if seed_weights is None
            else {k for k, v in seed_weights.items() if k in valid_nodes and float(v) > 0.0}
        )
        a_set = set(endpoint_a) & effective_seeds
        b_set = set(endpoint_b) & effective_seeds

        ranked_nodes = [n for n, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0.0]
        active_nodes = set(ranked_nodes[:top_nodes]) | effective_seeds

        edge_scores: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any]]] = {}
        one_hop_candidates: List[Tuple[float, List[Dict[str, Any]]]] = []
        for rec in self._cached_records:
            u, v = rec["source"], rec["target"]
            if u not in active_nodes or v not in active_nodes:
                continue
            su, sv = scores.get(u, 0.0), scores.get(v, 0.0)
            if su <= 0.0 or sv <= 0.0:
                continue
            edge_score = rec["weight"] * math.sqrt(su * sv)
            hop = self._hop_from_record(rec, edge_score)
            key = (u, rec["relation_type"], v)
            edge_scores[key] = (edge_score, hop)
            one_hop_candidates.append((edge_score, [hop]))

        one_hop_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1][0]["source"],
                item[1][0]["relation_type"],
                item[1][0]["target"],
            )
        )

        two_hop_candidates: List[Tuple[float, List[Dict[str, Any]]]] = []
        if max_hops >= 2:
            seen_undirected_2hop: Set[Tuple[Tuple[str, str, str], ...]] = set()
            for u in sorted(active_nodes, key=lambda n: (-scores.get(n, 0.0), n)):
                for mid, rec1 in self._cached_adj.get(u, []):
                    if mid not in active_nodes or mid == u:
                        continue
                    k1 = (rec1["source"], rec1["relation_type"], rec1["target"])
                    if k1 not in edge_scores:
                        continue
                    s1, hop1 = edge_scores[k1]
                    for v, rec2 in self._cached_adj.get(mid, []):
                        if v not in active_nodes or v in (u, mid):
                            continue
                        k2 = (rec2["source"], rec2["relation_type"], rec2["target"])
                        if k2 not in edge_scores or k2 == k1:
                            continue
                        canon = tuple(sorted((k1, k2)))
                        if canon in seen_undirected_2hop:
                            continue
                        seen_undirected_2hop.add(canon)
                        s2, hop2 = edge_scores[k2]
                        base = 0.5 * (s1 + s2)
                        spans_ab = (u in a_set and v in b_set) or (v in a_set and u in b_set)
                        both_seeds = u in effective_seeds and v in effective_seeds
                        bonus = 1.25 if spans_ab else (1.15 if both_seeds else 1.0)
                        # Orient starting from the Endpoint A / higher-PPR endpoint when canonicalizing.
                        if (v in a_set and u not in a_set) or (
                            (v in effective_seeds) == (u in effective_seeds)
                            and scores.get(v, 0.0) > scores.get(u, 0.0)
                        ):
                            path = [hop2, hop1]
                        else:
                            path = [hop1, hop2]
                        two_hop_candidates.append((base * bonus, path))

            two_hop_candidates.sort(
                key=lambda item: (
                    -item[0],
                    tuple((h["source"], h["relation_type"], h["target"]) for h in item[1]),
                )
            )

        if max_hops < 2 or not two_hop_candidates:
            return [p for _, p in one_hop_candidates[:top_k]]

        # Reserve slots for both top 1-hop PPR edges and top 2-hop bridge paths.
        bridge_quota = min(len(two_hop_candidates), max(1, top_k // 2))
        selected: List[Tuple[float, List[Dict[str, Any]]]] = []
        selected_sigs: Set[Tuple[Tuple[str, str, str], ...]] = set()

        for sc, path in two_hop_candidates[:bridge_quota]:
            sig = tuple((h["source"], h["relation_type"], h["target"]) for h in path)
            selected_sigs.add(sig)
            selected.append((sc, path))

        for sc, path in one_hop_candidates:
            if len(selected) >= top_k:
                break
            sig = tuple((h["source"], h["relation_type"], h["target"]) for h in path)
            if sig not in selected_sigs:
                selected_sigs.add(sig)
                selected.append((sc, path))

        for sc, path in two_hop_candidates[bridge_quota:]:
            if len(selected) >= top_k:
                break
            sig = tuple((h["source"], h["relation_type"], h["target"]) for h in path)
            if sig not in selected_sigs:
                selected_sigs.add(sig)
                selected.append((sc, path))

        selected.sort(key=lambda item: (-item[0], len(item[1])))
        return [p for _, p in selected[:top_k]]
