"""Live Neo4j integration test (skipped unless NEO4J_URI is reachable, e.g. `docker compose up -d neo4j`).

Writes only `bench1b`-prefixed ids and deletes them afterwards.
"""

from __future__ import annotations

import pytest

from src.config import settings
from src.ingestion.arxiv_fetcher import PaperMeta
from src.ingestion.chunker import StructureAwareChunker
from src.schema.ontology import EntityType, RelationType, ResolvedEntityNode, ResolvedGraphEdge
from src.stores.neo4j_store import Neo4jGraphStore

PREFIXES = ["bench1b", "paper::bench1b"]


@pytest.fixture()
def store():
    neo4j = pytest.importorskip("neo4j")
    try:
        driver = neo4j.GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
                                            connection_timeout=2.0)
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable at {settings.neo4j_uri}: {type(exc).__name__}")
    s = Neo4jGraphStore(driver=driver)
    s.delete_by_prefix(PREFIXES)
    s.init_schema()
    yield s
    s.delete_by_prefix(PREFIXES)
    s.close()


def test_live_idempotent_merge_and_provenance(store: Neo4jGraphStore) -> None:
    paper = PaperMeta(paper_id="bench1b_paper", arxiv_id="0000.00000", title="Bench", year=2024, pillar="architecture")
    chunks = StructureAwareChunker().chunk_paper("bench1b_paper", "Bench", 2024, "architecture",
                                                 "# A\n\nSrc extends Tgt here.\n\n# B\n\nSrc builds on Tgt.\n")
    nodes = [ResolvedEntityNode(entity_id=f"bench1b::{n}", entity_type=EntityType.ARCH_COMPONENT, canonical_name=n,
                                source_chunk_ids=[chunks[0].chunk_id]) for n in ("src", "tgt")]
    edges = [ResolvedGraphEdge(source_entity_id="bench1b::src", target_entity_id="bench1b::tgt",
                               relation_type=RelationType.EXTENDS, source_chunk_id=c.chunk_id, paper_id="bench1b_paper",
                               evidence_quote=c.raw_text, confidence=0.9) for c in chunks]
    counts = []
    for _ in range(3):
        store.upsert_paper_and_chunks(paper, chunks)
        store.upsert_resolved_nodes(nodes)
        store.upsert_resolved_edges(edges[:1])
        store.upsert_resolved_edges(edges[1:])
        store.upsert_chunk_mentions(chunks[0].chunk_id, ["bench1b::src"])
        rows = store._read(
            "MATCH (n) WHERE any(p IN $p WHERE n.entity_id STARTS WITH p OR n.chunk_id STARTS WITH p) "
            "OPTIONAL MATCH (n)-[r]->() RETURN count(DISTINCT n) AS nodes, count(r) AS rels", {"p": PREFIXES})
        counts.append((rows[0]["nodes"], rows[0]["rels"]))
    assert len(set(counts)) == 1
    rel = store.get_relationship("bench1b::src", "EXTENDS", "bench1b::tgt")
    assert rel["source_chunk_ids"] == [c.chunk_id for c in chunks]
    assert store.find_paths("bench1b::src", "bench1b::tgt", "EXTENDS", 1)


def test_live_expanded_templates_execute(store: Neo4jGraphStore) -> None:
    """Runs (not just EXPLAINs) the expanded T1/T3 and T5/T6 Cypher on a namespaced 3-hop chain."""
    from src.retrieval.cypher_templates import CypherTemplateCatalog, CypherTemplateId as T

    nodes = [ResolvedEntityNode(entity_id=f"bench1b::{n}", entity_type=t, canonical_name=n)
             for n, t in (("r1", EntityType.MODEL_FAMILY), ("v3", EntityType.MODEL_FAMILY),
                          ("mla", EntityType.ARCH_COMPONENT), ("kv", EntityType.BOTTLENECK))]
    store.upsert_resolved_nodes(nodes)
    edges = [ResolvedGraphEdge(source_entity_id=f"bench1b::{s}", target_entity_id=f"bench1b::{t}", relation_type=rt,
                               source_chunk_id=f"bench1b::c{i}", paper_id="bench1b_paper", evidence_quote=f"{s} {rt.value} {t} quote",
                               confidence=0.9)
             for i, (s, rt, t) in enumerate((("r1", RelationType.REQUIRES, "v3"), ("v3", RelationType.ADOPTS_COMPONENT, "mla"),
                                             ("mla", RelationType.MITIGATES, "kv")))]
    store.upsert_resolved_edges(edges)
    cat = CypherTemplateCatalog()
    t3 = cat.run(store, T.T3_BOTTLENECK_SOLVERS, {"bottleneck_id": "bench1b::r1"})
    assert ["REQUIRES", "ADOPTS_COMPONENT", "MITIGATES"] in [[h["relation_type"] for h in p] for p in t3.paths]
    t1 = cat.run(store, T.T1_COMPONENT_LINEAGE, {"root_entity_id": "bench1b::kv"})
    assert any(len(p) == 3 for p in t1.paths)
    t6 = cat.run(store, T.T6_DYNAMIC_SUBGRAPH_FALLBACK, {"seed_entity_ids": ["bench1b::v3"]}, query="mla kv")
    assert t6.paths and all(h["source_chunk_ids"] for p in t6.paths for h in p)


def test_export_graph_matches_store_counts(store):
    """GET /graph/full backend: the read-only export returns every Entity, and Chunks + plumbing on request."""
    plain = store.export_graph(include_chunks=False)
    assert len(plain["nodes"]) == store.count_nodes("Entity")
    assert not {e["relation_type"] for e in plain["edges"]} & {"MENTIONED_IN", "IN_PAPER"}
    full = store.export_graph(include_chunks=True)
    assert sum(1 for n in full["nodes"] if n["entity_type"] == "Chunk") == store.count_nodes("Chunk")
    assert len(full["edges"]) == store.count_relationships()
