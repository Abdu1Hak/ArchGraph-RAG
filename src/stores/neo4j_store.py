"""Idempotent Neo4j Graph Store + In-Memory Reference Store (Phase 1B).

Idempotency contract (both stores):
- Nodes MERGE on `entity_id` (`:Entity` plus one secondary label from `EntityType`).
- Edges MERGE on `(source_entity_id, relation_type, target_entity_id)`: one edge per triple,
  never parallel duplicates. `source_chunk_ids`, `paper_ids`, `evidence_quotes` and `contexts`
  accumulate as de-duplicated lists, so every chunk that justified an edge is preserved.
- Papers are `(:Entity:Paper {entity_id: "paper::" + paper_id})`, chunks are
  `(:Chunk {chunk_id})`, linked by `(:Chunk)-[:IN_PAPER]->(:Paper)`.
- Mentions are `(:Entity)-[:MENTIONED_IN]->(:Chunk)`.
Running the same ingestion N times yields the same graph as running it once.

Security: every value travels as a Cypher parameter. Labels and relationship types cannot be
parameterized in MERGE, so they are inlined ONLY from the `EntityType` / `RelationType` enums
(validated before string formatting), never from extracted text.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from src.ingestion.chunker import DocumentChunk
from src.schema.ontology import EntityType, RelationType, ResolvedEntityNode, ResolvedGraphEdge

logger = logging.getLogger(__name__)

_ENTITY_LABELS = {e.value for e in EntityType}
_RELATION_TYPES = {r.value for r in RelationType}

SCHEMA_STATEMENTS: Tuple[str, ...] = (
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
)


def _dedupe(items: Iterable[Any]) -> List[Any]:
    return [x for x in dict.fromkeys(i for i in items if i is not None and i != "")]


def _label(entity_type: EntityType | str) -> str:
    value = entity_type.value if isinstance(entity_type, EntityType) else str(entity_type)
    if value not in _ENTITY_LABELS:
        raise ValueError(f"Refusing non-ontology label: {value!r}")
    return value


def _rel_type(relation_type: RelationType | str) -> str:
    value = relation_type.value if isinstance(relation_type, RelationType) else str(relation_type)
    if value not in _RELATION_TYPES:
        raise ValueError(f"Refusing non-ontology relationship type: {value!r}")
    return value


def paper_entity_id(paper_id: str) -> str:
    return f"paper::{paper_id}"


def node_row(node: ResolvedEntityNode) -> Dict[str, Any]:
    return {
        "entity_id": node.entity_id,
        "entity_type": node.entity_type.value,
        "canonical_name": node.canonical_name,
        "aliases": _dedupe(node.aliases),
        "category": node.category,
        "description": node.description or "",
        "source_chunk_ids": _dedupe(node.source_chunk_ids),
        "paper_ids": _dedupe(node.paper_ids),
        "parameter_scales": _dedupe(node.parameter_scales),
        "confidence": float(node.confidence),
    }


def aggregate_edges(edges: Sequence[ResolvedGraphEdge]) -> "OrderedDict[Tuple[str, str, str], Dict[str, Any]]":
    """Collapse edges sharing (source, relation, target) into one row with list provenance."""
    rows: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
    for e in edges:
        key = (e.source_entity_id, _rel_type(e.relation_type), e.target_entity_id)
        row = rows.setdefault(key, {
            "source_entity_id": e.source_entity_id,
            "relation_type": key[1],
            "target_entity_id": e.target_entity_id,
            "source_chunk_ids": [], "paper_ids": [], "evidence_quotes": [], "contexts": [],
            "confidence": 0.0,
        })
        for field_name, value in (
            ("source_chunk_ids", e.source_chunk_id), ("paper_ids", e.paper_id),
            ("evidence_quotes", e.evidence_quote), ("contexts", e.model_or_metric_context),
        ):
            if value and value not in row[field_name]:
                row[field_name].append(value)
        row["confidence"] = max(row["confidence"], float(e.confidence))
    return rows


def chunk_row(chunk: DocumentChunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "section_path": chunk.section_path,
        "section_index": chunk.section_index,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
        "approx_token_count": chunk.approx_token_count,
        "contains_equation": chunk.contains_equation,
        "contains_table": chunk.contains_table,
        "text": chunk.raw_text,
    }


class GraphStore(Protocol):
    """Write + verification API shared by `Neo4jGraphStore` and `InMemoryGraphStore`."""

    def init_schema(self) -> None: ...
    def upsert_paper_and_chunks(self, paper_meta: Any, chunks: Sequence[DocumentChunk]) -> None: ...
    def upsert_resolved_nodes(self, nodes: Sequence[ResolvedEntityNode]) -> int: ...
    def upsert_resolved_edges(self, edges: Sequence[ResolvedGraphEdge]) -> int: ...
    def upsert_chunk_mentions(self, chunk_id: str, entity_ids: Sequence[str]) -> int: ...
    def count_nodes(self, label: Optional[str] = None) -> int: ...
    def count_relationships(self, rel_type: Optional[str] = None) -> int: ...
    def duplicate_relationship_groups(self) -> int: ...
    def get_relationship(self, source_id: str, rel_type: str, target_id: str) -> Optional[Dict[str, Any]]: ...
    def get_chunk_text(self, chunk_id: str) -> Optional[str]: ...
    def find_paths(self, source_id: str, target_id: str, rel_type: str, hops: int) -> List[List[Dict[str, Any]]]: ...


def _paper_props(paper_meta: Any) -> Dict[str, Any]:
    get = (lambda k: paper_meta.get(k)) if isinstance(paper_meta, dict) else (lambda k: getattr(paper_meta, k, None))
    pillar = get("pillar")
    return {
        "entity_id": paper_entity_id(get("paper_id")),
        "paper_id": get("paper_id"),
        "arxiv_id": get("arxiv_id"),
        "title": get("title"),
        "canonical_name": get("title"),
        "year": get("year"),
        "pillar": getattr(pillar, "value", pillar),
        "seed_set": bool(get("seed_set")),
        "entity_type": EntityType.PAPER.value,
    }


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

_UPSERT_PAPER_AND_CHUNKS = """
MERGE (p:Entity {entity_id: $paper.entity_id})
SET p:Paper,
    p.paper_id = $paper.paper_id, p.arxiv_id = $paper.arxiv_id, p.title = $paper.title,
    p.canonical_name = $paper.canonical_name, p.year = $paper.year, p.pillar = $paper.pillar,
    p.seed_set = $paper.seed_set, p.entity_type = $paper.entity_type
WITH p
UNWIND $chunks AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c += row
MERGE (c)-[:IN_PAPER]->(p)
RETURN count(c) AS written
"""

_UPSERT_NODES_TEMPLATE = """
UNWIND $rows AS row
MERGE (e:Entity {{entity_id: row.entity_id}})
ON CREATE SET e.aliases = [], e.source_chunk_ids = [], e.paper_ids = [], e.parameter_scales = []
SET e:`{label}`,
    e.entity_type = row.entity_type,
    e.canonical_name = row.canonical_name,
    e.category = coalesce(row.category, e.category),
    e.description = CASE WHEN coalesce(e.description, '') = '' THEN row.description ELSE e.description END,
    e.aliases = coalesce(e.aliases, []) + [x IN row.aliases WHERE NOT x IN coalesce(e.aliases, [])],
    e.source_chunk_ids = coalesce(e.source_chunk_ids, []) + [x IN row.source_chunk_ids WHERE NOT x IN coalesce(e.source_chunk_ids, [])],
    e.paper_ids = coalesce(e.paper_ids, []) + [x IN row.paper_ids WHERE NOT x IN coalesce(e.paper_ids, [])],
    e.parameter_scales = coalesce(e.parameter_scales, []) + [x IN row.parameter_scales WHERE NOT x IN coalesce(e.parameter_scales, [])],
    e.confidence = CASE WHEN e.confidence IS NULL OR row.confidence > e.confidence THEN row.confidence ELSE e.confidence END
RETURN count(e) AS written
"""

_UPSERT_EDGES_TEMPLATE = """
UNWIND $rows AS row
MATCH (s:Entity {{entity_id: row.source_entity_id}})
MATCH (t:Entity {{entity_id: row.target_entity_id}})
MERGE (s)-[r:`{rel_type}`]->(t)
ON CREATE SET r.source_chunk_ids = [], r.paper_ids = [], r.evidence_quotes = [], r.contexts = [], r.confidence = 0.0
SET r.source_chunk_ids = r.source_chunk_ids + [x IN row.source_chunk_ids WHERE NOT x IN r.source_chunk_ids],
    r.paper_ids = r.paper_ids + [x IN row.paper_ids WHERE NOT x IN r.paper_ids],
    r.evidence_quotes = r.evidence_quotes + [x IN row.evidence_quotes WHERE NOT x IN r.evidence_quotes],
    r.contexts = r.contexts + [x IN row.contexts WHERE NOT x IN r.contexts],
    r.confidence = CASE WHEN row.confidence > r.confidence THEN row.confidence ELSE r.confidence END
RETURN count(r) AS written
"""

_UPSERT_MENTIONS = """
MATCH (c:Chunk {chunk_id: $chunk_id})
UNWIND $entity_ids AS eid
MATCH (e:Entity {entity_id: eid})
MERGE (e)-[:MENTIONED_IN]->(c)
RETURN count(e) AS written
"""


class Neo4jGraphStore:
    """Idempotent writer/reader over the official `neo4j` Python driver."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        driver: Any = None,
    ) -> None:
        if driver is None:
            from neo4j import GraphDatabase

            from src.config import settings

            driver = GraphDatabase.driver(
                uri or settings.neo4j_uri,
                auth=(user or settings.neo4j_user, password or settings.neo4j_password),
            )
        self.driver = driver
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def _write(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return session.execute_write(lambda tx: [r.data() for r in tx.run(query, params)])

    def _read(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(lambda tx: [r.data() for r in tx.run(query, params)])

    @staticmethod
    def _written(records: List[Dict[str, Any]]) -> int:
        return int(records[0]["written"]) if records else 0

    def init_schema(self) -> None:
        for stmt in SCHEMA_STATEMENTS:
            self._write(stmt, {})

    def upsert_paper_and_chunks(self, paper_meta: Any, chunks: Sequence[DocumentChunk]) -> None:
        self._write(_UPSERT_PAPER_AND_CHUNKS, {"paper": _paper_props(paper_meta), "chunks": [chunk_row(c) for c in chunks]})

    def upsert_resolved_nodes(self, nodes: Sequence[ResolvedEntityNode]) -> int:
        by_label: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for n in nodes:
            by_label.setdefault(_label(n.entity_type), {})[n.entity_id] = node_row(n)
        written = 0
        for label, rows in by_label.items():
            written += self._written(self._write(_UPSERT_NODES_TEMPLATE.format(label=label), {"rows": list(rows.values())}))
        return written

    def upsert_resolved_edges(self, edges: Sequence[ResolvedGraphEdge]) -> int:
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for row in aggregate_edges(edges).values():
            by_type.setdefault(row["relation_type"], []).append(row)
        written = 0
        for rel_type, rows in by_type.items():
            written += self._written(self._write(_UPSERT_EDGES_TEMPLATE.format(rel_type=rel_type), {"rows": rows}))
        return written

    def upsert_chunk_mentions(self, chunk_id: str, entity_ids: Sequence[str]) -> int:
        return self._written(self._write(_UPSERT_MENTIONS, {"chunk_id": chunk_id, "entity_ids": _dedupe(entity_ids)}))

    # ---- verification queries -------------------------------------------------

    def count_nodes(self, label: Optional[str] = None) -> int:
        if label and label not in _ENTITY_LABELS | {"Entity", "Chunk"}:
            raise ValueError(f"Refusing unknown label: {label!r}")
        lbl = f":`{label}`" if label else ""
        return int(self._read(f"MATCH (n{lbl}) RETURN count(n) AS n", {})[0]["n"])

    def count_relationships(self, rel_type: Optional[str] = None) -> int:
        if rel_type and rel_type not in _RELATION_TYPES | {"IN_PAPER"}:
            raise ValueError(f"Refusing unknown relationship type: {rel_type!r}")
        rt = f":`{rel_type}`" if rel_type else ""
        return int(self._read(f"MATCH ()-[r{rt}]->() RETURN count(r) AS n", {})[0]["n"])

    def duplicate_relationship_groups(self) -> int:
        rows = self._read(
            "MATCH (s)-[r]->(t) WITH s, type(r) AS t_, t, count(r) AS c WHERE c > 1 RETURN count(*) AS n", {}
        )
        return int(rows[0]["n"])

    def get_relationship(self, source_id: str, rel_type: str, target_id: str) -> Optional[Dict[str, Any]]:
        rows = self._read(
            f"MATCH (s:Entity {{entity_id: $s}})-[r:`{_rel_type(rel_type)}`]->(t:Entity {{entity_id: $t}}) "
            "RETURN properties(r) AS props",
            {"s": source_id, "t": target_id},
        )
        return rows[0]["props"] if rows else None

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        rows = self._read("MATCH (c:Chunk {chunk_id: $id}) RETURN c.text AS text", {"id": chunk_id})
        return rows[0]["text"] if rows else None

    def find_paths(self, source_id: str, target_id: str, rel_type: str, hops: int) -> List[List[Dict[str, Any]]]:
        rows = self._read(
            f"MATCH p = (a:Entity {{entity_id: $a}})-[:`{_rel_type(rel_type)}`*{int(hops)}]->(b:Entity {{entity_id: $b}}) "
            "RETURN [r IN relationships(p) | {source: startNode(r).entity_id, target: endNode(r).entity_id, "
            "relation_type: type(r), source_chunk_ids: r.source_chunk_ids, evidence_quotes: r.evidence_quotes}] AS hops",
            {"a": source_id, "b": target_id},
        )
        return [row["hops"] for row in rows]

    def has_entity(self, entity_id: str) -> bool:
        return bool(self._read("MATCH (e:Entity {entity_id: $id}) RETURN 1 AS x LIMIT 1", {"id": entity_id}))

    def load_entities(self) -> List[ResolvedEntityNode]:
        """All entity nodes, used to hydrate the query-time EntityResolver in live mode."""
        rows = self._read(
            "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.entity_type AS entity_type, "
            "coalesce(e.canonical_name, e.title, e.entity_id) AS canonical_name, coalesce(e.aliases, []) AS aliases, "
            "e.category AS category, coalesce(e.description, '') AS description", {})
        out = []
        for r in rows:
            try:
                out.append(ResolvedEntityNode(entity_id=r["entity_id"], entity_type=EntityType(r["entity_type"]),
                                              canonical_name=r["canonical_name"], aliases=list(r["aliases"]),
                                              category=r["category"], description=r.get("description") or ""))
            except ValueError:
                logger.warning("Skipping entity with unknown type: %s", r)
        return out

    def neighbors(self, entity_id: str) -> List[Dict[str, Any]]:
        """1-hop semantic neighbors (both directions; excludes IN_PAPER / MENTIONED_IN plumbing)."""
        return self._read(
            "MATCH (e:Entity {entity_id: $id})-[r]-(o:Entity) WHERE NOT type(r) IN ['IN_PAPER', 'MENTIONED_IN'] "
            "RETURN startNode(r).entity_id AS source, type(r) AS relation_type, endNode(r).entity_id AS target, "
            "o.entity_id AS neighbor, o.canonical_name AS neighbor_name, coalesce(r.source_chunk_ids, []) AS source_chunk_ids, "
            "coalesce(r.evidence_quotes, []) AS evidence_quotes "
            "ORDER BY relation_type, neighbor",
            {"id": entity_id},
        )

    def export_graph(self, include_chunks: bool = False) -> Dict[str, Any]:
        """Whole graph for visualization (see `InMemoryGraphStore.export_graph` for the shape)."""
        nodes = self._read(
            "MATCH (e:Entity) OPTIONAL MATCH (e)-[m:MENTIONED_IN]->(:Chunk) "
            "WITH e, count(m) AS mentions OPTIONAL MATCH (ch:Chunk)-[:IN_PAPER]->(e) "
            "WITH e, mentions, count(ch) AS paper_chunks "
            "RETURN e.entity_id AS entity_id, coalesce(e.canonical_name, e.title, e.entity_id) AS canonical_name, "
            "coalesce(e.entity_type, CASE WHEN e:Paper THEN 'Paper' ELSE 'Entity' END) AS entity_type, "
            "e.category AS category, coalesce(e.aliases, []) AS aliases, e.paper_id AS paper_id, "
            "mentions + paper_chunks AS chunk_count ORDER BY entity_id", {})
        rel_filter = "" if include_chunks else "WHERE NOT type(r) IN ['IN_PAPER', 'MENTIONED_IN'] "
        edges = self._read(
            "MATCH (s)-[r]->(t) " + rel_filter +
            "RETURN coalesce(s.entity_id, s.chunk_id) AS source, type(r) AS relation_type, "
            "coalesce(t.entity_id, t.chunk_id) AS target, coalesce(r.source_chunk_ids, []) AS source_chunk_ids, "
            "coalesce(r.evidence_quotes, []) AS evidence_quotes ORDER BY source, relation_type, target", {})
        if include_chunks:
            nodes += self._read(
                "MATCH (c:Chunk) RETURN c.chunk_id AS entity_id, c.chunk_id AS canonical_name, 'Chunk' AS entity_type, "
                "null AS category, [] AS aliases, c.paper_id AS paper_id, c.section_path AS section_path, "
                "0 AS chunk_count ORDER BY entity_id", {})
        return {"nodes": nodes, "edges": edges}

    def delete_by_prefix(self, prefixes: Sequence[str]) -> None:
        """Remove benchmark/test data: entities and chunks whose id starts with any prefix."""
        self._write(
            "MATCH (n) WHERE any(p IN $prefixes WHERE n.entity_id STARTS WITH p OR n.chunk_id STARTS WITH p) "
            "DETACH DELETE n",
            {"prefixes": list(prefixes)},
        )


# ---------------------------------------------------------------------------
# In-memory reference implementation (offline tests, benchmarks, --offline-demo)
# ---------------------------------------------------------------------------


class InMemoryGraphStore:
    """Dictionary-backed store with the same MERGE/accumulate semantics as `Neo4jGraphStore`."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}  # entity_id or chunk_id -> {"labels": set, **props}
        self.rels: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
        self.schema_initialized = False

    def init_schema(self) -> None:
        self.schema_initialized = True

    def _merge_rel(self, source: str, rel_type: str, target: str) -> Dict[str, Any]:
        return self.rels.setdefault((source, rel_type, target), {})

    def upsert_paper_and_chunks(self, paper_meta: Any, chunks: Sequence[DocumentChunk]) -> None:
        props = _paper_props(paper_meta)
        node = self.nodes.setdefault(props["entity_id"], {"labels": set()})
        node["labels"].update({"Entity", "Paper"})
        node.update(props)
        for c in chunks:
            ch = self.nodes.setdefault(c.chunk_id, {"labels": {"Chunk"}})
            ch.update(chunk_row(c))
            self._merge_rel(c.chunk_id, "IN_PAPER", props["entity_id"])

    def upsert_resolved_nodes(self, nodes: Sequence[ResolvedEntityNode]) -> int:
        for n in nodes:
            row = node_row(n)
            label = _label(n.entity_type)
            node = self.nodes.setdefault(n.entity_id, {
                "labels": set(), "aliases": [], "source_chunk_ids": [], "paper_ids": [], "parameter_scales": [],
            })
            node["labels"].update({"Entity", label})
            node["entity_type"] = row["entity_type"]
            node["canonical_name"] = row["canonical_name"]
            node["category"] = row["category"] if row["category"] is not None else node.get("category")
            if not node.get("description"):
                node["description"] = row["description"]
            for key in ("aliases", "source_chunk_ids", "paper_ids", "parameter_scales"):
                existing = node.setdefault(key, [])
                existing.extend(x for x in row[key] if x not in existing)
            if node.get("confidence") is None or row["confidence"] > node["confidence"]:
                node["confidence"] = row["confidence"]
        return len({n.entity_id for n in nodes})

    def upsert_resolved_edges(self, edges: Sequence[ResolvedGraphEdge]) -> int:
        written = 0
        for (s, rt, t), row in aggregate_edges(edges).items():
            if s not in self.nodes or t not in self.nodes:
                continue  # mirrors MATCH semantics: no endpoint, no edge
            rel = self._merge_rel(s, rt, t)
            for key in ("source_chunk_ids", "paper_ids", "evidence_quotes", "contexts"):
                existing = rel.setdefault(key, [])
                existing.extend(x for x in row[key] if x not in existing)
            rel["confidence"] = max(rel.get("confidence", 0.0), row["confidence"])
            written += 1
        return written

    def upsert_chunk_mentions(self, chunk_id: str, entity_ids: Sequence[str]) -> int:
        if chunk_id not in self.nodes:
            return 0
        written = 0
        for eid in _dedupe(entity_ids):
            if eid in self.nodes:
                self._merge_rel(eid, "MENTIONED_IN", chunk_id)
                written += 1
        return written

    def count_nodes(self, label: Optional[str] = None) -> int:
        return sum(1 for n in self.nodes.values() if label is None or label in n["labels"])

    def count_relationships(self, rel_type: Optional[str] = None) -> int:
        return sum(1 for (_, rt, _) in self.rels if rel_type is None or rt == rel_type)

    def duplicate_relationship_groups(self) -> int:
        return 0  # keyed by (source, type, target): duplicates are unrepresentable by construction

    def get_relationship(self, source_id: str, rel_type: str, target_id: str) -> Optional[Dict[str, Any]]:
        rel = self.rels.get((source_id, rel_type, target_id))
        return dict(rel) if rel is not None else None

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        node = self.nodes.get(chunk_id)
        return node.get("text") if node and "Chunk" in node["labels"] else None

    def has_entity(self, entity_id: str) -> bool:
        return "Entity" in self.nodes.get(entity_id, {}).get("labels", set())

    def neighbors(self, entity_id: str) -> List[Dict[str, Any]]:
        """1-hop semantic neighbors (both directions; excludes IN_PAPER / MENTIONED_IN plumbing)."""
        out = []
        for (s, rt, t), props in self.rels.items():
            if rt in ("IN_PAPER", "MENTIONED_IN") or entity_id not in (s, t):
                continue
            other = t if s == entity_id else s
            out.append({"source": s, "relation_type": rt, "target": t, "neighbor": other,
                        "neighbor_name": self.nodes.get(other, {}).get("canonical_name"),
                        "source_chunk_ids": list(props.get("source_chunk_ids", [])),
                        "evidence_quotes": list(props.get("evidence_quotes", []))})
        return sorted(out, key=lambda r: (r["relation_type"], r["neighbor"]))

    def export_graph(self, include_chunks: bool = False) -> Dict[str, Any]:
        """Whole graph for visualization.

        nodes: Entity rows {entity_id, canonical_name, entity_type, category, aliases, paper_id, chunk_count}
        (chunk_count = MENTIONED_IN links, or the paper's chunks for a Paper) plus, with include_chunks,
        Chunk rows {entity_id: chunk_id, entity_type: "Chunk", paper_id, section_path}.
        edges: {source, relation_type, target, source_chunk_ids, evidence_quotes}; IN_PAPER / MENTIONED_IN
        plumbing only with include_chunks.
        """
        chunk_count: Dict[str, int] = {}
        for (s, rt, t) in self.rels:
            if rt == "MENTIONED_IN":
                chunk_count[s] = chunk_count.get(s, 0) + 1
            elif rt == "IN_PAPER":
                chunk_count[t] = chunk_count.get(t, 0) + 1
        nodes: List[Dict[str, Any]] = []
        for nid, n in sorted(self.nodes.items()):
            if "Entity" in n["labels"]:
                nodes.append({"entity_id": nid, "canonical_name": n.get("canonical_name") or n.get("title") or nid,
                              "entity_type": n.get("entity_type") or ("Paper" if "Paper" in n["labels"] else "Entity"),
                              "category": n.get("category"), "aliases": list(n.get("aliases", [])),
                              "paper_id": n.get("paper_id"), "chunk_count": chunk_count.get(nid, 0)})
            elif include_chunks and "Chunk" in n["labels"]:
                nodes.append({"entity_id": nid, "canonical_name": nid, "entity_type": "Chunk", "category": None,
                              "aliases": [], "paper_id": n.get("paper_id"), "section_path": n.get("section_path"),
                              "chunk_count": 0})
        edges = [{"source": s, "relation_type": rt, "target": t,
                  "source_chunk_ids": list(p.get("source_chunk_ids", [])),
                  "evidence_quotes": list(p.get("evidence_quotes", []))}
                 for (s, rt, t), p in self.rels.items()
                 if include_chunks or rt not in ("IN_PAPER", "MENTIONED_IN")]
        return {"nodes": nodes, "edges": sorted(edges, key=lambda e: (e["source"], e["relation_type"], e["target"]))}

    def find_paths(self, source_id: str, target_id: str, rel_type: str, hops: int) -> List[List[Dict[str, Any]]]:
        out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for (s, rt, t), props in self.rels.items():
            if rt == rel_type:
                out.setdefault(s, []).append((t, props))
        paths: List[List[Dict[str, Any]]] = []

        def walk(node: str, depth: int, acc: List[Dict[str, Any]]) -> None:
            if depth == hops:
                if node == target_id:
                    paths.append(list(acc))
                return
            for nxt, props in out.get(node, []):
                acc.append({
                    "source": node, "target": nxt, "relation_type": rel_type,
                    "source_chunk_ids": list(props.get("source_chunk_ids", [])),
                    "evidence_quotes": list(props.get("evidence_quotes", [])),
                })
                walk(nxt, depth + 1, acc)
                acc.pop()

        walk(source_id, 0, [])
        return paths
