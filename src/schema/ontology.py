"""Constrained Domain Ontology for LLM Architecture Knowledge Graph (Phase 1).

Design Decision:
----------------
Why strictly constrain to 8 EntityTypes and 12 RelationTypes?
An open-ended prompt ("Extract all entities and relationships") causes the LLM to
invent hundreds of synonymous edge labels (`USES`, `UTILIZES`, `EMPLOYS`, `IS_BUILT_WITH`),
fragmenting the graph so that parameterized multi-hop Cypher queries return empty sets.

By enforcing strict Python Enums via Pydantic and passing `model_json_schema()` to
Gemini's `response_format`, every extracted node and edge is guaranteed to conform to
our queryable ontology before touching Neo4j.
"""

from enum import Enum
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


logger = logging.getLogger(__name__)

# Bump EXTRACTION_SCHEMA_VERSION whenever the ontology/payload schema changes and
# EXTRACTION_PROMPT_VERSION whenever the extraction system prompt changes: both are part of
# every chunk's content_hash, so a bump invalidates the extraction cache automatically.
EXTRACTION_SCHEMA_VERSION = "kg-schema-v1"
EXTRACTION_PROMPT_VERSION = "kg-prompt-v1"


class PillarType(str, Enum):
    """The 5 core LLM engineering pillars covered by the corpus."""

    PRE_TRAINING = "pre_training"
    ARCHITECTURE = "architecture"
    MID_TRAINING = "mid_training"
    POST_TRAINING = "post_training"
    INFERENCE = "inference"


class EntityType(str, Enum):
    """Strictly constrained 8 node labels in the Neo4j Knowledge Graph."""

    PAPER = "Paper"
    MODEL_FAMILY = "ModelFamily"
    ARCH_COMPONENT = "ArchComponent"
    TRAINING_METHOD = "TrainingMethod"
    INFERENCE_TECHNIQUE = "InferenceTechnique"
    BOTTLENECK = "Bottleneck"
    HYPERPARAMETER_OR_METRIC = "HyperparameterOrMetric"
    BENCHMARK_OR_DATASET = "BenchmarkOrDataset"


class RelationType(str, Enum):
    """Strictly constrained 12 edge types in the Neo4j Knowledge Graph."""

    INTRODUCES = "INTRODUCES"
    ADOPTS_COMPONENT = "ADOPTS_COMPONENT"
    REPLACES_IN = "REPLACES_IN"
    EXTENDS = "EXTENDS"
    MITIGATES = "MITIGATES"
    CAUSES_BOTTLENECK = "CAUSES_BOTTLENECK"
    OUTPERFORMS_ON = "OUTPERFORMS_ON"
    EVALUATED_ON = "EVALUATED_ON"
    MODIFIES_HYPERPARAMETER = "MODIFIES_HYPERPARAMETER"
    REQUIRES = "REQUIRES"
    CITES_INSPIRED_BY = "CITES_INSPIRED_BY"
    MENTIONED_IN = "MENTIONED_IN"


class ComponentCategory(str, Enum):
    """Sub-categories for ArchComponent nodes to support corpus-wide aggregations."""

    ATTENTION = "attention"
    POS_ENCODING = "positional"
    NORMALIZATION = "norm"
    ACTIVATION_FFN = "activation"
    MOE_ROUTING = "moe_routing"
    KV_CACHE = "kv_cache"
    SSM_HYBRID = "ssm"
    OTHER = "other"


class ExtractedEntity(BaseModel):
    """Raw entity extracted from a single document chunk by Gemini."""

    entity_type: EntityType = Field(
        description="One of the 8 constrained ontology entity types."
    )
    canonical_name: str = Field(
        min_length=2,
        max_length=140,
        description="Primary technical name (e.g., 'Grouped-Query Attention', 'RoPE', 'DeepSeek-V2').",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Acronyms or alternate names mentioned in the text (e.g., ['GQA']).",
    )
    category: Optional[ComponentCategory] = Field(
        default=None,
        description="Required if entity_type is ArchComponent; otherwise optional.",
    )
    description: str = Field(
        default="",
        max_length=400,
        description="Concise 1-sentence technical definition grounded in the chunk.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence score between 0.0 and 1.0.",
    )

    @field_validator("canonical_name")
    @classmethod
    def clean_canonical_name(cls, v: str) -> str:
        cleaned = re.sub(r"\s+", " ", v).strip()
        if not cleaned:
            raise ValueError("canonical_name cannot be empty or whitespace")
        return cleaned


class ExtractedRelationship(BaseModel):
    """Directed relationship triple extracted from a single document chunk."""

    source_name: str = Field(
        min_length=2,
        description="Canonical name of the source entity (must appear in the chunk).",
    )
    source_type: EntityType = Field(
        description="EntityType of the source node."
    )
    relation_type: RelationType = Field(
        description="One of the 12 constrained ontology relationship types."
    )
    target_name: str = Field(
        min_length=2,
        description="Canonical name of the target entity (must appear in the chunk).",
    )
    target_type: EntityType = Field(
        description="EntityType of the target node."
    )
    evidence_quote: str = Field(
        min_length=10,
        max_length=500,
        description="Verbatim or near-verbatim sentence/formula from the chunk justifying this edge.",
    )
    model_or_metric_context: Optional[str] = Field(
        default=None,
        description="Optional qualifier such as model context for REPLACES_IN (e.g., 'LLaMA-1') or metric delta for OUTPERFORMS_ON (e.g., '-93.3% KV cache').",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship between 0.0 and 1.0.",
    )

    @model_validator(mode="after")
    def prevent_self_loops(self) -> "ExtractedRelationship":
        if (
            self.source_name.strip().lower() == self.target_name.strip().lower()
            and self.source_type == self.target_type
        ):
            raise ValueError(
                f"Self-loop relationship not allowed: {self.source_name} -> {self.target_name}"
            )
        return self


class ChunkExtractionPayload(BaseModel):
    """Schema passed to Gemini `response_format` for a single chunk extraction pass."""

    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="All domain entities found in the chunk.",
    )
    relationships: List[ExtractedRelationship] = Field(
        default_factory=list,
        description="All directed relationships between extracted entities supported by the chunk.",
    )


class ResolvedEntityNode(BaseModel):
    """Canonical entity node ready for idempotent MERGE into Neo4j."""

    entity_id: str = Field(
        description="Deterministic canonical node ID, e.g., 'arch_component::grouped_query_attention'"
    )
    entity_type: EntityType
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    description: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)
    paper_ids: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    parameter_scales: List[str] = Field(
        default_factory=list,
        description="Dense checkpoint sizes seen for a ModelFamily (e.g. ['8B', '70B']); metadata, not identity.",
    )


class ResolvedGraphEdge(BaseModel):
    """Canonical directed edge ready for idempotent MERGE into Neo4j."""

    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    source_chunk_id: str
    paper_id: str
    evidence_quote: str
    model_or_metric_context: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


def filter_extractions_by_confidence(
    payload: ChunkExtractionPayload,
    min_confidence: float = 0.65,
) -> ChunkExtractionPayload:
    """Filter out low-confidence entities and relationships before entity resolution."""
    valid_entities = [
        e for e in payload.entities if e.confidence >= min_confidence
    ]
    valid_relationships = [
        r for r in payload.relationships if r.confidence >= min_confidence
    ]
    return ChunkExtractionPayload(
        entities=valid_entities,
        relationships=valid_relationships,
    )


class ExtractionParseError(ValueError):
    """Raised when an LLM response is not a parseable JSON object (the whole chunk is unusable)."""


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(?P<body>[\s\S]*?)\s*```\s*$", re.IGNORECASE)


def _coerce_to_dict(raw: Union[str, bytes, Dict[str, Any], "ChunkExtractionPayload", None]) -> Dict[str, Any]:
    if isinstance(raw, ChunkExtractionPayload):
        return raw.model_dump(mode="json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str) or not raw.strip():
        raise ExtractionParseError("Empty or non-text extraction response")
    text = raw.strip()
    fence = _JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group("body")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError(f"Malformed JSON at line {exc.lineno} col {exc.colno}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ExtractionParseError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def sanitize_and_filter_extraction(
    raw_dict_or_json: Union[str, bytes, Dict[str, Any], ChunkExtractionPayload, None],
    min_confidence: float = 0.65,
    dropped: Optional[List[Dict[str, Any]]] = None,
) -> ChunkExtractionPayload:
    """Validate an extraction item by item, dropping only the items that fail.

    A single malformed entity, an out-of-ontology type (e.g. a prompt-injected
    `"entity_type": "SystemOverride"`), a self-loop, or a low-confidence item is logged and
    dropped; its valid siblings in the same chunk survive. Only a response that is not a JSON
    object at all raises `ExtractionParseError` (callers may retry on that).

    If `dropped` is given, one `{"kind", "index", "reason", "item"}` record is appended per
    dropped item for auditing.
    """
    data = _coerce_to_dict(raw_dict_or_json)

    def _drop(kind: str, index: int, reason: str, item: Any) -> None:
        logger.warning("Dropping extracted %s #%d: %s", kind, index, reason)
        if dropped is not None:
            dropped.append({"kind": kind, "index": index, "reason": reason, "item": item})

    def _items(key: str) -> List[Any]:
        value = data.get(key) or []
        if not isinstance(value, list):
            _drop(key, -1, f"'{key}' is {type(value).__name__}, expected list", value)
            return []
        return value

    entities: List[ExtractedEntity] = []
    for i, item in enumerate(_items("entities")):
        try:
            ent = ExtractedEntity.model_validate(item)
        except ValidationError as exc:
            _drop("entity", i, exc.errors()[0].get("msg", str(exc)), item)
            continue
        if ent.confidence < min_confidence:
            _drop("entity", i, f"confidence {ent.confidence} < {min_confidence}", item)
            continue
        entities.append(ent)

    relationships: List[ExtractedRelationship] = []
    for i, item in enumerate(_items("relationships")):
        try:
            rel = ExtractedRelationship.model_validate(item)
        except ValidationError as exc:
            _drop("relationship", i, exc.errors()[0].get("msg", str(exc)), item)
            continue
        if rel.confidence < min_confidence:
            _drop("relationship", i, f"confidence {rel.confidence} < {min_confidence}", item)
            continue
        relationships.append(rel)

    return ChunkExtractionPayload(entities=entities, relationships=relationships)