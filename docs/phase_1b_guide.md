# Phase 1B Engineering, Testing & Evaluation Guide
**From Raw Papers to an Idempotent, Cost-Guarded Knowledge Graph**

---

## Part 1: The Big Picture — How Phase 1A + 1B Fit Together

With **Phase 1A (`53/53 = 100%`)** and **Phase 1B (`21/21 = 100%`)** complete, our entire **Phase 1 Knowledge Graph Ingestion Pipeline** (`src/ingestion/pipeline.py`) now runs end-to-end:
1. **Manifest & arXiv Fetcher (`src/ingestion/arxiv_fetcher.py`)**: Loads paper metadata and converts PDFs to structured Markdown.
2. **Structure-Aware Chunker (`src/ingestion/chunker.py`)**: Slices Markdown into deterministic `chunk_id`s (`{paper_id}::s{sec:02d}::c{idx:02d}`) and SHA-256 `content_hash`es.
3. **Cost Budget Guard & SQLite Cache (`src/ingestion/cost_estimator.py`)**: Checks `data/cache/extractions.sqlite` first (cached chunks = `$0.00`) and aborts before any API call if projected uncached cost exceeds `MAX_EXTRACTION_BUDGET_USD`.
4. **Resilient Gemini 3.8 Flash Extractor (`src/ingestion/extractor.py`)**: Extracts entities and relationships using Gemini Structured Outputs and drops single invalid/self-loop items without failing valid sibling triples in the same chunk.
5. **3-Tier Entity Resolver (`src/ingestion/entity_resolver.py`)**: Resolves synonyms and acronyms while blocking version/precision collisions.
6. **Idempotent Neo4j Store (`src/stores/neo4j_store.py`)**: Uses parameterized Cypher `MERGE` to upsert nodes and edges, accumulating `source_chunk_ids[]`, `paper_ids[]`, and `evidence_quotes[]` on each edge.

---

## Part 2: Hands-On Testing & Benchmarking Manual

Run these commands in PowerShell to inspect and verify Phase 1A and Phase 1B:

```powershell
# 1. Phase 1A Scorecard (53/53 = 100% + 20/20 Tier-3 Embedding Stress Check)
.\.venv\Scripts\python.exe scripts/benchmark_phase1a.py

# 2. Phase 1B Scorecard (21/21 = 100% across Cache/Budget, Resilient Extraction, Idempotent MERGE, and 2-Hop E2E Audit)
.\.venv\Scripts\python.exe scripts/benchmark_phase1b.py --no-live-neo4j

# 3. End-to-End Offline Ingestion Demo across the 12-Paper Seed Set
.\.venv\Scripts\python.exe scripts/ingest_phase1.py --offline-demo

# 4. Full Pytest Unit Suite (68 tests)
.\.venv\Scripts\python.exe -m pytest tests/unit -v
```
