"""Offline fixtures for the remaining 38 manifest papers (used only for the full 50-paper corpus).

Each paper lives in `data/fixtures/full_corpus/<paper_id>.yaml`:
    markdown:       synthetic technical sections (facts paraphrased from the paper, banner-marked)
    appendix:       true for seed papers whose 12-paper text must stay untouched; the markdown is then
                    APPENDED to the seed fixture (earlier chunk_ids stay stable) only in full-corpus mode
    relationships:  [source, source_type, relation, target, target_type, evidence_quote, context?]
`_entity_hints.yaml` adds aliases / descriptions / categories to recorded entities by name.

None of this is loaded by the 12-paper default (`build_offline_corpus(seed_only=True)`), so every
Phase 1-5 scorecard sees exactly the same text and recordings as before.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "full_corpus"


@lru_cache(maxsize=1)
def load_full_corpus_fixtures() -> Dict[str, Any]:
    """{"markdown": {pid: md}, "appendix": {pid: md}, "relationships": {pid: [rel]}, "entity_hints": {name: hint}}."""
    from src.ingestion.offline_fixtures import SYNTHETIC_BANNER, _rel

    out: Dict[str, Any] = {"markdown": {}, "appendix": {}, "relationships": {}, "entity_hints": {}}
    for path in sorted(FIXTURE_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if path.stem == "_entity_hints":
            out["entity_hints"] = {k.lower(): v for k, v in (data.get("entities") or {}).items()}
            continue
        pid = data["paper_id"]
        md = data["markdown"].replace("{BANNER}", SYNTHETIC_BANNER)
        out["appendix" if data.get("appendix") else "markdown"][pid] = md
        rels: List[Dict[str, Any]] = []
        for row in data.get("relationships") or []:
            src, st, rt, tgt, tt, quote = row[:6]
            rels.append(_rel(src, st, rt, tgt, tt, quote, context=row[6] if len(row) > 6 else None))
        out["relationships"][pid] = rels
    return out


def full_corpus_markdown(paper_id: str, seed_markdown: Dict[str, str]) -> str | None:
    """Full-corpus text for a paper: seed fixture + appendix, or the 38-paper fixture; None if absent."""
    fx = load_full_corpus_fixtures()
    if paper_id in seed_markdown:
        extra = fx["appendix"].get(paper_id)
        return seed_markdown[paper_id] if not extra else seed_markdown[paper_id].rstrip("\n") + "\n" + extra
    return fx["markdown"].get(paper_id)


def full_corpus_recordings(seed_recordings: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Seed recordings plus every full-corpus recording (a new dict; the seed dict is not mutated)."""
    merged = {pid: list(rels) for pid, rels in seed_recordings.items()}
    for pid, rels in load_full_corpus_fixtures()["relationships"].items():
        merged.setdefault(pid, []).extend(rels)
    return merged
