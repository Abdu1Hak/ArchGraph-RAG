"""Deterministic Citation Gate (Phase 4) with the Crop-Then-Fallback repair ladder primitives.

Rules (pure Python, sub-millisecond):
1. Zero invented citations: every `[chunk_id]` must be in `allowed_chunk_ids`.
2. Claim coverage: every substantive claim unit (bullet line or sentence) carries >= 1 valid citation.
3. Exact resolution: every cited chunk_id resolves to a real chunk (paper, section, verbatim text).

Ladder (driven by the synthesizer): attempt -> at most 1 repair -> Step A crop offending
sentences (if >= 1 valid sentence survives) -> Step B deterministic extractive fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Anything that looks like a chunk citation, including malformed / invented ones.
CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\-]+::[A-Za-z0-9_.:\-]+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"(*$])")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

ChunkResolver = Callable[[Sequence[str]], Mapping[str, Any]]


@dataclass
class SentenceReport:
    index: int
    text: str
    is_claim: bool
    cited_ids: List[str]
    invalid_ids: List[str]

    @property
    def ok(self) -> bool:
        return not self.invalid_ids and (not self.is_claim or bool(self.cited_ids))


@dataclass
class ValidationResult:
    sentences: List[SentenceReport] = field(default_factory=list)
    invented_ids: List[str] = field(default_factory=list)       # rule 1
    unresolved_ids: List[str] = field(default_factory=list)     # rule 3
    uncited_claims: List[str] = field(default_factory=list)     # rule 2
    cited_ids: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.sentences) and all(s.ok for s in self.sentences) and any(s.cited_ids for s in self.sentences)

    @property
    def valid_claim_count(self) -> int:
        return sum(1 for s in self.sentences if s.ok and s.is_claim)

    def summary(self) -> str:
        return (f"valid={self.valid} claims={sum(s.is_claim for s in self.sentences)} invented={self.invented_ids} "
                f"unresolved={self.unresolved_ids} uncited={len(self.uncited_claims)}")


def split_units(answer: str) -> List[str]:
    """Bullet lines are single claims; prose lines are split into sentences. Citation-only
    fragments ('[id].') are merged into the preceding unit."""
    units: List[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = [line] if _BULLET_RE.match(line) else _SENT_SPLIT_RE.split(line)
        for piece in pieces:
            if units and not CITATION_RE.sub("", piece).strip(" .,;"):
                units[-1] = f"{units[-1]} {piece}"
            else:
                units.append(piece)
    return units


def is_claim(unit: str) -> bool:
    """Substantive claim = >= 4 words outside citations and not a heading/lead-in line ending in ':'."""
    body = CITATION_RE.sub("", unit).strip()
    if not body or body.endswith(":") or body.startswith("#"):
        return False
    return len(_WORD_RE.findall(body)) >= 4


def validate_citations(answer: str, allowed_chunk_ids: Iterable[str], resolve: Optional[ChunkResolver] = None) -> ValidationResult:
    import time

    t0 = time.perf_counter()
    allowed: Set[str] = set(allowed_chunk_ids)
    units = split_units(answer)
    all_cited = list(dict.fromkeys(c for u in units for c in CITATION_RE.findall(u)))
    resolved = set(resolve([c for c in all_cited if c in allowed]).keys()) if resolve else {c for c in all_cited if c in allowed}
    res = ValidationResult(cited_ids=all_cited)
    for i, u in enumerate(units):
        cited = list(dict.fromkeys(CITATION_RE.findall(u)))
        bad = [c for c in cited if c not in allowed or c not in resolved]
        rep = SentenceReport(i, u, is_claim(u), [c for c in cited if c not in bad], bad)
        res.sentences.append(rep)
    res.invented_ids = [c for c in all_cited if c not in allowed]
    res.unresolved_ids = [c for c in all_cited if c in allowed and c not in resolved]
    # Tweak A: Paragraph-level forward citation attribution.
    # When two consecutive sentences stem from the same source and the LLM places the valid
    # [chunk_id] at the end of the following sentence, inherit that trailing valid citation
    # (only when the answer contains zero invented/unresolved IDs). Trailing uncited claims
    # at the end of the response remain uncited.
    if not res.invented_ids and not res.unresolved_ids:
        for i in range(len(res.sentences) - 1):
            cur = res.sentences[i]
            if cur.is_claim and not cur.cited_ids and not cur.invalid_ids:
                for nxt in res.sentences[i + 1 :]:
                    if nxt.invalid_ids:
                        break
                    if nxt.cited_ids:
                        cur.cited_ids = list(nxt.cited_ids)
                        break
    for rep in res.sentences:
        if rep.is_claim and not rep.cited_ids and not rep.invalid_ids:
            res.uncited_claims.append(rep.text)
    res.elapsed_ms = (time.perf_counter() - t0) * 1000
    return res


def build_repair_prompt(original_answer: str, validation_result: ValidationResult, allowed_chunk_ids: Iterable[str]) -> str:
    problems = []
    for s in validation_result.sentences:
        if s.invalid_ids:
            problems.append(f"- Sentence {s.index + 1} cites ids that are not in the allowed list: {', '.join(s.invalid_ids)}")
        elif s.is_claim and not s.cited_ids:
            problems.append(f"- Sentence {s.index + 1} makes a claim without any citation: \"{s.text[:120]}\"")
    allowed = "\n".join(f"- {c}" for c in sorted(set(allowed_chunk_ids)))
    return (
        "REPAIR REQUEST: your previous answer failed citation validation.\n"
        "Rewrite it so that every claim ends with at least one [chunk_id] copied EXACTLY from the allowed list, "
        "remove any claim you cannot support, and never invent ids.\n\n"
        f"PROBLEMS:\n{chr(10).join(problems) or '- none recorded'}\n\n"
        f"ALLOWED CHUNK IDS:\n{allowed}\n\n"
        f"ORIGINAL ANSWER:\n{original_answer}\n"
    )


def crop_invalid_sentences(validation_result: ValidationResult) -> Tuple[str, int]:
    """Step A: drop only the offending units. Returns (cropped_answer, n_cropped); keeps non-claim lead-ins
    only when at least one valid cited claim survives."""
    kept = [s for s in validation_result.sentences if s.ok]
    cropped = len(validation_result.sentences) - len(kept)
    if not any(s.is_claim and s.cited_ids for s in kept):
        return "", cropped
    return "\n".join(s.text for s in kept), cropped


def extractive_fallback(fact_texts: Sequence[str], passages: Sequence[Tuple[str, str]], max_facts: int = 4) -> str:
    """Step B: answer built only from verbalized graph facts (already cited) and the top passage's
    best sentence, so every citation is valid by construction. `passages` = [(chunk_id, sentence)]."""
    lines = ["Extractive answer assembled directly from the retrieved evidence:"]
    lines += [f"- {f}" for f in list(fact_texts)[:max_facts]]
    for cid, sentence in list(passages)[:2]:
        s = sentence.strip().rstrip(".")
        if s:
            lines.append(f"- {s} [{cid}].")
    return "\n".join(lines) if len(lines) > 1 else ""
