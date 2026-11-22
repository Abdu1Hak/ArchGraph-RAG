"""Phase 1A scorecard: run gold cases against the current EntityResolver and StructureAwareChunker.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/benchmark_phase1a.py [--gold PATH] [--seeds PATH]

This script only reads src/ and data/manifest/; it never modifies them.

Test modes:
- Isolated: every resolver pair runs on a fresh seeded EntityResolver, in both orders
  (A then B, B then A). A case passes only if both orders give the expected outcome.
- Sequential: all synonym and adversarial pairs run in file order on ONE shared resolver,
  so aliases learned by earlier cases can leak into later ones. Results are reported as
  order effects and do not change the bucket scores.

Tier 3 (embedding similarity) is not exercised because no embed_fn is configured, so
results reflect seed aliases, Tier 1, Tier 2 and the version guard only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import StructureAwareChunker  # noqa: E402
from src.ingestion.entity_resolver import (  # noqa: E402
    EntityResolver,
    extract_version_signature,
    has_version_conflict,
    normalize_surface_key,
    slugify_entity_id,
    split_name_and_parenthetical_acronym,
)
from src.schema.ontology import (  # noqa: E402
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
)

DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "benchmark" / "phase1a_gold_cases.json"
DEFAULT_SEEDS_PATH = PROJECT_ROOT / "data" / "manifest" / "seed_aliases.yaml"
TIER2_LEXICAL_THRESHOLD = 0.91  # Hardcoded in EntityResolver.resolve_entity (Tier 2)
GLUED_DIGIT_RE = re.compile(r"[A-Za-z]\d")
SCALE_SUFFIX_RE = re.compile(r"\b\d+(?:\.\d+)?[BbMm]\b")
LINE = "=" * 92
SUBLINE = "-" * 92


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    surface: str
    entity_id: str
    entity_type: str
    canonical_name: str
    tier: str
    score: float

    def describe(self) -> str:
        return f'"{self.surface}" -> {self.entity_id} ({self.tier}, score={self.score})'


@dataclass
class Tier1Hit:
    surface: str
    key: str
    entity_id: str
    index: str  # "typed" or "global"
    from_parenthetical: bool


@dataclass
class PairRun:
    first: Resolution
    second: Resolution
    first_node_canonical: str
    first_node_surfaces: List[str]
    tier1_hits_for_second: List[Tier1Hit]

    @property
    def merged(self) -> bool:
        return self.first.entity_id == self.second.entity_id


@dataclass
class CaseResult:
    bucket: str
    number: int
    case_id: str
    title: str
    expected: str
    passed: bool
    actual_lines: List[str]
    hint: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------


class ResolverFactory:
    def __init__(self, seeds_path: Path) -> None:
        self.seeds_path = seeds_path
        baseline = self.fresh()
        # key -> entity_id for every surface defined by the seed file (global index)
        self.seed_global_keys: Dict[str, str] = dict(baseline.global_alias_index)

    def fresh(self) -> EntityResolver:
        return EntityResolver(seed_aliases_path=self.seeds_path)


def resolve(resolver: EntityResolver, etype: EntityType, surface: str, chunk_id: str) -> Resolution:
    node = resolver.resolve_entity(
        ExtractedEntity(entity_type=etype, canonical_name=surface, confidence=0.9),
        source_chunk_id=chunk_id,
        paper_id="benchmark_paper",
    )
    log = resolver.decision_logs[-1]
    return Resolution(
        surface=surface,
        entity_id=node.entity_id,
        entity_type=node.entity_type.value,
        canonical_name=node.canonical_name,
        tier=log.tier_used,
        score=log.similarity_score,
    )


def probe_tier1(resolver: EntityResolver, etype: EntityType, surface: str) -> List[Tier1Hit]:
    """Replicate the Tier 1 lookup (without mutating state) to see which keys would match."""
    primary, paren = split_name_and_parenthetical_acronym(surface)
    hits: List[Tier1Hit] = []
    for surf in [primary] + paren:
        key = normalize_surface_key(surf)
        if not key:
            continue
        typed = resolver.alias_index.get((etype, key))
        glob = resolver.global_alias_index.get(key)
        if typed:
            hits.append(Tier1Hit(surf, key, typed, "typed", surf in paren))
        elif glob:
            hits.append(Tier1Hit(surf, key, glob, "global", surf in paren))
    return hits


def run_pair(resolver: EntityResolver, etype: EntityType, first: str, second: str, tag: str) -> PairRun:
    r1 = resolve(resolver, etype, first, f"{tag}::c00")
    node1 = resolver.nodes[r1.entity_id]
    node1_surfaces = [node1.canonical_name] + list(node1.aliases)
    hits = probe_tier1(resolver, etype, second)
    r2 = resolve(resolver, etype, second, f"{tag}::c01")
    return PairRun(r1, r2, node1.canonical_name, node1_surfaces, hits)


def best_lexical_ratio(surface: str, node_surfaces: List[str]) -> Tuple[float, str]:
    primary, paren = split_name_and_parenthetical_acronym(surface)
    best, best_s = 0.0, ""
    for cand in [primary] + paren:
        for ns in node_surfaces:
            r = SequenceMatcher(None, normalize_surface_key(cand), normalize_surface_key(ns)).ratio()
            if r > best:
                best, best_s = r, ns
    return best, best_s


def fmt_sig(text: str) -> str:
    sig = sorted(extract_version_signature(text))
    return "{" + ", ".join(f"'{s}'" for s in sig) + "}" if sig else "{}"


# ---------------------------------------------------------------------------
# Root-cause hints
# ---------------------------------------------------------------------------


def hint_failed_merge(run: PairRun) -> str:
    second_primary, _ = split_name_and_parenthetical_acronym(run.second.surface)
    target = run.first_node_canonical
    if has_version_conflict(second_primary, target):
        sig_new, sig_old = fmt_sig(second_primary), fmt_sig(target)
        msg = (
            f"Version guard: has_version_conflict('{second_primary}', '{target}') is True because "
            f"extract_version_signature() returns {sig_new} vs {sig_old}"
        )
        empty_side = second_primary if sig_new == "{}" else target if sig_old == "{}" else None
        if empty_side and GLUED_DIGIT_RE.search(empty_side):
            msg += (
                f"; _VERSION_TOKEN_RE requires a separator before the digit, so glued '{empty_side}' "
                "carries no version and the one-sided rule declares a conflict"
            )
        if run.tier1_hits_for_second:
            hit = run.tier1_hits_for_second[0]
            msg += f" (Tier 1 had found key '{hit.key}' -> {hit.entity_id} but the guard rejected it)"
        return msg + "."

    ratio, against = best_lexical_ratio(run.second.surface, run.first_node_surfaces)
    msg = (
        f"No Tier 1 key match for '{normalize_surface_key(second_primary)}', and the best Tier 2 "
        f"SequenceMatcher ratio is {ratio:.3f} (vs '{against}') < {TIER2_LEXICAL_THRESHOLD}; Tier 3 is off"
    )
    if SCALE_SUFFIX_RE.search(run.second.surface) or SCALE_SUFFIX_RE.search(run.first.surface):
        msg += "; nothing strips parameter-scale suffixes (e.g. '8B') before lookup"
    return msg + "."


def hint_failed_split(run: PairRun, factory: ResolverFactory) -> str:
    merged_id = run.second.entity_id
    first_primary, _ = split_name_and_parenthetical_acronym(run.first.surface)
    second_primary, _ = split_name_and_parenthetical_acronym(run.second.surface)

    if run.second.tier == "TIER_1_ALIAS":
        hit = next((h for h in run.tier1_hits_for_second if h.entity_id == merged_id), None)
        if hit is None:
            return "Tier 1 alias lookup merged the pair (matching key not identified)."
        if factory.seed_global_keys.get(hit.key) == merged_id and not hit.from_parenthetical:
            return (
                f"seed_aliases.yaml maps '{hit.surface}' (key '{hit.key}') to {merged_id}, so Tier 1 merges "
                f"it and the version guard sees matching signatures {fmt_sig(second_primary)} vs "
                f"{fmt_sig(run.first_node_canonical)}."
            )
        if hit.from_parenthetical:
            return (
                f"split_name_and_parenthetical_acronym() treated '({hit.surface})' as an acronym alias; key "
                f"'{hit.key}' matched {merged_id} via the {hit.index} alias index, and the version guard only "
                f"compares primary names ('{second_primary}' vs '{run.first_node_canonical}'), which carry no versions."
            )
        return f"Tier 1 key '{hit.key}' matched {merged_id} via the {hit.index} alias index."

    if run.second.tier == "TIER_2_ACRONYM_LEXICAL":
        sa, sb = fmt_sig(first_primary), fmt_sig(second_primary)
        msg = f"Tier 2 lexical match (score {run.second.score} >= {TIER2_LEXICAL_THRESHOLD}) and the version guard passed: signatures {sa} vs {sb}"
        sigs = {sa, sb}
        if sigs == {"{'1'}", "{}"}:
            msg += "; has_version_conflict() lets an explicit {'1'} match a version-less name, so 'K' is treated like 'v1'"
        elif "{}" in sigs:
            glued = [s for s in (first_primary, second_primary) if GLUED_DIGIT_RE.search(s)]
            if glued:
                msg += f"; _VERSION_TOKEN_RE misses glued digits in {glued}, so no version/precision token is extracted"
        return msg + "."

    return f"Merged via {run.second.tier} (score {run.second.score})."


# ---------------------------------------------------------------------------
# Bucket runners
# ---------------------------------------------------------------------------


def run_pair_bucket(bucket: str, cases: List[dict], factory: ResolverFactory, expect_merge: bool) -> List[CaseResult]:
    results: List[CaseResult] = []
    expected = "MERGE (same entity_id)" if expect_merge else "SPLIT (different entity_ids)"
    for n, case in enumerate(cases, start=1):
        etype = EntityType(case["entity_type"])
        a, b = case["surface_a"], case["surface_b"]
        title = f'"{a}" vs "{b}"  [{etype.value}]'
        try:
            ab = run_pair(factory.fresh(), etype, a, b, f"{case['id']}_ab")
            ba = run_pair(factory.fresh(), etype, b, a, f"{case['id']}_ba")
        except Exception as exc:  # report, don't abort the scorecard
            results.append(CaseResult(bucket, n, case["id"], title, expected, False,
                                      [f"EXCEPTION: {type(exc).__name__}: {exc}"], "Resolver raised an exception."))
            continue

        runs = {"A->B": ab, "B->A": ba}
        ok = {k: (r.merged == expect_merge) for k, r in runs.items()}
        lines = []
        for label, r in runs.items():
            verdict = "MERGE" if r.merged else "SPLIT"
            lines.append(f"{label} {verdict}: {r.first.describe()}; {r.second.describe()}")
        hint = ""
        failing = [runs[k] for k, v in ok.items() if not v]
        if failing:
            r = failing[0]
            hint = hint_failed_merge(r) if expect_merge else hint_failed_split(r, factory)
        results.append(CaseResult(bucket, n, case["id"], title, expected, all(ok.values()), lines, hint))
    return results


def run_sequential(synonyms: List[dict], adversarial: List[dict], factory: ResolverFactory,
                   isolated: Dict[str, CaseResult]) -> List[str]:
    """Run all pairs A->B on one shared resolver; return lines describing outcome changes."""
    resolver = factory.fresh()
    diffs: List[str] = []
    for bucket, cases, expect_merge in (("synonym_merges", synonyms, True), ("adversarial_splits", adversarial, False)):
        for case in cases:
            etype = EntityType(case["entity_type"])
            try:
                r = run_pair(resolver, etype, case["surface_a"], case["surface_b"], f"{case['id']}_seq")
            except Exception as exc:
                diffs.append(f"  {case['id']}: EXCEPTION {type(exc).__name__}: {exc}")
                continue
            seq_pass = r.merged == expect_merge
            iso_pass = isolated[case["id"]].passed
            if seq_pass != iso_pass:
                direction = "PASS -> FAIL" if iso_pass else "FAIL -> PASS"
                diffs.append(
                    f"  [{bucket.upper()}] {case['id']} isolated vs sequential: {direction}\n"
                    f"      {r.first.describe()}\n"
                    f"      {r.second.describe()}"
                )
    return diffs


def run_cross_type_bucket(cases: List[dict], factory: ResolverFactory) -> List[CaseResult]:
    results: List[CaseResult] = []
    for n, case in enumerate(cases, start=1):
        stype, ttype = EntityType(case["source_type"]), EntityType(case["target_type"])
        title = (f'{stype.value}("{case["source_name"]}") --[{case["relation_type"]}]--> '
                 f'{ttype.value}("{case["target_name"]}")')
        expected = "EDGE_SURVIVES (distinct source/target ids with correct type prefixes)"
        resolver = factory.fresh()
        try:
            rel = ExtractedRelationship(
                source_name=case["source_name"], source_type=stype,
                relation_type=RelationType(case["relation_type"]),
                target_name=case["target_name"], target_type=ttype,
                evidence_quote=case["evidence_quote"], confidence=0.9,
            )
            edge = resolver.resolve_relationship(rel, source_chunk_id=f"{case['id']}::c00", paper_id="benchmark_paper")
        except Exception as exc:
            results.append(CaseResult("cross_type_edges", n, case["id"], title, expected, False,
                                      [f"EXCEPTION: {type(exc).__name__}: {exc}"], "Ontology validation or resolver raised."))
            continue

        src_log, tgt_log = resolver.decision_logs[-2], resolver.decision_logs[-1]
        src_node = resolver.nodes[src_log.resolved_entity_id]
        tgt_node = resolver.nodes[tgt_log.resolved_entity_id]
        exp_src_prefix = slugify_entity_id(stype, "x").split("::")[0]
        exp_tgt_prefix = slugify_entity_id(ttype, "x").split("::")[0]
        types_ok = src_node.entity_type == stype and tgt_node.entity_type == ttype
        passed = edge is not None and types_ok

        lines = [
            f"source -> {src_node.entity_id} [{src_node.entity_type.value}] ({src_log.tier_used})",
            f"target -> {tgt_node.entity_id} [{tgt_node.entity_type.value}] ({tgt_log.tier_used})",
            "edge   -> " + (f"{edge.source_entity_id} --[{edge.relation_type.value}]--> {edge.target_entity_id}"
                            if edge else "None (dropped as self-loop)"),
        ]
        hint = ""
        if not passed:
            def article(word: str) -> str:
                return "an" if word[0].upper() in "AEIOU" else "a"

            wrong = []
            if src_node.entity_type != stype:
                t = src_node.entity_type.value
                wrong.append(f"source '{case['source_name']}' ({stype.value}) resolved to {article(t)} {t} node")
            if tgt_node.entity_type != ttype:
                t = tgt_node.entity_type.value
                wrong.append(f"target '{case['target_name']}' ({ttype.value}) resolved to {article(t)} {t} node")
            hint = (
                "Tier 1 falls back to global_alias_index, which ignores entity_type, so "
                + " and ".join(wrong)
                + (f" (expected prefixes {exp_src_prefix}:: / {exp_tgt_prefix}::)" if wrong else "")
                + ("; resolve_relationship() then sees src == tgt and returns None." if edge is None else ".")
            )
        results.append(CaseResult("cross_type_edges", n, case["id"], title, expected, passed, lines, hint))
    return results


def hint_mention(text: str, missing: List[str], unexpected: List[str], resolver: EntityResolver) -> str:
    norm = f" {normalize_surface_key(text)} "
    parts: List[str] = []
    for eid in missing:
        keys = sorted(k for k, v in resolver.global_alias_index.items() if v == eid)
        punct = [k for k in keys if re.search(rf" {re.escape(k)}[\.]", norm)]
        plural = [k for k in keys if f" {k}s " in norm]
        if punct:
            parts.append(
                f"normalize_surface_key() keeps '.', so the chunk token is '{punct[0]}.' and never equals "
                f"the padded key ' {punct[0]} ' for {eid}"
            )
        elif plural:
            parts.append(
                f"normalize_surface_key() singularizes only the LAST token, so the seed alias became "
                f"'{plural[0]}' while the chunk keeps mid-sentence '{plural[0]}s' ({eid} missed)"
            )
        else:
            parts.append(f"no alias key of {eid} appears as a whole-word span in '{norm.strip()}'")
    for eid in unexpected:
        keys = sorted((k for k, v in resolver.global_alias_index.items() if v == eid and len(k) >= 3 and f" {k} " in norm), key=len)
        if keys:
            k = keys[0]
            longer = [lk for lk, lv in resolver.global_alias_index.items()
                      if lv != eid and k in lk and len(lk) > len(k) and f" {lk} " in norm]
            inside = f" inside '{longer[0]}'" if longer else ""
            parts.append(
                f"alias key '{k}' of {eid} matches{inside}; resolve_Text_mentions_in_chunk() has no "
                "longest-match/overlap suppression"
            )
        else:
            parts.append(f"{eid} matched unexpectedly")
    return "; ".join(parts) + "."


def run_chunk_mention_bucket(cases: List[dict], factory: ResolverFactory) -> List[CaseResult]:
    results: List[CaseResult] = []
    chunker = StructureAwareChunker()
    for n, case in enumerate(cases, start=1):
        text = case["text"]
        shown = text.replace("\n", "\\n")
        try:
            if case["kind"] == "mention":
                resolver = factory.fresh()
                found = resolver.resolve_Text_mentions_in_chunk(text)
                expected_ids, forbidden = case["expected_entity_ids"], case.get("forbidden_entity_ids", [])
                missing = [e for e in expected_ids if e not in found]
                unexpected = [e for e in forbidden if e in found]
                passed = not missing and not unexpected
                expected = "MENTIONS " + ", ".join(expected_ids) + (f" and NOT {', '.join(forbidden)}" if forbidden else "")
                lines = [f"mentioned_entity_ids = {found}"]
                if missing:
                    lines.append(f"missing    = {missing}")
                if unexpected:
                    lines.append(f"unexpected = {unexpected}")
                hint = "" if passed else hint_mention(text, missing, unexpected, resolver)
                title = f'mention("{shown}")'
            else:
                chunks = chunker.chunk_paper("benchmark_paper", "Benchmark Paper", 2024, "architecture",
                                             f"# Results\n\n{text}\n")
                actual = any(c.contains_equation for c in chunks)
                exp = bool(case["expected_contains_equation"])
                passed = actual == exp
                expected = f"contains_equation = {exp}"
                lines = [f"contains_equation = {actual} ({len(chunks)} chunk(s))"]
                hint = ""
                if not passed:
                    if actual and "$$" not in text:
                        hint = ("chunk_paper() sets has_eq = _EQUATION_BLOCK_RE.search(raw_text) or \"$\" in raw_text, "
                                "so any dollar sign (currency) flags an equation.")
                    else:
                        hint = "Neither _EQUATION_BLOCK_RE nor the '$' check fired for this math."
                title = f'chunker("{shown}")'
        except Exception as exc:
            results.append(CaseResult("chunk_and_mention_cases", n, case["id"], f'"{shown}"', "n/a", False,
                                      [f"EXCEPTION: {type(exc).__name__}: {exc}"], "Component raised an exception."))
            continue
        results.append(CaseResult("chunk_and_mention_cases", n, case["id"], title, expected, passed, lines, hint))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(gold_path: Path, seeds_path: Path, by_bucket: Dict[str, List[CaseResult]], seq_diffs: List[str]) -> None:
    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(p)

    print(LINE)
    print(" PHASE 1A SCORECARD")
    print(f" Gold cases : {rel(gold_path)}")
    print(f" Seeds      : {rel(seeds_path)}")
    print(" Mode       : isolated fresh resolver per pair, both orders must pass | Tier 3 embeddings OFF")
    print(LINE)
    print(f" {'Bucket':<28}{'Passed / Total':>16}{'Score':>12}")
    print(SUBLINE)
    total_pass = total = 0
    pcts = []
    for bucket, results in by_bucket.items():
        p, t = sum(r.passed for r in results), len(results)
        total_pass, total = total_pass + p, total + t
        pct = 100.0 * p / t if t else 0.0
        pcts.append(pct)
        print(f" {bucket:<28}{f'{p} / {t}':>16}{pct:>11.1f}%")
    print(SUBLINE)
    overall = 100.0 * total_pass / total if total else 0.0
    macro = sum(pcts) / len(pcts) if pcts else 0.0
    print(f" {'OVERALL PHASE 1A SCORE':<28}{f'{total_pass} / {total}':>16}{overall:>11.1f}%")
    print(f" {'(macro avg of buckets)':<28}{'':>16}{macro:>11.1f}%")
    print(f" Sequential order effects (one shared resolver): {len(seq_diffs)} pair outcome(s) differ from isolated runs")
    print(LINE)

    failed = [r for results in by_bucket.values() for r in results if not r.passed]
    print()
    print(f" FAILED CASE DIAGNOSTICS ({len(failed)})")
    print(LINE)
    for r in failed:
        print(f"[{r.bucket.upper()}] Case #{r.number} ({r.case_id}): {r.title}")
        print(f"  Expected: {r.expected}")
        print(f"  Actual:   {r.actual_lines[0]}")
        for extra in r.actual_lines[1:]:
            print(f"            {extra}")
        print(f"  Root Cause Hint: {r.hint}")
        print(SUBLINE)

    print()
    print(" SEQUENTIAL ORDER EFFECTS (all pairs A->B in file order on one shared resolver)")
    print(LINE)
    if seq_diffs:
        for d in seq_diffs:
            print(d)
    else:
        print("  None: every pair gave the same verdict as its isolated run.")
    print(LINE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1A benchmark scorecard (read-only).")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH, help="Gold cases JSON path.")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH, help="Seed aliases YAML path.")
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    buckets = gold["buckets"]
    factory = ResolverFactory(args.seeds)

    by_bucket: Dict[str, List[CaseResult]] = {
        "synonym_merges": run_pair_bucket("synonym_merges", buckets["synonym_merges"], factory, expect_merge=True),
        "adversarial_splits": run_pair_bucket("adversarial_splits", buckets["adversarial_splits"], factory, expect_merge=False),
        "cross_type_edges": run_cross_type_bucket(buckets["cross_type_edges"], factory),
        "chunk_and_mention_cases": run_chunk_mention_bucket(buckets["chunk_and_mention_cases"], factory),
    }
    isolated = {r.case_id: r for key in ("synonym_merges", "adversarial_splits") for r in by_bucket[key]}
    seq_diffs = run_sequential(buckets["synonym_merges"], buckets["adversarial_splits"], factory, isolated)

    print_report(args.gold, args.seeds, by_bucket, seq_diffs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
