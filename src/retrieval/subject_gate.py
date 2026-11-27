"""Subject-Entity Verification Gate (Phase 5 fix #5).

Problem: an in-domain question about a model the corpus does not cover ("How many parameters does
GPT-4 have?") retrieves related-but-wrong chunks (GPT-3, DeepSeek) and gets a confident, cited,
wrong answer. The citation gate cannot catch this: every citation is valid, it just answers a
different question.

Rule: extract the named external models/systems the question is *about*, and refuse unless each
one appears in the active indexed corpus (chunk text). Two kinds of names are extracted:
1. Versioned model names: a capitalized name followed by a version/size token
   ("GPT-4", "Gemini 1.5", "Mixtral 8x22B", "Claude 3"). Four-digit years ("WMT 2014") are ignored.
2. A small registry of well-known unversioned external model/system names (Claude, PaLM, ...).
Generic capitalized words (company names, "Transformer", benchmark names) are never extracted, so
"Meta's 2024 herd" or "DeepMind's recipe" are not refused for naming an organization.

The check is against the *indexed corpus*, not a hard-coded allow list: once a GPT-4 paper is
ingested, GPT-4 questions are answered.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List

_VERSIONED_RE = re.compile(
    r"(?<![\w-])([A-Z][A-Za-z]{1,20}(?:[ -][A-Z][a-z]{2,20})?)[ -]?"
    r"((?:[Vv]|[Rr])?\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?[Bb])?)(?![\w.])"
)
_YEAR_RE = re.compile(r"^(19|20)\d\d$")
# Unversioned names of external systems that are commonly asked about. Matching any of these only
# triggers a corpus lookup; it never refuses by itself.
EXTERNAL_SYSTEM_NAMES = (
    "ChatGPT", "Claude", "PaLM", "AlphaFold", "Mixtral", "Mistral", "Gemini", "Gemma", "Falcon",
    "BERT", "Grok", "Diffusion Transformer", "DiT", "Stable Diffusion", "Copilot",
)
_EXTERNAL_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in EXTERNAL_SYSTEM_NAMES) + r")(?![\w-])")


def normalize_phrase(text: str) -> str:
    t = text.lower().replace("\u2013", " ").replace("\u2014", " ")
    t = re.sub(r"[-_]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_subject_names(question: str) -> List[str]:
    names: List[str] = []
    for m in _VERSIONED_RE.finditer(question):
        base, ver = m.group(1), m.group(2)
        if _YEAR_RE.match(ver):
            continue
        names.append(f"{base} {ver}")
    names += [m.group(1) for m in _EXTERNAL_RE.finditer(question)]
    return list(dict.fromkeys(names))


def unverified_subjects(question: str, vector_store: Any) -> List[str]:
    """Named subjects of the question that do not occur anywhere in the indexed corpus text."""
    names = extract_subject_names(question)
    if not names or vector_store is None or not hasattr(vector_store, "text_exists"):
        return []
    return [n for n in names if not vector_store.text_exists(normalize_phrase(n))]


def refusal_for(names: Iterable[str]) -> str:
    listed = ", ".join(names)
    return (f"The indexed corpus does not cover {listed}, so I can't answer this without guessing. "
            "It covers 2017-2025 LLM architecture papers such as the Transformer, LLaMA, DeepSeek-V2/V3/R1, "
            "GQA, RoPE, YaRN, DPO, Chinchilla and PagedAttention.")
