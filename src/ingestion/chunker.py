"""Structure-Aware Document Chunker with Shared `chunk_id` Generation (Phase 1 & 2).

Design Decisions:
-----------------
1. Shared Deterministic Key (`chunk_id`):
   Every chunk gets a deterministic ID `{paper_id}::s{section_idx:02d}::c{chunk_idx:02d}`.
   This exact string is stored on Neo4j relationships (`r.source_chunk_ids`) AND as the
   PRIMARY KEY in PostgreSQL (`pgvector`), allowing zero-overhead joins between graph
   paths and verbatim text passages.

2. Why Not Fixed-Token Character Splitting?
   In LLM architecture papers, splitting every 500 characters slices right through:
   - Display LaTeX equations (`$$ ... $$` or `\\begin{equation} ... \\end{equation}`)
   - Architectural comparison tables (e.g., MHA vs MQA vs GQA KV-cache table)
   - Section boundaries (mixing "Background" with "Proposed Method")

   Our chunker:
   a) Parses the document into a hierarchical section tree (`Section 2 > 2.2 SwiGLU`).
   b) Treats LaTeX equation blocks and Markdown/LaTeX tables as atomic units.
   c) Packs atomic blocks into target windows (default ~1,800 chars / ~450 tokens)
      with section-bounded overlap and prepends a `[Paper | Year | Section]` header.
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import List, Optional, Tuple

from src.schema.ontology import EXTRACTION_PROMPT_VERSION, EXTRACTION_SCHEMA_VERSION


@dataclass(frozen=True)
class DocumentChunk:
    """A single structure-aware text chunk shared across Neo4j and pgvector."""

    chunk_id: str
    paper_id: str
    paper_title: str
    pub_year: int
    pillar: str
    section_path: str
    section_index: int
    chunk_index: int
    raw_text: str
    context_prefixed_text: str
    content_hash: str
    approx_token_count: int
    contains_equation: bool = False
    contains_table: bool = False
    mentioned_entity_ids: Tuple[str, ...] = field(default_factory=tuple)


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_FENCED_BLOCK_RE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$", re.MULTILINE)
_EQUATION_BLOCK_RE = re.compile(
    r"(\$\$[\s\S]*?\$\$|\\begin\{(?:equation\*?|align\*?|gather\*?)\}[\s\S]*?\\end\{(?:equation\*?|align\*?|gather\*?)\})",
    re.MULTILINE,
)
# LaTeX \( ... \) and \[ ... \] delimiters are unambiguous math.
_LATEX_DELIM_RE = re.compile(r"\\\((?:.|\n)+?\\\)|\\\[(?:.|\n)+?\\\]")
# Pandoc-style inline math: the opening '$' is followed by a non-space, the closing '$' is
# preceded by a non-space and NOT followed by a digit. This already rejects
# '$0.14 per million ... versus $0.55' (closing '$' preceded by a space).
_INLINE_MATH_RE = re.compile(r"(?<![\\$\w])\$(?=\S)(?P<body>[^$\n]{1,300}?)(?<=\S)\$(?!\d)")
_MATH_SIGNAL_RE = re.compile(r"[\\^_{}=<>|]|[A-Za-z]\(|\([A-Za-z]")
_SHORT_VARIABLE_RE = re.compile(r"[A-Za-z][A-Za-z0-9']{0,2}")
_CURRENCY_BODY_RE = re.compile(r"[\d.,]+\s*[KMBkmb]?")


def contains_equation(text: str) -> bool:
    """Return True for display/inline math, False for currency amounts like '$2M' or '$100,000'.

    True:  '$$ c_t^{KV} = W h_t $$', '\\begin{equation}...', '$O(N^2)$', '$d_k$', '$N$'.
    False: 'cost was $2M', '$0.14 per million input tokens versus $0.55', 'a $100,000 budget'.
    """
    if not text or ("$" not in text and "\\" not in text):
        return False
    if _EQUATION_BLOCK_RE.search(text) or _LATEX_DELIM_RE.search(text):
        return True
    for m in _INLINE_MATH_RE.finditer(text):
        body = m.group("body").strip()
        if _CURRENCY_BODY_RE.fullmatch(body):
            continue
        if _MATH_SIGNAL_RE.search(body) or _SHORT_VARIABLE_RE.fullmatch(body):
            return True
    return False


_BIB_TITLE = r"(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|works\s+cited|literature\s+cited|reference\s+list)"
_BIB_HEADING_RE = re.compile(
    rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*|__)?\s*{_BIB_TITLE}\s*(?:\*\*|__)?\s*:?\s*$", re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+.+|(?:\*\*|__)[^*_]{2,80}(?:\*\*|__)\s*)$")
_BACK_MATTER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:appendix|appendices|supplementary|supplemental|"
    r"[A-H](?:\.\d+)*\.?\s+[A-Z][a-z])", re.IGNORECASE)

_ATX_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")
_BOLD_HEADING_RE = re.compile(r"^\s*(\*\*|__)(.+?)\1\s*:?\s*$")
_CORE_SECTION_RE = re.compile(
    r"\b(?:"
    r"abstract|introduction|background|"
    r"methods?|methodology|architectures?|"
    r"experiments?|experimental\s+(?:setup|results?|evaluation|details)|results?|"
    r"ablations?|ablation\s+stud(?:y|ies)|"
    r"conclusions?"
    r")\b",
    re.IGNORECASE,
)
_APPENDIX_OR_SUPP_RE = re.compile(
    r"^(?:appendix|appendices|supplementary(?:\s+materials?)?|supplemental(?:\s+materials?)?)\b",
    re.IGNORECASE,
)
_NON_CORE_SECTION_RE = re.compile(
    r"^(?:"
    r"references?|reference\s+list|(?:selected\s+)?bibliography|works\s+cited|literature\s+cited|"
    r"(?:funding\s+(?:and\s+)?)?acknowledg?e?ments?(?:\s+and\s+funding)?|acknowlegements?|"
    r"ethics(?:\s+(?:and\s+safety\s+)?(?:statement|considerations?|checklist))?|"
    r"ethical\s+(?:statement|considerations?|checklist)|"
    r"(?:broader(?:\s+societal)?|societal)\s+impacts?(?:\s+statement)?|impact\s+statement|"
    r"(?:authors?['’]?\s+)?contributions?(?:\s+statement)?|(?:core\s+)?contributors?|author\s+list|"
    r"reproducibility(?:\s+(?:statement|checklist))?|"
    r"limitations?(?:\s+of\s+(?:the\s+)?(?:work|study|approach|method|model))?"
    r"(?:\s+and\s+(?:future\s+(?:work|directions)|broader\s+impacts?))?"
    r")\b",
    re.IGNORECASE,
)
_PLAIN_NON_CORE_EXACT_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:"
    r"references?|reference\s+list|bibliography|works\s+cited|literature\s+cited|"
    r"acknowledg?e?ments?|acknowlegements?|"
    r"ethics(?:\s+statement)?|broader\s+impacts?(?:\s+statement)?|impact\s+statement|"
    r"author\s+contributions?|contributors?|reproducibility(?:\s+statement)?|"
    r"limitations?|appendix|appendices|supplementary(?:\s+materials?)?|supplemental(?:\s+materials?)?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)


def _clean_heading_text(raw: str) -> str:
    """Strip inline Markdown emphasis and trailing colons from a heading title."""
    cleaned = re.sub(r"[*_`]+", "", raw).strip()
    cleaned = re.sub(r":+\s*$", "", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def _is_non_core_heading(line: str) -> bool:
    """Return True if a Markdown heading line begins a non-core trailing/back-matter section."""
    hashes = 0
    is_bold = False
    m_atx = _ATX_HEADING_RE.match(line)
    if m_atx:
        hashes = len(m_atx.group(1))
        title = _clean_heading_text(m_atx.group(2))
    else:
        m_bold = _BOLD_HEADING_RE.match(line)
        if m_bold:
            is_bold = True
            title = _clean_heading_text(m_bold.group(2))
        else:
            return bool(_PLAIN_NON_CORE_EXACT_RE.match(line.strip()))

    if not title:
        return False

    unnumbered = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()
    if _APPENDIX_OR_SUPP_RE.match(unnumbered):
        return True
    if re.match(r"^[A-Z]\.\d+(?:\.\d+)*\b", title):
        return True
    if re.match(r"^[A-Z]\.(?:\s|$)", title):
        # Preserve Roman-numeral core sections like "I. Introduction" or "V. Conclusion".
        if re.match(r"^[IVX]\.", title) and _CORE_SECTION_RE.search(title):
            return False
        return True
    if (1 <= hashes <= 3 or is_bold) and re.match(r"^[A-HJ-Z]\s+[A-Za-z]", title):
        return True
    if _CORE_SECTION_RE.search(unnumbered):
        return False
    return bool(_NON_CORE_SECTION_RE.match(unnumbered))


def strip_non_core_sections(markdown_text: str, min_position: float = 0.25) -> Tuple[str, int]:
    """Strip non-core trailing/back-matter sections once encountered (respecting fenced code blocks).

    Removes back-matter starting at the first qualifying heading at or after `min_position`
    of the document: References, Bibliography, Works/Literature Cited, Acknowledgments,
    Ethics Statement, Broader Impact/Impact Statement, Author Contributions/Contributors,
    Reproducibility Statement, Limitations, Appendix/Appendices, Supplementary Material, and
    lettered appendix headings (e.g. `Appendix A`, `# A. ...`), while preserving core
    sections such as Abstract, Introduction, Background, Method/Architecture,
    Experiments/Results, Ablations, and Conclusion. Returns `(text, removed_lines_count)`.
    """
    if not markdown_text:
        return markdown_text, 0
    lines = markdown_text.splitlines()
    n_lines = len(lines)
    in_fence = False
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and i >= n_lines * min_position and _is_non_core_heading(line):
            start = i
            break
    if start is None:
        return markdown_text, 0
    return "\n".join(lines[:start]), n_lines - start


def strip_bibliography(markdown_text: str, min_position: float = 0.3) -> Tuple[str, int]:
    """Remove a trailing References/Bibliography section; keep appendices that follow it.

    The reference list runs from the bibliography heading to the next appendix-like heading
    ("Appendix", "A Experimental Details", "Supplementary Material") or the end of the document.
    A heading in the first `min_position` of the document is ignored (e.g. a "Related Work /
    References" mention in a table of contents). Returns (text, number_of_removed_lines).
    """
    lines = markdown_text.splitlines()
    start = None
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and i >= len(lines) * min_position and _BIB_HEADING_RE.match(line):
            start = i  # keep the last match: the real bibliography is at the back
    if start is None:
        return markdown_text, 0
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _ANY_HEADING_RE.match(lines[j]) and _BACK_MATTER_RE.match(lines[j]):
            end = j
            break
    kept = lines[:start] + lines[end:]
    return "\n".join(kept), end - start


def compute_content_hash(
    text: str,
    schema_version: str = EXTRACTION_SCHEMA_VERSION,
    prompt_version: str = EXTRACTION_PROMPT_VERSION,
) -> str:
    """Return the extraction cache key: sha256(schema_version + prompt_version + normalized text)."""
    normalized = re.sub(r"\s+", " ", text.strip())
    payload = f"{schema_version}::{prompt_version}::{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimator (~4 chars per token for technical English + LaTeX)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _split_into_sections(markdown_text: str) -> List[Tuple[str, str]]:
    """Parse Markdown text into `(section_breadcrumb_path, section_body)` pairs."""
    lines = markdown_text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    heading_stack: List[Tuple[int, str]] = []
    current_path = "Abstract & Introduction"
    current_lines: List[str] = []

    in_fence = False
    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current_lines.append(line)
            continue
        m = None if in_fence else _HEADING_LINE_RE.match(line.strip())
        if m:
            # Flush previous section if it has substantive text
            body_text = "\n".join(current_lines).strip()
            if body_text:
                sections.append((current_path, current_lines))
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = " > ".join(h[1] for h in heading_stack)
            current_lines = []
        else:
            current_lines.append(line)

    if "\n".join(current_lines).strip():
        sections.append((current_path, current_lines))

    return [(path, "\n".join(body).strip()) for path, body in sections]


def _tokenize_atomic_blocks(section_body: str) -> List[str]:
    """Split a section into paragraphs while keeping equations and tables intact."""
    # First protect display equations with placeholders so double-newline splits don't break them
    protected_equations: List[str] = []

    def _stash_eq(match: re.Match) -> str:
        idx = len(protected_equations)
        protected_equations.append(match.group(0))
        return f"__ATOMIC_EQ_{idx}__"

    # Fenced code blocks are atomic too (blank lines inside them must not split the block)
    stashed_text = _FENCED_BLOCK_RE.sub(_stash_eq, section_body)
    stashed_text = _EQUATION_BLOCK_RE.sub(_stash_eq, stashed_text)

    # Group contiguous markdown table rows (`| ... |`) into single atomic blocks
    raw_lines = stashed_text.splitlines()
    blocks: List[str] = []
    current_block: List[str] = []
    in_table = False

    for line in raw_lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            if not in_table and current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            in_table = True
            current_block.append(line)
        else:
            if in_table:
                blocks.append("\n".join(current_block).strip())
                current_block = []
                in_table = False

            if not stripped:
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
            else:
                current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    # Restore equations and attach standalone equations to their preceding paragraph when small
    restored_blocks: List[str] = []
    for blk in blocks:
        if not blk:
            continue
        for idx, eq in enumerate(protected_equations):
            blk = blk.replace(f"__ATOMIC_EQ_{idx}__", eq)
        restored_blocks.append(blk)

    return restored_blocks


class StructureAwareChunker:
    """Chunks scientific papers by section while preserving equations and tables."""

    def __init__(
        self,
        target_chars: int = 1800,
        max_chars: int = 2600,
        overlap_blocks: int = 0,
        core_body_only: bool = False,
        max_chunks_per_paper: Optional[int] = None,
    ) -> None:
        if max_chunks_per_paper is not None and max_chunks_per_paper < 0:
            raise ValueError("max_chunks_per_paper must be >= 0 or None")
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_blocks = overlap_blocks
        self.core_body_only = core_body_only
        self.max_chunks_per_paper = max_chunks_per_paper

    def chunk_paper(
        self,
        paper_id: str,
        paper_title: str,
        pub_year: int,
        pillar: str,
        markdown_text: str,
    ) -> List[DocumentChunk]:
        """Split a paper into deterministic `DocumentChunk` objects (reference lists are stripped first)."""
        if self.max_chunks_per_paper == 0:
            return []
        if self.core_body_only:
            markdown_text, _ = strip_non_core_sections(markdown_text)
        markdown_text, _ = strip_bibliography(markdown_text)
        sections = _split_into_sections(markdown_text)
        all_chunks: List[DocumentChunk] = []

        for sec_idx, (section_path, section_body) in enumerate(sections):
            blocks = _tokenize_atomic_blocks(section_body)
            if not blocks:
                continue

            section_windows: List[List[str]] = []
            current_window: List[str] = []
            current_len = 0

            for blk in blocks:
                blk_len = len(blk)
                if current_window and (current_len + blk_len + 2 > self.target_chars):
                    section_windows.append(current_window)
                    # Carry over overlap block(s) within the same section if reasonable size
                    if (
                        self.overlap_blocks > 0
                        and len(current_window[-1]) < (self.target_chars // 2)
                    ):
                        current_window = current_window[-self.overlap_blocks :]
                        current_len = sum(len(x) for x in current_window)
                    else:
                        current_window = []
                        current_len = 0

                # If a single non-table, non-equation block exceeds max_chars, split by sentence
                if (
                    blk_len > self.max_chars
                    and not (blk.strip().startswith("|") and blk.strip().endswith("|"))
                    and not _EQUATION_BLOCK_RE.search(blk)
                ):
                    sentences = re.split(r"(?<=[.!?])\s+", blk)
                    for sent in sentences:
                        if current_window and (
                            current_len + len(sent) + 1 > self.target_chars
                        ):
                            section_windows.append(current_window)
                            current_window = []
                            current_len = 0
                        current_window.append(sent)
                        current_len += len(sent) + 1
                else:
                    current_window.append(blk)
                    current_len += blk_len + 2

            if current_window:
                section_windows.append(current_window)

            for chunk_idx, window_blocks in enumerate(section_windows):
                raw_text = "\n\n".join(window_blocks).strip()
                chunk_id = f"{paper_id}::s{sec_idx:02d}::c{chunk_idx:02d}"
                header = (
                    f"[Paper: {paper_title} ({pub_year}) | "
                    f"Pillar: {pillar} | Section: {section_path}]"
                )
                context_prefixed = f"{header}\n{raw_text}"
                has_eq = contains_equation(raw_text)
                has_tbl = bool(
                    re.search(r"^\|.*\|$", raw_text, flags=re.MULTILINE)
                )

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        paper_title=paper_title,
                        pub_year=pub_year,
                        pillar=pillar,
                        section_path=section_path,
                        section_index=sec_idx,
                        chunk_index=chunk_idx,
                        raw_text=raw_text,
                        context_prefixed_text=context_prefixed,
                        content_hash=compute_content_hash(context_prefixed),
                        approx_token_count=estimate_tokens(context_prefixed),
                        contains_equation=has_eq,
                        contains_table=has_tbl,
                    )
                )
                if self.max_chunks_per_paper is not None and len(all_chunks) >= self.max_chunks_per_paper:
                    return all_chunks

        return all_chunks
