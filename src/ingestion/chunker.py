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
from typing import List, Tuple


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


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_EQUATION_BLOCK_RE = re.compile(
    r"(\$\$[\s\S]*?\$\$|\\begin\{(?:equation\*?|align\*?|gather\*?)\}[\s\S]*?\\end\{(?:equation\*?|align\*?|gather\*?)\})",
    re.MULTILINE,
)


def compute_content_hash(text: str, schema_version: str = "v1") -> str:
    """Return deterministic SHA-256 hex digest for extraction caching."""
    normalized = re.sub(r"\s+", " ", text.strip())
    payload = f"{schema_version}::{normalized}".encode("utf-8")
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

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
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

    stashed_text = _EQUATION_BLOCK_RE.sub(_stash_eq, section_body)

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
        overlap_blocks: int = 1,
    ) -> None:
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_blocks = overlap_blocks

    def chunk_paper(
        self,
        paper_id: str,
        paper_title: str,
        pub_year: int,
        pillar: str,
        markdown_text: str,
    ) -> List[DocumentChunk]:
        """Split a paper into deterministic `DocumentChunk` objects."""
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
                has_eq = bool(
                    _EQUATION_BLOCK_RE.search(raw_text) or "$" in raw_text
                )
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

        return all_chunks
