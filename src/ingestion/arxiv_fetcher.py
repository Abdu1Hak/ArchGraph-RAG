"""Corpus Manifest Loader, Polite arXiv PDF Fetcher & PDF -> Markdown Converter (Phase 1B).

- `load_manifest()` reads `data/manifest/corpus_50_papers.yaml` (12 seed papers or all 50).
- `ArxivFetcher.download_pdf()` caches `https://arxiv.org/pdf/{arxiv_id}.pdf` to
  `data/raw_pdfs/{paper_id}.pdf`: single-threaded, custom User-Agent, timeout, retry with
  backoff, a delay between network requests (arXiv asks for ~3 s), %PDF magic check, atomic write.
- `ArxivFetcher.pdf_to_markdown()` uses `pymupdf4llm.to_markdown()` (falls back to plain
  PyMuPDF text) and caches cleaned Markdown at `data/cache/markdown/{paper_id}.md`.
- `load_or_synthesize_seed_markdown()` is the offline mode: cached Markdown if present,
  otherwise a deterministic synthetic fixture. It never touches the network.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

import httpx
import yaml
from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.offline_fixtures import SYNTHETIC_SEED_MARKDOWN, synthesize_generic_markdown
from src.schema.ontology import PillarType

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_PAPER_ID_RE = re.compile(r"^[a-z0-9_]+$")


class PaperMeta(BaseModel):
    """One corpus manifest entry."""

    paper_id: str = Field(pattern=_PAPER_ID_RE.pattern)
    arxiv_id: str = Field(pattern=_ARXIV_ID_RE.pattern)
    title: str
    year: int
    pillar: PillarType
    seed_set: bool = False

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"


def load_manifest(manifest_path: Optional[Path] = None, seed_only: bool = True) -> List[PaperMeta]:
    """Load the corpus manifest; `seed_only=True` keeps the `seed_set: true` papers."""
    path = manifest_path or settings.corpus_manifest_path
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    papers = [PaperMeta.model_validate(p) for p in data.get("papers", [])]
    ids = [p.paper_id for p in papers]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate paper_id(s) in manifest: {sorted(dupes)}")
    return [p for p in papers if p.seed_set] if seed_only else papers


def clean_markdown(text: str) -> str:
    """Normalize converter output: drop NULs/form feeds, trim trailing spaces, collapse blank runs."""
    text = text.replace("\x00", "").replace("\x0c", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


class PdfDownloadError(RuntimeError):
    pass


class ArxivFetcher:
    """Downloads and converts arXiv PDFs with on-disk caching at every stage."""

    def __init__(
        self,
        raw_pdf_dir: Optional[Path] = None,
        markdown_dir: Optional[Path] = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        request_delay_s: Optional[float] = None,
        user_agent: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.raw_pdf_dir = Path(raw_pdf_dir or settings.raw_pdf_dir)
        self.markdown_dir = Path(markdown_dir or settings.markdown_cache_dir)
        self.max_retries = max_retries
        self.request_delay_s = settings.arxiv_request_delay_s if request_delay_s is None else request_delay_s
        self._sleep = sleep
        self._last_request_at: Optional[float] = None
        self._client = http_client or httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent or settings.arxiv_user_agent},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArxivFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def pdf_path(self, paper: PaperMeta) -> Path:
        return self.raw_pdf_dir / f"{paper.paper_id}.pdf"

    def markdown_path(self, paper: PaperMeta) -> Path:
        return self.markdown_dir / f"{paper.paper_id}.md"

    def _polite_wait(self) -> None:
        if self._last_request_at is not None:
            remaining = self.request_delay_s - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = time.monotonic()

    def download_pdf(self, paper: PaperMeta) -> Path:
        """Return the cached PDF path, downloading it first if missing or empty."""
        dest = self.pdf_path(paper)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._polite_wait()
            try:
                resp = self._client.get(paper.pdf_url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise PdfDownloadError(f"HTTP {resp.status_code} from {paper.pdf_url}")
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    raise PdfDownloadError(f"Response from {paper.pdf_url} is not a PDF")
                tmp = dest.with_suffix(".pdf.part")
                tmp.write_bytes(resp.content)
                tmp.replace(dest)
                logger.info("Downloaded %s (%d bytes)", paper.pdf_url, len(resp.content))
                return dest
            except (httpx.HTTPError, PdfDownloadError) as exc:
                last_error = exc
                logger.warning("Download attempt %d/%d failed for %s: %s", attempt, self.max_retries, paper.paper_id, exc)
                if attempt < self.max_retries:
                    self._sleep(min(30.0, 2.0 ** attempt))
        raise PdfDownloadError(f"Failed to download {paper.pdf_url} after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def pdf_to_markdown(pdf_path: Path) -> str:
        """Convert a PDF to Markdown with pymupdf4llm, falling back to plain PyMuPDF text."""
        try:
            import pymupdf4llm

            return clean_markdown(pymupdf4llm.to_markdown(str(pdf_path)))
        except ImportError:
            logger.warning("pymupdf4llm not installed; falling back to plain PyMuPDF text extraction")
        try:
            import pymupdf
        except ImportError:  # PyMuPDF < 1.24 only ships the legacy 'fitz' name
            import fitz as pymupdf

        with pymupdf.open(str(pdf_path)) as doc:
            return clean_markdown("\n\n".join(page.get_text("text") for page in doc))

    def fetch_markdown(self, paper: PaperMeta) -> str:
        """Return cached Markdown, or download + convert + cache it."""
        md_path = self.markdown_path(paper)
        if md_path.exists() and md_path.stat().st_size > 0:
            return md_path.read_text(encoding="utf-8")
        markdown = self.pdf_to_markdown(self.download_pdf(paper))
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        return markdown


def load_or_synthesize_seed_markdown(
    paper: PaperMeta,
    markdown_dir: Optional[Path] = None,
    prefer_cached: bool = True,
) -> str:
    """Offline mode: cached real Markdown if present, else a deterministic synthetic fixture.

    Never uses the network and never writes synthetic text into the Markdown cache.
    `prefer_cached=False` always returns the synthetic fixture (used by `--offline-demo` and
    benchmarks so recorded extractions line up with the text).
    """
    md_path = Path(markdown_dir or settings.markdown_cache_dir) / f"{paper.paper_id}.md"
    if prefer_cached and md_path.exists() and md_path.stat().st_size > 0:
        return md_path.read_text(encoding="utf-8")
    fixture = SYNTHETIC_SEED_MARKDOWN.get(paper.paper_id)
    if fixture is not None:
        return fixture
    return synthesize_generic_markdown(paper.paper_id, paper.title, paper.year, paper.pillar.value)
