"""Centralized configuration and cost/threshold parameters for LLM Architecture KG-RAG."""

from pathlib import Path
from typing import Any, List, Tuple, Type

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini API Configuration (free tier: flash-lite 15 RPM / 250K TPM / 500 RPD per model)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    extraction_model: str = Field(default="gemini-3.5-flash-lite", alias="EXTRACTION_MODEL")
    router_model: str = Field(default="gemini-3.5-flash-lite", alias="ROUTER_MODEL")
    generation_model: str = Field(default="gemini-3.5-flash-lite", alias="GENERATION_MODEL")
    # Daily-quota fallback chain for text models, tried in order after the primary model.
    fallback_model_1: str = Field(default="gemini-3.5-flash-lite", alias="FALLBACK_MODEL_1")
    fallback_model_2: str = Field(default="gemini-3.1-flash-lite", alias="FALLBACK_MODEL_2")
    fallback_model_3: str = Field(default="gemma-4-31b-it", alias="FALLBACK_MODEL_3")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="llmkgrag2026", alias="NEO4J_PASSWORD")

    # PostgreSQL + pgvector Configuration
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="llmkgrag2026", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="llm_kg_rag", alias="POSTGRES_DB")

    # Phase 1: Extraction & Entity Resolution Thresholds
    max_extraction_budget_usd: float = Field(default=5.00, alias="MAX_EXTRACTION_BUDGET_USD")
    min_extraction_confidence: float = Field(default=0.65, alias="MIN_EXTRACTION_CONFIDENCE")
    er_auto_merge_threshold: float = Field(default=0.88, alias="ER_AUTO_MERGE_THRESHOLD")
    er_gray_zone_low_threshold: float = Field(default=0.82, alias="ER_GRAY_ZONE_LOW_THRESHOLD")

    # Phase 1B: Extraction cost model (USD per 1M tokens) and per-chunk token overheads
    extraction_input_usd_per_1m: float = Field(default=0.15, alias="EXTRACTION_INPUT_USD_PER_1M")
    extraction_output_usd_per_1m: float = Field(default=0.60, alias="EXTRACTION_OUTPUT_USD_PER_1M")
    extraction_prompt_overhead_tokens: int = Field(default=650, alias="EXTRACTION_PROMPT_OVERHEAD_TOKENS")
    extraction_output_tokens_per_chunk: int = Field(default=350, alias="EXTRACTION_OUTPUT_TOKENS_PER_CHUNK")

    # Phase 5: projected query-time cost model (USD per 1M tokens). These are ASSUMED list-style
    # rates for reporting only; override them in .env with your actual contract prices.
    router_input_usd_per_1m: float = Field(default=0.15, alias="ROUTER_INPUT_USD_PER_1M")
    router_output_usd_per_1m: float = Field(default=0.60, alias="ROUTER_OUTPUT_USD_PER_1M")
    generation_input_usd_per_1m: float = Field(default=1.25, alias="GENERATION_INPUT_USD_PER_1M")
    generation_output_usd_per_1m: float = Field(default=10.00, alias="GENERATION_OUTPUT_USD_PER_1M")
    embedding_usd_per_1m: float = Field(default=0.15, alias="EMBEDDING_USD_PER_1M")

    # Phase 1B: arXiv fetching
    arxiv_user_agent: str = Field(
        default="llm-arch-kg-rag/0.1 (research corpus builder; polite single-threaded fetcher)",
        alias="ARXIV_USER_AGENT",
    )
    arxiv_request_delay_s: float = Field(default=3.0, alias="ARXIV_REQUEST_DELAY_S")

    # Paths
    seed_aliases_path: Path = Field(
        default=PROJECT_ROOT / "data" / "manifest" / "seed_aliases.yaml"
    )
    corpus_manifest_path: Path = Field(
        default=PROJECT_ROOT / "data" / "manifest" / "corpus_50_papers.yaml"
    )
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    raw_pdf_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw_pdfs")

    @property
    def markdown_cache_dir(self) -> Path:
        return self.cache_dir / "markdown"

    @property
    def extraction_cache_path(self) -> Path:
        return self.cache_dir / "extractions.sqlite"

    @field_validator("gemini_api_key", "extraction_model", "router_model", "generation_model", "embedding_model",
                     "fallback_model_1", "fallback_model_2", "fallback_model_3", mode="before")
    @classmethod
    def _strip_quotes(cls, v: Any) -> Any:
        """Accept KEY="value", KEY='value' and stray whitespace in .env."""
        if isinstance(v, str):
            v = v.strip()
            while len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1].strip()
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # .env wins over inherited process environment variables, so swapping GEMINI_API_KEY in .env takes
        # effect on the next run even if an old key is still exported in the shell. Without a .env file
        # (containers), plain environment variables are used as usual.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @property
    def fallback_models(self) -> List[str]:
        """FALLBACK_MODEL_1..3, non-empty, de-duplicated, in order."""
        out: List[str] = []
        for m in (self.fallback_model_1, self.fallback_model_2, self.fallback_model_3):
            if m and m not in out:
                out.append(m)
        return out

    def text_model_chain(self, primary: str) -> List[str]:
        """Primary model first, then the fallback chain, without duplicates."""
        return list(dict.fromkeys([m for m in [primary] + self.fallback_models if m]))

    def reload(self) -> "Settings":
        """Re-read .env in place, so every module holding `settings` sees new values (e.g. a swapped key)."""
        fresh = type(self)()
        for name in type(self).model_fields:
            object.__setattr__(self, name, getattr(fresh, name))
        return self

    @property
    def postgres_dsn(self) -> str:
        """Return PostgreSQL connection string for psycopg v3."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
