"""Per-query API usage telemetry (tokens and requests) collected from every Gemini call.

Gemini wrappers call `record_usage(...)`; `RagService` opens a `usage_scope()` around each query and
turns the collected events into a `QueryUsageTelemetry`. The scope is a ContextVar, so concurrent
API requests in different threads never mix their counts. Outside a scope, recording is a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from pydantic import BaseModel, Field


@dataclass
class UsageEvent:
    component: str  # router | generator | embedder | extractor
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0  # live API requests (0 for cache hits)
    cache_hits: int = 0
    estimated: bool = False  # True when the API returned no token counts (embeddings)
    waited_s: float = 0.0


@dataclass
class UsageMeter:
    events: List[UsageEvent] = field(default_factory=list)

    def add(self, event: UsageEvent) -> None:
        self.events.append(event)

    def by(self, component: str) -> List[UsageEvent]:
        return [e for e in self.events if e.component == component]


class QueryUsageTelemetry(BaseModel):
    router_model: str = ""
    router_requests: int = 0
    router_input_tokens: int = 0
    router_output_tokens: int = 0
    reranker_model: str = ""
    reranker_requests: int = 0
    reranker_input_tokens: int = 0
    reranker_output_tokens: int = 0
    generator_model: str = ""
    generator_requests: int = 0
    generator_input_tokens: int = 0
    generator_output_tokens: int = 0
    embedding_model: str = ""
    embedding_requests_used: int = 0
    embedding_cache_hits: int = 0
    embedding_tokens_est: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    llm_requests_used: int = 0
    total_api_requests_used: int = 0
    rate_limit_wait_s: float = 0.0
    models_used: List[str] = Field(default_factory=list)
    note: str = ""


_CURRENT: ContextVar[Optional[UsageMeter]] = ContextVar("usage_meter", default=None)


@contextmanager
def usage_scope() -> Iterator[UsageMeter]:
    meter = UsageMeter()
    token = _CURRENT.set(meter)
    try:
        yield meter
    finally:
        _CURRENT.reset(token)


def record_usage(component: str, model: str, input_tokens: int = 0, output_tokens: int = 0, requests: int = 0,
                 cache_hits: int = 0, estimated: bool = False, waited_s: float = 0.0) -> None:
    meter = _CURRENT.get()
    if meter is not None:
        meter.add(UsageEvent(component, model, int(input_tokens or 0), int(output_tokens or 0), requests, cache_hits,
                             estimated, waited_s))


def summarize_usage(meter: UsageMeter, router_name: str = "", generator_name: str = "", embedding_name: str = "",
                    note: str = "") -> QueryUsageTelemetry:
    def last_model(comp: str, default: str) -> str:
        live = [e.model for e in meter.by(comp) if e.requests]
        return live[-1] if live else default

    r, rr, g, e = meter.by("router"), meter.by("reranker"), meter.by("generator"), meter.by("embedder")
    t = QueryUsageTelemetry(
        router_model=last_model("router", router_name),
        router_requests=sum(x.requests for x in r),
        router_input_tokens=sum(x.input_tokens for x in r),
        router_output_tokens=sum(x.output_tokens for x in r),
        reranker_model=last_model("reranker", ""),
        reranker_requests=sum(x.requests for x in rr),
        reranker_input_tokens=sum(x.input_tokens for x in rr),
        reranker_output_tokens=sum(x.output_tokens for x in rr),
        generator_model=last_model("generator", generator_name),
        generator_requests=sum(x.requests for x in g),
        generator_input_tokens=sum(x.input_tokens for x in g),
        generator_output_tokens=sum(x.output_tokens for x in g),
        embedding_model=last_model("embedder", embedding_name),
        embedding_requests_used=sum(x.requests for x in e),
        embedding_cache_hits=sum(x.cache_hits for x in e),
        embedding_tokens_est=sum(x.input_tokens for x in e),
        rate_limit_wait_s=round(sum(x.waited_s for x in meter.events), 3),
        models_used=list(dict.fromkeys(x.model for x in meter.events if x.requests)),
        note=note,
    )
    t.total_input_tokens = t.router_input_tokens + t.reranker_input_tokens + t.generator_input_tokens + t.embedding_tokens_est
    t.total_output_tokens = t.router_output_tokens + t.reranker_output_tokens + t.generator_output_tokens
    t.total_tokens = t.total_input_tokens + t.total_output_tokens
    t.llm_requests_used = t.router_requests + t.reranker_requests + t.generator_requests
    t.total_api_requests_used = t.llm_requests_used + t.embedding_requests_used
    return t


def interaction_usage(interaction: object) -> tuple:
    """(input, output, total) tokens from an Interactions response; zeros if absent."""
    u = getattr(interaction, "usage", None)
    if u is None:
        return 0, 0, 0
    inp = getattr(u, "total_input_tokens", None) or getattr(u, "input_tokens", None) or 0
    out = getattr(u, "total_output_tokens", None) or getattr(u, "output_tokens", None) or 0
    tot = getattr(u, "total_tokens", None) or (inp + out)
    return int(inp), int(out), int(tot)
