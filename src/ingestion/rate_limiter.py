"""Sliding-window RPM/TPM governor, 429 classification and daily-quota model fallback (free-tier safe).

One process-wide `RateLimitGovernor` is shared by the extractor, embedder, router and generator.

- Pacing: per model, a 60 s sliding window of (timestamp, tokens). Before a live request the governor
  sleeps exactly until 1 more request and `estimated_tokens` fit under the model's ceiling. Ceilings
  sit below the free-tier limits (flash-lite 13 RPM / 200K TPM vs 15 / 250K; gemma-4-31b-it 25 / 13K
  vs 30 / 16K; gemini-embedding-001 80 / 24K vs 100 / 30K). Cache hits never touch the governor.
- 429 / RESOURCE_EXHAUSTED: a per-minute spike sleeps (server retryDelay if given) and retries the
  same model. A daily (per-day) quota error, or a minute error that persists `max_minute_retries`
  times, marks the model exhausted for this process and steps to the next model in the chain.
- When every model in the chain is exhausted, `AllModelsExhaustedError` is raised. Callers commit
  what they have and exit (ingestion exits with code 4 and a resume banner).

Daily RPD is NOT counted locally: the per-key daily counter lives on Google's side and survives
restarts, so the server's 429 is the only reliable signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ModelLimits:
    max_rpm: int
    max_tpm: int


TEXT_LIMITS = ModelLimits(max_rpm=13, max_tpm=200_000)
DEFAULT_LIMITS: Dict[str, ModelLimits] = {
    "gemini-3.5-flash-lite": TEXT_LIMITS,
    "gemini-3.1-flash-lite": TEXT_LIMITS,
    "gemma-4-31b-it": ModelLimits(max_rpm=25, max_tpm=13_000),
    "gemini-embedding-001": ModelLimits(max_rpm=80, max_tpm=24_000),
}


class QuotaKind(str, Enum):
    MINUTE = "minute"
    DAILY = "daily"


class AllModelsExhaustedError(RuntimeError):
    """Every model in the chain hit its daily quota (or kept failing with 429) on the current key."""

    def __init__(self, models: Sequence[str], last_error: Optional[BaseException] = None) -> None:
        self.models = list(models)
        self.last_error = last_error
        super().__init__(f"Daily API quota exhausted for all models: {', '.join(self.models)}"
                         + (f" (last error: {last_error})" if last_error else ""))


_RETRY_RE = re.compile(r"retry[_ ]?delay\W+(\d+(?:\.\d+)?)s|retry in (\d+(?:\.\d+)?)\s*s", re.I)


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("body", "details", "response_json", "message", "status"):
        v = getattr(exc, attr, None)
        if v:
            parts.append(v if isinstance(v, str) else json.dumps(v, default=str))
    return " ".join(parts)


def classify_quota_error(exc: BaseException) -> Optional[Tuple[QuotaKind, Optional[float]]]:
    """(kind, retry_after_seconds) for a 429 / RESOURCE_EXHAUSTED error, else None.

    Works for google-genai `APIError` (`.code`, `.details`) and Interactions `GenAiError`
    (`.status_code`, JSON `.body`) by inspecting the status code and the quota ids in the error text,
    e.g. `GenerateRequestsPerDayPerProjectPerModel-FreeTier` vs `...PerMinute...`.
    """
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = _error_text(exc)
    if code != 429 and "RESOURCE_EXHAUSTED" not in text and not re.search(r"\b429\b", text):
        return None
    m = _RETRY_RE.search(text)
    retry = float(m.group(1) or m.group(2)) if m else None
    compact = re.sub(r"[\s_\-]", "", text.lower())
    kind = QuotaKind.DAILY if ("perday" in compact or "daily" in compact) else QuotaKind.MINUTE
    return kind, retry


class RateLimitGovernor:
    def __init__(
        self,
        limits: Optional[Dict[str, ModelLimits]] = None,
        default_limits: ModelLimits = TEXT_LIMITS,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_minute_retries: int = 3,
        minute_backoff_s: float = 20.0,
    ) -> None:
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        self.default_limits = default_limits
        self.window_s = window_s
        self.clock, self.sleep = clock, sleep
        self.max_minute_retries = max_minute_retries
        self.minute_backoff_s = minute_backoff_s
        self._windows: Dict[str, Deque[List[float]]] = {}
        self._lock = threading.Lock()
        self.exhausted: Set[str] = set()
        self.total_wait_s = 0.0
        self.paced_requests = 0

    def limits_for(self, model: str) -> ModelLimits:
        return self.limits.get(model, self.default_limits)

    def _prune(self, win: Deque[List[float]], now: float) -> None:
        while win and now - win[0][0] >= self.window_s:
            win.popleft()

    def wait_needed(self, model: str, estimated_tokens: int) -> float:
        """Seconds until one more request with `estimated_tokens` fits in the window (0 if it fits now)."""
        lim = self.limits_for(model)
        now = self.clock()
        win = self._windows.setdefault(model, deque())
        self._prune(win, now)
        est = min(int(estimated_tokens), lim.max_tpm)  # a single oversized request is allowed alone
        wait = 0.0
        if len(win) + 1 > lim.max_rpm:
            wait = max(wait, win[len(win) - lim.max_rpm][0] + self.window_s - now)
        used = sum(e[1] for e in win)
        if used + est > lim.max_tpm:
            freed = 0.0
            for t, tok in win:  # oldest first: how long until enough tokens leave the window
                freed += tok
                if used - freed + est <= lim.max_tpm:
                    wait = max(wait, t + self.window_s - now)
                    break
        return max(0.0, wait)

    def acquire(self, model: str, estimated_tokens: int = 0) -> float:
        """Block until the request fits, then reserve it. Returns seconds slept."""
        slept = 0.0
        with self._lock:
            while True:
                wait = self.wait_needed(model, estimated_tokens)
                if wait <= 0:
                    break
                logger.info("Rate pacing %s: sleeping %.1fs to stay under %s", model, wait, self.limits_for(model))
                self.sleep(wait)
                slept += wait
            self._windows[model].append([self.clock(), float(min(estimated_tokens, self.limits_for(model).max_tpm))])
        if slept:
            self.total_wait_s += slept
            self.paced_requests += 1
        return slept

    def record_actual(self, model: str, actual_tokens: int) -> None:
        """Replace the last reservation's estimate with the server-reported token count."""
        with self._lock:
            win = self._windows.get(model)
            if win and actual_tokens:
                win[-1][1] = float(actual_tokens)

    def window_usage(self, model: str) -> Tuple[int, int]:
        with self._lock:
            win = self._windows.setdefault(model, deque())
            self._prune(win, self.clock())
            return len(win), int(sum(e[1] for e in win))

    def call_with_fallback(
        self,
        models: Iterable[str],
        fn: Callable[[str], T],
        estimated_tokens: int = 0,
        actual_tokens: Optional[Callable[[T], int]] = None,
    ) -> Tuple[T, str]:
        """Run `fn(model)` on the first model of the chain that has quota. Returns (result, model_used)."""
        chain = [m for m in dict.fromkeys(models) if m]
        last: Optional[BaseException] = None
        for model in chain:
            if model in self.exhausted:
                continue
            minute_failures = 0
            while True:
                self.acquire(model, estimated_tokens)
                try:
                    result = fn(model)
                except Exception as exc:
                    q = classify_quota_error(exc)
                    if q is None:
                        raise
                    last = exc
                    kind, retry = q
                    if kind == QuotaKind.MINUTE and minute_failures < self.max_minute_retries:
                        minute_failures += 1
                        delay = retry if retry is not None else self.minute_backoff_s * minute_failures
                        logger.warning("429 per-minute limit on %s; retrying in %.0fs (%d/%d)", model, delay,
                                       minute_failures, self.max_minute_retries)
                        self.sleep(delay)
                        self.total_wait_s += delay
                        continue
                    reason = "daily quota" if kind == QuotaKind.DAILY else f"{minute_failures} consecutive 429s"
                    logger.warning("%s exhausted (%s); stepping to the next fallback model", model, reason)
                    self.exhausted.add(model)
                    break
                if actual_tokens is not None:
                    try:
                        self.record_actual(model, int(actual_tokens(result) or 0))
                    except Exception:  # usage metadata is optional
                        pass
                return result, model
        raise AllModelsExhaustedError(chain, last)


_GOVERNOR: Optional[RateLimitGovernor] = None
_GOV_LOCK = threading.Lock()


def get_governor() -> RateLimitGovernor:
    global _GOVERNOR
    with _GOV_LOCK:
        if _GOVERNOR is None:
            _GOVERNOR = RateLimitGovernor()
        return _GOVERNOR


def set_governor(governor: Optional[RateLimitGovernor]) -> None:
    """Install a governor (tests) or reset to a fresh default on next use (None)."""
    global _GOVERNOR
    with _GOV_LOCK:
        _GOVERNOR = governor
