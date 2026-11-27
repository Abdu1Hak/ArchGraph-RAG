"""Shared test fixtures."""

import pytest

from src.ingestion.rate_limiter import RateLimitGovernor, set_governor


class _VirtualClock:
    """Monotonic clock that only moves when the governor 'sleeps', so pacing is instant but exact."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


def make_test_governor() -> RateLimitGovernor:
    clock = _VirtualClock()
    return RateLimitGovernor(clock=clock, sleep=clock.sleep)


@pytest.fixture(autouse=True)
def _virtual_time_governor():
    """Every test gets a fresh governor on virtual time: fake API calls can never cause real 60 s
    pacing waits, and exhausted-model state never leaks between tests."""
    set_governor(make_test_governor())
    yield
    set_governor(None)
