"""Small in-memory sliding-window limiter for the public demo endpoints."""

from __future__ import annotations

from collections import defaultdict, deque
from math import ceil
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, limits: tuple[tuple[int, int], ...], now: float | None = None) -> int | None:
        """Record a request or return the number of seconds until it may be retried."""
        current_time = monotonic() if now is None else now
        longest_window = max((window for _limit, window in limits), default=0)
        with self._lock:
            events = self._requests[key]
            while events and events[0] <= current_time - longest_window:
                events.popleft()

            retry_after = 0
            for limit, window in limits:
                if limit <= 0:
                    continue
                recent = [event for event in events if event > current_time - window]
                if len(recent) >= limit:
                    retry_after = max(retry_after, ceil(recent[0] + window - current_time))
            if retry_after:
                return max(retry_after, 1)

            events.append(current_time)
            return None

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()


rate_limiter = SlidingWindowRateLimiter()


def _client_identifier(request: Request) -> str:
    """Use the proxy-appended client address when deployed behind Render-like proxies."""
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.rsplit(",", maxsplit=1)[-1].strip()
    return request.client.host if request.client else "unknown"


def enforce_request_rate_limit(request: Request, scope: str) -> None:
    if not settings.rate_limit_enabled:
        return

    if scope == "feedback":
        limits = ((settings.feedback_rate_limit_per_minute, 60),)
    else:
        limits = (
            (settings.api_rate_limit_per_minute, 60),
            (settings.api_rate_limit_per_hour, 3600),
        )
    client = _client_identifier(request)
    retry_after = rate_limiter.check(f"{scope}:{client}", limits)
    if retry_after is None:
        return

    logger.warning("event=rate_limited scope=%s client=%s retry_after=%s", scope, client, retry_after)
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="请求过于频繁，请稍后再试。",
        headers={"Retry-After": str(retry_after)},
    )
