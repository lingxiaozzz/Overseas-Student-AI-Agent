from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.config import settings

T = TypeVar("T")

def _is_retryable_error(exc: BaseException) -> bool:
    # We rely on message matching because upstream exceptions vary by package/version.
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "resource_exhausted",  # 429 quota exceeded
            "quota exceeded",
            "rate limit",
            "429",
            "unavailable",  # 503 high demand
            "503",
            "temporarily",
            "try again later",
        ]
    )


def _before_sleep(retry_state: RetryCallState) -> None:
    # Intentionally no logging here; keep core library silent.
    # Callers can add logging if needed.
    return None


async def with_retry(func: Callable[[], Awaitable[T]]) -> T:
    """Run an async callable with exponential backoff retries for transient LLM errors."""
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential_jitter(
            initial=settings.retry_initial_seconds,
            max=settings.retry_max_seconds,
        ),
        before_sleep=_before_sleep,
        reraise=True,
    ):
        with attempt:
            return await func()

    # Unreachable because reraise=True, but keeps type checkers happy.
    raise RuntimeError("Retry loop exited unexpectedly.")
