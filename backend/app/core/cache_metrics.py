"""DeepSeek context-cache telemetry without storing prompts or responses."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.core.logging import get_logger

logger = get_logger(__name__)


class _CacheMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests = 0
        self._prompt_tokens = 0
        self._cache_hit_tokens = 0
        self._cache_miss_tokens = 0
        self._output_tokens = 0

    def record(self, usage: Mapping[str, Any]) -> None:
        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        input_details = usage.get("input_tokens_details") or {}
        if not isinstance(input_details, Mapping):
            input_details = {}
        cache_hit_tokens = int(
            usage.get("prompt_cache_hit_tokens", input_details.get("cached_tokens", 0)) or 0
        )
        cache_miss_tokens = int(
            usage.get("prompt_cache_miss_tokens", max(prompt_tokens - cache_hit_tokens, 0)) or 0
        )

        with self._lock:
            self._requests += 1
            self._prompt_tokens += prompt_tokens
            self._cache_hit_tokens += cache_hit_tokens
            self._cache_miss_tokens += cache_miss_tokens
            self._output_tokens += output_tokens

        hit_rate = cache_hit_tokens / prompt_tokens if prompt_tokens else 0.0
        logger.info(
            "event=deepseek_usage prompt_tokens=%s cache_hit_tokens=%s "
            "cache_miss_tokens=%s output_tokens=%s cache_hit_rate=%.4f",
            prompt_tokens,
            cache_hit_tokens,
            cache_miss_tokens,
            output_tokens,
            hit_rate,
        )

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            hit_rate = self._cache_hit_tokens / self._prompt_tokens if self._prompt_tokens else 0.0
            return {
                "requests": self._requests,
                "prompt_tokens": self._prompt_tokens,
                "cache_hit_tokens": self._cache_hit_tokens,
                "cache_miss_tokens": self._cache_miss_tokens,
                "output_tokens": self._output_tokens,
                "cache_hit_rate": round(hit_rate, 4),
            }


cache_metrics = _CacheMetrics()


class DeepSeekUsageCallback(BaseCallbackHandler):
    """Collect usage emitted by DeepSeek's OpenAI-compatible chat endpoint."""

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, Mapping):
            cache_metrics.record(usage)

