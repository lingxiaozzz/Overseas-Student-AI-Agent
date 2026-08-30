"""Privacy-conscious local telemetry for product-facing agent turns."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings

_WRITE_LOCK = Lock()


def session_fingerprint(session_id: str) -> str:
    """Create a stable correlation key without persisting the raw session identifier."""
    return sha256(session_id.encode("utf-8")).hexdigest()[:16]


def cache_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | float]:
    """Calculate request-level cache use from process-lifetime counters."""
    keys = ("prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "output_tokens")
    values = {key: max(int(after.get(key, 0)) - int(before.get(key, 0)), 0) for key in keys}
    values["cache_hit_rate"] = round(
        values["cache_hit_tokens"] / values["prompt_tokens"] if values["prompt_tokens"] else 0.0,
        4,
    )
    return values


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{serialized}\n")


def append_agent_run(payload: dict[str, Any]) -> None:
    _append_jsonl(
        settings.agent_run_log_path,
        {
            "recorded_at": datetime.now(UTC).isoformat(),
            "event_type": "agent_run",
            **payload,
        },
    )


def append_feedback(event_id: str, rating: str) -> None:
    _append_jsonl(
        settings.feedback_log_path,
        {
            "recorded_at": datetime.now(UTC).isoformat(),
            "event_type": "user_feedback",
            "event_id": event_id,
            "rating": rating,
        },
    )
