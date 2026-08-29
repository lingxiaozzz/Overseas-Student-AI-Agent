"""Capture one DeepSeek cache summary for each completed evaluation run."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_FIELDS = (
    "requests",
    "prompt_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "output_tokens",
)
DEFAULT_CACHE_REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_reports" / "cache"


def fetch_cache_metrics(base_url: str, *, timeout: int = 5) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}/metrics/llm-cache"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def persist_cache_run(
    *,
    suite: str,
    base_url: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    output_dir: Path = DEFAULT_CACHE_REPORTS_DIR,
) -> Path | None:
    """Write one cache-delta JSON artifact when the backend exposes metrics."""
    if before is None or after is None:
        return None

    deltas = {field: int(after.get(field, 0)) - int(before.get(field, 0)) for field in CACHE_FIELDS}
    if any(value < 0 for value in deltas.values()):
        # The backend restarted mid-run, so a delta would be misleading.
        return None
    prompt_tokens = deltas["prompt_tokens"]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "base_url": base_url,
        **deltas,
        "cache_hit_rate": round(deltas["cache_hit_tokens"] / prompt_tokens, 4)
        if prompt_tokens
        else 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    path = output_dir / f"{suite}-cache-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
