"""Summarise privacy-conscious Agent run telemetry and binary user feedback."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "observability"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(ceil(len(ordered) * percentile) - 1, 0)]


def build_report(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    runs = _read_jsonl(data_dir / "agent_runs.jsonl")
    feedback = _read_jsonl(data_dir / "feedback.jsonl")
    successful_runs = [run for run in runs if run.get("outcome") == "success"]
    failures = [run for run in runs if run.get("outcome") == "error"]
    latencies = [int(run.get("elapsed_ms", 0) or 0) for run in runs]
    cache = [run.get("cache", {}) for run in successful_runs if isinstance(run.get("cache"), dict)]
    prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in cache)
    cache_hit_tokens = sum(int(item.get("cache_hit_tokens", 0) or 0) for item in cache)
    successful_event_ids = {str(run.get("event_id")) for run in successful_runs if run.get("event_id")}
    linked_feedback = [
        item
        for item in feedback
        if str(item.get("event_id")) in successful_event_ids
        and item.get("rating") in {"helpful", "not_helpful"}
    ]
    ratings = [item["rating"] for item in linked_feedback]
    feedback_event_ids = {str(item["event_id"]) for item in linked_feedback}

    return {
        "data_directory": str(data_dir),
        "runs": {
            "total": len(runs),
            "successful": len(successful_runs),
            "failed": len(failures),
            "success_rate": round(len(successful_runs) / len(runs), 4) if runs else 0.0,
        },
        "routing": dict(sorted(Counter(str(run.get("route", "unknown")) for run in successful_runs).items())),
        "latency_ms": {
            "average": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "agent": {
            "tool_use_rate": round(
                sum(int(run.get("tool_calls", 0) or 0) > 0 for run in successful_runs)
                / len(successful_runs),
                4,
            ) if successful_runs else 0.0,
            "replan_rate": round(
                sum(bool(run.get("replanned", False)) for run in successful_runs) / len(successful_runs),
                4,
            ) if successful_runs else 0.0,
            "evaluation_pass_rate": round(
                sum(bool((run.get("evaluation") or {}).get("passed", False)) for run in successful_runs)
                / len(successful_runs),
                4,
            ) if successful_runs else 0.0,
        },
        "cache": {
            "prompt_tokens": prompt_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "hit_rate": round(cache_hit_tokens / prompt_tokens, 4) if prompt_tokens else 0.0,
        },
        "feedback": {
            "total": len(ratings),
            "coverage": round(len(feedback_event_ids) / len(successful_runs), 4) if successful_runs else 0.0,
            "helpful_rate": round(ratings.count("helpful") / len(ratings), 4) if ratings else 0.0,
        },
        "error_types": dict(sorted(Counter(str(run.get("error_type", "unknown")) for run in failures).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise Agent telemetry and feedback JSONL files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = build_report(args.data_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
