from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import settings


_MEMORY_LOCK = Lock()
_SESSION_MEMORY: dict[str, deque[tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=settings.memory_max_turns)
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LOW_VALUE_LESSON_MARKERS = (
    "all planned subgoals completed",
    "step budget reached",
    "plan finalized",
    "no steps executed",
    "initial hierarchical plan",
    "continuing hierarchical execution",
    "single-step context acknowledgment",
)
_EVAL_SESSION_PREFIXES = ("route-eval", "eval-", "task-eval")


def _experience_path() -> Path:
    return settings.experience_memory_path


def _load_experiences_unlocked() -> list[dict[str, Any]]:
    path = _experience_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _save_experiences_unlocked(records: list[dict[str, Any]]) -> None:
    path = _experience_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 2}


def _preview(text: str, limit: int = 72) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def is_low_value_lesson(lesson: str) -> bool:
    lesson_lower = lesson.strip().lower()
    if not lesson_lower:
        return True
    return any(marker in lesson_lower for marker in _LOW_VALUE_LESSON_MARKERS)


def should_persist_experience(session_id: str, persist_experience: bool = True) -> bool:
    if not persist_experience:
        return False
    if not settings.experience_memory_enabled:
        return False
    session_lower = session_id.strip().lower()
    return not any(session_lower.startswith(prefix) for prefix in _EVAL_SESSION_PREFIXES)


def build_experience_lesson(
    goal: str,
    routes: list[str],
    tools: list[str],
    *,
    success: bool,
    replanned: bool,
    steps_used: int,
) -> str:
    route_path = " -> ".join(routes) if routes else "chat"
    tool_part = f"; tools={','.join(tools)}" if tools else ""
    strategies: list[str] = []

    if not success:
        strategies.append("avoid repeating failed empty-step trajectories")
    if replanned:
        strategies.append("replan once when a step stalls")
    if "rag" in routes:
        strategies.append("prefer retrieval for USYD/policy/onboarding facts")
    if tools:
        strategies.append(f"use tools ({', '.join(tools)}) for numeric/checklist tasks")
    if steps_used > 1:
        strategies.append("decompose multi-intent goals into ordered subgoals")
    if not strategies:
        strategies.append("keep single-intent requests as one-step plans")

    return (
        f"For goals like '{_preview(goal)}': route via {route_path}{tool_part}; "
        f"{'; '.join(strategies)}."
    )


def _fingerprint(goal: str, routes: list[str], tools: list[str]) -> str:
    goal_tokens = sorted(_tokenize(goal))
    return "|".join(
        [
            " ".join(goal_tokens),
            ",".join(routes),
            ",".join(sorted(tools)),
        ]
    )


def get_chat_history_text(session_id: str) -> str:
    with _MEMORY_LOCK:
        turns = list(_SESSION_MEMORY[session_id])

    if not turns:
        return "No prior conversation history."

    lines: list[str] = []
    for user_message, assistant_message in turns:
        lines.append(f"User: {user_message}")
        lines.append(f"Assistant: {assistant_message}")

    return "\n".join(lines)


def get_working_turn_count(session_id: str) -> int:
    with _MEMORY_LOCK:
        return len(_SESSION_MEMORY[session_id])


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    with _MEMORY_LOCK:
        _SESSION_MEMORY[session_id].append((user_message, assistant_message))


def append_experience(
    session_id: str,
    goal: str,
    lesson: str,
    *,
    routes: list[str] | None = None,
    tools: list[str] | None = None,
    success: bool = True,
    steps_used: int = 0,
    persist_experience: bool = True,
) -> dict[str, Any] | None:
    routes = routes or []
    tools = tools or []
    lesson = lesson.strip()

    if not should_persist_experience(session_id, persist_experience=persist_experience):
        return None
    if is_low_value_lesson(lesson):
        return None

    record = {
        "session_id": session_id,
        "goal": goal,
        "lesson": lesson,
        "routes": routes,
        "tools": tools,
        "success": success,
        "steps_used": steps_used,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    fingerprint = _fingerprint(goal, routes, tools)

    with _MEMORY_LOCK:
        records = _load_experiences_unlocked()
        for existing in reversed(records):
            existing_fp = _fingerprint(
                str(existing.get("goal", "")),
                list(existing.get("routes", []) or []),
                list(existing.get("tools", []) or []),
            )
            if existing_fp == fingerprint:
                # Refresh lesson/timestamp for the duplicate instead of appending.
                existing.update(record)
                _save_experiences_unlocked(records)
                return existing

        records.append(record)
        max_items = settings.experience_memory_max_items
        if len(records) > max_items:
            records = records[-max_items:]
        _save_experiences_unlocked(records)
    return record


def retrieve_experiences(
    query: str,
    session_id: str | None = None,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    limit = top_k or settings.experience_memory_top_k
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    with _MEMORY_LOCK:
        records = _load_experiences_unlocked()

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        lesson = str(record.get("lesson", ""))
        if is_low_value_lesson(lesson):
            continue
        text = f"{record.get('goal', '')} {lesson}"
        tokens = _tokenize(text)
        if not tokens:
            continue
        overlap = len(query_tokens & tokens)
        if overlap <= 0:
            continue
        score = overlap / len(query_tokens)
        if score < settings.experience_memory_min_score:
            continue
        # Prefer same-session experiences slightly.
        if session_id and record.get("session_id") == session_id:
            score += 0.2
        if record.get("success") is False:
            score += 0.05
        if record.get("tools"):
            score += 0.05
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:limit]]


def format_experience_context(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No prior experience lessons."

    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        goal = str(record.get("goal", "")).strip()
        lesson = str(record.get("lesson", "")).strip()
        routes = ",".join(record.get("routes", []) or []) or "n/a"
        tools = ",".join(record.get("tools", []) or []) or "n/a"
        lines.append(
            f"{index}. goal={goal}; lesson={lesson}; routes={routes}; tools={tools}"
        )
    return "\n".join(lines)
