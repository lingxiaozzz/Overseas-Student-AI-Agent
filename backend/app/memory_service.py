from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from app.config import settings


MemoryLayer = Literal["working", "long_term", "experience"]
MemoryOperation = Literal["read", "write"]
MemoryStatus = Literal[
    "hit", "miss", "wrote", "updated", "skipped", "deduped", "superseded", "expired"
]

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
_DURABLE_FACT_HINTS = (
    "i am",
    "i'm",
    "i will study",
    "i study",
    "i live",
    "i moved to",
    "my rent",
    "my budget",
    "international student",
    "usyd",
    "university of sydney",
    "weekly rent",
    "oshc",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _long_term_fact_attributes(fact: str) -> tuple[str, str]:
    """Extract a stable profile key where a new value should replace an old one."""
    compact = " ".join(fact.split()).strip()
    lower = compact.lower()

    city_match = re.search(r"\b(?:i live in|i moved to)\s+([^.,;]+)", compact, flags=re.IGNORECASE)
    if city_match:
        return "city", city_match.group(1).strip()

    rent_match = re.search(
        r"\b(?:my\s+)?rent\s+(?:is|=|around)\s*(\d+(?:\.\d+)?)\s*(aud|a\$|\$)?",
        lower,
    )
    if rent_match:
        return "rent_per_week_aud", f"{rent_match.group(1)} AUD/week"

    budget_match = re.search(
        r"\b(?:my\s+)?budget\s+(?:is|=|around)\s*(\d+(?:\.\d+)?)\s*(aud|a\$|\$)?",
        lower,
    )
    if budget_match:
        return "budget_per_week_aud", f"{budget_match.group(1)} AUD/week"

    if "university of sydney" in lower or "usyd" in lower:
        return "university", "USYD"

    # Unknown facts remain independently addressable rather than being incorrectly merged.
    return f"fact:{' '.join(sorted(_tokenize(compact)))}", compact


def _record_is_expired(record: dict[str, Any]) -> bool:
    ttl_days = settings.long_term_memory_ttl_days
    if ttl_days <= 0:
        return False
    updated_at = _parse_utc(record.get("updated_at") or record.get("created_at"))
    if updated_at is None:
        return False
    return (datetime.now(timezone.utc) - updated_at).days >= ttl_days


def _record_is_active(record: dict[str, Any]) -> bool:
    return record.get("status", "active") == "active" and not _record_is_expired(record)


def _experience_path() -> Path:
    return settings.experience_memory_path


def _long_term_path() -> Path:
    return settings.long_term_memory_path


def _preview(text: str, limit: int = 72) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 2}


def make_memory_event(
    *,
    layer: MemoryLayer,
    operation: MemoryOperation,
    status: MemoryStatus,
    detail: str,
    count: int = 0,
    items: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "operation": operation,
        "status": status,
        "detail": detail,
        "count": count,
        "items": list(items or []),
    }


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


def should_persist_long_term(session_id: str, persist_experience: bool = True) -> bool:
    if not persist_experience:
        return False
    if not settings.long_term_memory_enabled:
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


def _load_long_term_unlocked() -> dict[str, list[dict[str, Any]]]:
    path = _long_term_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for session_id, records in payload.items():
        if not isinstance(session_id, str) or not isinstance(records, list):
            continue
        cleaned[session_id] = [item for item in records if isinstance(item, dict)]
    return cleaned


def _save_long_term_unlocked(store: dict[str, list[dict[str, Any]]]) -> None:
    path = _long_term_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def get_working_turn_count(session_id: str) -> int:
    with _MEMORY_LOCK:
        return len(_SESSION_MEMORY[session_id])


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


def read_working_memory(session_id: str) -> tuple[str, dict[str, Any]]:
    turns = get_working_turn_count(session_id)
    text = get_chat_history_text(session_id)
    if turns <= 0:
        event = make_memory_event(
            layer="working",
            operation="read",
            status="miss",
            detail=f"No working-memory turns for session '{session_id}'.",
            count=0,
        )
        return text, event

    items = []
    with _MEMORY_LOCK:
        recent = list(_SESSION_MEMORY[session_id])[-2:]
    for user_message, assistant_message in recent:
        items.append(f"U:{_preview(user_message, 40)} | A:{_preview(assistant_message, 40)}")

    event = make_memory_event(
        layer="working",
        operation="read",
        status="hit",
        detail=f"Loaded {turns} working-memory turn(s) for session '{session_id}'.",
        count=turns,
        items=items,
    )
    return text, event


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    with _MEMORY_LOCK:
        _SESSION_MEMORY[session_id].append((user_message, assistant_message))


def write_working_memory(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> dict[str, Any]:
    append_turn(session_id, user_message, assistant_message)
    turns = get_working_turn_count(session_id)
    return make_memory_event(
        layer="working",
        operation="write",
        status="wrote",
        detail=f"Appended 1 turn to working memory (size={turns}/{settings.memory_max_turns}).",
        count=1,
        items=[f"U:{_preview(user_message, 48)} | A:{_preview(assistant_message, 48)}"],
    )


def extract_long_term_candidates(message: str) -> list[str]:
    """Heuristic durable-fact extractor for student profile / constraints."""
    text = " ".join(message.split()).strip()
    if len(text) < 8:
        return []

    clauses = [part.strip(" .;") for part in re.split(r"[.\n;]+", text) if part.strip()]
    candidates: list[str] = []
    for clause in clauses:
        lower = clause.lower()
        if len(clause) < 8 or len(clause) > 220:
            continue
        if lower.startswith(("what ", "how ", "can you", "could you", "please ")):
            continue
        if any(hint in lower for hint in _DURABLE_FACT_HINTS):
            candidates.append(clause)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[: settings.long_term_memory_max_facts_per_write]


def format_long_term_context(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No long-term profile facts."
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        fact = str(record.get("fact", "")).strip()
        if fact:
            key = str(record.get("key", "fact")).strip()
            confidence = float(record.get("confidence", 1.0))
            lines.append(f"{index}. {fact} (key={key}; confidence={confidence:.2f})")
    return "\n".join(lines) if lines else "No long-term profile facts."


def retrieve_long_term(
    session_id: str,
    query: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    limit = top_k or settings.long_term_memory_top_k
    query_tokens = _tokenize(query)

    with _MEMORY_LOCK:
        store = _load_long_term_unlocked()
        records = [record for record in store.get(session_id, []) if _record_is_active(record)]

    if not records:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        fact = str(record.get("fact", ""))
        tokens = _tokenize(fact)
        if not tokens:
            continue
        if query_tokens:
            overlap = len(query_tokens & tokens)
            score = overlap / len(query_tokens) if overlap else 0.0
        else:
            overlap = 0
            score = 0.0
        # Keep recent same-session facts discoverable even with weak overlap.
        score += 0.05
        if score < settings.long_term_memory_min_score and overlap <= 0:
            continue
        scored.append((score, record))

    if not scored:
        # Fallback: most recent facts for this session.
        return list(reversed(records))[:limit]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:limit]]


def read_long_term_memory(
    session_id: str,
    query: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    records = retrieve_long_term(session_id, query)
    context = format_long_term_context(records)
    facts = [str(item.get("fact", "")).strip() for item in records if str(item.get("fact", "")).strip()]
    if not records:
        event = make_memory_event(
            layer="long_term",
            operation="read",
            status="miss",
            detail=f"No long-term facts matched for session '{session_id}'.",
            count=0,
        )
    else:
        event = make_memory_event(
            layer="long_term",
            operation="read",
            status="hit",
            detail=f"Retrieved {len(records)} long-term fact(s) for session '{session_id}'.",
            count=len(records),
            items=[_preview(fact, 80) for fact in facts],
        )
    return records, context, event


def upsert_long_term_facts(
    session_id: str,
    facts: list[str],
    *,
    persist_experience: bool = True,
    source: str = "user_message",
) -> dict[str, Any]:
    cleaned = [fact.strip() for fact in facts if fact and fact.strip()]
    if not cleaned:
        return make_memory_event(
            layer="long_term",
            operation="write",
            status="skipped",
            detail="No durable facts extracted for long-term memory.",
            count=0,
        )
    if not should_persist_long_term(session_id, persist_experience=persist_experience):
        return make_memory_event(
            layer="long_term",
            operation="write",
            status="skipped",
            detail="Long-term write skipped (disabled or eval/demo session).",
            count=0,
            items=[_preview(item, 60) for item in cleaned[:3]],
        )

    wrote = 0
    updated = 0
    superseded = 0
    expired = 0
    with _MEMORY_LOCK:
        store = _load_long_term_unlocked()
        records = list(store.get(session_id, []))
        now = _utc_now()
        for item in records:
            if item.get("status", "active") == "active" and _record_is_expired(item):
                item["status"] = "expired"
                item["status_changed_at"] = now
                expired += 1

        for fact in cleaned:
            memory_key, value = _long_term_fact_attributes(fact)
            active_for_key = [
                item
                for item in records
                if item.get("status", "active") == "active"
                and str(item.get("key") or _long_term_fact_attributes(str(item.get("fact", "")))[0])
                == memory_key
            ]
            exact = next(
                (item for item in active_for_key if str(item.get("fact", "")).strip().lower() == fact.lower()),
                None,
            )
            if exact is not None:
                exact.update(
                    {
                        "key": memory_key,
                        "value": value,
                        "confidence": 0.95,
                        "status": "active",
                        "updated_at": now,
                        "source": source,
                    }
                )
                updated += 1
            else:
                for item in active_for_key:
                    item["status"] = "superseded"
                    item["superseded_at"] = now
                    superseded += 1
                record = {
                    "key": memory_key,
                    "value": value,
                    "fact": fact,
                    "confidence": 0.95,
                    "status": "active",
                    "source": source,
                    "created_at": now,
                    "updated_at": now,
                }
                records.append(record)
                wrote += 1

        max_items = settings.long_term_memory_max_items
        if len(records) > max_items:
            records = records[-max_items:]
        store[session_id] = records
        _save_long_term_unlocked(store)

    status: MemoryStatus = "updated" if updated and not wrote else "wrote"
    if updated and wrote:
        status = "wrote"
    return make_memory_event(
        layer="long_term",
        operation="write",
        status=status,
        detail=(
            f"Long-term memory write: created={wrote}, updated={updated}, "
            f"superseded={superseded}, expired={expired}."
        ),
        count=wrote + updated,
        items=[_preview(item, 70) for item in cleaned],
    )


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


def read_experience_memory(
    query: str,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    records = retrieve_experiences(query, session_id=session_id)
    context = format_experience_context(records)
    lessons = [
        str(item.get("lesson", "")).strip()
        for item in records
        if str(item.get("lesson", "")).strip()
    ]
    if not records:
        event = make_memory_event(
            layer="experience",
            operation="read",
            status="miss",
            detail="No experience lessons matched the current goal.",
            count=0,
        )
    else:
        event = make_memory_event(
            layer="experience",
            operation="read",
            status="hit",
            detail=f"Retrieved {len(records)} experience lesson(s).",
            count=len(records),
            items=[_preview(lesson, 80) for lesson in lessons],
        )
    return records, context, event


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
        "created_at": _utc_now(),
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
                existing.update(record)
                _save_experiences_unlocked(records)
                return existing

        records.append(record)
        max_items = settings.experience_memory_max_items
        if len(records) > max_items:
            records = records[-max_items:]
        _save_experiences_unlocked(records)
    return record


def write_experience_memory(
    session_id: str,
    goal: str,
    lesson: str,
    *,
    routes: list[str] | None = None,
    tools: list[str] | None = None,
    success: bool = True,
    steps_used: int = 0,
    persist_experience: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not should_persist_experience(session_id, persist_experience=persist_experience):
        return None, make_memory_event(
            layer="experience",
            operation="write",
            status="skipped",
            detail="Experience write skipped (disabled or eval/demo session).",
            count=0,
            items=[_preview(lesson, 70)] if lesson.strip() else [],
        )
    if is_low_value_lesson(lesson):
        return None, make_memory_event(
            layer="experience",
            operation="write",
            status="skipped",
            detail="Experience write skipped (low-value lesson).",
            count=0,
            items=[_preview(lesson, 70)] if lesson.strip() else [],
        )

    before_count = 0
    with _MEMORY_LOCK:
        before_count = len(_load_experiences_unlocked())

    record = append_experience(
        session_id=session_id,
        goal=goal,
        lesson=lesson,
        routes=routes,
        tools=tools,
        success=success,
        steps_used=steps_used,
        persist_experience=persist_experience,
    )
    if record is None:
        return None, make_memory_event(
            layer="experience",
            operation="write",
            status="skipped",
            detail="Experience write skipped.",
            count=0,
        )

    with _MEMORY_LOCK:
        after_count = len(_load_experiences_unlocked())
    status: MemoryStatus = "wrote" if after_count > before_count else "deduped"
    return record, make_memory_event(
        layer="experience",
        operation="write",
        status=status,
        detail=(
            "Experience lesson appended."
            if status == "wrote"
            else "Experience lesson deduped/updated existing fingerprint."
        ),
        count=1,
        items=[_preview(str(record.get("lesson", "")), 80)],
    )
