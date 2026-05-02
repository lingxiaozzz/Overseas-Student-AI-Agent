from collections import defaultdict, deque
from threading import Lock

from app.config import settings


_MEMORY_LOCK = Lock()
_SESSION_MEMORY: dict[str, deque[tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=settings.memory_max_turns)
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


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    with _MEMORY_LOCK:
        _SESSION_MEMORY[session_id].append((user_message, assistant_message))
