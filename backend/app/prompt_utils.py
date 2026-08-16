from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

_EMPTY_HISTORY = {
    "",
    "No prior conversation history.",
    "No prior turns.",
}


def history_to_messages(history: str) -> list[BaseMessage]:
    """Turn stored User/Assistant transcript into separate messages for prefix cache hits."""
    text = (history or "").strip()
    if text in _EMPTY_HISTORY:
        return []

    messages: list[BaseMessage] = []
    role: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal role, lines
        if role is None:
            return
        content = "\n".join(lines).strip()
        if content:
            messages.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
        role = None
        lines = []

    for line in text.splitlines():
        if line.startswith("User: "):
            flush()
            role = "human"
            lines = [line[6:]]
        elif line.startswith("Assistant: "):
            flush()
            role = "ai"
            lines = [line[11:]]
        elif role is None:
            role = "human"
            lines = [line]
        else:
            lines.append(line)
    flush()
    return messages


def cache_friendly_messages(system: str, history: str, user_content: str) -> list[BaseMessage]:
    """Stable system first, then prior turns, then the new user payload last."""
    return [
        SystemMessage(content=system),
        *history_to_messages(history),
        HumanMessage(content=user_content),
    ]
