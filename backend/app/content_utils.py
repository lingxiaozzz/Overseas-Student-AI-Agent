from typing import Any


def content_to_text(content: Any) -> str:
    """Normalize LangChain/Gemini message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return content_to_text(content["text"])
        if "content" in content:
            return content_to_text(content["content"])
    return str(content)
