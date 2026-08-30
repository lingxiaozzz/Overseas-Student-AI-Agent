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


def preferred_response_language(message: str) -> str:
    """Choose the response language from the user's original message."""
    return "zh-CN" if any("\u4e00" <= character <= "\u9fff" for character in message) else "en"


def response_language_instruction(language: str | None) -> str:
    """Return a compact instruction that keeps generated replies language-consistent."""
    if language == "zh-CN":
        return "Respond in Simplified Chinese. Keep source citations and source filenames unchanged."
    return "Respond in English. Keep source citations and source filenames unchanged."
