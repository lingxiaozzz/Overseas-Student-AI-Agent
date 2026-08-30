from app.core.llm import MissingApiKeyError, create_chat_model
from app.core.prompts import cache_friendly_messages
from app.core.retry import with_retry
from app.utils.content import content_to_text, preferred_response_language, response_language_instruction


SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Give practical, accurate, and friendly advice.
When the user asks for preparation steps, answer with a clear checklist.
If the question needs official confirmation, remind the user to check official university or government sources.
Use prior conversation history when it helps keep responses consistent and contextual."""


async def generate_chat_response(
    message: str,
    chat_history: str = "",
    response_language: str | None = None,
) -> str:
    model = create_chat_model(temperature=0.2)
    language = response_language or preferred_response_language(message)
    messages = cache_friendly_messages(
        f"{SYSTEM_PROMPT}\n\n{response_language_instruction(language)}",
        chat_history,
        message,
    )
    response = await with_retry(lambda: model.ainvoke(messages))
    return content_to_text(response.content)


__all__ = ["MissingApiKeyError", "generate_chat_response"]
