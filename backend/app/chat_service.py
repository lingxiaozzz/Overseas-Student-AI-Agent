from app.content_utils import content_to_text
from app.llm_service import MissingApiKeyError, create_chat_model
from app.prompt_utils import cache_friendly_messages
from app.retry_service import with_retry


SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Give practical, accurate, and friendly advice.
When the user asks for preparation steps, answer with a clear checklist.
If the question needs official confirmation, remind the user to check official university or government sources.
Use prior conversation history when it helps keep responses consistent and contextual."""


async def generate_chat_response(message: str, chat_history: str = "") -> str:
    model = create_chat_model(temperature=0.2)
    messages = cache_friendly_messages(SYSTEM_PROMPT, chat_history, message)
    response = await with_retry(lambda: model.ainvoke(messages))
    return content_to_text(response.content)


__all__ = ["MissingApiKeyError", "generate_chat_response"]
