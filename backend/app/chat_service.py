from langchain_core.prompts import ChatPromptTemplate

from app.content_utils import content_to_text
from app.llm_service import MissingApiKeyError, create_chat_model
from app.retry_service import with_retry


SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Give practical, accurate, and friendly advice.
When the user asks for preparation steps, answer with a clear checklist.
If the question needs official confirmation, remind the user to check official university or government sources.
Use prior conversation history when it helps keep responses consistent and contextual."""


async def generate_chat_response(message: str, chat_history: str = "") -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Conversation history:\n{chat_history}\n\nCurrent user message:\n{message}"),
        ]
    )
    model = create_chat_model(temperature=0.2)
    chain = prompt | model
    response = await with_retry(lambda: chain.ainvoke({"message": message, "chat_history": chat_history}))
    return content_to_text(response.content)


__all__ = ["MissingApiKeyError", "generate_chat_response"]
