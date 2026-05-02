from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Give practical, accurate, and friendly advice.
When the user asks for preparation steps, answer with a clear checklist.
If the question needs official confirmation, remind the user to check official university or government sources.
Use prior conversation history when it helps keep responses consistent and contextual."""


class MissingApiKeyError(RuntimeError):
    pass


async def generate_chat_response(message: str, chat_history: str = "") -> str:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Conversation history:\n{chat_history}\n\nCurrent user message:\n{message}"),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    chain = prompt | model
    response = await chain.ainvoke({"message": message, "chat_history": chat_history})

    if isinstance(response.content, str):
        return response.content

    return str(response.content)
