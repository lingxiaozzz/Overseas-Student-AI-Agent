from pydantic import BaseModel, Field
from typing import Literal


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        examples=["I am a new international student at USYD. What should I prepare before arrival?"],
    )


class ChatResponse(BaseModel):
    answer: str


class RetrievedContext(BaseModel):
    rank: int
    source: str
    score: float
    content_preview: str


class RagChatResponse(ChatResponse):
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]


class AgentChatResponse(RagChatResponse):
    route: Literal["chat", "rag"]
