from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        examples=["I am a new international student at USYD. What should I prepare before arrival?"],
    )


class ChatResponse(BaseModel):
    answer: str


class RagChatResponse(ChatResponse):
    sources: list[str]
