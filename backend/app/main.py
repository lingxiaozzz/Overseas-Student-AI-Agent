from fastapi import FastAPI, HTTPException, status

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(title=settings.app_name)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await generate_chat_response(request.message)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /chat.",
        ) from exc

    return ChatResponse(answer=answer)
