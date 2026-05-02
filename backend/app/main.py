from fastapi import FastAPI, HTTPException, status

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
from app.graph_service import run_agent_workflow
from app.rag_service import KnowledgeBaseNotFoundError, generate_rag_response
from app.schemas import AgentChatResponse, ChatRequest, ChatResponse, RagChatResponse


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


@app.post("/rag-chat", response_model=RagChatResponse)
async def rag_chat(request: ChatRequest) -> RagChatResponse:
    try:
        answer, sources, retrieved_contexts = await generate_rag_response(request.message)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /rag-chat.",
        ) from exc
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return RagChatResponse(answer=answer, sources=sources, retrieved_contexts=retrieved_contexts)


@app.post("/agent-chat", response_model=AgentChatResponse)
async def agent_chat(request: ChatRequest) -> AgentChatResponse:
    try:
        result = await run_agent_workflow(request.message)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /agent-chat.",
        ) from exc
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return AgentChatResponse(
        answer=result["answer"],
        route=result["route"],
        sources=result["sources"],
        retrieved_contexts=result["retrieved_contexts"],
    )
