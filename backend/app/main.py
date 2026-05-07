from fastapi import FastAPI, HTTPException, status

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
from app.graph_service import run_agent_workflow
from app.memory_service import append_turn, get_chat_history_text
from app.rag_service import KnowledgeBaseNotFoundError, generate_rag_response
from app.schemas import AgentChatResponse, ChatRequest, ChatResponse, RagChatResponse, ToolChatResponse
from app.tool_service import generate_tool_response


app = FastAPI(title=settings.app_name)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        chat_history = get_chat_history_text(request.session_id)
        answer = await generate_chat_response(request.message, chat_history=chat_history)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /chat.",
        ) from exc

    append_turn(request.session_id, request.message, answer)
    return ChatResponse(answer=answer)


@app.post("/rag-chat", response_model=RagChatResponse)
async def rag_chat(request: ChatRequest) -> RagChatResponse:
    try:
        chat_history = get_chat_history_text(request.session_id)
        answer, sources, retrieved_contexts = await generate_rag_response(
            request.message,
            chat_history=chat_history,
        )
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

    append_turn(request.session_id, request.message, answer)
    return RagChatResponse(answer=answer, sources=sources, retrieved_contexts=retrieved_contexts)


@app.post("/tool-chat", response_model=ToolChatResponse)
async def tool_chat(request: ChatRequest) -> ToolChatResponse:
    try:
        chat_history = get_chat_history_text(request.session_id)
        answer, used_tools = await generate_tool_response(request.message, chat_history=chat_history)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /tool-chat.",
        ) from exc

    append_turn(request.session_id, request.message, answer)
    return ToolChatResponse(answer=answer, used_tools=used_tools)


@app.post("/agent-chat", response_model=AgentChatResponse)
async def agent_chat(request: ChatRequest) -> AgentChatResponse:
    try:
        chat_history = get_chat_history_text(request.session_id)
        result = await run_agent_workflow(request.message, chat_history=chat_history)
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

    append_turn(request.session_id, request.message, result["answer"])
    return AgentChatResponse(
        answer=result["answer"],
        route=result["route"],
        router_reason=result["router_reason"],
        sources=result["sources"],
        retrieved_contexts=result["retrieved_contexts"],
        used_tools=result["used_tools"],
    )
