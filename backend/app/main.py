from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
from app.graph_service import run_agent_workflow
from app.logging_service import get_logger
from app.memory_service import append_turn, get_chat_history_text
from app.rag_service import KnowledgeBaseNotFoundError, generate_rag_response
from app.schemas import AgentChatResponse, ChatRequest, ChatResponse, RagChatResponse, ToolChatResponse
from app.tool_service import generate_tool_response


app = FastAPI(title=settings.app_name)
logger = get_logger(__name__)


def _trace_id_from_request(request: Request) -> str:
    provided = request.headers.get("x-trace-id")
    if provided:
        return provided
    return f"trace-{uuid4().hex[:12]}"


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    trace_id = _trace_id_from_request(http_request)
    start = perf_counter()
    try:
        chat_history = get_chat_history_text(request.session_id)
        answer = await generate_chat_response(request.message, chat_history=chat_history)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /chat.",
        ) from exc

    append_turn(request.session_id, request.message, answer)
    elapsed_ms = int((perf_counter() - start) * 1000)
    logger.info(
        f"trace_id={trace_id} endpoint=/chat session_id={request.session_id} elapsed_ms={elapsed_ms}"
    )
    return ChatResponse(answer=answer)


@app.post("/rag-chat", response_model=RagChatResponse)
async def rag_chat(request: ChatRequest, http_request: Request) -> RagChatResponse:
    trace_id = _trace_id_from_request(http_request)
    start = perf_counter()
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
    elapsed_ms = int((perf_counter() - start) * 1000)
    logger.info(
        f"trace_id={trace_id} endpoint=/rag-chat session_id={request.session_id} "
        f"sources={len(sources)} contexts={len(retrieved_contexts)} elapsed_ms={elapsed_ms}"
    )
    return RagChatResponse(answer=answer, sources=sources, retrieved_contexts=retrieved_contexts)


@app.post("/tool-chat", response_model=ToolChatResponse)
async def tool_chat(request: ChatRequest, http_request: Request) -> ToolChatResponse:
    trace_id = _trace_id_from_request(http_request)
    start = perf_counter()
    try:
        chat_history = get_chat_history_text(request.session_id)
        answer, used_tools = await generate_tool_response(request.message, chat_history=chat_history)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set GOOGLE_API_KEY in your .env file before using /tool-chat.",
        ) from exc

    append_turn(request.session_id, request.message, answer)
    elapsed_ms = int((perf_counter() - start) * 1000)
    logger.info(
        f"trace_id={trace_id} endpoint=/tool-chat session_id={request.session_id} "
        f"used_tools={used_tools} elapsed_ms={elapsed_ms}"
    )
    return ToolChatResponse(answer=answer, used_tools=used_tools)


@app.post("/agent-chat", response_model=AgentChatResponse)
async def agent_chat(request: ChatRequest, http_request: Request) -> AgentChatResponse:
    trace_id = _trace_id_from_request(http_request)
    start = perf_counter()
    try:
        chat_history = get_chat_history_text(request.session_id)
        result = await run_agent_workflow(request.message, chat_history=chat_history, trace_id=trace_id)
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
    elapsed_ms = int((perf_counter() - start) * 1000)
    logger.info(
        f"trace_id={trace_id} endpoint=/agent-chat session_id={request.session_id} "
        f"route={result['route']} used_tools={result['used_tools']} sources={len(result['sources'])} "
        f"contexts={len(result['retrieved_contexts'])} elapsed_ms={elapsed_ms}"
    )
    return AgentChatResponse(
        answer=result["answer"],
        route=result["route"],
        router_reason=result["router_reason"],
        sources=result["sources"],
        retrieved_contexts=result["retrieved_contexts"],
        used_tools=result["used_tools"],
    )
