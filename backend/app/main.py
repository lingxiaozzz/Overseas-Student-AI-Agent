from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status

from app.chat_service import generate_chat_response
from app.llm_service import MissingApiKeyError, llm_override
from app.config import settings
from app.graph_service import run_agent_workflow
from app.logging_service import get_logger
from app.memory_service import append_turn, get_chat_history_text, write_working_memory
from app.rag_service import KnowledgeBaseNotFoundError, generate_rag_response
from app.schemas import (
    ActionDecisionInfo,
    AgentChatResponse,
    AgentBudget,
    AgentMetrics,
    AgentPlan,
    ChatRequest,
    ChatResponse,
    EnvironmentInfo,
    EvaluationInfo,
    MemoryEvent,
    ObservationInfo,
    PlanStepResult,
    RagChatResponse,
    ReflectionInfo,
    ToolChatResponse,
)
from app.tool_service import generate_tool_response


app = FastAPI(title=settings.app_name)
logger = get_logger(__name__)


def _trace_id_from_request(request: Request) -> str:
    provided = request.headers.get("x-trace-id")
    if provided:
        return provided
    return f"trace-{uuid4().hex[:12]}"


def _persist_experience_from_request(request: Request) -> bool:
    raw = request.headers.get("x-persist-experience")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    trace_id = _trace_id_from_request(http_request)
    start = perf_counter()
    try:
        chat_history = get_chat_history_text(request.session_id)
        with llm_override(request.llm, request.model):
            answer = await generate_chat_response(request.message, chat_history=chat_history)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
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
        with llm_override(request.llm, request.model):
            answer, sources, retrieved_contexts = await generate_rag_response(
                request.message,
                chat_history=chat_history,
            )
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
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
        with llm_override(request.llm, request.model):
            answer, used_tools = await generate_tool_response(request.message, chat_history=chat_history)
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
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
        with llm_override(request.llm, request.model):
            result = await run_agent_workflow(
                request.message,
                chat_history=chat_history,
                trace_id=trace_id,
                session_id=request.session_id,
                persist_experience=_persist_experience_from_request(http_request),
            )
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    working_write = write_working_memory(request.session_id, request.message, result["answer"])
    memory_writes = list(result.get("memory_writes", []))
    memory_writes.append(working_write)
    memory_reads = list(result.get("memory_reads", []))
    elapsed_ms = int((perf_counter() - start) * 1000)
    steps_used = result.get("steps_used", len(result.get("step_results", [])))
    logger.info(
        f"trace_id={trace_id} endpoint=/agent-chat session_id={request.session_id} "
        f"route={result['route']} steps={steps_used} used_tools={result.get('used_tools', [])} "
        f"sources={len(result.get('sources', []))} contexts={len(result.get('retrieved_contexts', []))} "
        f"memory_reads={len(memory_reads)} memory_writes={len(memory_writes)} "
        f"elapsed_ms={elapsed_ms}"
    )
    step_results = result.get("step_results", [])
    raw_action_source = (
        step_results[-1].get("action_source", result.get("action_decision_source", "rule_fallback"))
        if step_results
        else result.get("action_decision_source", "rule_fallback")
    )
    action_source = (
        raw_action_source
        if raw_action_source in {"llm", "hint_fallback", "rule_fallback"}
        else "rule_fallback"
    )
    return AgentChatResponse(
        answer=result["answer"],
        route=result["route"],
        router_reason=result["router_reason"],
        sources=result.get("sources", []),
        retrieved_contexts=result.get("retrieved_contexts", []),
        used_tools=result.get("used_tools", []),
        plan=AgentPlan(
            goal=result.get("goal", request.message),
            subgoals=result.get("subgoals", []),
        ),
        steps=[
            PlanStepResult(
                step_index=item["step_index"],
                subgoal=item["subgoal"],
                route=item["route"],
                router_reason=item["router_reason"],
                answer_preview=item.get("answer_preview", ""),
                used_tools=item.get("used_tools", []),
                reward=float(item.get("reward", 0.0)),
                action_type=str(item.get("action_type", item["route"])),
            )
            for item in step_results
        ],
        reflection=ReflectionInfo(
            done=bool(result.get("reflect_done", True)),
            next_action=result.get("reflect_next_action", "finish"),
            progress=float(result.get("reflect_progress", 1.0)),
            lesson=result.get("reflect_lesson", ""),
            goal_achieved=bool(result.get("reflect_goal_achieved", False)),
            missing_info=str(result.get("reflect_missing_info", "")),
            judge_source=result.get("reflect_judge_source", "rule_fallback"),
        ),
        evaluation=EvaluationInfo(
            passed=bool(result.get("evaluation_passed", False)),
            score=float(result.get("evaluation_score", 0.0)),
            feedback=str(result.get("evaluation_feedback", "")),
            source=result.get("evaluation_source", "rule_fallback"),
            triggered_replan=bool(result.get("evaluation_triggered_replan", False)),
        ),
        metrics=AgentMetrics(
            steps_used=int(result.get("steps_used", len(step_results))),
            tool_calls=int(result.get("tool_calls", 0)),
            replanned=bool(result.get("replanned", False)),
            memory_hits=int(result.get("memory_hits", 0)),
            last_reward=float(result.get("last_reward", 0.0)),
            total_reward=float(result.get("total_reward", 0.0)),
        ),
        budget=AgentBudget(
            max_steps=int(result.get("max_agent_steps", settings.max_agent_steps)),
            steps_remaining=max(
                int(result.get("max_agent_steps", settings.max_agent_steps))
                - int(result.get("steps_used", len(step_results))),
                0,
            ),
            max_tool_calls=int(result.get("max_tool_calls", settings.max_tool_calls)),
            tool_calls_remaining=max(
                int(result.get("max_tool_calls", settings.max_tool_calls))
                - int(result.get("tool_calls", 0)),
                0,
            ),
            max_runtime_seconds=float(
                result.get("max_agent_runtime_seconds", settings.max_agent_runtime_seconds)
            ),
            elapsed_ms=int(result.get("elapsed_ms", 0)),
            stop_reason=result.get("budget_stop_reason") or None,
        ),
        memory_lessons=list(result.get("memory_lessons", [])),
        memory_reads=[MemoryEvent(**item) for item in memory_reads],
        memory_writes=[MemoryEvent(**item) for item in memory_writes],
        long_term_facts=list(result.get("long_term_facts", [])),
        environment=EnvironmentInfo(
            name=result.get("environment_name", "student_support"),
            action_space=list(result.get("action_space", ["chat", "rag", "tool"])),
        ),
        last_observation=ObservationInfo(
            goal=str((result.get("last_observation") or {}).get("goal", result.get("goal", ""))),
            current_subgoal=str((result.get("last_observation") or {}).get("current_subgoal", "")),
            step_index=int((result.get("last_observation") or {}).get("step_index", 0)),
            completed_steps=int((result.get("last_observation") or {}).get("completed_steps", 0)),
            available_actions=list(
                (result.get("last_observation") or {}).get(
                    "available_actions",
                    result.get("action_space", ["chat", "rag", "tool"]),
                )
            ),
            last_answer_preview=str(
                (result.get("last_observation") or {}).get("last_answer_preview", "")
            ),
            last_reward=float((result.get("last_observation") or {}).get("last_reward", 0.0)),
        ),
        last_action_decision=ActionDecisionInfo(
            action_type=(step_results[-1]["route"] if step_results else result.get("route", "chat")),
            content=str(step_results[-1]["subgoal"] if step_results else ""),
            reason=str(
                step_results[-1]["router_reason"]
                if step_results
                else result.get("router_reason", "")
            ),
            source=action_source,
        ),
    )
