from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
from app.logging_service import get_logger
from app.rag_service import generate_rag_response
from app.schemas import RetrievedContext
from app.retry_service import with_retry
from app.tool_service import generate_tool_response

Route = Literal["chat", "rag", "tool"]
ReflectAction = Literal["continue", "replan", "finish"]

RAG_HINT_KEYWORDS = {
    "usyd",
    "visa",
    "oshc",
    "accommodation",
    "arrival",
    "rent",
    "suburb",
    "campus",
    "orientation",
    "enrolment",
    "student health cover",
}

TOOL_HINT_KEYWORDS = {
    "budget",
    "cost",
    "calculate",
    "estimat",
    "how much",
    "checklist",
}

ROUTER_PROMPT = """You are a routing controller for an international student assistant.
Choose exactly one route:
- chat: general questions that do not require local knowledge base or tool execution.
- rag: questions about USYD/student visa/OSHC/accommodation/arrival that should use the local knowledge base.
- tool: user asks for numeric calculations, budgeting, or checklist generation that directly map to available tools.

State-machine policy (must follow):
1) If user only provides background facts and no explicit task request, route to chat.
2) Do not route to tool only because the message contains numbers.
3) Do not route to rag only because the message mentions a school/entity.
4) If user asks for potentially harmful or policy-bypass advice (e.g., fake visa shortcuts), route to rag so the assistant can ground response in official guidance.
5) For ambiguous planning requests tied to student onboarding in Sydney/USYD, prefer rag first.

Few-shot routing examples:
- "My rent is 450 AUD." -> chat (background statement, no calculation request)
- "USYD thing" -> rag (entity-specific factual lookup likely needed)
- "Help me pls" -> chat (intent too vague, needs clarification)

Return strict JSON with:
- route: one of chat, rag, tool
- reason: one short sentence explaining the routing choice."""

PLANNER_PROMPT = """You are a hierarchical planner for an international student assistant.
Decompose the user goal into executable subgoals.

Rules:
1) Simple single-intent questions should produce exactly 1 subgoal (copy/refine the user ask).
2) Complex multi-part goals may produce 2-4 subgoals.
3) Keep each subgoal concrete and independently actionable.
4) Do not invent unrelated tasks.
5) Prefer retrieval before generation when policy/onboarding facts are needed.

Return strict JSON with:
- goal: short restatement of the overall goal
- subgoals: ordered list of 1-4 subgoal strings"""

logger = get_logger(__name__)


class RouterDecision(BaseModel):
    route: Route
    reason: str


class PlanDecision(BaseModel):
    goal: str
    subgoals: list[str] = Field(default_factory=list)


class StepRecord(TypedDict, total=False):
    step_index: int
    subgoal: str
    route: Route
    router_reason: str
    answer: str
    answer_preview: str
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]
    used_tools: list[str]


class AgentState(TypedDict, total=False):
    message: str
    chat_history: str
    trace_id: str
    goal: str
    subgoals: list[str]
    current_step: int
    max_steps: int
    step_results: list[StepRecord]
    route: Route
    router_reason: str
    answer: str
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]
    used_tools: list[str]
    reflect_done: bool
    reflect_next_action: ReflectAction
    reflect_progress: float
    reflect_lesson: str
    replanned: bool
    steps_used: int
    tool_calls: int


SAFETY_HINT_KEYWORDS = {
    "fake visa",
    "visa shortcut",
    "shortcut",
    "bypass",
    "illegal",
    "exploit",
    "evade policy",
}

BACKGROUND_ONLY_PREFIXES = (
    "i will",
    "i am",
    "i'm",
    "my rent is",
    "my budget is",
    "i study",
    "i will study",
)

TASK_REQUEST_HINTS = {
    "can you",
    "could you",
    "please",
    "help me",
    "what should",
    "how do",
    "calculate",
    "estimate",
    "plan",
    "checklist",
}


def _is_context_only_message(message: str) -> bool:
    message_lower = message.strip().lower()
    if not message_lower:
        return False
    if "?" in message_lower:
        return False
    if any(hint in message_lower for hint in TASK_REQUEST_HINTS):
        return False
    return any(message_lower.startswith(prefix) for prefix in BACKGROUND_ONLY_PREFIXES)


def _should_force_rag_for_safety(message: str) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in SAFETY_HINT_KEYWORDS)


def _should_prefer_rag_for_ambiguous_plan(message: str) -> bool:
    message_lower = message.lower()
    has_plan_intent = "plan" in message_lower or "first month" in message_lower
    has_student_context = "student" in message_lower or "sydney" in message_lower or "usyd" in message_lower
    return has_plan_intent and has_student_context


def _keyword_route(message: str) -> Route:
    message_lower = message.lower()
    if _should_force_rag_for_safety(message):
        return "rag"
    if _is_context_only_message(message):
        return "chat"
    if _should_prefer_rag_for_ambiguous_plan(message):
        return "rag"
    if any(keyword in message_lower for keyword in TOOL_HINT_KEYWORDS):
        return "tool"
    if any(keyword in message_lower for keyword in RAG_HINT_KEYWORDS):
        return "rag"
    return "chat"


def _preview(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


async def _llm_route(message: str, chat_history: str) -> RouterDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nCurrent user message:\n{message}",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(RouterDecision)
    chain = prompt | model
    return await with_retry(lambda: chain.ainvoke({"message": message, "chat_history": chat_history}))


async def _decide_route(message: str, chat_history: str, trace_id: str) -> RouterDecision:
    if _should_force_rag_for_safety(message):
        reason = "Safety-sensitive or policy-bypass request detected; using rag for compliant guidance."
        logger.trace(f"trace_id={trace_id} route_decision=rag reason={reason}")
        return RouterDecision(route="rag", reason=reason)
    if _is_context_only_message(message):
        reason = "Background-only input without explicit task request; using chat for clarification/context."
        logger.trace(f"trace_id={trace_id} route_decision=chat reason={reason}")
        return RouterDecision(route="chat", reason=reason)
    if _should_prefer_rag_for_ambiguous_plan(message):
        reason = "Ambiguous onboarding planning request; applying retrieval-first (rag) strategy."
        logger.trace(f"trace_id={trace_id} route_decision=rag reason={reason}")
        return RouterDecision(route="rag", reason=reason)
    try:
        decision = await _llm_route(message, chat_history)
        logger.trace(
            f"trace_id={trace_id} route_decision={decision.route} reason={decision.reason}"
        )
        return decision
    except Exception:
        fallback_route = _keyword_route(message)
        logger.warning(
            f"trace_id={trace_id} route_fallback={fallback_route} reason=llm_router_unavailable"
        )
        return RouterDecision(
            route=fallback_route,
            reason="Fallback keyword router used because LLM routing was unavailable.",
        )


async def _llm_plan(message: str, chat_history: str) -> PlanDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nUser goal:\n{message}",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(PlanDecision)
    chain = prompt | model
    return await with_retry(lambda: chain.ainvoke({"message": message, "chat_history": chat_history}))


def _normalize_subgoals(message: str, plan: PlanDecision) -> list[str]:
    cleaned = [item.strip() for item in plan.subgoals if item and item.strip()]
    if not cleaned:
        cleaned = [message]
    # Keep planning bounded for latency and eval stability.
    return cleaned[: settings.max_plan_steps]


async def _plan_node(state: AgentState) -> AgentState:
    message = state["message"]
    chat_history = state.get("chat_history", "")
    trace_id = state.get("trace_id", "n/a")
    max_steps = state.get("max_steps", settings.max_plan_steps)

    # Keep trivial context-only turns as single-step plans.
    if _is_context_only_message(message):
        goal = message
        subgoals = [message]
        logger.trace(f"trace_id={trace_id} plan_mode=single_context subgoals=1")
        return {
            "goal": goal,
            "subgoals": subgoals,
            "current_step": 0,
            "max_steps": max_steps,
            "step_results": [],
            "replanned": False,
            "steps_used": 0,
            "tool_calls": 0,
            "reflect_done": False,
            "reflect_next_action": "continue",
            "reflect_progress": 0.0,
            "reflect_lesson": "Single-step context acknowledgment plan.",
        }

    try:
        plan = await _llm_plan(message, chat_history)
        goal = plan.goal.strip() or message
        subgoals = _normalize_subgoals(message, plan)
    except Exception:
        goal = message
        subgoals = [message]
        logger.warning(f"trace_id={trace_id} plan_fallback=single_step reason=llm_planner_unavailable")

    logger.trace(
        f"trace_id={trace_id} plan_goal={_preview(goal, 80)} subgoals={len(subgoals)}"
    )
    return {
        "goal": goal,
        "subgoals": subgoals,
        "current_step": 0,
        "max_steps": max_steps,
        "step_results": [],
        "replanned": False,
        "steps_used": 0,
        "tool_calls": 0,
        "reflect_done": False,
        "reflect_next_action": "continue",
        "reflect_progress": 0.0,
        "reflect_lesson": "Initial hierarchical plan created.",
    }


async def _act_node(state: AgentState) -> AgentState:
    subgoals = state.get("subgoals", [])
    current_step = state.get("current_step", 0)
    chat_history = state.get("chat_history", "")
    trace_id = state.get("trace_id", "n/a")

    if current_step >= len(subgoals):
        return {
            "route": state.get("route", "chat"),
            "router_reason": state.get("router_reason", "No remaining subgoals."),
        }

    subgoal = subgoals[current_step]
    decision = await _decide_route(subgoal, chat_history, trace_id)
    logger.trace(
        f"trace_id={trace_id} act_step={current_step + 1} route={decision.route} "
        f"subgoal={_preview(subgoal, 80)}"
    )
    return {
        "route": decision.route,
        "router_reason": decision.reason,
    }


async def _execute_node(state: AgentState) -> AgentState:
    subgoals = state.get("subgoals", [])
    current_step = state.get("current_step", 0)
    chat_history = state.get("chat_history", "")
    route = state.get("route", "chat")
    router_reason = state.get("router_reason", "")
    trace_id = state.get("trace_id", "n/a")

    if current_step >= len(subgoals):
        return {}

    subgoal = subgoals[current_step]
    sources: list[str] = []
    retrieved_contexts: list[RetrievedContext] = []
    used_tools: list[str] = []

    if route == "rag":
        answer, sources, retrieved_contexts = await generate_rag_response(
            subgoal,
            chat_history=chat_history,
        )
    elif route == "tool":
        answer, used_tools = await generate_tool_response(
            subgoal,
            chat_history=chat_history,
        )
    else:
        answer = await generate_chat_response(subgoal, chat_history=chat_history)

    step_results = list(state.get("step_results", []))
    step_results.append(
        {
            "step_index": current_step + 1,
            "subgoal": subgoal,
            "route": route,
            "router_reason": router_reason,
            "answer": answer,
            "answer_preview": _preview(answer),
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
            "used_tools": used_tools,
        }
    )
    steps_used = current_step + 1
    tool_calls = state.get("tool_calls", 0) + len(used_tools)
    logger.trace(
        f"trace_id={trace_id} execute_step={steps_used} route={route} tools={used_tools}"
    )
    return {
        "step_results": step_results,
        "current_step": current_step + 1,
        "steps_used": steps_used,
        "tool_calls": tool_calls,
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
        "used_tools": used_tools,
    }


async def _reflect_node(state: AgentState) -> AgentState:
    subgoals = state.get("subgoals", [])
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", settings.max_plan_steps)
    step_results = state.get("step_results", [])
    replanned = state.get("replanned", False)
    trace_id = state.get("trace_id", "n/a")

    total = max(len(subgoals), 1)
    progress = min(current_step / total, 1.0)
    last_answer = step_results[-1]["answer"] if step_results else ""

    if current_step >= len(subgoals) or current_step >= max_steps:
        next_action: ReflectAction = "finish"
        done = True
        lesson = "All planned subgoals completed or step budget reached."
    elif not last_answer.strip() and not replanned:
        # One-shot replan when execution produced an empty answer.
        next_action = "replan"
        done = False
        lesson = "Empty step result detected; triggering one-shot replan."
    else:
        next_action = "continue"
        done = False
        lesson = f"Completed {current_step}/{len(subgoals)} subgoals; continuing hierarchical execution."

    logger.trace(
        f"trace_id={trace_id} reflect_action={next_action} progress={progress:.2f} "
        f"step={current_step}/{len(subgoals)}"
    )
    return {
        "reflect_done": done,
        "reflect_next_action": next_action,
        "reflect_progress": progress,
        "reflect_lesson": lesson,
    }


async def _replan_node(state: AgentState) -> AgentState:
    message = state["message"]
    chat_history = state.get("chat_history", "")
    trace_id = state.get("trace_id", "n/a")
    remaining_budget = max(1, state.get("max_steps", settings.max_plan_steps) - state.get("current_step", 0))

    try:
        plan = await _llm_plan(
            f"Previous plan stalled. Create a simpler remaining plan for: {message}",
            chat_history,
        )
        new_subgoals = _normalize_subgoals(message, plan)[:remaining_budget]
        goal = plan.goal.strip() or state.get("goal", message)
    except Exception:
        new_subgoals = [message]
        goal = state.get("goal", message)
        logger.warning(f"trace_id={trace_id} replan_fallback=single_step")

    # Replace unfinished tail while keeping already completed step_results.
    completed = state.get("current_step", 0)
    logger.trace(
        f"trace_id={trace_id} replan_new_subgoals={len(new_subgoals)} completed={completed}"
    )
    return {
        "goal": goal,
        "subgoals": new_subgoals,
        "current_step": 0,
        "replanned": True,
        "reflect_next_action": "continue",
        "reflect_done": False,
        "reflect_lesson": "Replanned remaining work after a stalled step.",
        "step_results": [],
        "steps_used": 0,
        "tool_calls": 0,
    }


async def _finalize_node(state: AgentState) -> AgentState:
    step_results = state.get("step_results", [])
    if not step_results:
        return {
            "answer": "I could not complete the planned steps for this request.",
            "route": state.get("route", "chat"),
            "router_reason": state.get("router_reason", "No executed steps."),
            "sources": [],
            "retrieved_contexts": [],
            "used_tools": [],
            "steps_used": 0,
            "tool_calls": 0,
            "reflect_done": True,
            "reflect_next_action": "finish",
            "reflect_progress": 0.0,
            "reflect_lesson": state.get("reflect_lesson", "No steps executed."),
        }

    if len(step_results) == 1:
        answer = step_results[0]["answer"]
    else:
        parts = [
            f"### Step {item['step_index']}: {item['subgoal']}\n{item['answer']}"
            for item in step_results
        ]
        answer = "\n\n".join(parts)

    sources: list[str] = []
    retrieved_contexts: list[RetrievedContext] = []
    used_tools: list[str] = []
    for item in step_results:
        for source in item.get("sources", []):
            if source not in sources:
                sources.append(source)
        retrieved_contexts.extend(item.get("retrieved_contexts", []))
        for tool_name in item.get("used_tools", []):
            if tool_name not in used_tools:
                used_tools.append(tool_name)

    last = step_results[-1]
    return {
        "answer": answer,
        "route": last["route"],
        "router_reason": last["router_reason"],
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
        "used_tools": used_tools,
        "steps_used": state.get("steps_used", len(step_results)),
        "tool_calls": state.get("tool_calls", len(used_tools)),
        "reflect_done": True,
        "reflect_next_action": "finish",
        "reflect_progress": 1.0,
        "reflect_lesson": state.get("reflect_lesson", "Plan finalized."),
    }


def _after_reflect(state: AgentState) -> Literal["act", "replan", "finalize"]:
    action = state.get("reflect_next_action", "finish")
    if action == "continue":
        return "act"
    if action == "replan" and not state.get("replanned", False):
        return "replan"
    return "finalize"


@lru_cache(maxsize=1)
def _build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan_node)
    graph.add_node("act", _act_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("replan", _replan_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _after_reflect,
        {
            "act": "act",
            "replan": "replan",
            "finalize": "finalize",
        },
    )
    graph.add_edge("replan", "act")
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_agent_workflow(message: str, chat_history: str = "", trace_id: str = "n/a") -> AgentState:
    graph = _build_agent_graph()
    result = await graph.ainvoke(
        {
            "message": message,
            "chat_history": chat_history,
            "trace_id": trace_id,
            "max_steps": settings.max_plan_steps,
            "step_results": [],
            "current_step": 0,
            "replanned": False,
            "steps_used": 0,
            "tool_calls": 0,
        }
    )
    return result
