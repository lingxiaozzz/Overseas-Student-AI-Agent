from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
from app.logging_service import get_logger
from app.rag_service import generate_rag_response
from app.schemas import RetrievedContext
from app.retry_service import with_retry
from app.tool_service import generate_tool_response

Route = Literal["chat", "rag", "tool"]

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

logger = get_logger(__name__)


class RouterDecision(BaseModel):
    route: Route
    reason: str


class AgentState(TypedDict, total=False):
    message: str
    chat_history: str
    trace_id: str
    route: Route
    router_reason: str
    answer: str
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]
    used_tools: list[str]


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


async def _route_node(state: AgentState) -> AgentState:
    message = state["message"]
    chat_history = state.get("chat_history", "")
    trace_id = state.get("trace_id", "n/a")
    if _should_force_rag_for_safety(message):
        reason = "Safety-sensitive or policy-bypass request detected; using rag for compliant guidance."
        logger.trace(f"trace_id={trace_id} route_decision=rag reason={reason}")
        return {"route": "rag", "router_reason": reason}
    if _is_context_only_message(message):
        reason = "Background-only input without explicit task request; using chat for clarification/context."
        logger.trace(f"trace_id={trace_id} route_decision=chat reason={reason}")
        return {"route": "chat", "router_reason": reason}
    if _should_prefer_rag_for_ambiguous_plan(message):
        reason = "Ambiguous onboarding planning request; applying retrieval-first (rag) strategy."
        logger.trace(f"trace_id={trace_id} route_decision=rag reason={reason}")
        return {"route": "rag", "router_reason": reason}
    try:
        decision = await _llm_route(message, chat_history)
        logger.trace(
            f"trace_id={trace_id} route_decision={decision.route} reason={decision.reason}"
        )
        return {"route": decision.route, "router_reason": decision.reason}
    except Exception:
        fallback_route = _keyword_route(message)
        logger.warning(
            f"trace_id={trace_id} route_fallback={fallback_route} reason=llm_router_unavailable"
        )
        return {
            "route": fallback_route,
            "router_reason": "Fallback keyword router used because LLM routing was unavailable.",
        }


async def _chat_node(state: AgentState) -> AgentState:
    answer = await generate_chat_response(state["message"], chat_history=state.get("chat_history", ""))
    return {
        "answer": answer,
        "sources": [],
        "retrieved_contexts": [],
        "used_tools": [],
    }


async def _rag_node(state: AgentState) -> AgentState:
    answer, sources, retrieved_contexts = await generate_rag_response(
        state["message"],
        chat_history=state.get("chat_history", ""),
    )
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
        "used_tools": [],
    }


async def _tool_node(state: AgentState) -> AgentState:
    answer, used_tools = await generate_tool_response(
        state["message"],
        chat_history=state.get("chat_history", ""),
    )
    return {
        "answer": answer,
        "sources": [],
        "retrieved_contexts": [],
        "used_tools": used_tools,
    }


def _route_decision(state: AgentState) -> Route:
    return state["route"]


@lru_cache(maxsize=1)
def _build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", _route_node)
    graph.add_node("chat", _chat_node)
    graph.add_node("rag", _rag_node)
    graph.add_node("tool", _tool_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        _route_decision,
        {
            "chat": "chat",
            "rag": "rag",
            "tool": "tool",
        },
    )
    graph.add_edge("chat", END)
    graph.add_edge("rag", END)
    graph.add_edge("tool", END)

    return graph.compile()


async def run_agent_workflow(message: str, chat_history: str = "", trace_id: str = "n/a") -> AgentState:
    graph = _build_agent_graph()
    result = await graph.ainvoke({"message": message, "chat_history": chat_history, "trace_id": trace_id})
    return result
