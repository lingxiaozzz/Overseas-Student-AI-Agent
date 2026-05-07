from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.chat_service import MissingApiKeyError, generate_chat_response
from app.config import settings
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

Return strict JSON with:
- route: one of chat, rag, tool
- reason: one short sentence explaining the routing choice."""


class RouterDecision(BaseModel):
    route: Route
    reason: str


class AgentState(TypedDict, total=False):
    message: str
    chat_history: str
    route: Route
    router_reason: str
    answer: str
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]
    used_tools: list[str]


def _keyword_route(message: str) -> Route:
    message_lower = message.lower()
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
    try:
        decision = await _llm_route(message, chat_history)
        return {"route": decision.route, "router_reason": decision.reason}
    except Exception:
        fallback_route = _keyword_route(message)
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


async def run_agent_workflow(message: str, chat_history: str = "") -> AgentState:
    graph = _build_agent_graph()
    result = await graph.ainvoke({"message": message, "chat_history": chat_history})
    return result
