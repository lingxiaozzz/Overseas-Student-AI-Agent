from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.chat_service import generate_chat_response
from app.rag_service import generate_rag_response
from app.schemas import RetrievedContext
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
    "plan",
}


class AgentState(TypedDict, total=False):
    message: str
    chat_history: str
    route: Route
    answer: str
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]
    used_tools: list[str]


def _choose_route(message: str) -> Route:
    message_lower = message.lower()
    if any(keyword in message_lower for keyword in TOOL_HINT_KEYWORDS):
        return "tool"
    if any(keyword in message_lower for keyword in RAG_HINT_KEYWORDS):
        return "rag"

    return "chat"


async def _route_node(state: AgentState) -> AgentState:
    return {"route": _choose_route(state["message"])}


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
