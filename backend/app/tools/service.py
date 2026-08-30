import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.core.llm import create_chat_model
from app.core.prompts import cache_friendly_messages
from app.core.retry import with_retry
from app.utils.content import content_to_text, preferred_response_language, response_language_instruction


TOOL_SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
You MUST use available tools for calculations, budgeting, or checklist-style planning.
Do not invent numeric totals yourself when a tool can compute them.
Call the appropriate tool with reasonable assumptions, then explain the result clearly.
Use prior conversation history if relevant to the latest user request."""

FORCE_TOOL_PROMPT = """You must call exactly one relevant tool now.
Do not answer in plain text before calling a tool.
If the user asks about budget/cost/rent, call estimate_weekly_budget.
If the user asks about a checklist, call build_prearrival_checklist."""


@tool
def estimate_weekly_budget(
    rent_per_week_aud: float,
    groceries_per_week_aud: float = 120.0,
    transport_per_week_aud: float = 50.0,
    utilities_per_week_aud: float = 35.0,
) -> str:
    """Estimate weekly living cost in AUD for an international student."""
    total = rent_per_week_aud + groceries_per_week_aud + transport_per_week_aud + utilities_per_week_aud
    return (
        "Estimated weekly budget (AUD):\n"
        f"- Rent: {rent_per_week_aud:.2f}\n"
        f"- Groceries: {groceries_per_week_aud:.2f}\n"
        f"- Transport: {transport_per_week_aud:.2f}\n"
        f"- Utilities: {utilities_per_week_aud:.2f}\n"
        f"- Total: {total:.2f}"
    )


@tool
def build_prearrival_checklist(
    has_visa: bool,
    has_oshc: bool,
    has_accommodation: bool,
    has_enrolment_confirmation: bool,
) -> str:
    """Generate a pre-arrival checklist and highlight missing items."""
    status = {
        "Student visa approval": has_visa,
        "OSHC coverage": has_oshc,
        "Confirmed accommodation": has_accommodation,
        "Enrolment confirmation": has_enrolment_confirmation,
    }
    completed = [item for item, done in status.items() if done]
    missing = [item for item, done in status.items() if not done]

    lines = ["Pre-arrival checklist status:"]
    if completed:
        lines.append("- Completed:")
        lines.extend(f"  - {item}" for item in completed)
    if missing:
        lines.append("- Missing (priority):")
        lines.extend(f"  - {item}" for item in missing)
    else:
        lines.append("- All core items completed.")

    return "\n".join(lines)


TOOLS = [estimate_weekly_budget, build_prearrival_checklist]
TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}


def _create_tool_model() -> BaseChatModel:
    return create_chat_model(temperature=0.1)


def _extract_rent(message: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:aud|a\$|\$)?", message.lower())
    if match:
        return float(match.group(1))
    return 420.0


def _heuristic_tool_call(message: str) -> dict:
    message_lower = message.lower()
    if any(token in message_lower for token in ("checklist", "check list", "pre-arrival", "prearrival")):
        return {
            "name": "build_prearrival_checklist",
            "args": {
                "has_visa": False,
                "has_oshc": False,
                "has_accommodation": False,
                "has_enrolment_confirmation": False,
            },
            "id": "heuristic-checklist-1",
        }
    return {
        "name": "estimate_weekly_budget",
        "args": {"rent_per_week_aud": _extract_rent(message)},
        "id": "heuristic-budget-1",
    }


def _normalize_tool_call(tool_call: object) -> dict:
    if isinstance(tool_call, dict):
        return {
            "name": tool_call.get("name", ""),
            "args": tool_call.get("args", {}) or {},
            "id": tool_call.get("id") or "tool-call",
        }
    return {
        "name": getattr(tool_call, "name", ""),
        "args": getattr(tool_call, "args", {}) or {},
        "id": getattr(tool_call, "id", None) or "tool-call",
    }


def _run_tool_calls(tool_calls: list[object]) -> tuple[list[str], list[ToolMessage]]:
    used_tools: list[str] = []
    tool_messages: list[ToolMessage] = []
    for raw_call in tool_calls:
        tool_call = _normalize_tool_call(raw_call)
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        selected_tool = TOOLS_BY_NAME.get(tool_name)
        if selected_tool is None:
            continue
        tool_result = selected_tool.invoke(tool_args)
        used_tools.append(tool_name)
        tool_messages.append(
            ToolMessage(
                content=content_to_text(tool_result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )
    return used_tools, tool_messages


async def generate_tool_response(
    message: str,
    chat_history: str = "",
    max_tool_calls: int | None = None,
    response_language: str | None = None,
) -> tuple[str, list[str]]:
    base_model = _create_tool_model()
    soft_model = base_model.bind_tools(TOOLS)
    forced_model = base_model.bind_tools(TOOLS, tool_choice="any")

    language = response_language or preferred_response_language(message)
    messages = cache_friendly_messages(
        f"{TOOL_SYSTEM_PROMPT}\n\n{response_language_instruction(language)}",
        chat_history,
        message,
    )

    first_response = await with_retry(lambda: soft_model.ainvoke(messages))
    tool_calls = list(first_response.tool_calls or [])

    # Retry with forced tool choice when the model answers in plain text.
    if not tool_calls:
        forced_messages = [
            *messages,
            HumanMessage(content=FORCE_TOOL_PROMPT),
        ]
        first_response = await with_retry(lambda: forced_model.ainvoke(forced_messages))
        tool_calls = list(first_response.tool_calls or [])
        messages = forced_messages

    # Deterministic fallback so tool route never silently skips tools.
    if not tool_calls:
        tool_calls = [_heuristic_tool_call(message)]
        first_response = AIMessage(content="", tool_calls=tool_calls)

    if max_tool_calls is not None:
        tool_calls = tool_calls[: max(0, max_tool_calls)]

    used_tools, tool_messages = _run_tool_calls(tool_calls)
    if not used_tools:
        tool_calls = [_heuristic_tool_call(message)]
        first_response = AIMessage(content="", tool_calls=tool_calls)
        used_tools, tool_messages = _run_tool_calls(tool_calls)

    final_response = await with_retry(
        lambda: soft_model.ainvoke([*messages, first_response, *tool_messages])
    )
    return content_to_text(final_response.content), used_tools
