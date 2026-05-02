from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.chat_service import MissingApiKeyError
from app.config import settings


TOOL_SYSTEM_PROMPT = """You are an AI assistant for international students in Sydney.
Use available tools when the user asks for calculations, budgeting, or checklist-style planning.
If a tool is needed, call it with reasonable assumptions and explain those assumptions clearly.
Use prior conversation history if relevant to the latest user request."""


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


def _create_tool_model() -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


async def generate_tool_response(message: str, chat_history: str = "") -> tuple[str, list[str]]:
    model = _create_tool_model().bind_tools(TOOLS)
    messages = [
        SystemMessage(content=TOOL_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversation history:\n{chat_history}\n\nCurrent user message:\n{message}"),
    ]
    first_response = await model.ainvoke(messages)
    tool_calls = first_response.tool_calls or []

    if not tool_calls:
        return _to_text(first_response.content), []

    used_tools: list[str] = []
    tool_messages: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        selected_tool = TOOLS_BY_NAME.get(tool_name)
        if selected_tool is None:
            continue

        tool_result = selected_tool.invoke(tool_args)
        used_tools.append(tool_name)
        tool_messages.append(
            ToolMessage(
                content=_to_text(tool_result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    final_response = await model.ainvoke([*messages, first_response, *tool_messages])
    return _to_text(final_response.content), used_tools
