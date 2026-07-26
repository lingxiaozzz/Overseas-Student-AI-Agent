from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.chat_service import MissingApiKeyError
from app.config import settings
from app.environment import Action, clear_env, get_or_create_env, reset_env
from app.logging_service import get_logger
from app.memory_service import (
    append_experience,
    build_experience_lesson,
    format_experience_context,
    is_low_value_lesson,
    retrieve_experiences,
)
from app.schemas import RetrievedContext
from app.retry_service import with_retry

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
6) If prior experience lessons are provided, reuse useful strategy patterns and avoid previously failed approaches.

Return strict JSON with:
- goal: short restatement of the overall goal
- subgoals: ordered list of 1-4 subgoal strings"""

REFLECTOR_PROMPT = """You are a reflection judge for a hierarchical international-student agent.
Evaluate whether the latest execution step advances the overall goal.

Choose next_action:
- continue: more planned subgoals remain and current progress is healthy
- replan: current approach is stalled/wrong and a new remaining plan is needed
- finish: overall goal is sufficiently satisfied or no useful work remains

Rules:
1) Prefer finish when the user goal is adequately answered.
2) Prefer continue only if unfinished subgoals are still necessary.
3) Prefer replan only for clear failures (empty/irrelevant/blocked results).
4) lesson must be an actionable strategy (route/tool preference), not a status sentence.
5) missing_info should be brief; use empty string if none.

Return strict JSON with:
- done: boolean
- next_action: one of continue, replan, finish
- progress: number between 0 and 1
- goal_achieved: boolean
- missing_info: short string
- lesson: one short actionable sentence"""

logger = get_logger(__name__)


class RouterDecision(BaseModel):
    route: Route
    reason: str


class PlanDecision(BaseModel):
    goal: str
    subgoals: list[str] = Field(default_factory=list)


class ReflectionDecision(BaseModel):
    done: bool
    next_action: ReflectAction
    progress: float = 0.0
    goal_achieved: bool = False
    missing_info: str = ""
    lesson: str = ""


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
    reward: float
    action_type: str


class AgentState(TypedDict, total=False):
    message: str
    chat_history: str
    session_id: str
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
    experience_context: str
    memory_lessons: list[str]
    memory_hits: int
    persist_experience: bool
    environment_name: str
    action_space: list[str]
    last_reward: float
    total_reward: float
    reflect_goal_achieved: bool
    reflect_missing_info: str
    reflect_judge_source: str


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


async def _llm_plan(message: str, chat_history: str, experience_context: str) -> PlanDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT),
            (
                "human",
                "Conversation history:\n{chat_history}\n\n"
                "Prior experience lessons:\n{experience_context}\n\n"
                "User goal:\n{message}",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(PlanDecision)
    chain = prompt | model
    return await with_retry(
        lambda: chain.ainvoke(
            {
                "message": message,
                "chat_history": chat_history,
                "experience_context": experience_context,
            }
        )
    )


def _normalize_subgoals(message: str, plan: PlanDecision) -> list[str]:
    cleaned = [item.strip() for item in plan.subgoals if item and item.strip()]
    if not cleaned:
        cleaned = [message]
    # Keep planning bounded for latency and eval stability.
    return cleaned[: settings.max_plan_steps]


async def _plan_node(state: AgentState) -> AgentState:
    message = state["message"]
    chat_history = state.get("chat_history", "")
    session_id = state.get("session_id", "default")
    trace_id = state.get("trace_id", "n/a")
    max_steps = state.get("max_steps", settings.max_plan_steps)
    environment_name = state.get("environment_name", "student_support")
    observation = reset_env(
        trace_id=trace_id,
        goal=message,
        chat_history=chat_history,
        environment_name=environment_name,
    )
    action_space = observation.available_actions

    experience_records = retrieve_experiences(message, session_id=session_id)
    experience_context = format_experience_context(experience_records)
    memory_lessons = [
        str(item.get("lesson", "")).strip()
        for item in experience_records
        if str(item.get("lesson", "")).strip()
    ]
    memory_hits = len(experience_records)
    logger.trace(
        f"trace_id={trace_id} memory_hits={memory_hits} session_id={session_id}"
    )

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
            "experience_context": experience_context,
            "memory_lessons": memory_lessons,
            "memory_hits": memory_hits,
            "environment_name": environment_name,
            "action_space": action_space,
            "last_reward": 0.0,
            "total_reward": 0.0,
        }

    try:
        plan = await _llm_plan(message, chat_history, experience_context)
        goal = plan.goal.strip() or message
        subgoals = _normalize_subgoals(message, plan)
    except Exception:
        goal = message
        subgoals = [message]
        logger.warning(f"trace_id={trace_id} plan_fallback=single_step reason=llm_planner_unavailable")

    # Keep environment goal aligned with planner restatement.
    reset_env(
        trace_id=trace_id,
        goal=goal,
        chat_history=chat_history,
        environment_name=environment_name,
    )
    logger.trace(
        f"trace_id={trace_id} plan_goal={_preview(goal, 80)} subgoals={len(subgoals)} "
        f"env={environment_name} actions={action_space}"
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
        "experience_context": experience_context,
        "memory_lessons": memory_lessons,
        "memory_hits": memory_hits,
        "environment_name": environment_name,
        "action_space": action_space,
        "last_reward": 0.0,
        "total_reward": 0.0,
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
    route = state.get("route", "chat")
    router_reason = state.get("router_reason", "")
    trace_id = state.get("trace_id", "n/a")
    environment_name = state.get("environment_name", "student_support")

    if current_step >= len(subgoals):
        return {}

    subgoal = subgoals[current_step]
    env = get_or_create_env(trace_id, environment_name)
    step_result = await env.step(
        Action(type=route, content=subgoal, reason=router_reason)
    )
    info = step_result.info
    answer = str(info.get("answer", ""))
    sources = list(info.get("sources", []))
    retrieved_contexts = list(info.get("retrieved_contexts", []))
    used_tools = list(info.get("used_tools", []))
    reward = float(step_result.reward)

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
            "reward": reward,
            "action_type": step_result.action.type,
        }
    )
    steps_used = current_step + 1
    tool_calls = state.get("tool_calls", 0) + len(used_tools)
    total_reward = float(state.get("total_reward", 0.0)) + reward
    logger.trace(
        f"trace_id={trace_id} execute_step={steps_used} env={environment_name} "
        f"action={route} reward={reward:.2f} tools={used_tools}"
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
        "last_reward": reward,
        "total_reward": total_reward,
        "action_space": env.action_space(),
        "environment_name": environment_name,
    }


def _rule_reflect(state: AgentState) -> ReflectionDecision:
    subgoals = state.get("subgoals", [])
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", settings.max_plan_steps)
    step_results = state.get("step_results", [])
    replanned = state.get("replanned", False)
    total = max(len(subgoals), 1)
    progress = min(current_step / total, 1.0)
    last_answer = step_results[-1]["answer"] if step_results else ""

    if current_step >= len(subgoals) or current_step >= max_steps:
        return ReflectionDecision(
            done=True,
            next_action="finish",
            progress=1.0,
            goal_achieved=bool(last_answer.strip()),
            missing_info="",
            lesson=build_experience_lesson(
                state.get("goal", state.get("message", "")),
                routes=[item.get("route", "chat") for item in step_results],
                tools=[
                    tool_name
                    for item in step_results
                    for tool_name in item.get("used_tools", [])
                ],
                success=bool(last_answer.strip()),
                replanned=bool(replanned),
                steps_used=current_step,
            ),
        )
    if not last_answer.strip() and not replanned:
        return ReflectionDecision(
            done=False,
            next_action="replan",
            progress=progress,
            goal_achieved=False,
            missing_info="Latest step returned an empty answer.",
            lesson="Replan once after empty step outputs; avoid repeating the failed action path.",
        )
    return ReflectionDecision(
        done=False,
        next_action="continue",
        progress=progress,
        goal_achieved=False,
        missing_info="",
        lesson=f"Continue remaining subgoals ({current_step}/{len(subgoals)} completed).",
    )


def _clamp_progress(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _apply_reflect_guards(state: AgentState, decision: ReflectionDecision) -> ReflectionDecision:
    subgoals = state.get("subgoals", [])
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", settings.max_plan_steps)
    replanned = bool(state.get("replanned", False))
    step_results = state.get("step_results", [])
    last_answer = step_results[-1]["answer"] if step_results else ""
    remaining = max(len(subgoals) - current_step, 0)

    next_action = decision.next_action
    done = decision.done
    progress = _clamp_progress(decision.progress)

    if current_step >= len(subgoals) or current_step >= max_steps:
        next_action = "finish"
        done = True
        progress = 1.0
    elif not last_answer.strip() and not replanned:
        next_action = "replan"
        done = False
    elif next_action == "replan" and replanned:
        next_action = "continue" if remaining > 0 else "finish"
        done = next_action == "finish"
    elif next_action == "continue" and remaining <= 0:
        next_action = "finish"
        done = True
        progress = 1.0

    lesson = decision.lesson.strip()
    if not lesson or is_low_value_lesson(lesson):
        lesson = _rule_reflect(state).lesson

    return ReflectionDecision(
        done=done,
        next_action=next_action,
        progress=progress,
        goal_achieved=bool(decision.goal_achieved) if next_action == "finish" else False,
        missing_info=decision.missing_info.strip(),
        lesson=lesson,
    )


async def _llm_reflect(state: AgentState) -> ReflectionDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    step_results = state.get("step_results", [])
    latest = step_results[-1] if step_results else {}
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REFLECTOR_PROMPT),
            (
                "human",
                "Overall goal:\n{goal}\n\n"
                "Planned subgoals:\n{subgoals}\n\n"
                "Completed steps: {current_step}/{total_steps}\n"
                "Already replanned: {replanned}\n"
                "Latest subgoal:\n{latest_subgoal}\n"
                "Latest route: {latest_route}\n"
                "Latest tools: {latest_tools}\n"
                "Latest answer preview:\n{latest_answer}\n",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(ReflectionDecision)
    chain = prompt | model
    subgoals = state.get("subgoals", [])
    return await with_retry(
        lambda: chain.ainvoke(
            {
                "goal": state.get("goal", state.get("message", "")),
                "subgoals": "\n".join(f"- {item}" for item in subgoals) or "- none",
                "current_step": state.get("current_step", 0),
                "total_steps": len(subgoals),
                "replanned": bool(state.get("replanned", False)),
                "latest_subgoal": latest.get("subgoal", ""),
                "latest_route": latest.get("route", "unknown"),
                "latest_tools": ", ".join(latest.get("used_tools", [])) or "none",
                "latest_answer": _preview(str(latest.get("answer", "")), 500),
            }
        )
    )


async def _reflect_node(state: AgentState) -> AgentState:
    trace_id = state.get("trace_id", "n/a")
    judge_source = "llm"
    try:
        decision = await _llm_reflect(state)
        decision = _apply_reflect_guards(state, decision)
    except Exception:
        judge_source = "rule_fallback"
        decision = _rule_reflect(state)
        logger.warning(f"trace_id={trace_id} reflect_fallback=rule reason=llm_reflect_unavailable")

    logger.trace(
        f"trace_id={trace_id} reflect_action={decision.next_action} progress={decision.progress:.2f} "
        f"goal_achieved={decision.goal_achieved} source={judge_source}"
    )
    return {
        "reflect_done": decision.done,
        "reflect_next_action": decision.next_action,
        "reflect_progress": decision.progress,
        "reflect_lesson": decision.lesson,
        "reflect_goal_achieved": decision.goal_achieved,
        "reflect_missing_info": decision.missing_info,
        "reflect_judge_source": judge_source,
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
            state.get("experience_context", "No prior experience lessons."),
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
    session_id = state.get("session_id", "default")
    goal = state.get("goal", state.get("message", ""))
    persist_experience = bool(state.get("persist_experience", True))
    replanned = bool(state.get("replanned", False))
    environment_name = state.get("environment_name", "student_support")
    trace_id = state.get("trace_id", "n/a")
    clear_env(trace_id, environment_name)

    if not step_results:
        lesson = build_experience_lesson(
            goal,
            routes=[],
            tools=[],
            success=False,
            replanned=replanned,
            steps_used=0,
        )
        append_experience(
            session_id=session_id,
            goal=goal,
            lesson=lesson,
            routes=[],
            tools=[],
            success=False,
            steps_used=0,
            persist_experience=persist_experience,
        )
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
            "reflect_lesson": lesson,
            "memory_lessons": state.get("memory_lessons", []),
            "memory_hits": state.get("memory_hits", 0),
            "environment_name": environment_name,
            "action_space": state.get("action_space", ["chat", "rag", "tool"]),
            "last_reward": float(state.get("last_reward", 0.0)),
            "total_reward": float(state.get("total_reward", 0.0)),
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
    routes: list[str] = []
    for item in step_results:
        routes.append(item["route"])
        for source in item.get("sources", []):
            if source not in sources:
                sources.append(source)
        retrieved_contexts.extend(item.get("retrieved_contexts", []))
        for tool_name in item.get("used_tools", []):
            if tool_name not in used_tools:
                used_tools.append(tool_name)

    last = step_results[-1]
    steps_used = state.get("steps_used", len(step_results))
    reflect_lesson = str(state.get("reflect_lesson", "")).strip()
    if reflect_lesson and not is_low_value_lesson(reflect_lesson):
        lesson = reflect_lesson
    else:
        lesson = build_experience_lesson(
            goal,
            routes=routes,
            tools=used_tools,
            success=True,
            replanned=replanned,
            steps_used=steps_used,
        )
    append_experience(
        session_id=session_id,
        goal=goal,
        lesson=lesson,
        routes=routes,
        tools=used_tools,
        success=True,
        steps_used=steps_used,
        persist_experience=persist_experience,
    )
    return {
        "answer": answer,
        "route": last["route"],
        "router_reason": last["router_reason"],
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
        "used_tools": used_tools,
        "steps_used": steps_used,
        "tool_calls": state.get("tool_calls", len(used_tools)),
        "reflect_done": True,
        "reflect_next_action": "finish",
        "reflect_progress": 1.0,
        "reflect_lesson": lesson,
        "reflect_goal_achieved": bool(state.get("reflect_goal_achieved", True)),
        "reflect_missing_info": state.get("reflect_missing_info", ""),
        "reflect_judge_source": state.get("reflect_judge_source", "rule_fallback"),
        "memory_lessons": state.get("memory_lessons", []),
        "memory_hits": state.get("memory_hits", 0),
        "environment_name": environment_name,
        "action_space": state.get("action_space", ["chat", "rag", "tool"]),
        "last_reward": float(state.get("last_reward", 0.0)),
        "total_reward": float(state.get("total_reward", 0.0)),
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


async def run_agent_workflow(
    message: str,
    chat_history: str = "",
    trace_id: str = "n/a",
    session_id: str = "default",
    persist_experience: bool = True,
    environment_name: str = "student_support",
) -> AgentState:
    graph = _build_agent_graph()
    result = await graph.ainvoke(
        {
            "message": message,
            "chat_history": chat_history,
            "session_id": session_id,
            "trace_id": trace_id,
            "max_steps": settings.max_plan_steps,
            "step_results": [],
            "current_step": 0,
            "replanned": False,
            "steps_used": 0,
            "tool_calls": 0,
            "memory_lessons": [],
            "memory_hits": 0,
            "experience_context": "No prior experience lessons.",
            "persist_experience": persist_experience,
            "environment_name": environment_name,
            "action_space": ["chat", "rag", "tool"],
            "last_reward": 0.0,
            "total_reward": 0.0,
            "reflect_goal_achieved": False,
            "reflect_missing_info": "",
            "reflect_judge_source": "rule_fallback",
        }
    )
    return result
