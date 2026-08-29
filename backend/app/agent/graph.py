import asyncio
from functools import lru_cache
from time import monotonic
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agent.environment import Action, Observation, clear_env, get_or_create_env, reset_env
from app.api.schemas import RetrievedContext
from app.core.config import settings
from app.core.llm import create_chat_model
from app.core.logging import get_logger
from app.core.prompts import cache_friendly_messages
from app.core.retry import with_retry
from app.evaluation.service import compose_candidate_answer, llm_evaluate, rule_evaluate
from app.memory.service import (
    build_experience_lesson,
    extract_long_term_candidates,
    is_low_value_lesson,
    read_experience_memory,
    read_long_term_memory,
    read_working_memory,
    upsert_long_term_facts,
    write_experience_memory,
)

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
Propose soft subgoal hints for the actor. The actor will re-decide each step from fresh observations.

Rules:
1) Simple single-intent questions should produce exactly 1 hint (copy/refine the user ask).
2) Complex multi-part goals may produce 2-4 ordered hints.
3) Keep each hint concrete and independently actionable.
4) Do not invent unrelated tasks.
5) Prefer retrieval before generation when policy/onboarding facts are needed.
6) If prior experience lessons are provided, reuse useful strategy patterns and avoid previously failed approaches.
7) If long-term profile facts are provided, respect student constraints (budget, campus, housing, etc.).
8) Hints are guidance only; the actor may adapt, skip, or replace them based on observations.

Return strict JSON with:
- goal: short restatement of the overall goal
- subgoals: ordered list of 1-4 soft hint strings"""

ACTOR_PROMPT = """You are an Observation→Action policy for an international student assistant.
Given the latest observation, choose exactly ONE next action.

Available action types:
- rag: need official/onboarding/policy facts from the knowledge base
- tool: need calculation/lookup tools (budget, weather, exchange, etc.)
- chat: conversational acknowledgment or synthesis without retrieval/tools

Rules:
1) content must be a concrete executable instruction/question for this single step.
2) Prefer unused plan hints when they still help the goal.
3) Do not repeat a completed step unless the previous result failed or was empty.
4) Do not pack multiple intents into one action.
5) Respect available_actions from the observation.

Return strict JSON with:
- action_type: one of chat, rag, tool
- content: executable string for this step
- reason: one short sentence"""

REFLECTOR_PROMPT = """You are a reflection judge for a hierarchical international-student agent.
Evaluate whether the latest execution step advances the overall goal.

Choose next_action:
- continue: more work is still needed; the actor will observe again and choose the next action
- replan: current approach is stalled/wrong and a new remaining plan is needed
- finish: overall goal is sufficiently satisfied or no useful work remains

Rules:
1) Prefer finish when the user goal is adequately answered.
2) Prefer continue when unused plan hints remain or the latest answer is incomplete for a multi-part goal.
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


class NextActionDecision(BaseModel):
    action_type: Route
    content: str
    reason: str = ""


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
    observation_step_index: int
    action_source: str
    replan_round: int


class AgentState(TypedDict, total=False):
    message: str
    chat_history: str
    session_id: str
    trace_id: str
    goal: str
    subgoals: list[str]
    subgoal_hints: list[str]
    current_step: int
    max_steps: int
    max_agent_steps: int
    max_tool_calls: int
    max_agent_runtime_seconds: float
    started_at_monotonic: float
    elapsed_ms: int
    budget_stop_reason: str
    step_results: list[StepRecord]
    route: Route
    router_reason: str
    pending_action_content: str
    action_decision_source: str
    last_observation: dict[str, Any]
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
    long_term_context: str
    long_term_facts: list[str]
    memory_lessons: list[str]
    memory_hits: int
    memory_reads: list[dict]
    memory_writes: list[dict]
    persist_experience: bool
    environment_name: str
    action_space: list[str]
    last_reward: float
    total_reward: float
    reflect_goal_achieved: bool
    reflect_missing_info: str
    reflect_judge_source: str
    evaluation_passed: bool
    evaluation_score: float
    evaluation_feedback: str
    evaluation_source: str
    evaluation_triggered_replan: bool
    evaluation_next_action: Literal["finalize", "replan"]
    replan_round: int


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
    "hello",
    "hi,",
    "hi ",
    "hey",
    "thanks",
    "thank you",
)

BACKGROUND_INTRO_MARKERS = (
    "international student",
    "new usyd",
    "starting at usyd",
    "study at usyd",
    "will study at usyd",
)

PURE_CHAT_KEYWORDS = {
    "homesick",
    "lonely",
    "loneliness",
    "feel stressed",
    "stressed about",
    "coping tips",
    "coping strategies",
    "making friends",
    "feel lonely",
    "emotional support",
    "anxious",
    "anxiety",
    "just arrived",
    "new here",
    "new city",
}

PURE_CHAT_PHRASES = (
    "any advice",
    "any coping",
    "what can i do",
    "how can i make friends",
    "how can i cope",
    "any coping tips",
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
    if any(message_lower.startswith(prefix) for prefix in BACKGROUND_ONLY_PREFIXES):
        return True
    return any(marker in message_lower for marker in BACKGROUND_INTRO_MARKERS)


def _is_pure_chat_message(message: str) -> bool:
    message_lower = message.strip().lower()
    if not message_lower:
        return False
    if _should_prefer_rag_for_onboarding_knowledge(message):
        return False
    if _requires_budget_tool(message) or _requires_checklist_tool(message):
        return False
    if _should_force_rag_for_safety(message):
        return False
    if _is_context_only_message(message):
        return True
    if message_lower in {"hello", "hi", "hey", "thanks", "thank you", "uh help me pls", "??"}:
        return True
    if any(keyword in message_lower for keyword in PURE_CHAT_KEYWORDS):
        return True
    if any(phrase in message_lower for phrase in PURE_CHAT_PHRASES):
        policy_tokens = RAG_HINT_KEYWORDS | {"budget", "checklist", "calculate", "estimate"}
        if not any(token in message_lower for token in policy_tokens):
            return True
    if "?" in message_lower:
        emotional_tokens = ("feel", "lonely", "homesick", "stress", "anxious", "friend", "coping")
        policy_tokens = RAG_HINT_KEYWORDS | {"budget", "checklist", "calculate", "estimate"}
        has_emotional = any(token in message_lower for token in emotional_tokens)
        has_policy = any(token in message_lower for token in policy_tokens)
        if has_emotional and not has_policy:
            return True
    return False


def _should_force_rag_for_safety(message: str) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in SAFETY_HINT_KEYWORDS)


def _should_prefer_rag_for_onboarding_knowledge(message: str) -> bool:
    """Prefer retrieval for mixed onboarding questions before calculation tools.

    A request such as "prepare before arrival and estimate rent" needs factual
    arrival guidance first. A pure calculation like "estimate my weekly budget"
    should continue to use the tool route.
    """
    message_lower = message.strip().lower()
    onboarding_markers = (
        "arrival",
        "arrive",
        "settle",
        "settling",
        "orientation",
        "accommodation",
        "housing",
    )
    knowledge_request_markers = (
        "prepare",
        "plan",
        "settle",
        "what should",
        "how do",
        "first week",
        "first month",
    )
    return any(marker in message_lower for marker in onboarding_markers) and any(
        marker in message_lower for marker in knowledge_request_markers
    )


def _should_prefer_rag_for_ambiguous_plan(message: str) -> bool:
    if _requires_budget_tool(message) or _requires_checklist_tool(message):
        return False
    message_lower = message.lower()
    has_plan_intent = "plan" in message_lower or "first month" in message_lower
    has_student_context = "student" in message_lower or "sydney" in message_lower or "usyd" in message_lower
    return has_plan_intent and has_student_context


def _requires_checklist_tool(message: str) -> bool:
    message_lower = message.strip().lower()
    if not message_lower or _is_context_only_message(message):
        return False
    has_checklist_signal = "checklist" in message_lower
    has_build_signal = any(
        phrase in message_lower
        for phrase in ("build a", "build my", "create a", "generate a", "make a")
    ) and has_checklist_signal
    has_request_signal = ("?" in message_lower) or any(
        hint in message_lower for hint in TASK_REQUEST_HINTS
    )
    return (has_checklist_signal or has_build_signal) and has_request_signal


def _keyword_route(message: str) -> Route:
    message_lower = message.lower()
    if _should_force_rag_for_safety(message):
        return "rag"
    if _should_prefer_rag_for_onboarding_knowledge(message):
        return "rag"
    if _is_context_only_message(message) or _is_pure_chat_message(message):
        return "chat"
    if _requires_checklist_tool(message) or _requires_budget_tool(message):
        return "tool"
    if _should_prefer_rag_for_ambiguous_plan(message):
        return "rag"
    if any(keyword in message_lower for keyword in TOOL_HINT_KEYWORDS):
        return "tool"
    if any(keyword in message_lower for keyword in RAG_HINT_KEYWORDS):
        return "rag"
    return "chat"


def _requires_budget_tool(message: str) -> bool:
    message_lower = message.strip().lower()
    if not message_lower or _is_context_only_message(message):
        return False
    has_budget_intent = any(
        token in message_lower
        for token in ("budget", "cost", "calculate", "estimate", "how much")
    )
    has_budget_signal = has_budget_intent or "rent" in message_lower
    has_request_signal = ("?" in message_lower) or any(
        hint in message_lower for hint in TASK_REQUEST_HINTS
    )
    return has_budget_signal and has_request_signal


def _preview(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _observation_to_dict(observation: Observation) -> dict[str, Any]:
    extras = observation.extras or {}
    return {
        "goal": observation.goal,
        "current_subgoal": observation.current_subgoal,
        "step_index": observation.step_index,
        "completed_steps": observation.completed_steps,
        "available_actions": list(observation.available_actions),
        "last_answer_preview": str(extras.get("last_answer_preview", "")),
        "last_reward": float(extras.get("last_reward", 0.0) or 0.0),
        "last_tools": list(extras.get("last_tools", [])),
        "last_sources": list(extras.get("last_sources", [])),
    }


def _plan_hints(state: AgentState) -> list[str]:
    hints = state.get("subgoal_hints") or state.get("subgoals") or []
    return [item for item in hints if item and str(item).strip()]


def _step_budget_remaining(state: AgentState) -> int:
    return max(
        int(state.get("max_steps", settings.max_plan_steps)) - int(state.get("current_step", 0)),
        0,
    )


def _elapsed_ms(state: AgentState) -> int:
    started_at = float(state.get("started_at_monotonic", monotonic()))
    return max(int((monotonic() - started_at) * 1000), 0)


def _runtime_seconds_remaining(state: AgentState) -> float:
    budget = float(state.get("max_agent_runtime_seconds", settings.max_agent_runtime_seconds))
    return budget - (_elapsed_ms(state) / 1000)


def _execution_budget_stop_reason(state: AgentState) -> str:
    if state.get("budget_stop_reason"):
        return str(state["budget_stop_reason"])
    if _runtime_seconds_remaining(state) <= 0:
        return "runtime_budget_exhausted"
    if int(state.get("steps_used", 0)) >= int(
        state.get("max_agent_steps", settings.max_agent_steps)
    ):
        return "step_budget_exhausted"
    return ""


async def _llm_route(message: str, chat_history: str) -> RouterDecision:
    model = create_chat_model(temperature=0).with_structured_output(RouterDecision)
    messages = cache_friendly_messages(ROUTER_PROMPT, chat_history, message)
    return await with_retry(lambda: model.ainvoke(messages))


async def _decide_route(message: str, chat_history: str, trace_id: str) -> RouterDecision:
    if _should_force_rag_for_safety(message):
        reason = "Safety-sensitive or policy-bypass request detected; using rag for compliant guidance."
        logger.trace(f"trace_id={trace_id} route_decision=rag reason={reason}")
        return RouterDecision(route="rag", reason=reason)
    if _should_prefer_rag_for_onboarding_knowledge(message):
        reason = "Onboarding knowledge request detected; applying retrieval-first (rag) strategy."
        logger.trace(f"trace_id={trace_id} route_decision=rag reason={reason}")
        return RouterDecision(route="rag", reason=reason)
    if _is_context_only_message(message) or _is_pure_chat_message(message):
        reason = "Conversational or background-only input; using chat without retrieval/tools."
        logger.trace(f"trace_id={trace_id} route_decision=chat reason={reason}")
        return RouterDecision(route="chat", reason=reason)
    if _requires_checklist_tool(message):
        reason = "Explicit checklist generation request; using tool."
        logger.trace(f"trace_id={trace_id} route_decision=tool reason={reason}")
        return RouterDecision(route="tool", reason=reason)
    if _requires_budget_tool(message):
        reason = "Explicit budget calculation request; using tool."
        logger.trace(f"trace_id={trace_id} route_decision=tool reason={reason}")
        return RouterDecision(route="tool", reason=reason)
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


async def _llm_plan(
    message: str,
    chat_history: str,
    experience_context: str,
    long_term_context: str,
) -> PlanDecision:
    model = create_chat_model(temperature=0).with_structured_output(PlanDecision)
    messages = cache_friendly_messages(
        PLANNER_PROMPT,
        chat_history,
        (
            "Long-term profile facts:\n"
            f"{long_term_context}\n\n"
            "Prior experience lessons:\n"
            f"{experience_context}\n\n"
            "User goal:\n"
            f"{message}"
        ),
    )
    return await with_retry(lambda: model.ainvoke(messages))


def _normalize_subgoals(message: str, plan: PlanDecision) -> list[str]:
    cleaned = [item.strip() for item in plan.subgoals if item and item.strip()]
    if not cleaned:
        cleaned = [message]
    # Keep planning bounded for latency and eval stability.
    return cleaned[: settings.max_plan_steps]


async def _plan_node(state: AgentState) -> AgentState:
    message = state["message"]
    session_id = state.get("session_id", "default")
    trace_id = state.get("trace_id", "n/a")
    max_steps = state.get("max_steps", settings.max_plan_steps)
    environment_name = state.get("environment_name", "student_support")

    memory_reads: list[dict] = list(state.get("memory_reads", []))
    memory_writes: list[dict] = list(state.get("memory_writes", []))

    chat_history, working_read = read_working_memory(session_id)
    # Prefer explicitly provided history when callers inject it (tests/tools).
    if state.get("chat_history") and state.get("chat_history") != "No prior conversation history.":
        chat_history = state["chat_history"]
    memory_reads.append(working_read)

    _long_term_records, long_term_context, long_term_read = read_long_term_memory(
        session_id, message
    )
    memory_reads.append(long_term_read)
    long_term_facts = [
        str(item.get("fact", "")).strip()
        for item in _long_term_records
        if str(item.get("fact", "")).strip()
    ]

    experience_records, experience_context, experience_read = read_experience_memory(
        message, session_id=session_id
    )
    memory_reads.append(experience_read)
    memory_lessons = [
        str(item.get("lesson", "")).strip()
        for item in experience_records
        if str(item.get("lesson", "")).strip()
    ]
    memory_hits = sum(int(event.get("count", 0)) for event in memory_reads if event.get("status") == "hit")
    logger.trace(
        f"trace_id={trace_id} memory_hits={memory_hits} "
        f"reads={[(e.get('layer'), e.get('status'), e.get('count')) for e in memory_reads]} "
        f"session_id={session_id}"
    )

    observation = reset_env(
        trace_id=trace_id,
        goal=message,
        chat_history=chat_history,
        environment_name=environment_name,
    )
    action_space = observation.available_actions

    memory_payload = {
        "chat_history": chat_history,
        "experience_context": experience_context,
        "long_term_context": long_term_context,
        "long_term_facts": long_term_facts,
        "memory_lessons": memory_lessons,
        "memory_hits": memory_hits,
        "memory_reads": memory_reads,
        "memory_writes": memory_writes,
    }

    # Keep trivial context-only and pure-chat turns as single-step plans.
    if _is_context_only_message(message) or _is_pure_chat_message(message):
        goal = message
        subgoals = [message]
        plan_mode = "single_context" if _is_context_only_message(message) else "pure_chat"
        logger.trace(f"trace_id={trace_id} plan_mode={plan_mode} subgoals=1")
        return {
            "goal": goal,
            "subgoals": subgoals,
            "subgoal_hints": subgoals,
            "current_step": 0,
            "max_steps": max_steps,
            "step_results": [],
            "replanned": False,
            "steps_used": 0,
            "tool_calls": 0,
            "pending_action_content": "",
            "action_decision_source": "hint_fallback",
            "last_observation": _observation_to_dict(observation),
            "reflect_done": False,
            "reflect_next_action": "continue",
            "reflect_progress": 0.0,
            "reflect_lesson": "Single-step conversational plan.",
            "environment_name": environment_name,
            "action_space": action_space,
            "last_reward": 0.0,
            "total_reward": 0.0,
            **memory_payload,
        }

    # For pure chat requests, avoid multi-subgoal planning to reduce step overrun.
    # This also aligns with routing rules: when keyword routing says `chat`, we keep a 1-step plan.
    if _keyword_route(message) == "chat":
        goal = message
        subgoals = [message]
        logger.trace(f"trace_id={trace_id} plan_mode=keyword_chat subgoals=1")
    else:
        try:
            plan = await _llm_plan(message, chat_history, experience_context, long_term_context)
            goal = plan.goal.strip() or message
            subgoals = _normalize_subgoals(message, plan)
        except Exception:
            goal = message
            subgoals = [message]
            logger.warning(
                f"trace_id={trace_id} plan_fallback=single_step reason=llm_planner_unavailable"
            )

    # Keep environment goal aligned with planner restatement.
    observation = reset_env(
        trace_id=trace_id,
        goal=goal,
        chat_history=chat_history,
        environment_name=environment_name,
    )
    logger.trace(
        f"trace_id={trace_id} plan_goal={_preview(goal, 80)} subgoal_hints={len(subgoals)} "
        f"env={environment_name} actions={action_space}"
    )
    return {
        "goal": goal,
        "subgoals": subgoals,
        "subgoal_hints": subgoals,
        "current_step": 0,
        "max_steps": max_steps,
        "step_results": [],
        "replanned": False,
        "steps_used": 0,
        "tool_calls": 0,
        "pending_action_content": "",
        "action_decision_source": "hint_fallback",
        "last_observation": _observation_to_dict(observation),
        "reflect_done": False,
        "reflect_next_action": "continue",
        "reflect_progress": 0.0,
        "reflect_lesson": "Initial soft plan hints created for Observation→Action loop.",
        "environment_name": environment_name,
        "action_space": action_space,
        "last_reward": 0.0,
        "total_reward": 0.0,
        **memory_payload,
    }


async def _llm_next_action(
    *,
    goal: str,
    chat_history: str,
    observation: Observation,
    hints: list[str],
    hint_index: int,
    step_results: list[StepRecord],
    experience_context: str,
) -> NextActionDecision:
    completed = "\n".join(
        f"- step {item.get('step_index')}: [{item.get('route')}] {_preview(str(item.get('subgoal', '')), 120)} "
        f"-> {_preview(str(item.get('answer', '')), 160)}"
        for item in step_results
    ) or "- none"
    unused_hints = hints[hint_index:]
    model = create_chat_model(temperature=0).with_structured_output(NextActionDecision)
    extras = observation.extras or {}
    messages = cache_friendly_messages(
        ACTOR_PROMPT,
        chat_history or "",
        (
            f"Overall goal:\n{goal}\n\n"
            "Prior experience lessons:\n"
            f"{experience_context or 'No prior experience lessons.'}\n\n"
            "Soft plan hints:\n"
            f"{chr(10).join(f'- {item}' for item in hints) or '- none'}\n\n"
            "Unused hints:\n"
            f"{chr(10).join(f'- {item}' for item in unused_hints) or '- none'}\n\n"
            "Completed steps:\n"
            f"{completed}\n\n"
            "Current observation:\n"
            f"- step_index: {observation.step_index}\n"
            f"- completed_steps: {observation.completed_steps}\n"
            f"- available_actions: {', '.join(observation.available_actions)}\n"
            f"- last_answer_preview: {str(extras.get('last_answer_preview', '')) or 'none'}\n"
            f"- last_reward: {float(extras.get('last_reward', 0.0) or 0.0)}\n"
            f"- last_tools: {', '.join(extras.get('last_tools', [])) or 'none'}\n"
        ),
    )
    return await with_retry(lambda: model.ainvoke(messages))


async def _fallback_next_action(
    state: AgentState,
    observation: Observation,
    *,
    source: str = "hint_fallback",
) -> tuple[NextActionDecision, str]:
    hints = _plan_hints(state)
    current_step = int(state.get("current_step", 0))
    goal = state.get("goal", state.get("message", ""))
    content = hints[current_step] if current_step < len(hints) else goal
    decision = await _decide_route(content, state.get("chat_history", ""), state.get("trace_id", "n/a"))
    return (
        NextActionDecision(
            action_type=decision.route,
            content=content,
            reason=f"{source}: {decision.reason}",
        ),
        source,
    )


async def _act_node(state: AgentState) -> AgentState:
    chat_history = state.get("chat_history", "")
    trace_id = state.get("trace_id", "n/a")
    environment_name = state.get("environment_name", "student_support")
    current_step = int(state.get("current_step", 0))
    max_steps = int(state.get("max_steps", settings.max_plan_steps))
    goal = state.get("goal", state.get("message", ""))
    hints = _plan_hints(state)
    step_results = list(state.get("step_results", []))

    env = get_or_create_env(trace_id, environment_name)
    observation = env.observe()
    observation_payload = _observation_to_dict(observation)
    budget_stop_reason = _execution_budget_stop_reason(state)

    if budget_stop_reason:
        logger.warning(f"trace_id={trace_id} act_skip={budget_stop_reason}")
        return {
            "route": state.get("route", "chat"),
            "router_reason": f"Agent execution stopped: {budget_stop_reason}.",
            "pending_action_content": "",
            "action_decision_source": "rule_fallback",
            "last_observation": observation_payload,
            "action_space": observation.available_actions,
            "budget_stop_reason": budget_stop_reason,
            "elapsed_ms": _elapsed_ms(state),
        }

    if current_step >= max_steps:
        logger.trace(f"trace_id={trace_id} act_skip=max_steps current_step={current_step}")
        return {
            "route": state.get("route", "chat"),
            "router_reason": "Max steps reached; no further action selected.",
            "pending_action_content": "",
            "action_decision_source": "rule_fallback",
            "last_observation": observation_payload,
            "action_space": observation.available_actions,
        }

    source = "llm"
    try:
        user_message = state.get("message", goal)
        actor_experience_context = state.get("experience_context", "No prior experience lessons.")
        if (
            _is_context_only_message(user_message)
            or _is_pure_chat_message(user_message)
            or _should_force_rag_for_safety(user_message)
        ):
            actor_experience_context = "No prior experience lessons."

        decision = await _llm_next_action(
            goal=goal,
            chat_history=chat_history,
            observation=observation,
            hints=hints,
            hint_index=current_step,
            step_results=step_results,
            experience_context=actor_experience_context,
        )
        content = (decision.content or "").strip() or (
            hints[current_step] if current_step < len(hints) else goal
        )
        original_action_type = decision.action_type
        forced_action_type = original_action_type
        # Hard guards (order matters):
        # 1) Pure chat / context-only inputs stay chat.
        # 2) Safety-sensitive and onboarding-knowledge inputs stay rag.
        # 3) Explicit checklist / budget requests stay tool.
        # 4) Remaining ambiguous onboarding plans prefer rag.
        if _is_pure_chat_message(user_message):
            forced_action_type = "chat"
        elif current_step == 0 and _is_context_only_message(user_message):
            forced_action_type = "chat"
        elif _should_force_rag_for_safety(user_message):
            forced_action_type = "rag"
        elif _should_prefer_rag_for_onboarding_knowledge(user_message):
            forced_action_type = "rag"
        elif _requires_checklist_tool(user_message):
            forced_action_type = "tool"
        elif _requires_budget_tool(user_message):
            forced_action_type = "tool"
        elif _should_prefer_rag_for_ambiguous_plan(user_message):
            forced_action_type = "rag"

        action_type = forced_action_type

        if action_type not in observation.available_actions:
            routed = await _decide_route(content, chat_history, trace_id)
            action_type = routed.route
            reason = f"Corrected invalid action_type; {routed.reason}"
        else:
            reason = decision.reason.strip() or "Observation→Action policy selected next step."
            if forced_action_type != original_action_type:
                reason = f"action_forced_by_rules (route={forced_action_type}); {reason}"

        decision = NextActionDecision(action_type=action_type, content=content, reason=reason)
    except Exception:
        decision, source = await _fallback_next_action(state, observation, source="hint_fallback")
        logger.warning(f"trace_id={trace_id} act_fallback={source} reason=llm_actor_unavailable")

    if (
        decision.action_type == "tool"
        and int(state.get("tool_calls", 0)) >= int(state.get("max_tool_calls", settings.max_tool_calls))
    ):
        logger.warning(f"trace_id={trace_id} act_skip=tool_budget_exhausted")
        return {
            "route": decision.action_type,
            "router_reason": "Agent execution stopped: tool_budget_exhausted.",
            "pending_action_content": "",
            "action_decision_source": "rule_fallback",
            "last_observation": observation_payload,
            "action_space": observation.available_actions,
            "budget_stop_reason": "tool_budget_exhausted",
            "elapsed_ms": _elapsed_ms(state),
        }

    logger.trace(
        f"trace_id={trace_id} act_step={current_step + 1} route={decision.action_type} "
        f"source={source} content={_preview(decision.content, 80)}"
    )
    return {
        "route": decision.action_type,
        "router_reason": decision.reason,
        "pending_action_content": decision.content,
        "action_decision_source": source,
        "last_observation": observation_payload,
        "action_space": observation.available_actions,
    }


async def _execute_node(state: AgentState) -> AgentState:
    current_step = state.get("current_step", 0)
    route = state.get("route", "chat")
    router_reason = state.get("router_reason", "")
    trace_id = state.get("trace_id", "n/a")
    environment_name = state.get("environment_name", "student_support")
    max_steps = int(state.get("max_steps", settings.max_plan_steps))
    action_content = (state.get("pending_action_content") or "").strip()

    budget_stop_reason = _execution_budget_stop_reason(state)
    if budget_stop_reason or current_step >= max_steps or not action_content:
        return {
            "last_observation": state.get("last_observation", {}),
            "budget_stop_reason": budget_stop_reason,
            "elapsed_ms": _elapsed_ms(state),
        }

    if route == "tool" and int(state.get("tool_calls", 0)) >= int(
        state.get("max_tool_calls", settings.max_tool_calls)
    ):
        return {
            "last_observation": state.get("last_observation", {}),
            "budget_stop_reason": "tool_budget_exhausted",
            "elapsed_ms": _elapsed_ms(state),
        }

    env = get_or_create_env(trace_id, environment_name)
    try:
        remaining_tool_calls = max(
            int(state.get("max_tool_calls", settings.max_tool_calls)) - int(state.get("tool_calls", 0)),
            0,
        )
        step_result = await asyncio.wait_for(
            env.step(
                Action(
                    type=route,
                    content=action_content,
                    reason=router_reason,
                    tool_call_limit=remaining_tool_calls if route == "tool" else None,
                )
            ),
            timeout=max(_runtime_seconds_remaining(state), 0.001),
        )
    except TimeoutError:
        logger.warning(f"trace_id={trace_id} execute_timeout=runtime_budget_exhausted")
        return {
            "last_observation": state.get("last_observation", {}),
            "pending_action_content": "",
            "budget_stop_reason": "runtime_budget_exhausted",
            "elapsed_ms": _elapsed_ms(state),
        }
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
            "subgoal": action_content,
            "route": route,
            "router_reason": router_reason,
            "answer": answer,
            "answer_preview": _preview(answer),
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
            "used_tools": used_tools,
            "reward": reward,
            "action_type": step_result.action.type,
            "observation_step_index": int(
                state.get("last_observation", {}).get("step_index", current_step)
            ),
            "action_source": str(state.get("action_decision_source", "rule_fallback")),
            "replan_round": int(state.get("replan_round", 0)),
        }
    )
    steps_used = int(state.get("steps_used", 0)) + 1
    tool_calls = state.get("tool_calls", 0) + len(used_tools)
    total_reward = float(state.get("total_reward", 0.0)) + reward
    logger.trace(
        f"trace_id={trace_id} execute_step={steps_used} env={environment_name} "
        f"action={route} reward={reward:.2f} tools={used_tools} elapsed_ms={_elapsed_ms(state)}"
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
        "last_observation": _observation_to_dict(step_result.observation),
        "pending_action_content": "",
        "elapsed_ms": _elapsed_ms(state),
    }


def _rule_reflect(state: AgentState) -> ReflectionDecision:
    hints = _plan_hints(state)
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", settings.max_plan_steps)
    step_results = state.get("step_results", [])
    replanned = state.get("replanned", False)
    planned_total = max(len(hints), 1)
    progress = min(current_step / planned_total, 1.0)
    last_answer = step_results[-1]["answer"] if step_results else ""
    budget_left = _step_budget_remaining(state)
    total_steps_used = int(state.get("steps_used", current_step))

    if _execution_budget_stop_reason(state):
        return ReflectionDecision(
            done=True,
            next_action="finish",
            progress=1.0,
            goal_achieved=bool(last_answer.strip()),
            missing_info=str(state.get("budget_stop_reason", "")),
            lesson="Execution stopped by an agent budget; return the completed work without another action.",
        )
    if current_step >= max_steps:
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
                steps_used=total_steps_used,
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
    user_message = str(state.get("message", state.get("goal", "")))
    if _is_pure_chat_message(user_message) and last_answer.strip() and current_step >= 1:
        return ReflectionDecision(
            done=True,
            next_action="finish",
            progress=1.0,
            goal_achieved=True,
            missing_info="",
            lesson=build_experience_lesson(
                state.get("goal", state.get("message", "")),
                routes=[item.get("route", "chat") for item in step_results],
                tools=[
                    tool_name
                    for item in step_results
                    for tool_name in item.get("used_tools", [])
                ],
                success=True,
                replanned=bool(replanned),
                steps_used=total_steps_used,
            ),
        )
    if _should_finish_after_successful_step(state):
        return ReflectionDecision(
            done=True,
            next_action="finish",
            progress=1.0,
            goal_achieved=True,
            missing_info="",
            lesson=build_experience_lesson(
                state.get("goal", state.get("message", "")),
                routes=[item.get("route", "chat") for item in step_results],
                tools=[
                    tool_name
                    for item in step_results
                    for tool_name in item.get("used_tools", [])
                ],
                success=True,
                replanned=bool(replanned),
                steps_used=total_steps_used,
            ),
        )
    if current_step < len(hints) and budget_left > 0:
        return ReflectionDecision(
            done=False,
            next_action="continue",
            progress=progress,
            goal_achieved=False,
            missing_info="",
            lesson=f"Continue Observation→Action loop ({current_step}/{len(hints)} hints covered).",
        )
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
            steps_used=total_steps_used,
        ),
    )


def _clamp_progress(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _should_finish_after_successful_step(state: AgentState) -> bool:
    step_results = list(state.get("step_results", []))
    if not step_results:
        return False

    latest = step_results[-1]
    route = str(latest.get("route", "chat"))
    answer = str(latest.get("answer", "")).strip()
    used_tools = list(latest.get("used_tools", []))
    sources = list(latest.get("sources", []))
    all_used_tools = [
        tool_name
        for item in step_results
        for tool_name in list(item.get("used_tools", []))
    ]
    hints = _plan_hints(state)
    current_step = int(state.get("current_step", 0))
    is_single_intent = len(hints) <= 1
    user_message = str(state.get("message", state.get("goal", "")))

    if not answer:
        return False
    if _is_pure_chat_message(user_message):
        return route == "chat" and current_step >= 1
    if _requires_checklist_tool(user_message) and "build_prearrival_checklist" not in all_used_tools:
        return False
    if _requires_budget_tool(user_message) and "estimate_weekly_budget" not in all_used_tools:
        return False
    if route == "tool" and used_tools:
        return True
    if route == "rag" and sources:
        return True
    if route == "chat" and is_single_intent and current_step >= 1:
        return True
    return False


def _apply_reflect_guards(state: AgentState, decision: ReflectionDecision) -> ReflectionDecision:
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", settings.max_plan_steps)
    replanned = bool(state.get("replanned", False))
    step_results = state.get("step_results", [])
    last_answer = step_results[-1]["answer"] if step_results else ""
    budget_left = _step_budget_remaining(state)
    user_message = str(state.get("message", state.get("goal", "")))

    next_action = decision.next_action
    done = decision.done
    progress = _clamp_progress(decision.progress)

    budget_stop_reason = _execution_budget_stop_reason(state)
    if budget_stop_reason:
        next_action = "finish"
        done = True
        progress = 1.0
    elif current_step >= max_steps:
        next_action = "finish"
        done = True
        progress = 1.0
    elif _is_pure_chat_message(user_message) and last_answer.strip() and current_step >= 1:
        next_action = "finish"
        done = True
        progress = 1.0
    elif not last_answer.strip() and not replanned:
        next_action = "replan"
        done = False
    elif next_action == "continue" and _should_finish_after_successful_step(state):
        next_action = "finish"
        done = True
        progress = 1.0
    elif next_action == "replan" and _is_pure_chat_message(user_message) and last_answer.strip():
        next_action = "finish"
        done = True
        progress = 1.0
    elif next_action == "replan" and replanned:
        next_action = "continue" if budget_left > 0 else "finish"
        done = next_action == "finish"
    elif next_action == "continue" and budget_left <= 0:
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
    step_results = state.get("step_results", [])
    latest = step_results[-1] if step_results else {}
    hints = _plan_hints(state)
    observation = state.get("last_observation", {})
    model = create_chat_model(temperature=0).with_structured_output(ReflectionDecision)
    messages = cache_friendly_messages(
        REFLECTOR_PROMPT,
        str(state.get("chat_history", "")),
        (
            f"Overall goal:\n{state.get('goal', state.get('message', ''))}\n\n"
            "Soft plan hints:\n"
            f"{chr(10).join(f'- {item}' for item in hints) or '- none'}\n\n"
            f"Completed steps: {state.get('current_step', 0)}/{state.get('max_steps', settings.max_plan_steps)} "
            f"(hints={len(hints)})\n"
            f"Already replanned: {bool(state.get('replanned', False))}\n"
            f"Latest action content:\n{latest.get('subgoal', '')}\n"
            f"Latest route: {latest.get('route', 'unknown')}\n"
            f"Latest tools: {', '.join(latest.get('used_tools', [])) or 'none'}\n"
            f"Latest answer preview:\n{_preview(str(latest.get('answer', '')), 500)}\n"
            "Latest observation preview:\n"
            f"completed_steps={observation.get('completed_steps', 0)}; "
            f"last_reward={observation.get('last_reward', 0.0)}; "
            f"last_answer_preview={_preview(str(observation.get('last_answer_preview', '')), 200)}\n"
        ),
    )
    return await with_retry(lambda: model.ainvoke(messages))


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
            state.get("long_term_context", "No long-term profile facts."),
        )
        new_subgoals = _normalize_subgoals(message, plan)[:remaining_budget]
        goal = plan.goal.strip() or state.get("goal", message)
    except Exception:
        new_subgoals = [message]
        goal = state.get("goal", message)
        logger.warning(f"trace_id={trace_id} replan_fallback=single_step")

    completed = state.get("current_step", 0)
    next_replan_round = int(state.get("replan_round", 0)) + 1
    logger.trace(
        f"trace_id={trace_id} replan_new_hints={len(new_subgoals)} completed={completed} "
        f"replan_round={next_replan_round}"
    )
    return {
        "goal": goal,
        "subgoals": new_subgoals,
        "subgoal_hints": new_subgoals,
        "current_step": 0,
        "replanned": True,
        "replan_round": next_replan_round,
        "reflect_next_action": "continue",
        "reflect_done": False,
        "reflect_lesson": "Replanned soft hints after a stalled Observation→Action step.",
        "pending_action_content": "",
        "action_decision_source": "hint_fallback",
    }


async def _finalize_node(state: AgentState) -> AgentState:
    step_results = state.get("step_results", [])
    session_id = state.get("session_id", "default")
    goal = state.get("goal", state.get("message", ""))
    message = state.get("message", goal)
    persist_experience = bool(state.get("persist_experience", True))
    replanned = bool(state.get("replanned", False))
    environment_name = state.get("environment_name", "student_support")
    trace_id = state.get("trace_id", "n/a")
    elapsed_ms = _elapsed_ms(state)
    clear_env(trace_id, environment_name)

    memory_writes: list[dict] = list(state.get("memory_writes", []))

    def _persist_memories(
        *,
        lesson: str,
        routes: list[str],
        tools: list[str],
        success: bool,
        steps_used: int,
    ) -> list[dict]:
        writes = list(memory_writes)
        _record, experience_write = write_experience_memory(
            session_id=session_id,
            goal=goal,
            lesson=lesson,
            routes=routes,
            tools=tools,
            success=success,
            steps_used=steps_used,
            persist_experience=persist_experience,
        )
        writes.append(experience_write)

        candidates = extract_long_term_candidates(message)
        if not candidates and success:
            # Soft fallback: keep a compact goal constraint if it looks durable.
            candidates = extract_long_term_candidates(goal)
        long_term_write = upsert_long_term_facts(
            session_id,
            candidates,
            persist_experience=persist_experience,
            source="finalize",
        )
        writes.append(long_term_write)
        logger.trace(
            f"trace_id={trace_id} memory_writes="
            f"{[(e.get('layer'), e.get('status'), e.get('count')) for e in writes]}"
        )
        return writes

    if not step_results:
        lesson = build_experience_lesson(
            goal,
            routes=[],
            tools=[],
            success=False,
            replanned=replanned,
            steps_used=0,
        )
        writes = _persist_memories(
            lesson=lesson,
            routes=[],
            tools=[],
            success=False,
            steps_used=0,
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
            "memory_reads": state.get("memory_reads", []),
            "memory_writes": writes,
            "long_term_facts": state.get("long_term_facts", []),
            "environment_name": environment_name,
            "action_space": state.get("action_space", ["chat", "rag", "tool"]),
            "last_reward": float(state.get("last_reward", 0.0)),
            "total_reward": float(state.get("total_reward", 0.0)),
            "evaluation_passed": bool(state.get("evaluation_passed", False)),
            "evaluation_score": float(state.get("evaluation_score", 0.0)),
            "evaluation_feedback": state.get("evaluation_feedback", "No evaluation available."),
            "evaluation_source": state.get("evaluation_source", "rule_fallback"),
            "evaluation_triggered_replan": bool(state.get("evaluation_triggered_replan", False)),
            "elapsed_ms": elapsed_ms,
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

    # "route" in API response should represent the primary execution mode of the task,
    # not necessarily the last sub-step's route (to avoid summary-step route drift).
    if used_tools:
        primary_route: Route = "tool"
    elif any(r == "rag" for r in routes) or retrieved_contexts:
        primary_route = "rag"
    else:
        primary_route = "chat"

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
    writes = _persist_memories(
        lesson=lesson,
        routes=routes,
        tools=used_tools,
        success=True,
        steps_used=steps_used,
    )
    return {
        "answer": answer,
        "route": primary_route,
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
        "memory_reads": state.get("memory_reads", []),
        "memory_writes": writes,
        "long_term_facts": state.get("long_term_facts", []),
        "environment_name": environment_name,
        "action_space": state.get("action_space", ["chat", "rag", "tool"]),
        "last_reward": float(state.get("last_reward", 0.0)),
        "total_reward": float(state.get("total_reward", 0.0)),
        "evaluation_passed": bool(state.get("evaluation_passed", True)),
        "evaluation_score": float(state.get("evaluation_score", 0.0)),
        "evaluation_feedback": state.get("evaluation_feedback", "No evaluation available."),
        "evaluation_source": state.get("evaluation_source", "rule_fallback"),
        "evaluation_triggered_replan": bool(state.get("evaluation_triggered_replan", False)),
        "elapsed_ms": elapsed_ms,
    }



def _after_reflect(state: AgentState) -> Literal["act", "replan", "evaluate"]:
    action = state.get("reflect_next_action", "finish")
    if action == "continue":
        return "act"
    if action == "replan" and not state.get("replanned", False):
        return "replan"
    return "evaluate"


async def _evaluate_node(state: AgentState) -> AgentState:
    trace_id = state.get("trace_id", "n/a")
    goal = state.get("goal", state.get("message", ""))
    step_results = state.get("step_results", [])
    candidate = compose_candidate_answer(step_results)
    used_tools: list[str] = []
    sources: list[str] = []
    for item in step_results:
        for tool_name in item.get("used_tools", []):
            if tool_name not in used_tools:
                used_tools.append(tool_name)
        for source in item.get("sources", []):
            if source not in sources:
                sources.append(source)

    plan_summary = (
        f"hints={state.get('subgoal_hints', state.get('subgoals', []))}; "
        f"routes={[item.get('route') for item in step_results]}; "
        f"tools={used_tools}; "
        f"action_sources={[item.get('action_source') for item in step_results]}"
    )

    source = "llm"
    try:
        decision = await llm_evaluate(goal=goal, answer=candidate, plan_summary=plan_summary)
    except Exception:
        source = "rule_fallback"
        decision = rule_evaluate(
            goal=goal,
            answer=candidate,
            used_tools=used_tools,
            sources=sources,
        )
        logger.warning(f"trace_id={trace_id} evaluate_fallback=rule reason=llm_evaluator_unavailable")

    already_replanned = bool(state.get("replanned", False))
    previously_triggered = bool(state.get("evaluation_triggered_replan", False))
    user_message = str(state.get("message", state.get("goal", "")))
    triggered_replan = (
        (not decision.passed)
        and (not already_replanned)
        and not _is_pure_chat_message(user_message)
    )
    next_action: Literal["finalize", "replan"] = "replan" if triggered_replan else "finalize"

    logger.trace(
        f"trace_id={trace_id} evaluate_score={decision.score:.2f} passed={decision.passed} "
        f"source={source} next={next_action}"
    )
    return {
        "answer": candidate,
        "evaluation_passed": decision.passed,
        "evaluation_score": float(decision.score),
        "evaluation_feedback": decision.feedback,
        "evaluation_source": source,
        "evaluation_triggered_replan": previously_triggered or triggered_replan,
        "evaluation_next_action": next_action,
        "replanned": True if triggered_replan else already_replanned,
        "used_tools": used_tools,
        "sources": sources,
    }


def _after_evaluate(state: AgentState) -> Literal["finalize", "replan"]:
    if _execution_budget_stop_reason(state):
        return "finalize"
    if state.get("evaluation_next_action") == "replan":
        return "replan"
    return "finalize"


@lru_cache(maxsize=1)
def _build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan_node)
    graph.add_node("act", _act_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("evaluate", _evaluate_node)
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
            "evaluate": "evaluate",
        },
    )
    graph.add_conditional_edges(
        "evaluate",
        _after_evaluate,
        {
            "finalize": "finalize",
            "replan": "replan",
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
            "max_agent_steps": settings.max_agent_steps,
            "max_tool_calls": settings.max_tool_calls,
            "max_agent_runtime_seconds": settings.max_agent_runtime_seconds,
            "started_at_monotonic": monotonic(),
            "elapsed_ms": 0,
            "budget_stop_reason": "",
            "step_results": [],
            "current_step": 0,
            "replanned": False,
            "steps_used": 0,
            "tool_calls": 0,
            "memory_lessons": [],
            "memory_hits": 0,
            "memory_reads": [],
            "memory_writes": [],
            "long_term_facts": [],
            "long_term_context": "No long-term profile facts.",
            "experience_context": "No prior experience lessons.",
            "persist_experience": persist_experience,
            "environment_name": environment_name,
            "action_space": ["chat", "rag", "tool"],
            "subgoal_hints": [],
            "pending_action_content": "",
            "action_decision_source": "rule_fallback",
            "last_observation": {},
            "last_reward": 0.0,
            "total_reward": 0.0,
            "reflect_goal_achieved": False,
            "reflect_missing_info": "",
            "reflect_judge_source": "rule_fallback",
            "evaluation_passed": False,
            "evaluation_score": 0.0,
            "evaluation_feedback": "",
            "evaluation_source": "rule_fallback",
            "evaluation_triggered_replan": False,
            "evaluation_next_action": "finalize",
            "replan_round": 0,
        }
    )
    return result
