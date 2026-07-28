from functools import lru_cache
from typing import Any, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.chat_service import MissingApiKeyError
from app.config import settings
from app.environment import Action, Observation, clear_env, get_or_create_env, reset_env
from app.evaluator_service import compose_candidate_answer, llm_evaluate, rule_evaluate
from app.logging_service import get_logger
from app.memory_service import (
    build_experience_lesson,
    extract_long_term_candidates,
    is_low_value_lesson,
    read_experience_memory,
    read_long_term_memory,
    read_working_memory,
    upsert_long_term_facts,
    write_experience_memory,
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


async def _llm_plan(
    message: str,
    chat_history: str,
    experience_context: str,
    long_term_context: str,
) -> PlanDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT),
            (
                "human",
                "Conversation history (working memory):\n{chat_history}\n\n"
                "Long-term profile facts:\n{long_term_context}\n\n"
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
                "long_term_context": long_term_context,
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

    # Keep trivial context-only turns as single-step plans.
    if _is_context_only_message(message):
        goal = message
        subgoals = [message]
        logger.trace(f"trace_id={trace_id} plan_mode=single_context subgoals=1")
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
            "reflect_lesson": "Single-step context acknowledgment plan.",
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
    step_results: list[StepRecord],
    experience_context: str,
) -> NextActionDecision:
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    completed = "\n".join(
        f"- step {item.get('step_index')}: [{item.get('route')}] {_preview(str(item.get('subgoal', '')), 120)} "
        f"-> {_preview(str(item.get('answer', '')), 160)}"
        for item in step_results
    ) or "- none"
    unused_hints = hints[len(step_results) :]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ACTOR_PROMPT),
            (
                "human",
                "Overall goal:\n{goal}\n\n"
                "Conversation history:\n{chat_history}\n\n"
                "Prior experience lessons:\n{experience_context}\n\n"
                "Soft plan hints:\n{hints}\n\n"
                "Unused hints:\n{unused_hints}\n\n"
                "Completed steps:\n{completed}\n\n"
                "Current observation:\n"
                "- step_index: {step_index}\n"
                "- completed_steps: {completed_steps}\n"
                "- available_actions: {available_actions}\n"
                "- last_answer_preview: {last_answer_preview}\n"
                "- last_reward: {last_reward}\n"
                "- last_tools: {last_tools}\n",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(NextActionDecision)
    chain = prompt | model
    extras = observation.extras or {}
    return await with_retry(
        lambda: chain.ainvoke(
            {
                "goal": goal,
                "chat_history": chat_history or "No prior turns.",
                "experience_context": experience_context or "No prior experience lessons.",
                "hints": "\n".join(f"- {item}" for item in hints) or "- none",
                "unused_hints": "\n".join(f"- {item}" for item in unused_hints) or "- none",
                "completed": completed,
                "step_index": observation.step_index,
                "completed_steps": observation.completed_steps,
                "available_actions": ", ".join(observation.available_actions),
                "last_answer_preview": str(extras.get("last_answer_preview", "")) or "none",
                "last_reward": float(extras.get("last_reward", 0.0) or 0.0),
                "last_tools": ", ".join(extras.get("last_tools", [])) or "none",
            }
        )
    )


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
        decision = await _llm_next_action(
            goal=goal,
            chat_history=chat_history,
            observation=observation,
            hints=hints,
            step_results=step_results,
            experience_context=state.get("experience_context", "No prior experience lessons."),
        )
        content = (decision.content or "").strip() or (
            hints[current_step] if current_step < len(hints) else goal
        )
        original_action_type = decision.action_type
        forced_action_type = _keyword_route(content)
        # If the step-specific hint becomes too generic (and falls back to `chat`),
        # recover the intended routing from the original user message.
        overall_action_type = _keyword_route(state.get("message", goal))
        if forced_action_type == "chat" and overall_action_type != "chat":
            forced_action_type = overall_action_type
        # Hard-guard action type to match router semantics.
        # This prevents experience/memory from drifting context-only/safety requests into tool/rag.
        if forced_action_type != original_action_type:
            action_type = forced_action_type
        else:
            action_type = original_action_type

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

    if current_step >= max_steps or not action_content:
        return {"last_observation": state.get("last_observation", {})}

    env = get_or_create_env(trace_id, environment_name)
    step_result = await env.step(
        Action(type=route, content=action_content, reason=router_reason)
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
        "last_observation": _observation_to_dict(step_result.observation),
        "pending_action_content": "",
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
            steps_used=current_step,
        ),
    )


def _clamp_progress(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _apply_reflect_guards(state: AgentState, decision: ReflectionDecision) -> ReflectionDecision:
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", settings.max_plan_steps)
    replanned = bool(state.get("replanned", False))
    step_results = state.get("step_results", [])
    last_answer = step_results[-1]["answer"] if step_results else ""
    budget_left = _step_budget_remaining(state)

    next_action = decision.next_action
    done = decision.done
    progress = _clamp_progress(decision.progress)

    if current_step >= max_steps:
        next_action = "finish"
        done = True
        progress = 1.0
    elif not last_answer.strip() and not replanned:
        next_action = "replan"
        done = False
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
    if not settings.google_api_key:
        raise MissingApiKeyError("GOOGLE_API_KEY is not set.")

    step_results = state.get("step_results", [])
    latest = step_results[-1] if step_results else {}
    hints = _plan_hints(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REFLECTOR_PROMPT),
            (
                "human",
                "Overall goal:\n{goal}\n\n"
                "Soft plan hints:\n{subgoals}\n\n"
                "Completed steps: {current_step}/{max_steps} (hints={hint_count})\n"
                "Already replanned: {replanned}\n"
                "Latest action content:\n{latest_subgoal}\n"
                "Latest route: {latest_route}\n"
                "Latest tools: {latest_tools}\n"
                "Latest answer preview:\n{latest_answer}\n"
                "Latest observation preview:\n{observation}\n",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    ).with_structured_output(ReflectionDecision)
    chain = prompt | model
    observation = state.get("last_observation", {})
    return await with_retry(
        lambda: chain.ainvoke(
            {
                "goal": state.get("goal", state.get("message", "")),
                "subgoals": "\n".join(f"- {item}" for item in hints) or "- none",
                "current_step": state.get("current_step", 0),
                "max_steps": state.get("max_steps", settings.max_plan_steps),
                "hint_count": len(hints),
                "replanned": bool(state.get("replanned", False)),
                "latest_subgoal": latest.get("subgoal", ""),
                "latest_route": latest.get("route", "unknown"),
                "latest_tools": ", ".join(latest.get("used_tools", [])) or "none",
                "latest_answer": _preview(str(latest.get("answer", "")), 500),
                "observation": (
                    f"completed_steps={observation.get('completed_steps', 0)}; "
                    f"last_reward={observation.get('last_reward', 0.0)}; "
                    f"last_answer_preview={_preview(str(observation.get('last_answer_preview', '')), 200)}"
                ),
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
            state.get("long_term_context", "No long-term profile facts."),
        )
        new_subgoals = _normalize_subgoals(message, plan)[:remaining_budget]
        goal = plan.goal.strip() or state.get("goal", message)
    except Exception:
        new_subgoals = [message]
        goal = state.get("goal", message)
        logger.warning(f"trace_id={trace_id} replan_fallback=single_step")

    completed = state.get("current_step", 0)
    logger.trace(
        f"trace_id={trace_id} replan_new_hints={len(new_subgoals)} completed={completed}"
    )
    return {
        "goal": goal,
        "subgoals": new_subgoals,
        "subgoal_hints": new_subgoals,
        "current_step": 0,
        "replanned": True,
        "reflect_next_action": "continue",
        "reflect_done": False,
        "reflect_lesson": "Replanned soft hints after a stalled Observation→Action step.",
        "step_results": [],
        "steps_used": 0,
        "tool_calls": 0,
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
    triggered_replan = (not decision.passed) and (not already_replanned)
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
        }
    )
    return result
