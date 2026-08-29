from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

from app.chat_service import generate_chat_response
from app.rag_service import generate_rag_response
from app.schemas import RetrievedContext
from app.tool_service import generate_tool_response

ActionType = Literal["chat", "rag", "tool"]

_ENV_LOCK = Lock()
_ENV_SESSIONS: dict[str, "BaseEnvironment"] = {}


@dataclass
class Observation:
    """Normalized environment observation for hierarchical agents."""

    goal: str
    current_subgoal: str
    chat_history: str
    step_index: int
    completed_steps: int
    available_actions: list[str]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Normalized action issued by the agent policy/router."""

    type: ActionType
    content: str
    reason: str = ""
    tool_call_limit: int | None = None


@dataclass
class StepResult:
    """Gym-style transition returned by environment.step()."""

    observation: Observation
    action: Action
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Observation-Action abstraction for pluggable task environments."""

    name: str = "base"

    @abstractmethod
    def reset(self, goal: str, chat_history: str = "") -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> list[str]:
        raise NotImplementedError


class StudentSupportEnvironment(BaseEnvironment):
    """Student-support environment where actions map to chat/rag/tool backends."""

    name = "student_support"

    def __init__(self) -> None:
        self._goal = ""
        self._chat_history = ""
        self._step_index = 0
        self._completed_steps = 0
        self._current_subgoal = ""
        self._last_info: dict[str, Any] = {}

    def action_space(self) -> list[str]:
        return ["chat", "rag", "tool"]

    def reset(self, goal: str, chat_history: str = "") -> Observation:
        self._goal = goal
        self._chat_history = chat_history
        self._step_index = 0
        self._completed_steps = 0
        self._current_subgoal = goal
        self._last_info = {}
        return self.observe()

    def observe(self) -> Observation:
        last_answer = str(self._last_info.get("answer", "") or "")
        preview = " ".join(last_answer.split())
        if len(preview) > 240:
            preview = preview[:237] + "..."
        return Observation(
            goal=self._goal,
            current_subgoal=self._current_subgoal,
            chat_history=self._chat_history,
            step_index=self._step_index,
            completed_steps=self._completed_steps,
            available_actions=self.action_space(),
            extras={
                "environment": self.name,
                "last_tools": list(self._last_info.get("used_tools", [])),
                "last_sources": list(self._last_info.get("sources", [])),
                "last_reward": float(self._last_info.get("reward", 0.0) or 0.0),
                "last_answer_preview": preview,
            },
        )

    async def step(self, action: Action) -> StepResult:
        self._step_index += 1
        self._current_subgoal = action.content

        sources: list[str] = []
        retrieved_contexts: list[RetrievedContext] = []
        used_tools: list[str] = []

        if action.type == "rag":
            answer, sources, retrieved_contexts = await generate_rag_response(
                action.content,
                chat_history=self._chat_history,
            )
        elif action.type == "tool":
            answer, used_tools = await generate_tool_response(
                action.content,
                chat_history=self._chat_history,
                max_tool_calls=action.tool_call_limit,
            )
        else:
            answer = await generate_chat_response(
                action.content,
                chat_history=self._chat_history,
            )

        self._completed_steps += 1
        reward = 1.0 if answer.strip() else 0.0
        self._last_info = {
            "answer": answer,
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
            "used_tools": used_tools,
            "reward": reward,
        }
        observation = self.observe()
        return StepResult(
            observation=observation,
            action=action,
            reward=reward,
            done=False,
            info=self._last_info,
        )


def create_environment(name: str = "student_support") -> BaseEnvironment:
    environments: dict[str, type[BaseEnvironment]] = {
        StudentSupportEnvironment.name: StudentSupportEnvironment,
    }
    env_cls = environments.get(name)
    if env_cls is None:
        supported = ", ".join(sorted(environments))
        raise ValueError(f"Unknown environment '{name}'. Supported: {supported}")
    return env_cls()


def _session_key(trace_id: str, environment_name: str) -> str:
    return f"{environment_name}:{trace_id}"


def get_or_create_env(trace_id: str, environment_name: str = "student_support") -> BaseEnvironment:
    key = _session_key(trace_id, environment_name)
    with _ENV_LOCK:
        env = _ENV_SESSIONS.get(key)
        if env is None:
            env = create_environment(environment_name)
            _ENV_SESSIONS[key] = env
        return env


def reset_env(
    trace_id: str,
    goal: str,
    chat_history: str = "",
    environment_name: str = "student_support",
) -> Observation:
    env = get_or_create_env(trace_id, environment_name)
    return env.reset(goal=goal, chat_history=chat_history)


def clear_env(trace_id: str, environment_name: str = "student_support") -> None:
    key = _session_key(trace_id, environment_name)
    with _ENV_LOCK:
        _ENV_SESSIONS.pop(key, None)
