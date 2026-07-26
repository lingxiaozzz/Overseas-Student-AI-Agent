from pydantic import BaseModel, Field
from typing import Literal


Route = Literal["chat", "rag", "tool"]
ReflectAction = Literal["continue", "replan", "finish"]


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        examples=["I am a new international student at USYD. What should I prepare before arrival?"],
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        examples=["student-001"],
        description="Conversation session identifier for short-term memory.",
    )


class ChatResponse(BaseModel):
    answer: str


class RetrievedContext(BaseModel):
    rank: int
    source: str
    score: float
    content_preview: str


class RagChatResponse(ChatResponse):
    sources: list[str]
    retrieved_contexts: list[RetrievedContext]


class ToolChatResponse(ChatResponse):
    used_tools: list[str]


class AgentPlan(BaseModel):
    goal: str
    subgoals: list[str]


class PlanStepResult(BaseModel):
    step_index: int
    subgoal: str
    route: Route
    router_reason: str
    answer_preview: str
    used_tools: list[str]
    reward: float = 0.0
    action_type: str = "chat"


class ReflectionInfo(BaseModel):
    done: bool
    next_action: ReflectAction
    progress: float
    lesson: str
    goal_achieved: bool = False
    missing_info: str = ""
    judge_source: Literal["llm", "rule_fallback"] = "rule_fallback"


class AgentMetrics(BaseModel):
    steps_used: int
    tool_calls: int
    replanned: bool
    memory_hits: int = 0
    last_reward: float = 0.0
    total_reward: float = 0.0


class EnvironmentInfo(BaseModel):
    name: str
    action_space: list[str]


class AgentChatResponse(RagChatResponse):
    route: Route
    router_reason: str
    used_tools: list[str]
    plan: AgentPlan
    steps: list[PlanStepResult]
    reflection: ReflectionInfo
    metrics: AgentMetrics
    memory_lessons: list[str] = Field(default_factory=list)
    environment: EnvironmentInfo
