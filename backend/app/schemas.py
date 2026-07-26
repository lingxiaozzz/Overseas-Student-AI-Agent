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


class ReflectionInfo(BaseModel):
    done: bool
    next_action: ReflectAction
    progress: float
    lesson: str


class AgentMetrics(BaseModel):
    steps_used: int
    tool_calls: int
    replanned: bool


class AgentChatResponse(RagChatResponse):
    route: Route
    router_reason: str
    used_tools: list[str]
    plan: AgentPlan
    steps: list[PlanStepResult]
    reflection: ReflectionInfo
    metrics: AgentMetrics
