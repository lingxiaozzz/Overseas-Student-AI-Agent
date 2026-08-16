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
    llm: str | None = Field(
        default=None,
        examples=["gemini", "deepseek"],
        description="Chat provider. Leave empty to use LLM_PROVIDER (default: deepseek). Gemini remains optional.",
    )
    model: str | None = Field(
        default=None,
        examples=["gemini-2.5-flash", "deepseek-v4-flash", "deepseek-chat"],
        description="Chat model id. Default DeepSeek is deepseek-v4-flash; also deepseek-chat. Gemini option: gemini-2.5-flash.",
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


class ObservationInfo(BaseModel):
    goal: str = ""
    current_subgoal: str = ""
    step_index: int = 0
    completed_steps: int = 0
    available_actions: list[str] = Field(default_factory=list)
    last_answer_preview: str = ""
    last_reward: float = 0.0


class ActionDecisionInfo(BaseModel):
    action_type: Route = "chat"
    content: str = ""
    reason: str = ""
    source: Literal["llm", "hint_fallback", "rule_fallback"] = "rule_fallback"


class MemoryEvent(BaseModel):
    layer: Literal["working", "long_term", "experience"]
    operation: Literal["read", "write"]
    status: Literal["hit", "miss", "wrote", "updated", "skipped", "deduped"]
    detail: str = ""
    count: int = 0
    items: list[str] = Field(default_factory=list)


class EvaluationInfo(BaseModel):
    passed: bool
    score: float
    feedback: str
    source: Literal["llm", "rule_fallback"] = "rule_fallback"
    triggered_replan: bool = False


class AgentChatResponse(RagChatResponse):
    route: Route
    router_reason: str
    used_tools: list[str]
    plan: AgentPlan
    steps: list[PlanStepResult]
    reflection: ReflectionInfo
    evaluation: EvaluationInfo
    metrics: AgentMetrics
    memory_lessons: list[str] = Field(default_factory=list)
    memory_reads: list[MemoryEvent] = Field(default_factory=list)
    memory_writes: list[MemoryEvent] = Field(default_factory=list)
    long_term_facts: list[str] = Field(default_factory=list)
    environment: EnvironmentInfo
    last_observation: ObservationInfo = Field(default_factory=ObservationInfo)
    last_action_decision: ActionDecisionInfo = Field(default_factory=ActionDecisionInfo)
