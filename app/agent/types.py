from dataclasses import dataclass, field

from app.schemas.agent import AgentRunState


@dataclass(frozen=True)
class ProviderSelection:
    provider: str
    model: str | None
    configured: bool
    reason: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    async_task: bool = False
    argument_hint: str | None = None


@dataclass(frozen=True)
class LLMToolDecision:
    action: str
    message: str | None = None
    plan_summary: str | None = None
    tool_name: str | None = None
    tool_arguments: dict = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None


@dataclass
class AgentStepRecord:
    step_id: str
    step_index: int
    kind: str
    status: str
    title: str
    message: str | None = None
    details: dict = field(default_factory=dict)
    tool_name: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    tool_call: ToolCallRecord | None = None


@dataclass
class AgentRunRecord:
    session_id: str
    run_id: str
    user_message: str
    message: str
    final_state: AgentRunState
    parent_run_id: str | None = None
    root_run_id: str | None = None
    trigger_kind: str = "new"
    plan_summary: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    cancellation_requested: bool = False
    provider: str | None = None
    model: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: list[AgentStepRecord] = field(default_factory=list)
    request_payload: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)
