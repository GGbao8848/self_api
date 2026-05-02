from typing import Literal

from pydantic import BaseModel, Field


AgentRunState = Literal[
    "accepted",
    "running",
    "waiting_task",
    "interrupted",
    "completed",
    "requires_provider",
    "clarification_required",
    "failed",
    "cancelled",
]


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    provider: str | None = None
    model: str | None = None
    stream: bool = False
    async_run: bool = False
    max_steps: int | None = Field(default=None, ge=1, le=64)
    tool_name: str | None = None
    tool_arguments: dict = Field(default_factory=dict)
    parent_run_id: str | None = None
    trigger_kind: str = "new"


class AgentToolSpecResponse(BaseModel):
    name: str
    description: str
    async_task: bool = False
    argument_hint: str | None = None


class AgentToolCallResponse(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    result: dict | None = None
    error: str | None = None


class AgentStepResponse(BaseModel):
    step_id: str
    step_index: int
    kind: str
    status: str
    title: str
    message: str | None = None
    details: dict = Field(default_factory=dict)
    tool_name: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    tool_call: AgentToolCallResponse | None = None


class AgentChatResponse(BaseModel):
    session_id: str
    run_id: str
    user_message: str | None = None
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
    tool_calls: list[AgentToolCallResponse] = Field(default_factory=list)
    steps: list[AgentStepResponse] = Field(default_factory=list)
    request_payload: dict = Field(default_factory=dict)
    checkpoint: dict = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    session_id: str
    run_id: str
    user_message: str | None = None
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
    tool_calls: list[AgentToolCallResponse] = Field(default_factory=list)
    steps: list[AgentStepResponse] = Field(default_factory=list)
    request_payload: dict = Field(default_factory=dict)
    checkpoint: dict = Field(default_factory=dict)


class AgentSessionResponse(BaseModel):
    session_id: str
    runs: list[AgentRunResponse] = Field(default_factory=list)


class AgentToolsResponse(BaseModel):
    total: int
    items: list[AgentToolSpecResponse] = Field(default_factory=list)


class AgentRunCancelResponse(BaseModel):
    status: str = "ok"
    run: AgentRunResponse


class AgentRunRetryRequest(BaseModel):
    message: str | None = Field(default=None)
    async_run: bool = True
    max_steps: int | None = Field(default=None, ge=1, le=64)


class AgentRunContinueRequest(BaseModel):
    message: str = Field(min_length=1)
    async_run: bool = True
    max_steps: int | None = Field(default=None, ge=1, le=64)


class AgentRunResumeRequest(BaseModel):
    max_steps: int | None = Field(default=None, ge=1, le=64)


class AgentRunEventResponse(BaseModel):
    event: str
    run: AgentRunResponse
