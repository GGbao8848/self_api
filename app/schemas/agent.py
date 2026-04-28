from typing import Literal

from pydantic import BaseModel, Field


AgentRunState = Literal[
    "completed",
    "requires_provider",
    "clarification_required",
    "failed",
]


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    provider: str | None = None
    model: str | None = None
    stream: bool = False
    tool_name: str | None = None
    tool_arguments: dict = Field(default_factory=dict)


class AgentToolSpecResponse(BaseModel):
    name: str
    description: str
    async_task: bool = False


class AgentToolCallResponse(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    result: dict | None = None
    error: str | None = None


class AgentChatResponse(BaseModel):
    session_id: str
    run_id: str
    message: str
    final_state: AgentRunState
    provider: str | None = None
    model: str | None = None
    tool_calls: list[AgentToolCallResponse] = Field(default_factory=list)


class AgentRunResponse(BaseModel):
    session_id: str
    run_id: str
    message: str
    final_state: AgentRunState
    provider: str | None = None
    model: str | None = None
    tool_calls: list[AgentToolCallResponse] = Field(default_factory=list)


class AgentSessionResponse(BaseModel):
    session_id: str
    runs: list[AgentRunResponse] = Field(default_factory=list)


class AgentToolsResponse(BaseModel):
    total: int
    items: list[AgentToolSpecResponse] = Field(default_factory=list)
