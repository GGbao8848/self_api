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
    tool_name: str | None = None
    tool_arguments: dict = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None


@dataclass
class AgentRunRecord:
    session_id: str
    run_id: str
    message: str
    final_state: AgentRunState
    provider: str | None = None
    model: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
