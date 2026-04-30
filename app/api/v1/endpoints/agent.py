from fastapi import APIRouter, Depends, HTTPException

from app.agent.runtime import agent_runtime
from app.agent.sessions import agent_session_store
from app.agent.tools import get_tool_specs
from app.agent.types import AgentRunRecord
from app.core.security import require_api_auth
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentRunResponse,
    AgentSessionResponse,
    AgentSessionSummaryResponse,
    AgentToolCallResponse,
    AgentToolsResponse,
    AgentToolSpecResponse,
)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    dependencies=[Depends(require_api_auth)],
)


def _serialize_run(run: AgentRunRecord) -> AgentRunResponse:
    return AgentRunResponse(
        session_id=run.session_id,
        run_id=run.run_id,
        user_message=run.user_message,
        message=run.message,
        final_state=run.final_state,
        provider=run.provider,
        model=run.model,
        tool_calls=[
            AgentToolCallResponse(
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=tool_call.result,
                error=tool_call.error,
            )
            for tool_call in run.tool_calls
        ],
    )


def _get_session_summary(session_id: str) -> AgentSessionSummaryResponse:
    return agent_runtime.get_session_summary(session_id)


@router.post("/chat", response_model=AgentChatResponse)
def chat(payload: AgentChatRequest) -> AgentChatResponse:
    run = agent_runtime.chat(payload)
    serialized = _serialize_run(run)
    return AgentChatResponse(
        **serialized.model_dump(),
        summary=_get_session_summary(run.session_id),
    )


@router.get("/sessions")
def list_sessions() -> list[dict[str, str | int]]:
    items: list[dict[str, str | int]] = []
    for session_id, runs in reversed(agent_session_store.list_sessions()):
        if not runs:
            items.append(
                {
                    "id": session_id,
                    "messageCount": 0,
                    "preview": "New Chat",
                }
            )
            continue
        first_message = next((run.user_message for run in runs if run.user_message), "New Chat")
        items.append(
            {
                "id": session_id,
                "messageCount": len(runs),
                "preview": first_message,
            }
        )
    return items


@router.get("/runs/{run_id}", response_model=AgentRunResponse)
def get_run(run_id: str) -> AgentRunResponse:
    run = agent_session_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"agent run not found: {run_id}")
    return _serialize_run(run)


@router.get("/sessions/{session_id}", response_model=AgentSessionResponse)
def get_session(session_id: str) -> AgentSessionResponse:
    runs = agent_session_store.list_session_runs(session_id)
    return AgentSessionResponse(
        session_id=session_id,
        runs=[_serialize_run(run) for run in runs],
        summary=_get_session_summary(session_id),
    )


@router.get("/tools", response_model=AgentToolsResponse)
def list_tools() -> AgentToolsResponse:
    items = [
        AgentToolSpecResponse(
            name=tool.name,
            description=tool.description,
            async_task=tool.async_task,
            argument_hint=tool.argument_hint,
        )
        for tool in get_tool_specs()
    ]
    return AgentToolsResponse(total=len(items), items=items)
