import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agent.runtime import agent_runtime
from app.agent.sessions import agent_session_store
from app.agent.tools import get_tool_specs
from app.agent.types import AgentRunRecord
from app.core.security import require_api_auth
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentRunCancelResponse,
    AgentRunContinueRequest,
    AgentRunEventResponse,
    AgentRunResumeRequest,
    AgentRunRetryRequest,
    AgentRunResponse,
    AgentSessionResponse,
    AgentStepResponse,
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
        parent_run_id=run.parent_run_id,
        root_run_id=run.root_run_id,
        trigger_kind=run.trigger_kind,
        plan_summary=run.plan_summary,
        created_at=run.created_at,
        updated_at=run.updated_at,
        finished_at=run.finished_at,
        cancellation_requested=run.cancellation_requested,
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
        steps=[
            AgentStepResponse(
                step_id=step.step_id,
                step_index=step.step_index,
                kind=step.kind,
                status=step.status,
                title=step.title,
                message=step.message,
                details=step.details,
                tool_name=step.tool_name,
                task_id=step.task_id,
                task_type=step.task_type,
                started_at=step.started_at,
                finished_at=step.finished_at,
                tool_call=AgentToolCallResponse(
                    name=step.tool_call.name,
                    arguments=step.tool_call.arguments,
                    result=step.tool_call.result,
                    error=step.tool_call.error,
                )
                if step.tool_call is not None
                else None,
            )
            for step in run.steps
        ],
        request_payload=run.request_payload,
        checkpoint=run.checkpoint,
    )


@router.post("/chat", response_model=AgentChatResponse)
def chat(payload: AgentChatRequest) -> AgentChatResponse:
    run = agent_runtime.chat(payload)
    serialized = _serialize_run(run)
    return AgentChatResponse(**serialized.model_dump())


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


@router.post("/runs/{run_id}/cancel", response_model=AgentRunCancelResponse)
def cancel_run(run_id: str) -> AgentRunCancelResponse:
    run = agent_runtime.cancel_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"agent run not found: {run_id}")
    return AgentRunCancelResponse(run=_serialize_run(run))


@router.post("/runs/{run_id}/retry", response_model=AgentChatResponse)
def retry_run(run_id: str, payload: AgentRunRetryRequest) -> AgentChatResponse:
    try:
        run = agent_runtime.retry_run(
            run_id,
            message=payload.message,
            async_run=payload.async_run,
            max_steps=payload.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail=f"agent run not found: {run_id}")
    serialized = _serialize_run(run)
    return AgentChatResponse(**serialized.model_dump())


@router.post("/runs/{run_id}/continue", response_model=AgentChatResponse)
def continue_run(run_id: str, payload: AgentRunContinueRequest) -> AgentChatResponse:
    try:
        run = agent_runtime.continue_run(
            run_id,
            message=payload.message,
            async_run=payload.async_run,
            max_steps=payload.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail=f"agent run not found: {run_id}")
    serialized = _serialize_run(run)
    return AgentChatResponse(**serialized.model_dump())


@router.post("/runs/{run_id}/resume", response_model=AgentChatResponse)
def resume_run(run_id: str, payload: AgentRunResumeRequest) -> AgentChatResponse:
    try:
        run = agent_runtime.resume_run(
            run_id,
            max_steps=payload.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail=f"agent run not found: {run_id}")
    serialized = _serialize_run(run)
    return AgentChatResponse(**serialized.model_dump())


@router.get("/sessions/{session_id}", response_model=AgentSessionResponse)
def get_session(session_id: str) -> AgentSessionResponse:
    runs = agent_session_store.list_session_runs(session_id)
    return AgentSessionResponse(
        session_id=session_id,
        runs=[_serialize_run(run) for run in runs],
    )


@router.get("/runs/{run_id}/events")
def stream_run_events(
    run_id: str,
    interval_seconds: float = Query(default=0.25, ge=0.1, le=5.0),
) -> StreamingResponse:
    run = agent_session_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"agent run not found: {run_id}")

    def _event_generator():
        last_signature: str | None = None
        while True:
            current = agent_session_store.get_run(run_id)
            if current is None:
                break
            serialized = _serialize_run(current)
            payload = AgentRunEventResponse(
                event="snapshot",
                run=serialized,
            ).model_dump()
            signature = json.dumps(payload, ensure_ascii=True, sort_keys=True)
            if signature != last_signature:
                yield f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"
                last_signature = signature
            if current.final_state in {
                "completed",
                "failed",
                "cancelled",
                "interrupted",
                "clarification_required",
                "requires_provider",
            }:
                end_payload = AgentRunEventResponse(
                    event="end",
                    run=serialized,
                ).model_dump()
                yield f"event: end\ndata: {json.dumps(end_payload, ensure_ascii=True)}\n\n"
                break
            time.sleep(interval_seconds)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
