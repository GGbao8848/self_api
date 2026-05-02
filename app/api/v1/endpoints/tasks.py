from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import require_api_auth
from app.schemas.preprocess import AsyncTaskStatusResponse
from app.schemas.tasks import TaskCancelResponse, TaskListResponse, TaskSummaryResponse
from app.services.task_manager import cancel_task, get_task, list_tasks


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[Depends(require_api_auth)],
)


@router.get("", response_model=TaskListResponse)
def get_tasks(
    task_type: str | None = Query(default=None),
    state: Literal["pending", "running", "interrupted", "succeeded", "failed", "cancelled"] | None = Query(
        default=None
    ),
    limit: int = Query(default=100, ge=1, le=500),
) -> TaskListResponse:
    raw_items = list_tasks(task_type=task_type, state=state, limit=limit)
    items = [AsyncTaskStatusResponse(**task) for task in raw_items]
    summary = TaskSummaryResponse()
    summary.total = len(raw_items)
    for task in raw_items:
        task_state = task["state"]
        if task_state == "pending":
            summary.pending += 1
        elif task_state == "running":
            summary.running += 1
        elif task_state == "interrupted":
            summary.interrupted += 1
        elif task_state == "succeeded":
            summary.succeeded += 1
        elif task_state == "failed":
            summary.failed += 1
        elif task_state == "cancelled":
            summary.cancelled += 1
    return TaskListResponse(total=len(items), items=items, summary=summary)


@router.get("/{task_id}", response_model=AsyncTaskStatusResponse)
def get_task_detail(task_id: str) -> AsyncTaskStatusResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return AsyncTaskStatusResponse(**task)


@router.post("/{task_id}/cancel", response_model=TaskCancelResponse)
def cancel_task_detail(task_id: str) -> TaskCancelResponse:
    task = cancel_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return TaskCancelResponse(task=AsyncTaskStatusResponse(**task))
