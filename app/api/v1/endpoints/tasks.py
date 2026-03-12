from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import require_api_auth
from app.schemas.preprocess import AsyncTaskStatusResponse
from app.schemas.tasks import TaskCancelResponse, TaskListResponse
from app.services.task_manager import cancel_task, get_task, list_tasks

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[Depends(require_api_auth)],
)


@router.get("", response_model=TaskListResponse)
def get_tasks(
    task_type: str | None = Query(default=None),
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = Query(
        default=None
    ),
    limit: int = Query(default=100, ge=1, le=500),
) -> TaskListResponse:
    items = [
        AsyncTaskStatusResponse(**task)
        for task in list_tasks(task_type=task_type, state=state, limit=limit)
    ]
    return TaskListResponse(total=len(items), items=items)


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
