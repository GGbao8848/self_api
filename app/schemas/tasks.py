from pydantic import BaseModel, Field

from app.schemas.preprocess import AsyncTaskStatusResponse


class TaskSummaryResponse(BaseModel):
    total: int = 0
    pending: int = 0
    running: int = 0
    interrupted: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0


class TaskListResponse(BaseModel):
    total: int
    items: list[AsyncTaskStatusResponse] = Field(default_factory=list)
    summary: TaskSummaryResponse | None = None


class TaskCancelResponse(BaseModel):
    status: str = "ok"
    task: AsyncTaskStatusResponse
