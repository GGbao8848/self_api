from pydantic import BaseModel, Field

from app.schemas.preprocess import AsyncTaskStatusResponse

class TaskListResponse(BaseModel):
    total: int
    items: list[AsyncTaskStatusResponse] = Field(default_factory=list)


class TaskCancelResponse(BaseModel):
    status: str = "ok"
    task: AsyncTaskStatusResponse
