from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.preprocess import AsyncTaskStatusResponse


class TrainingWorkflowLaunchRequest(BaseModel):
    """训练工作流启动请求。

    ``run_target`` 为 ``None`` 时保持向后兼容：三项 SSH 核心字段齐全则视为远端，否则本地。
    显式 ``local``：本次按本地执行，转发给 n8n 的远端字段一律置空（SSH 仍可保存在前端仅作备忘）。
    显式 ``remote_sftp``：本次按远端执行，必须提供 ``remote_host`` / ``remote_username`` / ``remote_private_key_path``。
    """

    self_api_url: str
    workspace_root_dir: str
    project_name: str
    detector_name: str
    original_dataset_dir: str
    yolo_train_env: str = "yolo_pose"
    yolo_train_model: str = "yolo11s.pt"
    yolo_train_epochs: int = 5
    yolo_train_imgsz: int = 640
    split_mode: str = "train_val"
    run_target: Literal["local", "remote_sftp"] | None = None
    remote_host: str | None = None
    remote_username: str | None = None
    remote_private_key_path: str | None = None
    remote_project_root_dir: str | None = None


class TrainingWorkflowLaunchResponse(BaseModel):
    status: str = "accepted"
    workflow_mode: str
    webhook_url: str
    request_payload: dict = Field(default_factory=dict)
    upstream_status_code: int
    upstream_response: dict = Field(default_factory=dict)


class TaskListResponse(BaseModel):
    total: int
    items: list[AsyncTaskStatusResponse] = Field(default_factory=list)


class TaskCancelResponse(BaseModel):
    status: str = "ok"
    task: AsyncTaskStatusResponse
