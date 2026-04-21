from typing import Literal

from pydantic import BaseModel, Field


class SystemComponentStatus(BaseModel):
    name: str
    status: Literal["ok", "degraded", "not_configured"]
    detail: str | None = None


class SystemStatusResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: list[SystemComponentStatus] = Field(default_factory=list)


class SystemInfoResponse(BaseModel):
    app_name: str
    app_version: str
    app_env: str
    api_v1_prefix: str
    public_base_url: str | None = None
    auth_enabled: bool
    session_cookie_secure: bool
    restrict_file_access: bool
    explicit_file_access_roots: bool
    storage_root: str
    file_access_roots: list[str] = Field(default_factory=list)
    cors_allow_origins: list[str] = Field(default_factory=list)


class ValidateYoloEnvRequest(BaseModel):
    yolo_train_env: str = Field(description="Env name, env directory, or python path")


class ValidateYoloEnvResponse(BaseModel):
    status: Literal["ok", "failed"]
    mode: Literal["conda_env_name", "python_path"]
    yolo_train_env: str
    resolved_python: str | None = None
    command: str
    exit_code: int
    stdout: str
    stderr: str
