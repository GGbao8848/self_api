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
    auth_enabled: bool
    storage_root: str
    file_access_roots: list[str] = Field(default_factory=list)
