from typing import Literal

from pydantic import BaseModel, Field


class ArtifactSummary(BaseModel):
    artifact_id: str
    kind: Literal["file", "directory"]
    source: str
    file_name: str
    size_bytes: int | None = None
    content_type: str | None = None
    created_at: str
    task_id: str | None = None
    task_type: str | None = None


class ArtifactListResponse(BaseModel):
    total: int
    items: list[ArtifactSummary] = Field(default_factory=list)


class UploadArtifactResponse(BaseModel):
    status: str = "ok"
    artifact: ArtifactSummary
