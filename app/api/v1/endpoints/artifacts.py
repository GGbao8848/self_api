from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.security import require_api_auth
from app.schemas.artifacts import ArtifactListResponse, ArtifactSummary
from app.services.artifact_store import get_artifact, list_artifacts, prepare_download, summarize_record

router = APIRouter(
    prefix="/artifacts",
    tags=["artifacts"],
    dependencies=[Depends(require_api_auth)],
)


@router.get("", response_model=ArtifactListResponse)
def get_artifacts(
    source: str | None = Query(default=None),
    task_id: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> ArtifactListResponse:
    items = [
        ArtifactSummary(**summarize_record(record))
        for record in list_artifacts(source=source, task_id=task_id, kind=kind, limit=limit)
    ]
    return ArtifactListResponse(total=len(items), items=items)


@router.get("/{artifact_id}", response_model=ArtifactSummary)
def get_artifact_detail(artifact_id: str) -> ArtifactSummary:
    record = get_artifact(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_id}")
    return ArtifactSummary(**summarize_record(record))


@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str) -> FileResponse:
    try:
        file_path, file_name, media_type = prepare_download(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=file_path, filename=file_name, media_type=media_type)
