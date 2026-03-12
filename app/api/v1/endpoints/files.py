from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.security import require_api_auth
from app.schemas.artifacts import UploadArtifactResponse
from app.services.artifact_store import save_upload, summarize_record

router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(require_api_auth)],
)


@router.post("/upload", response_model=UploadArtifactResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)) -> UploadArtifactResponse:
    try:
        artifact = await save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadArtifactResponse(artifact=summarize_record(artifact))
