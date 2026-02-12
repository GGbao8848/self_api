from fastapi import APIRouter, HTTPException

from app.schemas.preprocess import (
    DeduplicateRequest,
    DeduplicateResponse,
    SlidingWindowCropRequest,
    SlidingWindowCropResponse,
)
from app.services.deduplicate import run_deduplication
from app.services.sliding_window import run_sliding_window_crop

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


@router.post("/sliding-window-crop", response_model=SlidingWindowCropResponse)
def sliding_window_crop(payload: SlidingWindowCropRequest) -> SlidingWindowCropResponse:
    try:
        return run_sliding_window_crop(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/deduplicate", response_model=DeduplicateResponse)
def deduplicate(payload: DeduplicateRequest) -> DeduplicateResponse:
    try:
        return run_deduplication(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
