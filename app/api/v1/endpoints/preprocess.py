from fastapi import APIRouter, HTTPException

from app.schemas.preprocess import (
    SplitYoloDatasetRequest,
    SplitYoloDatasetResponse,
    SlidingWindowCropRequest,
    SlidingWindowCropResponse,
    XmlToYoloRequest,
    XmlToYoloResponse,
)
from app.services.sliding_window import run_sliding_window_crop
from app.services.split_yolo_dataset import run_split_yolo_dataset
from app.services.xml_to_yolo import run_xml_to_yolo

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


@router.post("/sliding-window-crop", response_model=SlidingWindowCropResponse)
def sliding_window_crop(payload: SlidingWindowCropRequest) -> SlidingWindowCropResponse:
    try:
        return run_sliding_window_crop(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/xml-to-yolo", response_model=XmlToYoloResponse)
def xml_to_yolo(payload: XmlToYoloRequest) -> XmlToYoloResponse:
    try:
        return run_xml_to_yolo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/split-yolo-dataset", response_model=SplitYoloDatasetResponse)
def split_yolo_dataset(payload: SplitYoloDatasetRequest) -> SplitYoloDatasetResponse:
    try:
        return run_split_yolo_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
