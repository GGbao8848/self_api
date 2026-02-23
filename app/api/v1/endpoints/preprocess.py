from fastapi import APIRouter, HTTPException, Request

from app.schemas.preprocess import (
    AsyncTaskStatusResponse,
    AsyncTaskSubmitResponse,
    CopyPathAsyncRequest,
    CopyPathRequest,
    CopyPathResponse,
    MovePathAsyncRequest,
    MovePathRequest,
    MovePathResponse,
    SplitYoloDatasetAsyncRequest,
    SplitYoloDatasetRequest,
    SplitYoloDatasetResponse,
    SlidingWindowCropAsyncRequest,
    SlidingWindowCropRequest,
    SlidingWindowCropResponse,
    UnzipArchiveAsyncRequest,
    UnzipArchiveRequest,
    UnzipArchiveResponse,
    XmlToYoloAsyncRequest,
    XmlToYoloRequest,
    XmlToYoloResponse,
    YoloSlidingWindowCropAsyncRequest,
    YoloSlidingWindowCropRequest,
    YoloSlidingWindowCropResponse,
    ZipFolderAsyncRequest,
    ZipFolderRequest,
    ZipFolderResponse,
)
from app.services.file_operations import (
    run_copy_path,
    run_move_path,
    run_unzip_archive,
    run_zip_folder,
)
from app.services.sliding_window import run_sliding_window_crop
from app.services.split_yolo_dataset import run_split_yolo_dataset
from app.services.task_manager import get_task, submit_task
from app.services.xml_to_yolo import run_xml_to_yolo
from app.services.yolo_sliding_window import run_yolo_sliding_window_crop

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


@router.post("/sliding-window-crop", response_model=SlidingWindowCropResponse)
def sliding_window_crop(payload: SlidingWindowCropRequest) -> SlidingWindowCropResponse:
    try:
        return run_sliding_window_crop(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/sliding-window-crop/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def sliding_window_crop_async(
    payload: SlidingWindowCropAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "sliding_window_crop"
    sync_payload = SlidingWindowCropRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_sliding_window_crop(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=AsyncTaskStatusResponse,
    name="get_preprocess_task_status",
)
def get_preprocess_task_status(task_id: str) -> AsyncTaskStatusResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return AsyncTaskStatusResponse(**task)


@router.post("/xml-to-yolo", response_model=XmlToYoloResponse)
def xml_to_yolo(payload: XmlToYoloRequest) -> XmlToYoloResponse:
    try:
        return run_xml_to_yolo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/xml-to-yolo/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def xml_to_yolo_async(
    payload: XmlToYoloAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "xml_to_yolo"
    sync_payload = XmlToYoloRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_xml_to_yolo(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.post("/split-yolo-dataset", response_model=SplitYoloDatasetResponse)
def split_yolo_dataset(payload: SplitYoloDatasetRequest) -> SplitYoloDatasetResponse:
    try:
        return run_split_yolo_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/split-yolo-dataset/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def split_yolo_dataset_async(
    payload: SplitYoloDatasetAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "split_yolo_dataset"
    sync_payload = SplitYoloDatasetRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_split_yolo_dataset(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.post("/zip-folder", response_model=ZipFolderResponse)
def zip_folder(payload: ZipFolderRequest) -> ZipFolderResponse:
    try:
        return run_zip_folder(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/zip-folder/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def zip_folder_async(
    payload: ZipFolderAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "zip_folder"
    sync_payload = ZipFolderRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_zip_folder(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.post("/unzip-archive", response_model=UnzipArchiveResponse)
def unzip_archive(payload: UnzipArchiveRequest) -> UnzipArchiveResponse:
    try:
        return run_unzip_archive(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/unzip-archive/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def unzip_archive_async(
    payload: UnzipArchiveAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "unzip_archive"
    sync_payload = UnzipArchiveRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_unzip_archive(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.post("/move-path", response_model=MovePathResponse)
def move_path(payload: MovePathRequest) -> MovePathResponse:
    try:
        return run_move_path(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/move-path/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def move_path_async(
    payload: MovePathAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "move_path"
    sync_payload = MovePathRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_move_path(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.post("/copy-path", response_model=CopyPathResponse)
def copy_path(payload: CopyPathRequest) -> CopyPathResponse:
    try:
        return run_copy_path(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/copy-path/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def copy_path_async(
    payload: CopyPathAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "copy_path"
    sync_payload = CopyPathRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_copy_path(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )


@router.post("/yolo-sliding-window-crop", response_model=YoloSlidingWindowCropResponse)
def yolo_sliding_window_crop(
    payload: YoloSlidingWindowCropRequest,
) -> YoloSlidingWindowCropResponse:
    try:
        return run_yolo_sliding_window_crop(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/yolo-sliding-window-crop/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def yolo_sliding_window_crop_async(
    payload: YoloSlidingWindowCropAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "yolo_sliding_window_crop"
    sync_payload = YoloSlidingWindowCropRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_yolo_sliding_window_crop(sync_payload).model_dump(),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    status_url = str(request.url_for("get_preprocess_task_status", task_id=task_id))
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=status_url,
        callback_url=str(payload.callback_url) if payload.callback_url else None,
    )
