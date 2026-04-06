from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.url_builder import build_route_url
from app.core.security import require_api_auth
from app.schemas.preprocess import (
    AggregateNestedDatasetAsyncRequest,
    AggregateNestedDatasetRequest,
    AggregateNestedDatasetResponse,
    AsyncTaskStatusResponse,
    AsyncTaskSubmitResponse,
    CleanNestedDatasetAsyncRequest,
    CleanNestedDatasetRequest,
    CleanNestedDatasetResponse,
    CopyPathAsyncRequest,
    CopyPathRequest,
    CopyPathResponse,
    DiscoverLeafDirsAsyncRequest,
    DiscoverLeafDirsRequest,
    DiscoverLeafDirsResponse,
    MovePathAsyncRequest,
    MovePathRequest,
    MovePathResponse,
    RemoteTransferAsyncRequest,
    RemoteTransferRequest,
    RemoteTransferResponse,
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
    BuildYoloYamlAsyncRequest,
    BuildYoloYamlRequest,
    BuildYoloYamlResponse,
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
from app.services.remote_transfer import run_remote_transfer
from app.services.nested_dataset import (
    run_aggregate_nested_dataset,
    run_clean_nested_dataset,
    run_discover_leaf_dirs,
)
from app.services.sliding_window import run_sliding_window_crop
from app.services.split_yolo_dataset import run_split_yolo_dataset
from app.services.task_manager import get_task, submit_task
from app.services.build_yolo_yaml import run_build_yolo_yaml
from app.services.xml_to_yolo import run_xml_to_yolo
from app.services.yolo_sliding_window import run_yolo_sliding_window_crop

router = APIRouter(
    prefix="/preprocess",
    tags=["preprocess"],
    dependencies=[Depends(require_api_auth)],
)


def _build_async_submit_response(
    request: Request,
    *,
    task_id: str,
    task_type: str,
    callback_url: str | None,
) -> AsyncTaskSubmitResponse:
    return AsyncTaskSubmitResponse(
        task_id=task_id,
        task_type=task_type,
        status_url=build_route_url(
            request,
            "get_preprocess_task_status",
            task_id=task_id,
        ),
        callback_url=callback_url,
    )


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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = SlidingWindowCropRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_sliding_window_crop(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
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


@router.post("/discover-leaf-dirs", response_model=DiscoverLeafDirsResponse)
def discover_leaf_dirs(payload: DiscoverLeafDirsRequest) -> DiscoverLeafDirsResponse:
    try:
        return run_discover_leaf_dirs(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/discover-leaf-dirs/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def discover_leaf_dirs_async(
    payload: DiscoverLeafDirsAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "discover_leaf_dirs"
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = DiscoverLeafDirsRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_discover_leaf_dirs(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )


@router.post("/clean-nested-dataset", response_model=CleanNestedDatasetResponse)
def clean_nested_dataset(payload: CleanNestedDatasetRequest) -> CleanNestedDatasetResponse:
    try:
        return run_clean_nested_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/clean-nested-dataset/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def clean_nested_dataset_async(
    payload: CleanNestedDatasetAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "clean_nested_dataset"
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = CleanNestedDatasetRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_clean_nested_dataset(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )


@router.post("/aggregate-nested-dataset", response_model=AggregateNestedDatasetResponse)
def aggregate_nested_dataset(
    payload: AggregateNestedDatasetRequest,
) -> AggregateNestedDatasetResponse:
    try:
        return run_aggregate_nested_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/aggregate-nested-dataset/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def aggregate_nested_dataset_async(
    payload: AggregateNestedDatasetAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "aggregate_nested_dataset"
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = AggregateNestedDatasetRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_aggregate_nested_dataset(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )


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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = XmlToYoloRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_xml_to_yolo(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = SplitYoloDatasetRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_split_yolo_dataset(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = ZipFolderRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_zip_folder(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = UnzipArchiveRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_unzip_archive(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = MovePathRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_move_path(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )


@router.post("/copy-path", response_model=CopyPathResponse)
def copy_path(payload: CopyPathRequest) -> CopyPathResponse:
    try:
        return run_copy_path(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/remote-transfer", response_model=RemoteTransferResponse)
def remote_transfer(payload: RemoteTransferRequest) -> RemoteTransferResponse:
    try:
        return run_remote_transfer(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/remote-transfer/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def remote_transfer_async(
    payload: RemoteTransferAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "remote_transfer"
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = RemoteTransferRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_remote_transfer(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )


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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = CopyPathRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_copy_path(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
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
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = YoloSlidingWindowCropRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_yolo_sliding_window_crop(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )


@router.post("/build-yolo-yaml", response_model=BuildYoloYamlResponse)
def build_yolo_yaml(payload: BuildYoloYamlRequest) -> BuildYoloYamlResponse:
    try:
        return run_build_yolo_yaml(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/build-yolo-yaml/async",
    response_model=AsyncTaskSubmitResponse,
    status_code=202,
)
def build_yolo_yaml_async(
    payload: BuildYoloYamlAsyncRequest,
    request: Request,
) -> AsyncTaskSubmitResponse:
    task_type = "build_yolo_yaml"
    callback_url = str(payload.callback_url) if payload.callback_url else None
    sync_payload = BuildYoloYamlRequest(
        **payload.model_dump(exclude={"callback_url", "callback_timeout_seconds"})
    )
    task_id = submit_task(
        task_type=task_type,
        runner=lambda: run_build_yolo_yaml(sync_payload).model_dump(),
        callback_url=callback_url,
        callback_timeout_seconds=payload.callback_timeout_seconds,
    )
    return _build_async_submit_response(
        request,
        task_id=task_id,
        task_type=task_type,
        callback_url=callback_url,
    )
