from typing import Literal
from json import dumps, loads
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import get_settings
from app.core.security import require_api_auth
from app.schemas.preprocess import AsyncTaskStatusResponse
from app.schemas.tasks import (
    TaskCancelResponse,
    TaskListResponse,
    TrainingWorkflowLaunchRequest,
    TrainingWorkflowLaunchResponse,
)
from app.services.task_manager import cancel_task, get_task, list_tasks


def _resolve_training_workflow_mode_and_remote_fields(
    payload: TrainingWorkflowLaunchRequest,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """根据 run_target（及向后兼容规则）得到 workflow_mode 与发往 n8n 的远端四元组。"""
    host = (payload.remote_host or "").strip() or None
    user = (payload.remote_username or "").strip() or None
    key_path = (payload.remote_private_key_path or "").strip() or None
    remote_root_raw = (payload.remote_project_root_dir or "").strip()
    remote_root = remote_root_raw or None
    has_full_ssh = bool(host and user and key_path)

    if payload.run_target == "local":
        return "local", None, None, None, None

    if payload.run_target == "remote_sftp":
        if not has_full_ssh:
            raise HTTPException(
                status_code=400,
                detail="remote_sftp 需要填写 remote_host、remote_username、remote_private_key_path",
            )
        remote_dir_out = remote_root or payload.workspace_root_dir
        return "remote_sftp", host, user, key_path, remote_dir_out

    # legacy: run_target is None — 与历史行为一致（不完整 SSH 时仍回退 remote_project_root_dir）
    if has_full_ssh:
        remote_dir_out = remote_root or payload.workspace_root_dir
        return "remote_sftp", host, user, key_path, remote_dir_out
    rproj_legacy = remote_root or payload.workspace_root_dir
    return "local", host, user, key_path, rproj_legacy


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[Depends(require_api_auth)],
)


@router.get("", response_model=TaskListResponse)
def get_tasks(
    task_type: str | None = Query(default=None),
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = Query(
        default=None
    ),
    limit: int = Query(default=100, ge=1, le=500),
) -> TaskListResponse:
    items = [
        AsyncTaskStatusResponse(**task)
        for task in list_tasks(task_type=task_type, state=state, limit=limit)
    ]
    return TaskListResponse(total=len(items), items=items)


@router.get("/{task_id}", response_model=AsyncTaskStatusResponse)
def get_task_detail(task_id: str) -> AsyncTaskStatusResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return AsyncTaskStatusResponse(**task)


@router.post("/{task_id}/cancel", response_model=TaskCancelResponse)
def cancel_task_detail(task_id: str) -> TaskCancelResponse:
    task = cancel_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return TaskCancelResponse(task=AsyncTaskStatusResponse(**task))


@router.post("/launch-training-workflow", response_model=TrainingWorkflowLaunchResponse)
def launch_training_workflow(
    payload: TrainingWorkflowLaunchRequest,
) -> TrainingWorkflowLaunchResponse:
    settings = get_settings()
    n8n_base_url = (settings.n8n_base_url or "").strip().rstrip("/")
    if not n8n_base_url:
        raise HTTPException(status_code=500, detail="SELF_API_N8N_BASE_URL is not configured")

    webhook_url = f"{n8n_base_url}/webhook/self-api-train"
    workflow_mode, rh, ru, rk, rproj = _resolve_training_workflow_mode_and_remote_fields(
        payload
    )
    request_payload = {
        "self_api_url": payload.self_api_url,
        "workspace_root_dir": payload.workspace_root_dir,
        "project_name": payload.project_name,
        "detector_name": payload.detector_name,
        "original_dataset_dir": payload.original_dataset_dir,
        "yolo_train_env": payload.yolo_train_env,
        "yolo_train_model": payload.yolo_train_model,
        "yolo_train_epochs": payload.yolo_train_epochs,
        "yolo_train_imgsz": payload.yolo_train_imgsz,
        "split_mode": payload.split_mode,
        "remote_host": rh,
        "remote_username": ru,
        "remote_private_key_path": rk,
        "remote_project_root_dir": rproj,
    }
    body = dumps(request_payload).encode("utf-8")
    req = Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            parsed = loads(raw_text) if raw_text.strip() else {}
            return TrainingWorkflowLaunchResponse(
                workflow_mode=workflow_mode,
                webhook_url=webhook_url,
                request_payload=request_payload,
                upstream_status_code=int(resp.getcode() or 0),
                upstream_response=parsed if isinstance(parsed, dict) else {"data": parsed},
            )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=502,
            detail=f"n8n webhook returned {exc.code}: {detail or 'unknown error'}",
        ) from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"failed to reach n8n webhook: {exc}") from exc
