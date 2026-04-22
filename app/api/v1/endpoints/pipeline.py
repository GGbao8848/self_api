"""Pipeline REST API。

端点：
  POST /pipeline/run                启动新的 pipeline 运行，返回 run_id
  GET  /pipeline/{run_id}           查询运行状态（含当前暂停的审核点数据）
  POST /pipeline/{run_id}/confirm   人工确认/中止当前审核点
  POST /pipeline/{run_id}/abort     强制中止整条流程
  GET  /pipeline/{run_id}/events    Server-Sent Events 流，状态变化时推送快照
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

logger = logging.getLogger(__name__)

from app.core.security import require_api_auth
from app.graph.pipeline import compiled_graph
from app.graph.sops import apply_sop_defaults, list_sops
from app.graph.state import DEFAULT_GATES, PipelineState, StepGateConfig
from app.schemas.pipeline import (
    PipelineLinkedTaskStatus,
    PipelineConfirmRequest,
    PipelineOverallProgress,
    PipelineProgressSnapshot,
    PipelineRunRequest,
    PipelineStatusResponse,
    PipelineStepProgress,
    PipelineStepStatus,
    SopListResponse,
    SopRunRequest,
    SopSummary,
)
from app.services.task_manager import get_task
from app.services.task_manager import get_pipeline_progress

router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    dependencies=[Depends(require_api_auth)],
)


def _build_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _build_initial_state(req: PipelineRunRequest, run_id: str) -> PipelineState:
    """将 PipelineRunRequest 转为初始 PipelineState。"""
    step_gates: dict[str, StepGateConfig] = {}
    for step_name, default_mode in DEFAULT_GATES.items():
        user_gate = (req.step_gates or {}).get(step_name)
        mode = user_gate.mode if user_gate else default_mode
        step_gates[step_name] = StepGateConfig(mode=mode, confirmed=False, params_override={})

    return PipelineState(
        run_id=run_id,
        self_api_url=req.self_api_url,
        original_dataset=req.original_dataset,
        detector_name=req.detector_name,
        project_root_dir=req.project_root_dir,
        execution_mode=req.execution_mode,
        yolo_train_env=req.yolo_train_env,
        yolo_train_model=req.yolo_train_model,
        yolo_train_epochs=req.yolo_train_epochs,
        yolo_train_imgsz=req.yolo_train_imgsz,
        yolo_export_after_train=req.yolo_export_after_train,
        enable_sliding_window=req.enable_sliding_window,
        split_mode=req.split_mode,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        remote_host=req.remote_host,
        remote_username=req.remote_username,
        remote_private_key_path=req.remote_private_key_path,
        remote_project_root_dir=req.remote_project_root_dir,
        class_name_map=req.class_name_map,
        final_classes=req.final_classes,
        class_index_map=req.class_index_map,
        training_names=req.training_names,
        full_access=req.full_access,
        step_gates=step_gates,
        step_results={},
        completed=False,
        error=None,
        current_step=None,
        pending_review=None,
        train_task_id=None,
        crop_output_dir=None,
        export_file_path=None,
        yaml_path=None,
        labels_dir=None,
        split_output_dir=None,
        dataset_version=None,
        discovered_classes=None,
    )


def _get_graph_state(run_id: str) -> Any:
    config = _build_config(run_id)
    try:
        return compiled_graph.get_state(config)
    except Exception:
        return None


def _extract_pending_review(gs: Any) -> dict[str, Any] | None:
    """从 LangGraph 快照中提取当前暂停的 interrupt payload（若有）。

    LangGraph 把 interrupt(value) 的 value 存在 gs.tasks[*].interrupts[*].value。
    这里只取第一个 pending task 的第一个 interrupt；对本项目管线（串行 11 节点）够用。
    """
    tasks = getattr(gs, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()
        for it in interrupts:
            value = getattr(it, "value", None)
            if isinstance(value, dict):
                return value
    return None


def _is_waiting_for_confirmation(gs: Any) -> bool:
    """判断当前 run 是否在等待人工确认。

    兼容两种表现：
    - gs.next 非空（常见）
    - gs.tasks[*].interrupts 存在 value（某些节点内多次 interrupt 的场景）
    """
    if bool(getattr(gs, "next", ())):
        return True
    return _extract_pending_review(gs) is not None


def _extract_revision(gs: Any) -> int:
    metadata = getattr(gs, "metadata", None) or {}
    step = metadata.get("step")
    return int(step) if isinstance(step, int | float) else 0


def _extract_snapshot_id(gs: Any) -> str | None:
    config = getattr(gs, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    checkpoint_id = configurable.get("checkpoint_id")
    return str(checkpoint_id) if checkpoint_id else None


BASE_PROGRESS_STEPS: list[str] = [
    "label_transform_review",
    "split_dataset",
    "publish_transfer",
    "train",
    "poll_train",
    "review_result",
    "export_model",
    "model_infer",
]

DISPLAY_STEP_TO_TECH_STEP: dict[str, str] = {
    "label_transform_review": "review_labels",
}


def _resolve_display_steps(state: PipelineState, step_results: dict[str, PipelineStepStatus]) -> list[str]:
    steps = list(BASE_PROGRESS_STEPS)
    if state.get("enable_sliding_window") or "crop_window" in step_results or "augment_only" in step_results:
        steps[2:2] = ["crop_window", "augment_only"]
    if not state.get("yolo_export_after_train") and "export_model" not in step_results:
        steps = [step for step in steps if step != "export_model"]
    return steps


def _resolve_step_result_for_progress(
    display_step: str,
    step_results: dict[str, PipelineStepStatus],
) -> PipelineStepStatus | None:
    if display_step == "label_transform_review":
        return step_results.get("review_labels")
    return step_results.get(display_step)


def _clamp_percent(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _build_model_task_progress(
    *,
    active_step: str | None,
    display_step: str,
    interrupted: bool,
    model_task: PipelineLinkedTaskStatus | None,
) -> PipelineStepProgress | None:
    if display_step not in {"train", "poll_train", "review_result", "model_infer"}:
        return None

    if interrupted and active_step == display_step:
        return PipelineStepProgress(
            step=display_step,
            percent=12,
            hint="等待人工确认",
            tone="waiting",
            indeterminate=False,
        )

    if display_step == "poll_train" and model_task is None and active_step == "poll_train":
        return PipelineStepProgress(
            step=display_step,
            percent=72,
            hint="训练任务已提交，等待完成",
            tone="running",
            indeterminate=True,
        )
    if model_task is None or active_step != display_step:
        return None

    if model_task.state == "pending":
        queue_position = model_task.queue_position or 0
        return PipelineStepProgress(
            step=display_step,
            percent=20 if queue_position > 0 else 12,
            hint=f"队列中，前面还有 {queue_position} 个任务" if queue_position > 0 else "任务已创建，等待启动",
            tone="queued" if queue_position > 0 else "pending",
            indeterminate=False,
        )
    if model_task.state == "running":
        return PipelineStepProgress(
            step=display_step,
            percent=76,
            hint="模型任务执行中",
            tone="running",
            indeterminate=True,
        )
    if model_task.state == "succeeded":
        return PipelineStepProgress(step=display_step, percent=100, hint="模型任务已完成", tone="ok")
    if model_task.state == "failed":
        return PipelineStepProgress(step=display_step, percent=100, hint="模型任务失败", tone="failed")
    if model_task.state == "cancelled":
        return PipelineStepProgress(step=display_step, percent=100, hint="模型任务已取消", tone="skipped")
    return None


def _build_step_progress(
    *,
    display_step: str,
    active_step: str | None,
    interrupted: bool,
    step_result: PipelineStepStatus | None,
    model_task: PipelineLinkedTaskStatus | None,
    runtime_progress: dict[str, Any] | None,
) -> PipelineStepProgress:
    model_progress = _build_model_task_progress(
        active_step=active_step,
        display_step=display_step,
        interrupted=interrupted,
        model_task=model_task,
    )
    if model_progress is not None:
        return model_progress

    if step_result is not None:
        if step_result.status == "ok":
            return PipelineStepProgress(step=display_step, percent=100, hint="已完成", tone="ok")
        if step_result.status == "failed":
            return PipelineStepProgress(step=display_step, percent=100, hint="执行失败", tone="failed")
        if step_result.status == "skipped":
            return PipelineStepProgress(step=display_step, percent=100, hint="已跳过", tone="skipped")

    if runtime_progress is not None:
        tone = "running"
        if interrupted and active_step == display_step:
            tone = "waiting"
        elif runtime_progress.get("indeterminate"):
            tone = "running"
        percent = int(runtime_progress.get("percent") or 0)
        hint = str(runtime_progress.get("message") or runtime_progress.get("stage") or "处理中")
        if percent >= 100 and tone != "waiting":
            tone = "ok"
        return PipelineStepProgress(
            step=display_step,
            percent=percent,
            hint=hint,
            tone=tone,
            indeterminate=bool(runtime_progress.get("indeterminate", False)),
        )

    if interrupted and active_step == display_step:
        return PipelineStepProgress(step=display_step, percent=12, hint="等待人工确认", tone="waiting")
    if active_step == display_step:
        return PipelineStepProgress(step=display_step, percent=62, hint="处理中", tone="running", indeterminate=True)
    return PipelineStepProgress(step=display_step, percent=0, hint="未开始", tone="pending")


def _build_progress_snapshot(
    *,
    state: PipelineState,
    active_step: str | None,
    interrupted: bool,
    step_results: dict[str, PipelineStepStatus],
    model_task: PipelineLinkedTaskStatus | None,
    runtime_progress: dict[str, dict[str, Any]],
) -> PipelineProgressSnapshot:
    ordered_steps = _resolve_display_steps(state, step_results)
    progress_steps: dict[str, PipelineStepProgress] = {}
    total_percent = 0
    completed_steps = 0
    overall_tone = "pending"
    overall_indeterminate = False

    for step in ordered_steps:
        tech_step = DISPLAY_STEP_TO_TECH_STEP.get(step, step)
        step_progress = _build_step_progress(
            display_step=step,
            active_step=active_step if active_step != tech_step else step,
            interrupted=interrupted,
            step_result=_resolve_step_result_for_progress(step, step_results),
            model_task=model_task,
            runtime_progress=runtime_progress.get(step) or runtime_progress.get(tech_step),
        )
        progress_steps[step] = step_progress
        total_percent += step_progress.percent
        if step_progress.percent >= 100:
            completed_steps += 1
        overall_indeterminate = overall_indeterminate or step_progress.indeterminate
        if step_progress.tone == "failed":
            overall_tone = "failed"
        elif step_progress.tone == "running" and overall_tone != "failed":
            overall_tone = "running"
        elif step_progress.tone == "waiting" and overall_tone not in {"failed", "running"}:
            overall_tone = "waiting"
        elif step_progress.tone == "ok" and overall_tone == "pending":
            overall_tone = "ok"

    overall_percent = _clamp_percent(total_percent / len(ordered_steps)) if ordered_steps else 0
    if state.get("completed") and not state.get("error"):
        overall_hint = "全部步骤已完成"
    elif state.get("completed") and state.get("error"):
        overall_hint = "流程已结束，存在错误"
    elif interrupted and active_step:
        overall_hint = f"等待确认：{active_step}"
    elif active_step:
        overall_hint = f"正在执行：{active_step}"
    else:
        overall_hint = "等待开始"

    return PipelineProgressSnapshot(
        ordered_steps=ordered_steps,
        steps=progress_steps,
        overall=PipelineOverallProgress(
            percent=overall_percent,
            hint=overall_hint,
            tone=overall_tone,
            indeterminate=overall_indeterminate and overall_percent < 100,
            active_step=active_step,
            completed_steps=completed_steps,
            total_steps=len(ordered_steps),
        ),
    )


def _to_status_response(run_id: str) -> PipelineStatusResponse:
    gs = _get_graph_state(run_id)
    # LangGraph 对未知 thread_id 也返回空 StateSnapshot，需要额外校验
    if gs is None or not gs.values or not gs.values.get("run_id"):
        raise HTTPException(status_code=404, detail=f"pipeline run not found: {run_id}")

    state: PipelineState = gs.values
    interrupted = _is_waiting_for_confirmation(gs)

    step_results: dict[str, PipelineStepStatus] = {}
    for step, result in (state.get("step_results") or {}).items():
        step_results[step] = PipelineStepStatus(
            status=result.get("status", "unknown"),
            summary=result.get("summary", ""),
            data=result.get("data") or {},
        )

    pending_review = state.get("pending_review")
    if interrupted and not pending_review:
        pending_review = _extract_pending_review(gs)
    active_step = pending_review.get("step") if isinstance(pending_review, dict) else None
    if not active_step:
        active_step = state.get("current_step")

    model_task = None
    train_task_id = state.get("train_task_id")
    if train_task_id:
        task = get_task(train_task_id)
        if task is not None:
            model_task = PipelineLinkedTaskStatus(
                task_id=task["task_id"],
                task_type=task["task_type"],
                state=task["state"],
                queue_position=task.get("queue_position"),
            )
    progress = _build_progress_snapshot(
        state=state,
        active_step=active_step,
        interrupted=interrupted,
        step_results=step_results,
        model_task=model_task,
        runtime_progress=get_pipeline_progress(run_id),
    )

    return PipelineStatusResponse(
        run_id=run_id,
        current_step=state.get("current_step"),
        active_step=active_step,
        completed=state.get("completed", False),
        error=state.get("error"),
        revision=_extract_revision(gs),
        snapshot_id=_extract_snapshot_id(gs),
        pending_review=pending_review,
        step_results=step_results,
        interrupted=interrupted,
        model_task=model_task,
        progress=progress,
        initial_params={
            "original_dataset": state.get("original_dataset"),
            "detector_name": state.get("detector_name"),
            "project_root_dir": state.get("project_root_dir"),
            "execution_mode": state.get("execution_mode"),
            "yolo_train_env": state.get("yolo_train_env"),
            "yolo_train_model": state.get("yolo_train_model"),
            "yolo_train_epochs": state.get("yolo_train_epochs"),
            "yolo_train_imgsz": state.get("yolo_train_imgsz"),
            "yolo_export_after_train": state.get("yolo_export_after_train"),
            "enable_sliding_window": state.get("enable_sliding_window"),
            "full_access": state.get("full_access"),
            "class_name_map": state.get("class_name_map"),
            "final_classes": state.get("final_classes"),
        }
    )


def _wait_state_ready(run_id: str, timeout_s: float = 3.0) -> None:
    """后台线程起步时，等待状态可见；快任务则尽量等到 completed/interrupted。"""
    deadline = time.monotonic() + timeout_s
    seen_state = False
    while time.monotonic() < deadline:
        gs = _get_graph_state(run_id)
        if gs and gs.values and gs.values.get("run_id"):
            seen_state = True
            if gs.values.get("completed") or _is_waiting_for_confirmation(gs):
                return
        time.sleep(0.05)
    if seen_state:
        return


@router.post("/run", response_model=PipelineStatusResponse, status_code=202)
def run_pipeline(req: PipelineRunRequest) -> PipelineStatusResponse:
    """启动新的 pipeline 运行。

    - `full_access=False`（HITL）：同步推进到第一个 interrupt 后返回。
    - `full_access=True`：派出后台线程跑整条流程（含训练），立即返回 run_id；
      调用方通过 GET /pipeline/{run_id} 或 SSE 查进度。
    """
    run_id = str(uuid.uuid4())
    initial_state = _build_initial_state(req, run_id)
    config = _build_config(run_id)

    if req.full_access:
        def _run_in_background() -> None:
            try:
                compiled_graph.invoke(initial_state, config)
            except Exception:
                logger.exception("pipeline background run %s failed", run_id)

        threading.Thread(target=_run_in_background, daemon=True, name=f"pipeline-{run_id[:8]}").start()
        _wait_state_ready(run_id)
    else:
        try:
            compiled_graph.invoke(initial_state, config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"pipeline invoke error: {exc}") from exc

    return _to_status_response(run_id)


@router.get("/sops", response_model=SopListResponse)
def list_pipeline_sops() -> SopListResponse:
    """列出所有预设 SOP 模板。"""
    sops = [SopSummary(**item) for item in list_sops()]
    return SopListResponse(sops=sops)


@router.post(
    "/sops/{sop_id}/run",
    response_model=PipelineStatusResponse,
    status_code=202,
)
def run_pipeline_from_sop(sop_id: str, req: SopRunRequest) -> PipelineStatusResponse:
    """用 SOP 默认值 + 用户 overrides 启动 pipeline run。

    用户字段优先覆盖 SOP defaults；step_gates 深合并（用户优先）。
    """
    user_payload = req.model_dump(exclude_none=True)
    if "step_gates" in user_payload and user_payload["step_gates"]:
        user_payload["step_gates"] = {
            step: gate.model_dump() if hasattr(gate, "model_dump") else gate
            for step, gate in user_payload["step_gates"].items()
        }
    try:
        merged, missing = apply_sop_defaults(sop_id, user_payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"SOP {sop_id} requires fields: {missing}",
        )

    try:
        merged_req = PipelineRunRequest(**merged)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid merged payload: {exc}") from exc

    return run_pipeline(merged_req)


@router.get("/{run_id}", response_model=PipelineStatusResponse)
def get_pipeline_status(run_id: str) -> PipelineStatusResponse:
    """查询 pipeline 运行状态。interrupted=true 时表示正在等待人工确认。"""
    return _to_status_response(run_id)


@router.post("/{run_id}/confirm", response_model=PipelineStatusResponse)
def confirm_pipeline_step(run_id: str, body: PipelineConfirmRequest) -> PipelineStatusResponse:
    """人工确认（或中止）当前审核点，流程继续推进到下一个 interrupt 或结束。"""
    gs = _get_graph_state(run_id)
    if gs is None or not gs.values or not gs.values.get("run_id"):
        raise HTTPException(status_code=404, detail=f"pipeline run not found: {run_id}")
    if not _is_waiting_for_confirmation(gs):
        raise HTTPException(status_code=409, detail="pipeline is not waiting for confirmation")

    config = _build_config(run_id)
    resume_payload = {
        "decision": body.decision,
        "params_override": body.params_override,
    }
    try:
        compiled_graph.invoke(Command(resume=resume_payload), config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline resume error: {exc}") from exc
    _wait_state_ready(run_id, timeout_s=1.2)
    return _to_status_response(run_id)


@router.post("/{run_id}/abort", response_model=PipelineStatusResponse)
def abort_pipeline(run_id: str) -> PipelineStatusResponse:
    """强制中止 pipeline（等效于在当前审核点发送 abort decision）。"""
    gs = _get_graph_state(run_id)
    if gs is None or not gs.values or not gs.values.get("run_id"):
        raise HTTPException(status_code=404, detail=f"pipeline run not found: {run_id}")

    config = _build_config(run_id)
    if gs.next:
        try:
            compiled_graph.invoke(
                Command(resume={"decision": "abort", "params_override": {}}),
                config,
            )
        except Exception:
            pass
    else:
        # 非 interrupt 等待态也允许中止：写入终止标志，后续边路由会在当前节点结束后尽快停止。
        try:
            compiled_graph.update_state(
                config,
                {
                    "error": "用户手动中止流程",
                    "completed": True,
                },
            )
        except Exception:
            pass

    return _to_status_response(run_id)


def _snapshot_signature(resp: PipelineStatusResponse) -> str:
    """用于 SSE 去重：只有签名变化才发一条 event。"""
    step_states = {name: s.status for name, s in resp.step_results.items()}
    progress_steps = {
        name: {
            "percent": step.percent,
            "hint": step.hint,
            "tone": step.tone,
            "indeterminate": step.indeterminate,
        }
        for name, step in resp.progress.steps.items()
    }
    signature = {
        "revision": resp.revision,
        "snapshot_id": resp.snapshot_id,
        "active_step": resp.active_step,
        "current_step": resp.current_step,
        "completed": resp.completed,
        "interrupted": resp.interrupted,
        "error": resp.error,
        "step_states": step_states,
        "model_task_state": resp.model_task.state if resp.model_task else None,
        "model_task_queue_position": resp.model_task.queue_position if resp.model_task else None,
        "progress_percent": resp.progress.overall.percent,
        "progress_active_step": resp.progress.overall.active_step,
        "progress_hint": resp.progress.overall.hint,
        "progress_steps": progress_steps,
    }
    return json.dumps(signature, sort_keys=True, ensure_ascii=False)


@router.get("/{run_id}/events")
async def stream_pipeline_events(
    run_id: str,
    request: Request,
    poll_interval: float = 0.25,
    max_duration: float = 3600.0,
) -> StreamingResponse:
    """Server-Sent Events：状态变化时推送一次快照，替代前端 HTTP 轮询。

    Query params：
      - poll_interval：服务端内部轮询周期（秒），默认 0.25
      - max_duration：最长保持连接时间（秒），默认 3600；到期后自动关闭
    """
    gs = _get_graph_state(run_id)
    if gs is None or not gs.values or not gs.values.get("run_id"):
        raise HTTPException(status_code=404, detail=f"pipeline run not found: {run_id}")

    poll_interval = max(0.2, min(poll_interval, 30.0))
    max_duration = max(10.0, min(max_duration, 24 * 3600.0))

    async def event_stream() -> AsyncIterator[bytes]:
        last_sig: str | None = None
        elapsed = 0.0
        yield b": connected\n\n"  # SSE comment, 立即刷出

        while elapsed < max_duration:
            if await request.is_disconnected():
                break

            try:
                resp = _to_status_response(run_id)
            except HTTPException as exc:
                payload = json.dumps({"error": exc.detail}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
                break

            sig = _snapshot_signature(resp)
            if sig != last_sig:
                last_sig = sig
                body = resp.model_dump_json()
                yield f"event: snapshot\ndata: {body}\n\n".encode("utf-8")

            if resp.completed:
                yield b"event: end\ndata: {\"reason\": \"completed\"}\n\n"
                break

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if elapsed >= max_duration:
            yield b"event: end\ndata: {\"reason\": \"timeout\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx/反代 缓冲
            "Connection": "keep-alive",
        },
    )
