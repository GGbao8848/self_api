"""Pipeline REST API。

端点：
  POST /pipeline/run          启动新的 pipeline 运行，返回 run_id
  GET  /pipeline/{run_id}     查询运行状态（含当前暂停的审核点数据）
  POST /pipeline/{run_id}/confirm  人工确认/中止当前审核点
  POST /pipeline/{run_id}/abort    强制中止整条流程
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command

from app.core.security import require_api_auth
from app.graph.pipeline import compiled_graph
from app.graph.state import DEFAULT_GATES, PipelineState, StepGateConfig
from app.schemas.pipeline import (
    PipelineConfirmRequest,
    PipelineRunRequest,
    PipelineStatusResponse,
    PipelineStepStatus,
)

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
        split_mode=req.split_mode,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        remote_host=req.remote_host,
        remote_username=req.remote_username,
        remote_private_key_path=req.remote_private_key_path,
        remote_project_root_dir=req.remote_project_root_dir,
        class_name_map=req.class_name_map,
        final_classes=req.final_classes,
        full_access=req.full_access,
        step_gates=step_gates,
        step_results={},
        completed=False,
        error=None,
        current_step=None,
        pending_review=None,
        train_task_id=None,
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


def _to_status_response(run_id: str) -> PipelineStatusResponse:
    gs = _get_graph_state(run_id)
    # LangGraph 对未知 thread_id 也返回空 StateSnapshot，需要额外校验
    if gs is None or not gs.values or not gs.values.get("run_id"):
        raise HTTPException(status_code=404, detail=f"pipeline run not found: {run_id}")

    state: PipelineState = gs.values
    interrupted = bool(gs.next)  # LangGraph: gs.next 非空表示被 interrupt 暂停

    step_results: dict[str, PipelineStepStatus] = {}
    for step, result in (state.get("step_results") or {}).items():
        step_results[step] = PipelineStepStatus(
            status=result.get("status", "unknown"),
            summary=result.get("summary", ""),
            data=result.get("data") or {},
        )

    return PipelineStatusResponse(
        run_id=run_id,
        current_step=state.get("current_step"),
        completed=state.get("completed", False),
        error=state.get("error"),
        pending_review=state.get("pending_review"),
        step_results=step_results,
        interrupted=interrupted,
    )


@router.post("/run", response_model=PipelineStatusResponse, status_code=202)
def run_pipeline(req: PipelineRunRequest) -> PipelineStatusResponse:
    """启动新的 pipeline 运行。立即返回 run_id，流程在后台推进直到第一个 interrupt。"""
    run_id = str(uuid.uuid4())
    initial_state = _build_initial_state(req, run_id)
    config = _build_config(run_id)

    try:
        compiled_graph.invoke(initial_state, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline invoke error: {exc}") from exc

    return _to_status_response(run_id)


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
    if not gs.next:
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

    return _to_status_response(run_id)
