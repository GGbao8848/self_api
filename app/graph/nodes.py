"""Pipeline 各阶段节点函数。

每个节点遵循统一模式：
1. 检查 gate（若 manual 且非 full_access → interrupt）
2. 从 state 中读取参数（支持 params_override 覆盖）
3. 调用对应 service
4. 写回 step_results 并更新关键路径字段
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from langgraph.types import interrupt

from app.graph.state import DEFAULT_GATES, GateMode, PipelineState, StepGateConfig, StepResult


# ─── 辅助 ────────────────────────────────────────────────────────────────────


def _get_gate(state: PipelineState, step: str) -> StepGateConfig:
    gates = state.get("step_gates") or {}
    if step in gates:
        return gates[step]
    mode: GateMode = DEFAULT_GATES.get(step, "auto")
    return StepGateConfig(mode=mode, confirmed=False, params_override={})


def _maybe_interrupt(state: PipelineState, step: str, review_data: dict[str, Any]) -> dict[str, Any] | None:
    """若该步骤需要人工确认，触发 interrupt 并返回用户的输入；否则返回 None。"""
    if state.get("full_access"):
        return None
    gate = _get_gate(state, step)
    if gate["mode"] != "manual":
        return None

    user_response: dict[str, Any] = interrupt({
        "step": step,
        "review": review_data,
        "instructions": "确认后点击 confirm（可在 params_override 中修改参数），中止请发 abort。",
    })
    return user_response


def _merge_override(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _set_step_result(state: PipelineState, step: str, result: StepResult) -> dict[str, Any]:
    step_results = dict(state.get("step_results") or {})
    step_results[step] = result
    return {"step_results": step_results, "current_step": step}


# ─── 节点函数 ──────────────────────────────────────────────────────────────────


def node_healthcheck(state: PipelineState) -> dict[str, Any]:
    """检查 self_api 是否可达。"""
    self_api_url = (state.get("self_api_url") or "").rstrip("/")
    try:
        with urlopen(f"{self_api_url}/api/v1/healthz", timeout=5):
            pass
    except (URLError, OSError) as exc:
        return {
            **_set_step_result(state, "healthcheck", StepResult(
                status="failed", summary=f"API 不可达: {exc}", data={}
            )),
            "error": f"healthcheck failed: {exc}",
            "completed": True,
        }
    return _set_step_result(state, "healthcheck", StepResult(
        status="ok", summary=f"{self_api_url} 可达", data={}
    ))


def node_discover_classes(state: PipelineState) -> dict[str, Any]:
    """扫描原始数据集 XML，发现所有类名；在审核点让用户填写 class_name_map。"""
    from app.services.discover_xml_classes import run_discover_xml_classes
    from app.schemas.preprocess import DiscoverXmlClassesRequest

    req = DiscoverXmlClassesRequest(input_dir=state["original_dataset"])
    resp = run_discover_xml_classes(req)

    review_data = {
        "total_xml_files": resp.total_xml_files,
        "total_classes": resp.total_classes,
        "class_names": resp.class_names,
        "class_counts": resp.class_counts,
        "hint": (
            "如需合并类名（多→一），请在确认时提供 class_name_map，"
            "例如 {\"louyou1\": \"louyou\", \"louyou2\": \"louyou\"}，"
            "以及 final_classes（目标类列表，决定索引顺序），例如 [\"louyou\"]。"
        ),
    }

    user_response = _maybe_interrupt(state, "discover_classes", review_data)

    override: dict[str, Any] = {}
    if user_response:
        if user_response.get("decision") == "abort":
            return {
                **_set_step_result(state, "discover_classes", StepResult(
                    status="skipped", summary="用户中止", data={}
                )),
                "error": "用户在 discover_classes 中止流程",
                "completed": True,
            }
        override = user_response.get("params_override") or {}

    class_name_map: dict[str, str] | None = override.get("class_name_map") or state.get("class_name_map")
    final_classes: list[str] | None = override.get("final_classes") or state.get("final_classes")

    return {
        **_set_step_result(state, "discover_classes", StepResult(
            status="ok",
            summary=f"发现 {resp.total_classes} 个类名",
            data={"class_names": resp.class_names, "class_counts": resp.class_counts},
        )),
        "discovered_classes": resp.class_names,
        "class_name_map": class_name_map,
        "final_classes": final_classes,
    }


def node_xml_to_yolo(state: PipelineState) -> dict[str, Any]:
    """调用 xml-to-yolo 服务（同步，含 class_name_map 支持）。"""
    from app.services.xml_to_yolo import run_xml_to_yolo
    from app.schemas.preprocess import XmlToYoloRequest

    req = XmlToYoloRequest(
        input_dir=state["original_dataset"],
        classes=state.get("final_classes"),
        class_name_map=state.get("class_name_map"),
        recursive=True,
        write_classes_file=True,
    )
    try:
        resp = run_xml_to_yolo(req)
    except (ValueError, OSError) as exc:
        return {
            **_set_step_result(state, "xml_to_yolo", StepResult(
                status="failed", summary=str(exc), data={}
            )),
            "error": str(exc),
            "completed": True,
        }

    return {
        **_set_step_result(state, "xml_to_yolo", StepResult(
            status="ok",
            summary=f"转换 {resp.converted_files}/{resp.total_xml_files} 个文件，共 {resp.total_boxes} 个框",
            data=resp.model_dump(exclude={"details"}),
        )),
        "labels_dir": resp.labels_dir,
    }


def node_review_labels(state: PipelineState) -> dict[str, Any]:
    """xml_to_yolo 之后的人工审核：展示 label 统计，确认后进入数据集划分。"""
    xml_result = (state.get("step_results") or {}).get("xml_to_yolo", {})

    review_data = {
        "xml_to_yolo_summary": xml_result.get("summary", ""),
        "classes": xml_result.get("data", {}).get("classes", []),
        "class_to_id": xml_result.get("data", {}).get("class_to_id", {}),
        "labels_dir": state.get("labels_dir"),
        "hint": "确认 label 转换结果无误后继续；如有问题可中止后调整 class_name_map 再重新运行。",
    }

    user_response = _maybe_interrupt(state, "review_labels", review_data)
    if user_response and user_response.get("decision") == "abort":
        return {
            **_set_step_result(state, "review_labels", StepResult(
                status="skipped", summary="用户中止", data={}
            )),
            "error": "用户在 review_labels 中止流程",
            "completed": True,
        }

    return _set_step_result(state, "review_labels", StepResult(
        status="ok", summary="label 审核通过", data={}
    ))


def node_split_dataset(state: PipelineState) -> dict[str, Any]:
    """数据集 train/val 划分。"""
    from app.services.split_yolo_dataset import run_split_yolo_dataset
    from app.schemas.preprocess import SplitYoloDatasetRequest
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    detector_name = state.get("detector_name", "unknown")
    output_dir = f"{state['original_dataset']}/{detector_name}_{ts}"

    req = SplitYoloDatasetRequest(
        input_dir=state["original_dataset"],
        output_dir=output_dir,
        mode=state.get("split_mode", "train_val"),
        train_ratio=state.get("train_ratio", 0.85),
        val_ratio=state.get("val_ratio", 0.15),
        shuffle=True,
        seed=42,
        copy_files=True,
    )
    try:
        resp = run_split_yolo_dataset(req)
    except (ValueError, OSError) as exc:
        return {
            **_set_step_result(state, "split_dataset", StepResult(
                status="failed", summary=str(exc), data={}
            )),
            "error": str(exc),
            "completed": True,
        }

    version = resp.output_dir.rstrip("/").split("/")[-1]
    return {
        **_set_step_result(state, "split_dataset", StepResult(
            status="ok",
            summary=f"划分完成 → {resp.output_dir}",
            data=resp.model_dump(),
        )),
        "split_output_dir": resp.output_dir,
        "dataset_version": version,
    }


def node_crop_augment(state: PipelineState) -> dict[str, Any]:
    """滑窗裁剪 + 数据增强（仅宽图执行裁剪；增强作用于 train split）。"""
    from app.services.yolo_sliding_window import run_yolo_sliding_window_crop
    from app.services.yolo_augment import run_yolo_augment
    from app.schemas.preprocess import YoloSlidingWindowCropRequest, YoloAugmentRequest

    split_dir = state.get("split_output_dir", "")
    summaries: list[str] = []

    for split in ("train", "val"):
        input_split = f"{split_dir}/{split}"
        output_split = f"{split_dir}/crop/{split}"
        try:
            run_yolo_sliding_window_crop(YoloSlidingWindowCropRequest(
                images_dir=f"{input_split}/images",
                labels_dir=f"{input_split}/labels",
                output_dir=output_split,
                only_wide=True,
            ))
            summaries.append(f"{split} 裁剪完成")
        except (ValueError, OSError, FileNotFoundError):
            summaries.append(f"{split} 无需裁剪（目录不存在或无宽图），跳过")

    try:
        run_yolo_augment(YoloAugmentRequest(
            input_dir=f"{split_dir}/train",
        ))
        summaries.append("train 增强完成")
    except (ValueError, OSError) as exc:
        summaries.append(f"train 增强跳过: {exc}")

    return _set_step_result(state, "crop_augment", StepResult(
        status="ok",
        summary="；".join(summaries),
        data={"split_dir": split_dir},
    ))


def node_build_yaml(state: PipelineState) -> dict[str, Any]:
    """生成 Ultralytics data.yaml。"""
    from app.services.build_yolo_yaml import run_build_yolo_yaml
    from app.schemas.preprocess import BuildYoloYamlRequest

    split_dir = state.get("split_output_dir", "")
    version = state.get("dataset_version", "dataset")
    detector_name = state.get("detector_name", "unknown")
    project_root_dir = state.get("project_root_dir", "").rstrip("/")

    yaml_path = f"{split_dir}/{version}.yaml"
    replace_from = split_dir.rsplit("/", 1)[0] if "/" in split_dir else split_dir
    replace_to = f"{project_root_dir}/{detector_name}/dataset"

    req = BuildYoloYamlRequest(
        input_dir=split_dir,
        output_yaml_path=yaml_path,
        path_prefix_replace_from=replace_from,
        path_prefix_replace_to=replace_to,
        classes_file=f"{state['original_dataset']}/classes.txt",
    )
    try:
        resp = run_build_yolo_yaml(req)
    except (ValueError, OSError) as exc:
        return {
            **_set_step_result(state, "build_yaml", StepResult(
                status="failed", summary=str(exc), data={}
            )),
            "error": str(exc),
            "completed": True,
        }

    return {
        **_set_step_result(state, "build_yaml", StepResult(
            status="ok",
            summary=f"YAML 已生成: {resp.output_yaml_path}",
            data=resp.model_dump(),
        )),
        "yaml_path": resp.output_yaml_path,
    }


def node_publish_transfer(state: PipelineState) -> dict[str, Any]:
    """数据发布/传输审核点 + 实际 publish 调用。

    local 模式：仅执行 publish_yolo_dataset（在本地落盘）。
    remote 模式：zip → SFTP → 远端解压。
    """
    execution_mode = state.get("execution_mode", "local")

    review_data = {
        "execution_mode": execution_mode,
        "yaml_path": state.get("yaml_path"),
        "split_output_dir": state.get("split_output_dir"),
        "hint": "确认数据集已就绪，即将发布/传输到训练目录。",
    }
    user_response = _maybe_interrupt(state, "publish_transfer", review_data)
    if user_response and user_response.get("decision") == "abort":
        return {
            **_set_step_result(state, "publish_transfer", StepResult(
                status="skipped", summary="用户中止", data={}
            )),
            "error": "用户在 publish_transfer 中止流程",
            "completed": True,
        }

    from app.services.publish_yolo_dataset import run_publish_yolo_dataset
    from app.schemas.preprocess import PublishYoloDatasetRequest

    req = PublishYoloDatasetRequest(
        input_dir=state.get("split_output_dir", ""),
        project_root_dir=state.get("project_root_dir", ""),
        detector_name=state.get("detector_name", ""),
        publish_mode=execution_mode if execution_mode in ("local", "remote_sftp") else "local",
        remote_host=state.get("remote_host"),
        remote_username=state.get("remote_username"),
        remote_private_key_path=state.get("remote_private_key_path"),
        remote_project_root_dir=state.get("remote_project_root_dir"),
    )
    try:
        resp = run_publish_yolo_dataset(req)
    except (ValueError, OSError) as exc:
        return {
            **_set_step_result(state, "publish_transfer", StepResult(
                status="failed", summary=str(exc), data={}
            )),
            "error": str(exc),
            "completed": True,
        }

    return {
        **_set_step_result(state, "publish_transfer", StepResult(
            status="ok",
            summary=f"发布完成 → {resp.output_yaml_path}",
            data=resp.model_dump(),
        )),
        "yaml_path": resp.output_yaml_path,
    }


def node_train(state: PipelineState) -> dict[str, Any]:
    """启动训练前最终确认 + 提交异步训练任务。"""
    import datetime
    from app.services.task_manager import submit_task
    from app.schemas.preprocess import YoloTrainRequest

    yaml_path = state.get("yaml_path", "")
    detector_name = state.get("detector_name", "unknown")
    project_root_dir = state.get("project_root_dir", "").rstrip("/")
    version = state.get("dataset_version", datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    project = f"{project_root_dir}/{detector_name}/runs/detect"

    review_data = {
        "yaml_path": yaml_path,
        "project": project,
        "name": version,
        "model": state.get("yolo_train_model", "yolo11s.pt"),
        "epochs": state.get("yolo_train_epochs", 100),
        "imgsz": state.get("yolo_train_imgsz", 640),
        "yolo_train_env": state.get("yolo_train_env", ""),
        "hint": "即将启动异步训练，确认参数无误后点击 confirm。",
    }
    user_response = _maybe_interrupt(state, "train", review_data)
    if user_response and user_response.get("decision") == "abort":
        return {
            **_set_step_result(state, "train", StepResult(
                status="skipped", summary="用户中止", data={}
            )),
            "error": "用户在 train 中止流程",
            "completed": True,
        }

    override = (user_response or {}).get("params_override") or {}
    final_params = _merge_override(
        {
            "yaml_path": yaml_path,
            "project_root_dir": project_root_dir,
            "project": project,
            "name": version,
            "yolo_train_env": state.get("yolo_train_env", ""),
            "model": state.get("yolo_train_model", "yolo11s.pt"),
            "epochs": state.get("yolo_train_epochs", 100),
            "imgsz": state.get("yolo_train_imgsz", 640),
        },
        override,
    )

    from app.services.yolo_train import run_yolo_train

    def _runner() -> dict:
        return run_yolo_train(YoloTrainRequest(**final_params)).model_dump()

    task_id = submit_task(task_type="yolo_train", runner=_runner)

    return {
        **_set_step_result(state, "train", StepResult(
            status="ok",
            summary=f"训练任务已提交，task_id={task_id}",
            data={"task_id": task_id, **final_params},
        )),
        "train_task_id": task_id,
    }


def node_poll_train(state: PipelineState) -> dict[str, Any]:
    """轮询训练任务直到完成或失败（最长等待 12 小时，每 30 秒查一次）。"""
    from app.services.task_manager import get_task

    task_id = state.get("train_task_id")
    if not task_id:
        return _set_step_result(state, "poll_train", StepResult(
            status="failed", summary="train_task_id 缺失", data={}
        ))

    max_polls = 1440  # 12h / 30s
    for _ in range(max_polls):
        task = get_task(task_id)
        if task is None:
            break
        if task["state"] in ("succeeded", "failed", "cancelled"):
            status = "ok" if task["state"] == "succeeded" else "failed"
            return _set_step_result(state, "poll_train", StepResult(
                status=status,
                summary=f"训练 {task['state']}",
                data=task.get("result") or {},
            ))
        time.sleep(30)

    return _set_step_result(state, "poll_train", StepResult(
        status="failed", summary="训练超时或任务丢失", data={}
    ))


def node_review_result(state: PipelineState) -> dict[str, Any]:
    """训练完成后人工验收审核点。"""
    train_result = (state.get("step_results") or {}).get("poll_train", {})

    review_data = {
        "train_summary": train_result.get("summary", ""),
        "train_data": train_result.get("data", {}),
        "train_task_id": state.get("train_task_id"),
        "yaml_path": state.get("yaml_path"),
        "hint": "请确认训练结果，通过后流程结束；如需重训请中止后调整参数重新运行。",
    }
    user_response = _maybe_interrupt(state, "review_result", review_data)
    if user_response and user_response.get("decision") == "abort":
        return {
            **_set_step_result(state, "review_result", StepResult(
                status="skipped", summary="用户中止验收", data={}
            )),
            "completed": True,
        }

    return {
        **_set_step_result(state, "review_result", StepResult(
            status="ok", summary="训练结果验收通过", data={}
        )),
        "completed": True,
    }
