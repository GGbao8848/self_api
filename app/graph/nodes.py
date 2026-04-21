"""Pipeline 各阶段节点函数。

每个节点遵循统一模式：
1. 检查 gate（若 manual 且非 full_access → interrupt）
2. 从 state 中读取参数（支持 params_override 覆盖）
3. 调用对应 service
4. 写回 step_results 并更新关键路径字段
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from langgraph.types import interrupt

from app.graph.state import DEFAULT_GATES, GateMode, PipelineState, StepGateConfig, StepResult

logger = logging.getLogger(__name__)


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
            "在下方表单中填写：XML 类名重命名/合并、再选择「按列表顺序」或「显式类别 id」；"
            "也可展开高级 JSON 覆盖。与 xml-to-yolo API 一致。"
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

    def _pick_ov(key: str) -> Any:
        if key in override:
            return override[key]
        return state.get(key)

    class_name_map = _pick_ov("class_name_map")
    final_classes = _pick_ov("final_classes")
    class_index_map = _pick_ov("class_index_map")
    training_names = _pick_ov("training_names")

    return {
        **_set_step_result(state, "discover_classes", StepResult(
            status="ok",
            summary=f"发现 {resp.total_classes} 个类名",
            data={"class_names": resp.class_names, "class_counts": resp.class_counts},
        )),
        "discovered_classes": resp.class_names,
        "class_name_map": class_name_map,
        "final_classes": final_classes,
        "class_index_map": class_index_map,
        "training_names": training_names,
    }


def node_xml_to_yolo(state: PipelineState) -> dict[str, Any]:
    """调用 xml-to-yolo 服务（同步，含 class_name_map 支持）。"""
    from app.services.xml_to_yolo import run_xml_to_yolo
    from app.schemas.preprocess import XmlToYoloRequest

    final_classes = state.get("final_classes")
    class_index_map = state.get("class_index_map")
    if final_classes is not None and class_index_map is not None:
        return {
            **_set_step_result(state, "xml_to_yolo", StepResult(
                status="failed",
                summary="final_classes 与 class_index_map 不能同时设置",
                data={},
            )),
            "error": "final_classes 与 class_index_map 互斥",
            "completed": True,
        }

    req = XmlToYoloRequest(
        input_dir=state["original_dataset"],
        classes=final_classes,
        class_name_map=state.get("class_name_map"),
        class_index_map=class_index_map,
        training_names=state.get("training_names"),
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
    xml_data = xml_result.get("data", {}) if isinstance(xml_result, dict) else {}

    review_data = {
        "xml_to_yolo_summary": xml_result.get("summary", ""),
        "classes": xml_data.get("classes", []),
        "class_to_id": xml_data.get("class_to_id", {}),
        "class_name_map": state.get("class_name_map"),
        "final_classes": state.get("final_classes"),
        "class_index_map": state.get("class_index_map"),
        "training_names": state.get("training_names"),
        "labels_dir": state.get("labels_dir"),
        "hint": (
            "确认当前 classes.txt（类别名与索引）是否正确；如需调整可直接修改，"
            "确认后将自动重建 labels 与 classes.txt。"
        ),
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

    override: dict[str, Any] = user_response.get("params_override") if user_response else {}
    override = override or {}
    touched_mapping = any(
        key in override
        for key in ("class_name_map", "final_classes", "class_index_map", "training_names")
    )

    if not touched_mapping:
        return _set_step_result(state, "review_labels", StepResult(
            status="ok", summary="label 审核通过", data={}
        ))

    from app.schemas.preprocess import XmlToYoloRequest
    from app.services.xml_to_yolo import run_xml_to_yolo

    def _pick_ov(key: str) -> Any:
        if key in override:
            return override[key]
        return state.get(key)

    class_name_map = _pick_ov("class_name_map")
    final_classes = _pick_ov("final_classes")
    class_index_map = _pick_ov("class_index_map")
    training_names = _pick_ov("training_names")

    if final_classes is not None and class_index_map is not None:
        return {
            **_set_step_result(state, "review_labels", StepResult(
                status="failed",
                summary="final_classes 与 class_index_map 不能同时设置",
                data={},
            )),
            "error": "review_labels 参数冲突：final_classes 与 class_index_map 互斥",
            "completed": True,
        }

    req = XmlToYoloRequest(
        input_dir=state["original_dataset"],
        classes=final_classes,
        class_name_map=class_name_map,
        class_index_map=class_index_map,
        training_names=training_names,
        recursive=True,
        write_classes_file=True,
    )
    try:
        resp = run_xml_to_yolo(req)
    except (ValueError, OSError) as exc:
        return {
            **_set_step_result(state, "review_labels", StepResult(
                status="failed", summary=f"重建 labels 失败: {exc}", data={}
            )),
            "error": str(exc),
            "completed": True,
        }

    step_results = dict(state.get("step_results") or {})
    step_results["xml_to_yolo"] = StepResult(
        status="ok",
        summary=f"转换 {resp.converted_files}/{resp.total_xml_files} 个文件，共 {resp.total_boxes} 个框",
        data=resp.model_dump(exclude={"details"}),
    )
    step_results["review_labels"] = StepResult(
        status="ok",
        summary="已按审核修改重建 labels/classes.txt",
        data={
            "classes": resp.classes,
            "class_to_id": resp.class_to_id,
            "classes_file": resp.classes_file,
        },
    )

    return {
        "step_results": step_results,
        "current_step": "review_labels",
        "labels_dir": resp.labels_dir,
        "class_name_map": class_name_map,
        "final_classes": final_classes,
        "class_index_map": class_index_map,
        "training_names": training_names,
    }


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
    """滑窗裁剪（train+val 切小图，避免数据泄漏）+ 对 crop/train 做增强。

    原始 split 下是大图；此步完成后 crop/ 即构成实际的数据集版本来源：
      {split_dir}/
        classes.txt
        train/{images,labels}              ← 原始大图 train（不参与训练）
        val/{images,labels}                ← 原始大图 val
        crop/
          classes.txt                       ← 从 split_dir 复制
          train/
            images/, labels/                ← 原始 train 大图滑窗而来
            augment/
              images/, labels/              ← 对 crop/train 的离线增强
          val/
            images/, labels/                ← 原始 val 大图滑窗而来

    数据泄漏防护：sliding_window 分别对 split_dir/train、split_dir/val 切窗口，
    同一张大图的所有 patches 始终归属同一 split。
    最终 publish_transfer 将以 {split_dir}/crop 为 input_dir 发布数据集版本。
    """
    import shutil as _shutil
    from pathlib import Path as _Path
    from app.services.yolo_sliding_window import run_yolo_sliding_window_crop
    from app.services.yolo_augment import run_yolo_augment
    from app.schemas.preprocess import YoloSlidingWindowCropRequest, YoloAugmentRequest

    split_dir = state.get("split_output_dir", "")
    summaries: list[str] = []
    data_payload: dict[str, Any] = {"split_dir": split_dir}

    if not split_dir or not _Path(split_dir).is_dir():
        return {
            **_set_step_result(state, "crop_augment", StepResult(
                status="failed",
                summary=f"split_output_dir 不存在: {split_dir!r}",
                data=data_payload,
            )),
            "error": f"crop_augment: invalid split_output_dir {split_dir!r}",
            "completed": True,
        }

    crop_root = f"{split_dir}/crop"
    data_payload["crop_root"] = crop_root

    if _Path(crop_root).exists():
        _shutil.rmtree(crop_root, ignore_errors=True)

    try:
        crop_resp = run_yolo_sliding_window_crop(YoloSlidingWindowCropRequest(
            input_dir=split_dir,
            output_dir=crop_root,
            only_wide=True,
        ))
        summaries.append(
            f"滑窗裁剪：{crop_resp.processed_images}/{crop_resp.input_images} 张处理，"
            f"跳过 {crop_resp.skipped_images}，生成 {crop_resp.generated_crops} 个窗口"
        )
        data_payload["crop"] = {
            "output_dir": crop_resp.output_dir,
            "input_images": crop_resp.input_images,
            "processed_images": crop_resp.processed_images,
            "skipped_images": crop_resp.skipped_images,
            "generated_crops": crop_resp.generated_crops,
            "generated_labels": crop_resp.generated_labels,
        }
    except Exception as exc:
        logger.warning("crop_augment: sliding_window 失败", exc_info=True)
        return {
            **_set_step_result(state, "crop_augment", StepResult(
                status="failed",
                summary=f"滑窗裁剪失败：{type(exc).__name__}: {exc}",
                data=data_payload,
            )),
            "error": f"crop_augment sliding_window failed: {exc}",
            "completed": True,
        }

    src_classes = _Path(split_dir) / "classes.txt"
    dst_classes = _Path(crop_root) / "classes.txt"
    if src_classes.exists():
        try:
            dst_classes.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(src_classes, dst_classes)
            data_payload["classes_copied_to"] = str(dst_classes)
        except OSError as exc:
            logger.warning("crop_augment: 复制 classes.txt 失败: %s", exc)
            data_payload["classes_copy_error"] = str(exc)
    else:
        logger.warning("crop_augment: split_dir 下未找到 classes.txt: %s", src_classes)

    train_crop_dir = f"{crop_root}/train"
    if not _Path(f"{train_crop_dir}/images").is_dir():
        summaries.append("train 增强跳过：crop/train/images 不存在（train 无宽图？）")
        return _set_step_result(state, "crop_augment", StepResult(
            status="ok",
            summary="；".join(summaries),
            data=data_payload,
        ))

    try:
        aug_resp = run_yolo_augment(YoloAugmentRequest(
            input_dir=train_crop_dir,
            output_dir=f"{train_crop_dir}/augment",
        ))
        summaries.append(
            f"train 增强：处理 {aug_resp.processed_images} 张，"
            f"生成 {aug_resp.generated_images} 张（跳过 {aug_resp.skipped_images}）"
        )
        data_payload["augment"] = {
            "output_dir": aug_resp.output_dir,
            "processed_images": aug_resp.processed_images,
            "skipped_images": aug_resp.skipped_images,
            "generated_images": aug_resp.generated_images,
            "generated_labels": aug_resp.generated_labels,
        }
    except Exception as exc:
        logger.warning("crop_augment: augment 失败", exc_info=True)
        summaries.append(f"train 增强跳过：{type(exc).__name__}: {exc}")
        data_payload["augment_error"] = f"{type(exc).__name__}: {exc}"

    return _set_step_result(state, "crop_augment", StepResult(
        status="ok",
        summary="；".join(summaries),
        data=data_payload,
    ))


def node_publish_transfer(state: PipelineState) -> dict[str, Any]:
    """数据发布/传输审核点 + 实际 publish 调用。

    数据集版本 = 小图 + 增强（只发布 crop/，不含原始大图）：
      input_dir 优先取 {split_output_dir}/crop，若不存在则退回 split_output_dir
      （后一种仅用于 crop_augment 被跳过的异常路径）。

    local 模式：publish_yolo_dataset 在本地生成 dataset_version.yaml（含 classes 与递归
      发现的所有 images 路径）。此前的 build_yolo_yaml 独立节点已弃用，yaml 生成统一
      由 publish_yolo_dataset 负责。
    remote 模式：zip → SFTP → 远端解压。
    """
    from pathlib import Path as _Path

    execution_mode = state.get("execution_mode", "local")
    split_output_dir = (state.get("split_output_dir") or "").rstrip("/")
    crop_root = f"{split_output_dir}/crop" if split_output_dir else ""
    publish_input_dir = crop_root if (crop_root and _Path(crop_root).is_dir()) else split_output_dir

    review_data = {
        "execution_mode": execution_mode,
        "split_output_dir": split_output_dir,
        "publish_input_dir": publish_input_dir,
        "project_root_dir": state.get("project_root_dir", ""),
        "detector_name": state.get("detector_name", ""),
        "note": (
            "数据集版本将由 publish_yolo_dataset 基于该目录生成 yaml 并落盘；"
            "发布源为小图+增强的 crop/ 子目录。"
        ),
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
    override = (user_response or {}).get("params_override") or {}
    publish_params = _merge_override(
        {
            "input_dir": publish_input_dir,
            "project_root_dir": state.get("project_root_dir", ""),
            "detector_name": state.get("detector_name", ""),
            "publish_mode": execution_mode if execution_mode in ("local", "remote_sftp") else "local",
            "remote_host": state.get("remote_host"),
            "remote_username": state.get("remote_username"),
            "remote_private_key_path": state.get("remote_private_key_path"),
            "remote_project_root_dir": state.get("remote_project_root_dir"),
        },
        override,
    )

    from app.services.publish_yolo_dataset import run_publish_yolo_dataset
    from app.schemas.preprocess import PublishYoloDatasetRequest

    req = PublishYoloDatasetRequest(**publish_params)
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

    new_version = resp.output_yaml_path.rsplit("/", 1)[-1].rsplit(".", 1)[0] \
        if resp.output_yaml_path else state.get("dataset_version")

    return {
        **_set_step_result(state, "publish_transfer", StepResult(
            status="ok",
            summary=f"发布完成 → {resp.output_yaml_path}",
            data=resp.model_dump(),
        )),
        "yaml_path": resp.output_yaml_path,
        "dataset_version": new_version,
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

    task_id = submit_task(
        task_type="yolo_train",
        runner=_runner,
        queue_key="local_yolo_train",
    )

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
    train_data = train_result.get("data", {}) if isinstance(train_result, dict) else {}

    review_data = {
        "train_summary": train_result.get("summary", ""),
        "train_data": train_data,
        "train_task_id": state.get("train_task_id"),
        "yaml_path": state.get("yaml_path"),
        "yolo_export_after_train": bool(state.get("yolo_export_after_train", False)),
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

    export_payload: dict[str, Any] = {}
    if (
        state.get("execution_mode") == "local"
        and state.get("yolo_export_after_train", False)
        and train_result.get("status") == "ok"
    ):
        from app.schemas.preprocess import YoloExportRequest
        from app.services.yolo_export import run_yolo_export

        project = str(train_data.get("project") or "").rstrip("/")
        run_name = str(train_data.get("name") or "").strip()
        best_pt_path = f"{project}/{run_name}/weights/best.pt" if project and run_name else ""
        try:
            export_resp = run_yolo_export(
                YoloExportRequest(
                    best_pt_path=best_pt_path,
                    project_root_dir=state.get("project_root_dir", ""),
                    yolo_train_env=state.get("yolo_train_env", ""),
                    overwrite=True,
                )
            )
            export_payload = {
                "export_status": export_resp.status,
                "export_file_path": export_resp.export_file_path,
                "dataset_yaml": export_resp.dataset_yaml,
                "imgsz": export_resp.imgsz,
                "exit_code": export_resp.exit_code,
            }
        except Exception as exc:
            export_payload = {"export_status": "failed", "error": str(exc)}

    return {
        **_set_step_result(state, "review_result", StepResult(
            status="ok", summary="训练结果验收通过", data=export_payload
        )),
    }


def node_model_infer(state: PipelineState) -> dict[str, Any]:
    """模型导出后推理：人工确认参数后执行批量推理。"""
    from app.schemas.preprocess import YoloInferRequest
    from app.services.yolo_infer import run_yolo_infer

    export_result = (state.get("step_results") or {}).get("review_result", {})
    export_data = export_result.get("data", {}) if isinstance(export_result, dict) else {}
    default_model_path = str(export_data.get("export_file_path") or "").strip()
    if not default_model_path:
        train_result = (state.get("step_results") or {}).get("poll_train", {})
        train_data = train_result.get("data", {}) if isinstance(train_result, dict) else {}
        run_project = str(train_data.get("project") or "").rstrip("/")
        run_name = str(train_data.get("name") or "").strip()
        if run_project and run_name:
            from pathlib import Path

            weights_dir = Path(run_project) / run_name / "weights"
            if weights_dir.is_dir():
                ts_files = sorted(weights_dir.glob("*.torchscript"))
                prefer_non_best = [p for p in ts_files if p.name != "best.torchscript"]
                if prefer_non_best:
                    default_model_path = str(prefer_non_best[0].resolve())
                elif ts_files:
                    default_model_path = str(ts_files[0].resolve())
    export_imgsz = export_data.get("imgsz")
    default_infer_imgsz = int(export_imgsz) if isinstance(export_imgsz, int) else int(state.get("yolo_train_imgsz", 640))

    review_data = {
        "yolo_train_env": state.get("yolo_train_env", ""),
        "model_path": default_model_path,
        "source_path": state.get("original_dataset", ""),
        "project": f"{state.get('project_root_dir', '').rstrip('/')}/{state.get('detector_name', 'unknown')}/runs/infer",
        "name": str(state.get("dataset_version") or "infer_latest"),
        "imgsz": default_infer_imgsz,
        "conf": 0.4,
        "iou": 0.1,
        "classes": [0, 1],
        "recursive": True,
        "save_labels": True,
        "save_no_detect": True,
        "add_conf_prefix": True,
        "draw_label": True,
        "hint": "可修改推理输入与阈值参数，确认后执行批量推理。",
    }
    user_response = _maybe_interrupt(state, "model_infer", review_data)
    if user_response and user_response.get("decision") == "abort":
        return {
            **_set_step_result(state, "model_infer", StepResult(
                status="skipped", summary="用户跳过模型推理", data={}
            )),
            "completed": True,
        }

    override = (user_response or {}).get("params_override") or {}
    final_params = _merge_override(
        {
            "yolo_train_env": review_data["yolo_train_env"],
            "model_path": review_data["model_path"],
            "source_path": review_data["source_path"],
            "project": review_data["project"],
            "name": review_data["name"],
            "imgsz": review_data["imgsz"],
            "conf": review_data["conf"],
            "iou": review_data["iou"],
            "classes": review_data["classes"],
            "recursive": review_data["recursive"],
            "save_labels": review_data["save_labels"],
            "save_no_detect": review_data["save_no_detect"],
            "add_conf_prefix": review_data["add_conf_prefix"],
            "draw_label": review_data["draw_label"],
            "overwrite": True,
        },
        override,
    )
    if not final_params.get("model_path"):
        return {
            **_set_step_result(state, "model_infer", StepResult(
                status="failed",
                summary="缺少 model_path（未找到导出模型，请在参数中手动填写）",
                data=final_params,
            )),
            "error": "model_infer 缺少 model_path",
            "completed": True,
        }

    try:
        infer_resp = run_yolo_infer(YoloInferRequest(**final_params))
    except Exception as exc:
        return {
            **_set_step_result(state, "model_infer", StepResult(
                status="failed", summary=f"模型推理失败：{exc}", data=final_params
            )),
            "error": f"model_infer failed: {exc}",
            "completed": True,
        }

    return {
        **_set_step_result(state, "model_infer", StepResult(
            status="ok",
            summary=f"模型推理完成：{infer_resp.detected_images}/{infer_resp.total_images} 张命中",
            data=infer_resp.model_dump(),
        )),
        "completed": True,
    }
