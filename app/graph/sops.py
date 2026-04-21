"""预设 SOP（Standard Operating Procedure）模板。

每个 SOP 模板描述：
  - 业务场景（scenario）：用于用户选择
  - 默认参数 overrides：相对于 PipelineRunRequest 默认值的差异
  - 默认 step_gates：推荐的 auto/manual 组合
  - full_access 默认值

使用方式（REST）：
  GET  /api/v1/pipeline/sops               → 列出全部 SOP
  POST /api/v1/pipeline/sops/{sop_id}/run  → 用 SOP 默认值 + 用户 overrides 启动
"""

from __future__ import annotations

from typing import Any

from app.graph.state import GateMode


SopId = str


class _Sop(dict):
    """简单的 dict 包装，用于类型提示。"""


SOP_REGISTRY: dict[SopId, _Sop] = {
    "local-small-baseline": _Sop(
        id="local-small-baseline",
        name="小图本地 baseline 训练",
        description=(
            "适用于：图像较小（~640）、数据量不大、在本机 GPU 上训练的 baseline。"
            "默认启用 discover/review_labels/train/review_result 人工审核点。"
        ),
        defaults={
            "execution_mode": "local",
            "split_mode": "train_val",
            "train_ratio": 0.85,
            "val_ratio": 0.15,
            "yolo_train_model": "yolo11s.pt",
            "yolo_train_epochs": 100,
            "yolo_train_imgsz": 640,
            "enable_sliding_window": False,
            "full_access": False,
        },
        step_gates={
            "discover_classes": "manual",
            "review_labels": "manual",
            "publish_transfer": "auto",
            "train": "manual",
            "review_result": "manual",
        },
    ),
    "local-large-sliding-window": _Sop(
        id="local-large-sliding-window",
        name="大图滑窗裁剪 · 本地训练",
        description=(
            "适用于：横向长条图（例如 TV 类检测），需要滑窗裁剪为 640/800 小图后再训练。"
            "默认 imgsz=800，epochs=150，其他审核点与 baseline 相同。"
            "SOP 展示链路：标注检查与转换修改 → 数据集划分 → 滑窗裁剪 → 增强"
            " → 发布数据集 → 训练参数审核修改并确认 → 训练 → 模型导出 → 模型推理。"
            "其中模型导出为独立 export_model 阶段：训练结果验收通过后自动执行 yolo-export"
            "（读取 args.yaml 的 imgsz，并按 data 对应数据集 yaml 名命名 torchscript）；"
            "模型推理发生在 model_infer 阶段（支持单图、递归目录、多路径列表输入）。"
        ),
        defaults={
            "execution_mode": "local",
            "split_mode": "train_val",
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "yolo_train_model": "yolo11s.pt",
            "yolo_train_epochs": 150,
            "yolo_train_imgsz": 800,
            "yolo_export_after_train": True,
            "enable_sliding_window": True,
            "full_access": False,
        },
        step_gates={
            "discover_classes": "manual",
            "review_labels": "manual",
            "publish_transfer": "auto",
            "train": "manual",
            "review_result": "manual",
        },
    ),
    "remote-slurm-iter": _Sop(
        id="remote-slurm-iter",
        name="远程 Slurm 迭代训练",
        description=(
            "适用于：数据集通过 SFTP 传输至远端，远端通过 sbatch 提交训练任务。"
            "requires remote_host / remote_username / remote_private_key_path / remote_project_root_dir。"
        ),
        defaults={
            "execution_mode": "remote_slurm",
            "split_mode": "train_val",
            "train_ratio": 0.85,
            "val_ratio": 0.15,
            "yolo_train_model": "yolo11s.pt",
            "yolo_train_epochs": 150,
            "yolo_train_imgsz": 640,
            "enable_sliding_window": False,
            "full_access": False,
        },
        step_gates={
            "discover_classes": "manual",
            "review_labels": "manual",
            "publish_transfer": "manual",
            "train": "manual",
            "review_result": "manual",
        },
        required_fields=(
            "remote_host",
            "remote_username",
            "remote_private_key_path",
            "remote_project_root_dir",
        ),
    ),
    "full-auto-smoke": _Sop(
        id="full-auto-smoke",
        name="全自动 smoke 测试",
        description=(
            "全部步骤 auto、full_access=True，仅用于 CI 或快速冒烟验证，"
            "不建议对正式数据使用（无人工把关）。"
        ),
        defaults={
            "execution_mode": "local",
            "split_mode": "train_val",
            "train_ratio": 0.85,
            "val_ratio": 0.15,
            "yolo_train_model": "yolo11n.pt",
            "yolo_train_epochs": 1,
            "yolo_train_imgsz": 320,
            "enable_sliding_window": False,
            "full_access": True,
        },
        step_gates={},
    ),
}


def list_sops() -> list[dict[str, Any]]:
    """返回所有 SOP 的摘要列表（不含内部字段）。"""
    out: list[dict[str, Any]] = []
    for sop in SOP_REGISTRY.values():
        out.append({
            "id": sop["id"],
            "name": sop["name"],
            "description": sop["description"],
            "defaults": sop["defaults"],
            "step_gates": sop["step_gates"],
            "required_fields": tuple(sop.get("required_fields", ())),
        })
    return out


def get_sop(sop_id: SopId) -> dict[str, Any] | None:
    sop = SOP_REGISTRY.get(sop_id)
    if sop is None:
        return None
    return dict(sop)


def apply_sop_defaults(
    sop_id: SopId,
    user_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """将 SOP 的默认值合并进用户 payload（用户字段优先），返回 (merged_payload, missing_required)。

    Returns:
        merged_payload: 合并后的 payload，可直接构造 PipelineRunRequest
        missing_required: 缺失的必填字段名列表（SOP.required_fields ∩ 空字段）
    """
    sop = SOP_REGISTRY.get(sop_id)
    if sop is None:
        raise KeyError(f"Unknown SOP id: {sop_id}")

    merged: dict[str, Any] = dict(sop["defaults"])
    for key, value in user_payload.items():
        if value is None:
            continue
        merged[key] = value

    existing_gates: dict[str, Any] = dict(user_payload.get("step_gates") or {})
    sop_gates: dict[str, GateMode] = sop["step_gates"]
    for step_name, mode in sop_gates.items():
        if step_name not in existing_gates:
            existing_gates[step_name] = {"mode": mode}
    if existing_gates:
        merged["step_gates"] = existing_gates

    missing: list[str] = []
    for field in sop.get("required_fields", ()):
        if not merged.get(field):
            missing.append(field)

    return merged, missing
