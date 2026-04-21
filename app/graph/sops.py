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


def _merge_profiles(
    base: dict[str, GateMode],
    updates: dict[str, dict[str, GateMode]],
) -> dict[str, dict[str, GateMode]]:
    out: dict[str, dict[str, GateMode]] = {}
    for profile, patch in updates.items():
        merged = dict(base)
        merged.update(patch)
        out[profile] = merged
    return out


SOP_REGISTRY: dict[SopId, _Sop] = {
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
            "split_dataset": "manual",
            "crop_window": "manual",
            "augment_only": "manual",
            "publish_transfer": "auto",
            "train": "manual",
            "review_result": "manual",
        },
        review_profile_default="balanced",
        review_profiles=_merge_profiles(
            base={
                "discover_classes": "manual",
                "review_labels": "manual",
                "split_dataset": "manual",
                "crop_window": "manual",
                "augment_only": "manual",
                "publish_transfer": "auto",
                "train": "manual",
                "review_result": "manual",
                "model_infer": "manual",
            },
            updates={
                "strict": {
                    "healthcheck": "manual",
                    "xml_to_yolo": "manual",
                    "publish_transfer": "manual",
                    "export_model": "manual",
                    "poll_train": "manual",
                },
                "balanced": {},
                "auto": {
                    "discover_classes": "auto",
                    "review_labels": "auto",
                    "split_dataset": "auto",
                    "crop_window": "auto",
                    "augment_only": "auto",
                    "publish_transfer": "auto",
                    "train": "auto",
                    "review_result": "auto",
                    "model_infer": "auto",
                    "xml_to_yolo": "auto",
                    "export_model": "auto",
                    "healthcheck": "auto",
                    "poll_train": "auto",
                },
            },
        ),
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
            "review_profile_default": sop.get("review_profile_default", "balanced"),
            "review_profiles": sop.get("review_profiles", {}),
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

    review_profile = user_payload.get("review_profile")
    profile_gates = sop.get("review_profiles", {})
    if review_profile and review_profile in profile_gates:
        sop_gates: dict[str, GateMode] = dict(profile_gates[review_profile])
    else:
        default_profile = sop.get("review_profile_default")
        sop_gates = dict(profile_gates.get(default_profile, sop["step_gates"]))

    existing_gates: dict[str, Any] = dict(user_payload.get("step_gates") or {})
    for step_name, mode in sop_gates.items():
        if step_name not in existing_gates:
            existing_gates[step_name] = {"mode": mode}
    if existing_gates:
        merged["step_gates"] = existing_gates
    merged.pop("review_profile", None)

    missing: list[str] = []
    for field in sop.get("required_fields", ()):
        if not merged.get(field):
            missing.append(field)

    return merged, missing
