"""Pipeline 共享状态定义。

每次 pipeline 运行对应一个独立的 PipelineState，由 LangGraph MemorySaver
以 thread_id = run_id 持久化（进程内）。

Gate 系统：
- 每个步骤有独立的 GateMode（"auto" | "manual"）。
- full_access=True 时所有步骤跳过 interrupt，自动运行。
- 某步骤为 "manual" 且 full_access=False 时，节点调用 interrupt() 暂停，
  等待调用者通过 POST /pipeline/{run_id}/confirm 恢复。
"""

from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict


GateMode = Literal["auto", "manual"]

STEP_NAMES = [
    "healthcheck",
    "discover_classes",    # 转换前审核类别
    "xml_to_yolo",
    "review_labels",       # 转换后审核 label 分布
    "split_dataset",
    "crop_augment",
    "publish_transfer",    # 内置 data.yaml 生成（原 build_yaml 已并入此步）
    "train",
    "review_result",       # 训练完成后人工验收
    "model_infer",         # 模型导出后执行批量推理
]

DEFAULT_GATES: dict[str, GateMode] = {
    "healthcheck":       "auto",
    "discover_classes":  "manual",   # 展示类名，让用户确认/填写映射
    "xml_to_yolo":       "auto",
    "review_labels":     "manual",   # 展示 label 统计，确认后才划分
    "split_dataset":     "auto",
    "crop_augment":      "auto",
    "publish_transfer":  "manual",   # 传输/发布前确认（含 yaml 生成）
    "train":             "manual",   # 训练启动前最终确认
    "review_result":     "manual",   # 训练完成后人工验收
    "model_infer":       "manual",   # 推理参数确认
}


class StepGateConfig(TypedDict):
    mode: GateMode          # "auto" | "manual"
    confirmed: bool         # 该步骤是否已被人工确认（interrupt resume 后设为 True）
    params_override: dict[str, Any]   # 人工在确认时可修改的参数覆盖


class StepResult(TypedDict, total=False):
    status: str             # "ok" | "failed" | "skipped"
    summary: str            # 人类可读摘要（展示在审核点）
    data: dict[str, Any]    # 原始返回数据


class PipelineState(TypedDict, total=False):
    # ── 运行标识 ──────────────────────────────────────────────
    run_id: str
    sop_type: Literal["local_small", "remote_slurm", "local_large"]

    # ── 核心输入参数 ──────────────────────────────────────────
    self_api_url: str
    original_dataset: str           # XML + 图像所在的根目录
    detector_name: str
    project_root_dir: str
    execution_mode: Literal["local", "remote_sftp", "remote_slurm"]

    # 训练参数
    yolo_train_env: str
    yolo_train_model: str
    yolo_train_epochs: int
    yolo_train_imgsz: int
    yolo_export_after_train: bool
    split_mode: Literal["train_val", "train_val_test", "train_only"]
    train_ratio: float
    val_ratio: float

    # 远端参数（remote_sftp / remote_slurm 时使用）
    remote_host: str | None
    remote_username: str | None
    remote_private_key_path: str | None
    remote_project_root_dir: str | None

    # ── Gate 配置 ─────────────────────────────────────────────
    full_access: bool                        # True = 跳过所有 manual gate
    step_gates: dict[str, StepGateConfig]   # 每步 gate 配置

    # ── 类别映射（discover_classes → xml_to_yolo 之间填写） ──
    discovered_classes: list[str] | None        # discover 步骤扫描出的原始类名
    class_name_map: dict[str, str] | None       # 用户填写的映射，e.g. {louyou1: louyou}
    final_classes: list[str] | None             # 映射后的目标类列表，e.g. [louyou]（与 class_index_map 二选一）
    class_index_map: dict[str, int] | None      # 逻辑类名 → YOLO 索引（与 final_classes/classes 二选一）
    training_names: list[str] | None            # 写入 classes.txt / yaml 的显示名（可选）

    # ── 各阶段输出 ────────────────────────────────────────────
    step_results: dict[str, StepResult]

    # xml_to_yolo 输出的关键路径，供后续步骤引用
    labels_dir: str | None
    dataset_version: str | None              # split 后版本目录名
    split_output_dir: str | None
    yaml_path: str | None                    # publish_transfer 内置生成
    train_task_id: str | None                # async train task id

    # ── 流程控制 ──────────────────────────────────────────────
    current_step: str | None
    pending_review: dict[str, Any] | None    # interrupt 时暴露给用户的快照
    error: str | None
    completed: bool
