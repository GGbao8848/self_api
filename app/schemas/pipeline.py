"""Pipeline REST API 的请求/响应 schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GateMode = Literal["auto", "manual"]


class StepGateInput(BaseModel):
    mode: GateMode = "manual"


class PipelineRunRequest(BaseModel):
    """启动一次 pipeline 运行。"""

    original_dataset: str = Field(description="原始数据集根目录（含 images/ 和 xmls/）")
    detector_name: str = Field(description="检测器名称，用于目录命名，e.g. nzxj_louyou")
    project_root_dir: str = Field(description="项目根目录（训练结果写入此目录下）")

    execution_mode: Literal["local", "remote_sftp", "remote_slurm"] = Field(
        default="local", description="执行模式"
    )

    yolo_train_env: str = Field(description="训练所用 conda 环境名，e.g. yolo_pose")
    yolo_train_model: str = Field(default="yolo11s.pt", description="YOLO 底模")
    yolo_train_epochs: int = Field(default=100, ge=1)
    yolo_train_imgsz: int = Field(default=640, ge=1)
    split_mode: Literal["train_val", "train_val_test", "train_only"] = Field(default="train_val")
    train_ratio: float = Field(default=0.85, gt=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    # 远端参数（remote 模式时填写）
    remote_host: str | None = None
    remote_username: str | None = None
    remote_private_key_path: str | None = None
    remote_project_root_dir: str | None = None

    # 类别映射（可提前填写，也可在 discover_classes 审核点填写）
    class_name_map: dict[str, str] | None = Field(
        default=None,
        description="类名重映射，e.g. {\"louyou1\": \"louyou\"}",
    )
    final_classes: list[str] | None = Field(
        default=None,
        description="目标类列表（决定索引顺序），e.g. [\"louyou\"]",
    )

    # Gate 配置
    full_access: bool = Field(
        default=False,
        description="True = 跳过所有人工审核点，自动运行所有步骤",
    )
    step_gates: dict[str, StepGateInput] | None = Field(
        default=None,
        description="按步骤覆盖 gate 模式，key 为步骤名，value 为 {mode: auto|manual}",
    )

    self_api_url: str = Field(
        default="http://127.0.0.1:8666",
        description="self_api 服务地址（用于 healthcheck）",
    )


class PipelineConfirmRequest(BaseModel):
    """人工确认某个审核点，可选覆盖参数。"""

    decision: Literal["confirm", "abort"] = "confirm"
    params_override: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "可覆盖当前步骤的参数，例如在 discover_classes 步骤时提供：\n"
            "  class_name_map: {louyou1: louyou, louyou2: louyou}\n"
            "  final_classes: [louyou]"
        ),
    )


class PipelineStepStatus(BaseModel):
    status: str
    summary: str
    data: dict[str, Any] = {}


class PipelineStatusResponse(BaseModel):
    run_id: str
    current_step: str | None
    completed: bool
    error: str | None
    pending_review: dict[str, Any] | None = None
    step_results: dict[str, PipelineStepStatus] = {}
    interrupted: bool = Field(description="是否在等待人工确认（interrupt 暂停中）")


class SopSummary(BaseModel):
    """SOP 预设模板摘要。"""

    id: str
    name: str
    description: str
    defaults: dict[str, Any]
    step_gates: dict[str, GateMode] = {}
    required_fields: list[str] = []


class SopListResponse(BaseModel):
    sops: list[SopSummary]


class SopRunRequest(BaseModel):
    """使用 SOP 启动 run；用户仅需填写 SOP 默认值中未覆盖的字段。

    用户字段会覆盖 SOP 默认值；step_gates 深合并（以用户优先）。
    """

    original_dataset: str = Field(description="原始数据集根目录")
    detector_name: str = Field(description="检测器名称")
    project_root_dir: str = Field(description="项目根目录")
    yolo_train_env: str = Field(description="conda 训练环境名")

    execution_mode: Literal["local", "remote_sftp", "remote_slurm"] | None = None
    yolo_train_model: str | None = None
    yolo_train_epochs: int | None = Field(default=None, ge=1)
    yolo_train_imgsz: int | None = Field(default=None, ge=1)
    split_mode: Literal["train_val", "train_val_test", "train_only"] | None = None
    train_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    val_ratio: float | None = Field(default=None, ge=0.0, le=1.0)

    remote_host: str | None = None
    remote_username: str | None = None
    remote_private_key_path: str | None = None
    remote_project_root_dir: str | None = None

    class_name_map: dict[str, str] | None = None
    final_classes: list[str] | None = None

    full_access: bool | None = None
    step_gates: dict[str, StepGateInput] | None = None
    self_api_url: str | None = None
