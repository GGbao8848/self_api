# Workflow Map

> 项目已 **完全脱离 n8n**，LangGraph 是唯一编排引擎。所有 SOP 均通过 `/api/v1/pipeline/*` 发起。

## LangGraph 管线接口

| 操作 | 端点 |
|------|------|
| 列出 SOP 模板 | `GET /api/v1/pipeline/sops` |
| 用 SOP 模板启动 | `POST /api/v1/pipeline/sops/{sop_id}/run` |
| 启动自定义 run | `POST /api/v1/pipeline/run` |
| 查询状态 | `GET /api/v1/pipeline/{run_id}` |
| 实时事件流（SSE）| `GET /api/v1/pipeline/{run_id}/events` |
| 人工确认/覆盖参数 | `POST /api/v1/pipeline/{run_id}/confirm` |
| 终止 run | `POST /api/v1/pipeline/{run_id}/abort` |

## 管线节点顺序

1. `healthcheck` — API 可达性检测
2. `discover_classes`（HITL 审核点）— 扫描 XML，返回全部类名及频次；可在审核时填写 `class_name_map` / `final_classes`
3. `xml_to_yolo` — XML → YOLO TXT 转换（含多对一重命名）
4. `review_labels`（HITL 审核点）— 展示 label 分布，确认后才划分
5. `split_dataset` — 按比例划分 train/val
6. `crop_augment` — 滑窗裁剪 + 数据增强
7. `build_yaml` — 生成 YOLO `data.yaml`
8. `publish_transfer`（HITL 审核点）— 本地落盘 / zip + SFTP + 远端解压
9. `train`（HITL 审核点）— 启动异步训练任务
10. `poll_train` — 轮询训练任务状态（30 s 间隔，最长 12 h）
11. `review_result`（HITL 审核点）— 查看训练结果，决定是否接受

每个节点都支持 **gate** 机制：`auto`（跳过确认）或 `manual`（等待人工点击确认后才执行）。设置 `full_access: true` 可跳过所有门控，全自动运行。

## 预设 SOP

| sop_id | 场景 |
|--------|------|
| `local-small-baseline` | 小图（~640）+ 本机 GPU 的 baseline |
| `local-large-sliding-window` | 长条/大图滑窗裁剪 + 本机训练（imgsz=800） |
| `remote-slurm-iter` | 数据经 SFTP 推送至远端 + Slurm sbatch 训练 |
| `full-auto-smoke` | CI/冒烟：全 auto + `full_access=true`（不建议生产数据使用） |

## 从 SOP 映射到调用方式

| 场景 | 推荐方案 |
|------|---------|
| 新数据集端到端训练（含 HITL 审核） | `POST /pipeline/sops/local-small-baseline/run` |
| 长条大图训练 | `POST /pipeline/sops/local-large-sliding-window/run` |
| 远程 Slurm 集群训练 | `POST /pipeline/sops/remote-slurm-iter/run` |
| 仅需快速自动跑完 | `full_access: true` 或 `sops/full-auto-smoke/run` |
| 自定义 gate/参数 | `POST /pipeline/run` 带 `step_gates` / `class_name_map` / etc. |

## 何时调用各端点

- **`GET /pipeline/sops`**：面向用户列出可选模板。
- **`POST /pipeline/sops/{id}/run`**：最常用入口，只需填必填字段（original_dataset / detector_name / project_root_dir / yolo_train_env）。
- **`POST /pipeline/run`**：需要精细化覆盖 gate 时用。
- **`GET /pipeline/{id}/events`**：前端/agent 实时监听状态变化，优于轮询。
- **`POST /pipeline/{id}/confirm`**：在 HITL 审核点响应，可附 `params_override`。
- **`POST /pipeline/{id}/abort`**：终止 run。

Use this file when the user asks for:

- workflow mapping / node-by-node endpoint order
- HITL / human review gate configuration
- async orchestration with polling or SSE
- choosing a SOP template for a given scenario
