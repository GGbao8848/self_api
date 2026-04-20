# Workflow Map

## 当前编排方案：LangGraph 管线（推荐）

项目已从 n8n 迁移到 **LangGraph** 作为编排引擎，原 n8n 工作流仅保留 Webhook 触发版本供参考。

### LangGraph 管线接口

| 操作 | 端点 |
|------|------|
| 启动管线 | `POST /api/v1/pipeline/run` |
| 查询状态 | `GET /api/v1/pipeline/{run_id}` |
| 人工确认/覆盖步骤参数 | `POST /api/v1/pipeline/{run_id}/confirm` |
| 终止管线 | `POST /api/v1/pipeline/{run_id}/abort` |

### 管线节点顺序

1. `healthcheck` — API 可达性检测
2. `discover_classes` — 扫描 XML，返回全部类名及频次（支持 class_name_map 预览）
3. `xml_to_yolo` — XML → YOLO TXT 转换（含 class_name_map 多对一重命名）
4. `review_labels`（HITL 审核点）— 人工确认 label 目录
5. `split_dataset` — 按比例划分 train/val
6. `crop_augment` — 滑窗裁剪 + 数据增强
7. `build_yaml` — 生成 YOLO data.yaml
8. `publish_transfer` — 打包、zip、远程 SFTP 传输、远端解压
9. `train` — 本地 yolo-train/async 或 远程 remote-sbatch-yolo-train
10. `poll_train` — 轮询训练任务状态（30 s 间隔）
11. `review_result`（HITL 审核点）— 查看训练结果，决定是否接受

每个节点都支持 **gate** 机制：`auto`（跳过确认）或 `manual`（等待人工点击确认后才执行）。设置 `full_access: true` 可跳过所有门控，全自动运行。

---

## 保留的 n8n 工作流（Webhook 触发）

- `docs/n8n/模型训练工作流（Webhook）.json` — Webhook 触发的完整训练管线（本地训练 + 远程 Slurm，yolo-train 已更新为 /async + 轮询节点）

### 何时参考 n8n 文件

- 用户需要查看旧有 n8n 节点顺序或具体参数值时
- 比较 Webhook 驱动模式与 LangGraph REST 模式的差异时

---

## 从 SOP 映射到编排方案

| 场景 | 推荐方案 |
|------|---------|
| 新数据集端到端训练（含 HITL 审核） | LangGraph `/pipeline/run` |
| 仅需快速自动跑完 | LangGraph + `full_access: true` |
| 已有旧 n8n 实例、需要 Webhook 触发 | `docs/n8n/模型训练工作流（Webhook）.json` |

Use them when the user asks for:

- workflow mapping / node-by-node endpoint order
- HITL / human review gate configuration
- async orchestration with polling
- a JSON workflow that matches one of the standard SOPs
