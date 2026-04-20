# self_api 架构文档（脱离 n8n · LangGraph 重构版）

> **状态**：目标架构（Target Architecture）  
> **版本**：v1.0 · 2026-04-20  
> **核心目标**：把 n8n + skill + API 三层冗余合并为 **一个代码优先、带 HITL 的 LangGraph 管线**，保留现有数据处理、发布、训练能力。

---

## 1. 设计初衷

### 1.1 旧架构的痛点

```
┌───────────────────────────────────────────────────────────────┐
│ 用户 / n8n UI                                                  │
│   ↓ 触发（Manual / Webhook）                                   │
│ n8n 工作流 (JSON)   ← 每种 SOP 一条，节点一多就难维护          │
│   ↓ HTTP                                                       │
│ self_api (FastAPI)  ← 每个服务一个 /xxx 与 /xxx/async 端点    │
│   ↓                                                           │
│ services/*.py       ← 实际业务                                │
│                                                              │
│ skills/             ← 另一套“说明书”，和 n8n 工作流重复     │
└───────────────────────────────────────────────────────────────┘
```

问题：

1. **多份流程**：n8n 小图/大图/远程/本地，至少 4 条 JSON，参数同步靠复制。
2. **人工审核只能建新流**：每加一个人工审核点，就要再写一条 n8n 工作流。
3. **n8n 表达能力有限**：循环/轮询/条件分支只能用 Switch + Wait 拼，可读性差。
4. **两套编排语义**：skills SOP 文档 和 n8n JSON 经常不一致。

### 1.2 新架构目标

- **一套代码定义全部 SOP**（不同 SOP = 同一图的不同 gate 配置）。
- **Human-in-the-Loop 原生化**：任何节点都可 `interrupt()` 暂停，人工确认后 `resume`。
- **审核点 = 通用门控机制**：`auto` 自动跑 / `manual` 等确认 / `full_access` 全局跳过。
- **REST 面向最终用户**：前端与智能体都只对接 `/api/v1/pipeline/*`。

---

## 2. 架构总览

```
┌────────────────────────────────────────────────────────────────────────┐
│                       前端 / 智能体 / CLI                              │
│                 (train-ui · 外部 agent · curl)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP/JSON (JWT 鉴权)
          ┌─────────────────────────┼──────────────────────────┐
          ▼                         ▼                          ▼
  ┌──────────────┐         ┌───────────────┐           ┌──────────────┐
  │ /pipeline/*  │         │ /preprocess/* │           │ /tasks/*     │
  │ 编排入口     │         │ 原子能力      │           │ 异步任务状态 │
  │ (LangGraph)  │         │ (FastAPI)     │           │              │
  └───────┬──────┘         └───────┬───────┘           └──────┬───────┘
          │                        │                          │
          ▼                        ▼                          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    app/graph/   (编排层)                        │
  │  PipelineState (TypedDict)                                      │
  │  StepGateConfig (auto/manual + params_override)                 │
  │  StateGraph: healthcheck → ... → review_result                  │
  │  MemorySaver (thread_id=run_id 持久化)                          │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │ 直接函数调用（同进程）
                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                   app/services/   (原子能力)                    │
  │  xml_to_yolo · discover_xml_classes · split_yolo_dataset        │
  │  yolo_sliding_window · yolo_augment · build_yolo_yaml           │
  │  publish_yolo_dataset · yolo_train · remote_sbatch_yolo_train   │
  │  task_manager · remote_transfer · remote_unzip · ...            │
  └──────────────────────────────────────────────────────────────────┘
```

### 2.1 三层职责

| 层 | 目录 | 职责 |
|---|---|---|
| **编排层** | `app/graph/` | 定义 SOP 顺序、审核点、条件分支；**唯一** 的 workflow 定义 |
| **API 层** | `app/api/v1/endpoints/` | Pipeline 入口、原子能力 REST、鉴权、任务查询 |
| **能力层** | `app/services/` | 纯业务函数，**不感知编排**，可单独调用 |

---

## 3. 目标目录结构

```
self_api/
├── app/
│   ├── main.py                     # FastAPI 启动入口
│   ├── api/v1/
│   │   ├── router.py               # 聚合所有路由
│   │   └── endpoints/
│   │       ├── pipeline.py         # ★ 新核心：/pipeline/run|status|confirm|abort
│   │       ├── preprocess.py       # 保留：原子能力（给智能体/调试用）
│   │       ├── tasks.py            # 异步任务状态查询
│   │       ├── auth.py             # JWT 登录
│   │       ├── files.py            # 文件浏览/下载
│   │       ├── artifacts.py        # 训练产物下载
│   │       └── system.py           # /healthz
│   │
│   ├── graph/                      # ★ 编排层（LangGraph）
│   │   ├── state.py                # PipelineState + Gate 定义
│   │   ├── nodes.py                # 11 个节点函数
│   │   ├── pipeline.py             # build_graph() + compiled_graph 单例
│   │   └── sops.py                 # 预设 SOP 模板（4 个），减少用户填参
│   │
│   ├── schemas/
│   │   ├── pipeline.py             # PipelineRunRequest / Confirm / Status
│   │   ├── preprocess.py           # 原子能力 schema
│   │   └── ...
│   │
│   ├── services/                   # 原子能力（不变）
│   │   ├── discover_xml_classes.py # ★ 新：扫描 XML 类名，配合 class_name_map
│   │   ├── xml_to_yolo.py          # 支持 class_name_map 多对一
│   │   ├── split_yolo_dataset.py
│   │   ├── yolo_sliding_window.py
│   │   ├── yolo_augment.py
│   │   ├── build_yolo_yaml.py
│   │   ├── publish_yolo_dataset.py
│   │   ├── yolo_train.py
│   │   ├── remote_sbatch_yolo_train.py
│   │   ├── remote_transfer.py / remote_unzip.py
│   │   └── task_manager.py         # 本地异步任务池
│   │
│   ├── core/
│   │   ├── config.py               # 环境变量 & 配置
│   │   ├── security.py             # JWT 生成/校验
│   │   ├── path_safety.py          # 路径白名单
│   │   └── logging.py
│   │
│   └── static/
│       └── train-ui/               # 轻量前端：可视化 pipeline + 审核点
│
├── docs/
│   ├── architecture.md             # ← 本文
│   ├── pipeline-api.md             # REST 用法 + curl 示例
│   └── migration-from-n8n.md       # 从 n8n JSON 迁移指引
│
├── skills/
│   ├── sop-workflow/               # 高层编排技能（改引用 LangGraph）
│   ├── data-preprocess/            # 原子能力技能（保留）
│   └── ultralytics-yolo-modes/     # 训练参数技能（保留）
│
├── tests/
│   ├── test_pipeline_graph.py      # ★ 待补：图结构 + gate + interrupt
│   ├── test_discover_xml_classes.py
│   └── test_services_*.py
│
├── compose.yaml                    # Docker 编排（UID/GID 映射 + 卷）
├── Dockerfile
├── pyproject.toml                  # 依赖：+langgraph
└── README.md
```

**关键变更**：

- ✅ 删除 `docs/n8n/` 里所有“SOP 级别”工作流 JSON（仅保留 Webhook 触发版本作为对外兼容）。
- ✅ 新增 `app/graph/` 作为唯一编排定义。
- ✅ `app/api/v1/endpoints/pipeline.py` 成为**面向用户的主入口**；`preprocess.py` 退化为"给智能体/脚本用的原子能力面板"。

---

## 4. 核心抽象

### 4.1 PipelineState（单一共享状态）

```python
class PipelineState(TypedDict, total=False):
    run_id: str
    # 输入
    original_dataset, detector_name, project_root_dir, execution_mode, ...
    # 训练参数
    yolo_train_model, yolo_train_epochs, yolo_train_imgsz, ...
    # 远端参数
    remote_host, remote_username, remote_private_key_path, ...
    # 类别映射
    class_name_map, final_classes, discovered_classes
    # Gate
    full_access: bool
    step_gates: dict[step_name, StepGateConfig]
    # 各阶段产出
    step_results: dict[step_name, StepResult]
    labels_dir, split_output_dir, yaml_path, train_task_id, ...
    # 流程控制
    current_step, completed, error, pending_review
```

每次 `/pipeline/run` 产生一个唯一 `run_id`，LangGraph `MemorySaver` 以 `thread_id=run_id` 持久化状态（后续可切到 Postgres/SQLite Checkpointer）。

### 4.2 StepGateConfig（通用门控）

```python
class StepGateConfig(TypedDict):
    mode: "auto" | "manual"
    confirmed: bool
    params_override: dict[str, Any]
```

**门控判定**：

```
if state.full_access:           # 全局放行
    continue
elif gate.mode == "manual":     # 等待人工
    user = interrupt({...})      # LangGraph 原生暂停
    apply user.params_override
else:                            # auto
    continue
```

**默认配置**（见 `app/graph/state.py::DEFAULT_GATES`）：

| 步骤 | 默认 mode | 理由 |
|---|---|---|
| healthcheck | auto | 只是探测 |
| **discover_classes** | **manual** | 展示类名让用户配 `class_name_map` |
| xml_to_yolo | auto | 纯转换 |
| **review_labels** | **manual** | 审核转换结果 |
| split_dataset | auto | |
| crop_augment | auto | |
| build_yaml | auto | |
| **publish_transfer** | **manual** | 启动远程 SFTP 前确认 |
| **train** | **manual** | 训练前最终确认（会消耗 GPU） |
| **review_result** | **manual** | 训练完成后验收 |

### 4.3 节点模式（统一写法）

```python
def node_xxx(state: PipelineState) -> dict[str, Any]:
    # 1. 组装审核数据
    review_data = {...}
    # 2. 门控
    user = _maybe_interrupt(state, "xxx", review_data)
    if user and user["decision"] == "abort":
        return {..., "completed": True}
    override = (user or {}).get("params_override") or {}
    # 3. 合并参数 → 调 service
    final_params = _merge_override(defaults, override)
    resp = run_xxx_service(...)
    # 4. 写回 state
    return _set_step_result(state, "xxx", StepResult(...))
```

---

## 5. 流程图（数据 → 训练完整 SOP）

```
 ┌─────────────┐
 │ /pipeline/  │
 │   run       │ ── run_id=xxx, 初始 state
 └──────┬──────┘
        ▼
 ┌─────────────┐      ┌───────────────┐
 │ healthcheck │─failed→ END (error)
 └──────┬──────┘
        ▼
 ┌─────────────────────┐   manual
 │ discover_classes    │────────→ interrupt → 用户填 class_name_map / final_classes
 └──────┬──────────────┘
        ▼
 ┌─────────────┐
 │ xml_to_yolo │  ← 应用 class_name_map 多对一
 └──────┬──────┘
        ▼
 ┌─────────────┐   manual
 │review_labels│────────→ interrupt → 用户确认 label 统计
 └──────┬──────┘
        ▼
 ┌──────────────┐
 │ split_dataset│  train/val 拆分（时间戳版本）
 └──────┬───────┘
        ▼
 ┌─────────────┐
 │ crop_augment│  滑窗裁剪 + train 增强
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │ build_yaml  │  生成 data.yaml（含路径前缀替换）
 └──────┬──────┘
        ▼
 ┌─────────────────┐   manual
 │ publish_transfer│────→ interrupt → 用户确认发布（local/remote_sftp）
 └──────┬──────────┘
        ▼
 ┌──────────┐   manual
 │  train   │─────→ interrupt → 用户确认训练参数 → submit async task
 └─────┬────┘
       ▼
 ┌────────────┐
 │ poll_train │  每 30s 查 /tasks/{id}，最长 12h
 └─────┬──────┘
       ▼
 ┌────────────────┐   manual
 │ review_result  │─────→ interrupt → 用户验收
 └─────┬──────────┘
       ▼
      END
```

---

## 6. REST API 对外契约

### 6.1 端点

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/v1/pipeline/run` | 启动 run，返回 `run_id` |
| GET | `/api/v1/pipeline/{run_id}` | 查询状态 & `interrupted=true` 时的 review 数据 |
| POST | `/api/v1/pipeline/{run_id}/confirm` | 人工确认 / 覆盖参数 / abort |
| POST | `/api/v1/pipeline/{run_id}/abort` | 强制终止 |

### 6.2 典型时序

```
Client                             Server
  │                                   │
  │── POST /pipeline/run ────────────▶│  run_id=abc
  │                                   │  运行到 discover_classes 停住
  │◀── 202 {interrupted:true,         │
  │      pending_review:{class_names:[louyou1,louyou2,...]}}
  │                                   │
  │── POST /pipeline/abc/confirm ────▶│
  │   {decision:confirm,              │
  │    params_override:{              │
  │      class_name_map:{louyou1:louyou,louyou2:louyou},
  │      final_classes:[louyou]}}     │
  │                                   │  继续到 review_labels 停住
  │◀── {interrupted:true, pending_review:{...}}
  │                                   │
  │── POST confirm ──────────────────▶│  (连续几次 confirm)
  │                                   │  ...直到 train 提交后跑 poll_train
  │                                   │  ...直到 review_result 停住
  │── GET /pipeline/abc ─────────────▶│
  │◀── {completed:false, interrupted:true, current_step:review_result,
  │     step_results:{train:{...},poll_train:{summary:"succeeded"}}}
  │                                   │
  │── POST confirm ──────────────────▶│  验收通过
  │◀── {completed:true, error:null}
```

### 6.3 与旧 `/preprocess/*` 的关系

- `/preprocess/xml-to-yolo`、`/preprocess/yolo-train/async` 等**全部保留**。
- 它们是**原子能力**，被 LangGraph nodes 直接 import 调用（非 HTTP 自调）。
- 也可继续给脚本、curl、智能体单独调。
- **新项目推荐**：业务入口一律走 `/pipeline/*`，只有排查/重跑单步时才调 `/preprocess/*`。

---

## 7. 与智能体 / Skill 的协作

新定位：

| 组件 | 新定位 |
|---|---|
| **sop-workflow skill** | 给用户出"plan → confirm"的长 SOP 文字；confirm 后调 `POST /pipeline/run` |
| **data-preprocess skill** | 单步调试 `/preprocess/*` 原子能力 |
| **ultralytics-yolo-modes skill** | 生成 YOLO CLI 命令（纯本地训练/离线场景） |
| **n8n-workflow-map 参考** | 仅保留 LangGraph 管线 + 可选 Webhook JSON |

智能体对接建议：

```
  用户自然语言
      │
      ▼
  sop-workflow → 产出计划 + 询问 full_access
      │
      ▼ 用户 confirm
  POST /pipeline/run {full_access:true 或 step_gates:{...}}
      │
      ▼  loop:
  GET /pipeline/{id}
      │ if interrupted: 展示 pending_review，让用户做决定
      │ if completed: 展示 step_results
      ▼
  POST /pipeline/{id}/confirm
```

---

## 8. 迁移路线图

| 阶段 | 任务 | 状态 |
|---|---|---|
| **P0 打地基** | 新增 `app/graph/`、`/pipeline/*` 端点、`class_name_map`、`discover_xml_classes` | ✅ 已完成 |
| P0 | 删除旧 n8n JSON（保留 Webhook 一份） | ✅ 已完成 |
| **P1 测试** | `tests/test_pipeline_graph_api.py`（gate/interrupt/resume/abort/full_access）+ `test_discover_xml_classes_api.py` + `test_xml_to_yolo_class_name_map_api.py` | ✅ 已完成 |
| **P2 前端** | `app/static/pipeline-ui/`（独立页 `/pipeline-ui`）：启动表单 + run 列表 + pending_review 卡片 + confirm/abort | ✅ 已完成 |
| **P3 SOP 预设** | `app/graph/sops.py` 注册 4 个模板（local-small-baseline / local-large-sliding-window / remote-slurm-iter / full-auto-smoke），对外暴露 `GET /pipeline/sops` 与 `POST /pipeline/sops/{sop_id}/run`；前端 SOP 选择器 | ✅ 已完成 |
| **P4 持久化** | 可选：把 `MemorySaver` 换成 SQLite/Postgres checkpointer，让 run 跨进程重启可恢复 | 🔴 规划 |
| P4 | 可选：Server-Sent Events `/pipeline/{id}/events` 实时推送，替代轮询 | 🔴 规划 |
| **P5 清理** | 彻底删除 `docs/n8n/模型训练工作流（Webhook）.json`（确认外部无依赖后） | 🔴 规划 |
| P5 | Skill 文档统一以 LangGraph 为主叙事 | 🟡 部分 |

---

## 9. 关键权衡与决策记录

1. **为什么选 LangGraph 而不是 Prefect/Airflow/Dagster？**  
   - Native `interrupt()` + checkpointer 直接支持 HITL；其他框架要自己造。
   - 轻量、同进程、无额外服务；Prefect/Airflow 需要独立 scheduler。
   - 以 Python 函数为一等公民，与现有 services/*.py 直接复用。

2. **为什么原子能力 REST 仍保留？**  
   - 调试/重跑/绕过审核时有用；也留给外部脚本兼容。
   - 对编排层性能无损（LangGraph 直接函数调用，不走 HTTP）。

3. **为什么 gate 配置内嵌在 state，而不是独立资源？**  
   - 一次 run 的配置天然属于该 run；无需引入"配置版本"概念。
   - `full_access` + `step_gates` 组合足够表达所有场景（全自动 / 精选几步 / 全人工）。

4. **MemorySaver 进程内 vs 持久化？**  
   - P0 阶段内存足够（run 寿命通常 < 12h，重启可重跑）。
   - P4 规划切 SQLite，获得跨进程恢复能力。

5. **class_name_map vs `classes` 字段**  
   - `class_name_map` 解决"XML 里叫 louyou1/2/3，训练时要合并成 louyou 并固定索引 0"。
   - `final_classes` 决定索引顺序；两者独立，组合表达完整映射。

---

## 10. 快速使用（给开发者）

**启动**：

```bash
make run                 # 开发模式：uvicorn --reload
# 或
docker compose up -d     # 生产：含 UID/GID 映射、.ssh 卷挂载
```

**最简 run**（全自动）：

```bash
TOKEN=$(curl -s -X POST http://localhost:8666/api/v1/auth/login \
  -d '{"username":"admin","password":"xxx"}' | jq -r .access_token)

curl -X POST http://localhost:8666/api/v1/pipeline/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "original_dataset": "/data/louyou_raw",
    "detector_name": "zxj_louyou",
    "project_root_dir": "/data/workspace",
    "execution_mode": "local",
    "yolo_train_env": "yolo_pose",
    "yolo_train_epochs": 100,
    "full_access": true
  }'
```

**带人工审核的 run**：

去掉 `full_access`，轮询 `GET /pipeline/{run_id}`，看到 `interrupted:true` 就 `POST /{run_id}/confirm`。
