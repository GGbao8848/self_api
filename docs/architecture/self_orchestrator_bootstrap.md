# self_orchestrator 快速建设指南

本文档用于指导新建 `self_orchestrator` 项目。目标不是一次性做成“大而全”平台，而是在最短路径上搭起一个可运行、可扩展、可对接 `self_api` 的 LangGraph 编排层。

## 1. 定位

`self_orchestrator` 是编排层，不直接执行图像/数据集处理，而是负责：

- 接收上层请求
- 组织 LangGraph 工作流
- 调用 `self_api` HTTP 接口
- 提交异步任务并轮询状态
- 汇总结果并返回给调用方

与 `self_api` 的职责边界如下：

- `self_api`: 业务执行层，负责真实文件处理
- `self_orchestrator`: 流程编排层，负责决策、串联、状态推进与错误处理

## 2. 首版目标

首版只做最小闭环，范围应与 `self_api` 当前一期能力匹配。

必须具备：

- 能登录 `self_api` 并持有 Bearer Token
- 能调用同步接口
- 能调用 `/async` 接口并轮询 `status_url`
- 能把多个步骤串成一个 LangGraph workflow
- 能输出统一结果对象
- 能记录基础日志与失败原因

首版不要做：

- 不自己再造任务队列
- 不自己实现复杂多租户
- 不引入数据库驱动的长期任务恢复
- 不把文件存储改造成对象存储链路

## 3. 推荐仓库结构

```text
self_orchestrator/
├── app/
│   ├── graph/
│   │   ├── state.py              # LangGraph state 定义
│   │   ├── nodes.py              # graph node 逻辑
│   │   ├── router.py             # 条件分支/路由
│   │   └── workflows.py          # 组装 graph
│   ├── clients/
│   │   └── self_api_client.py    # self_api HTTP 封装
│   ├── services/
│   │   ├── auth.py               # token 获取与刷新
│   │   ├── polling.py            # 异步状态轮询
│   │   └── runner.py             # workflow 执行入口
│   ├── schemas/
│   │   ├── requests.py           # 外部请求模型
│   │   └── responses.py          # 外部响应模型
│   ├── core/
│   │   ├── config.py             # 环境变量
│   │   └── logging.py            # 日志配置
│   ├── api/
│   │   └── v1/
│   │       ├── router.py
│   │       └── endpoints/
│   │           ├── workflows.py  # 触发工作流
│   │           └── system.py     # healthz/readiness
│   └── main.py
├── tests/
│   ├── test_self_api_client.py
│   ├── test_polling.py
│   ├── test_workflow_api.py
│   └── test_graph_nodes.py
├── .env.example
├── Makefile
├── pyproject.toml
└── README.md
```

如果首版只打算作为内部服务使用，结构可以保持简洁，但不要把 graph、HTTP client、API endpoint 全堆在一个文件里，否则二期接数据库或队列时会很难拆。

## 4. 推荐最小功能切片

建议按照下面顺序实现，避免一上来就写复杂 agent。

### 4.1 切片一：打通 `self_api`

能力：

- 登录 `POST /api/v1/auth/login`
- 调同步接口
- 调异步接口
- 轮询 `GET status_url`

完成标准：

- 可以从一个 Python 函数里完成“提交任务 -> 轮询 -> 返回结果”

### 4.2 切片二：包装成 LangGraph

能力：

- 定义统一 state
- 把“参数校验 -> 提交任务 -> 轮询 -> 结果整理”做成 graph
- 支持成功分支与失败分支

完成标准：

- 能运行一个最小 workflow，例如 `xml_to_yolo_workflow`

### 4.3 切片三：暴露 HTTP 服务

能力：

- 提供 `POST /api/v1/workflows/...`
- 提供 `GET /api/v1/healthz`
- 提供 `GET /api/v1/readiness`

完成标准：

- 外部系统不用直接理解 LangGraph，只需要调 HTTP

### 4.4 切片四：补运维最小项

能力：

- 超时控制
- 重试策略
- 请求日志
- 基础错误码
- `.env.example`

完成标准：

- 能在开发和准生产环境稳定复现

## 5. 推荐环境变量

```env
SELF_ORCH_APP_ENV=dev
SELF_ORCH_HOST=0.0.0.0
SELF_ORCH_PORT=8777

SELF_API_BASE_URL=http://127.0.0.1:8666
SELF_API_USERNAME=admin
SELF_API_PASSWORD=change-me
SELF_API_LOGIN_PATH=/api/v1/auth/login

SELF_ORCH_HTTP_TIMEOUT_SECONDS=30
SELF_ORCH_TASK_POLL_INTERVAL_SECONDS=2
SELF_ORCH_TASK_POLL_TIMEOUT_SECONDS=1800
SELF_ORCH_LOG_LEVEL=INFO
```

如果 `self_api` 开启了 `SELF_API_PUBLIC_BASE_URL`，则 `status_url` 应直接使用返回值，不要在 `self_orchestrator` 里自行拼接任务查询地址。

## 6. 与 self_api 的对接约定

首版必须遵守以下约定：

- 短任务可直接调用同步接口
- 长任务优先调用 `/async`
- 异步任务以 `task_id + status_url` 为唯一跟踪入口
- 输入输出路径基于共享目录或共享挂载
- 结果以 `self_api` 返回值为准，不在编排层重复发明业务字段

推荐一个统一的 client 接口：

```python
class SelfApiClient:
    def login(self) -> str: ...
    def call_sync(self, path: str, payload: dict) -> dict: ...
    def call_async(self, path: str, payload: dict) -> dict: ...
    def poll_task(self, status_url: str) -> dict: ...
```

这样后面无论是 LangGraph node、脚本工具还是 HTTP API，都只依赖一个 client 抽象。

## 7. LangGraph 设计建议

首版不要把 graph 设计成泛化“万能 agent”，而要先从固定流程开始。

推荐 state：

```python
class WorkflowState(TypedDict, total=False):
    workflow_name: str
    request_id: str
    input_payload: dict
    self_api_token: str
    submitted_task: dict
    task_status: dict
    result: dict
    error: str
```

推荐 node 切分：

- `validate_input`
- `ensure_auth`
- `submit_self_api_task`
- `poll_self_api_task`
- `normalize_result`
- `handle_failure`

推荐第一批 workflow：

- `xml_to_yolo_workflow`
- `split_yolo_dataset_workflow`
- `nested_dataset_cleanup_workflow`

这三类都能直接复用当前 `self_api` 能力，且最容易形成稳定闭环。

## 8. 创建步骤

### 8.1 初始化仓库

```bash
mkdir self_orchestrator
cd self_orchestrator
python -m venv .venv
source .venv/bin/activate
```

### 8.2 初始化基础文件

至少创建：

- `pyproject.toml`
- `app/main.py`
- `app/core/config.py`
- `app/clients/self_api_client.py`
- `app/graph/state.py`
- `app/graph/nodes.py`
- `app/graph/workflows.py`
- `tests/`
- `.env.example`
- `Makefile`

### 8.3 先实现 client，再实现 graph

顺序必须是：

1. 先写 `SelfApiClient`
2. 再写轮询服务
3. 再写 LangGraph nodes
4. 最后暴露 HTTP API

原因很简单：如果连 `self_api` 的调用和状态轮询都没封装稳定，直接写 graph 只会把问题藏深。

### 8.4 最先打通一个真实链路

首个验收链路建议选：

1. `POST /api/v1/auth/login`
2. `POST /api/v1/preprocess/xml-to-yolo/async`
3. 轮询任务完成
4. 返回输出目录和统计信息

这是最小、最清晰、最能代表“编排层有价值”的链路。

## 9. 测试要求

首版至少要有以下测试：

- `self_api` 登录成功/失败测试
- 异步任务提交测试
- 轮询成功、失败、超时测试
- graph 节点状态推进测试
- workflow API 集成测试

不要只测 API endpoint，不测 client 和 polling。对这个项目来说，最容易出问题的地方不是 FastAPI 路由，而是外部调用链路。

## 10. 运行与验收

最低验收标准：

- 本地能启动
- 能连通 `self_api`
- 能跑通至少一个异步 workflow
- 失败时能返回明确错误
- README 能指导他人 15 分钟内启动

建议在 README 中放一个最短示例：

```bash
curl -X POST "http://127.0.0.1:8777/api/v1/workflows/xml-to-yolo" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "/shared/project_a/raw_dataset"
  }'
```

## 11. 首版完成定义

满足以下条件，就可以认为 `self_orchestrator` 首版建立完成：

- 有单独仓库与清晰目录
- 有配置化的 `self_api` client
- 有至少一个可运行 LangGraph workflow
- 有基础 HTTP API
- 有基础测试
- 有 README 和 `.env.example`

## 12. 二期再做什么

等首版稳定后，再推进这些能力：

- workflow 执行记录入库
- 幂等键
- 重试与补偿
- API 和 worker 解耦
- 队列化执行
- 更细粒度可观测性

这些能力应与 `self_api` 二期节奏一起推进，不建议在新仓库初始化阶段一次性塞进去。
