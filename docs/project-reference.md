# 项目详细说明

本文档承接根 README 中迁出的详细信息，包括项目范围、目录结构、Docker 维护命令、Agent API 与后续扩展建议。

## 最小可交付范围

- 完整 API 服务骨架（`FastAPI`）
- API 版本化路由（`/api/v1`）
- 健康检查（`/api/v1/healthz`）
- 请求/响应模型与参数校验（Pydantic）
- 统一错误处理（非法输入返回 `400`）
- 内建 Agent API 骨架（`/api/v1/agent/chat`、run/session、tool registry）
- 核心业务服务分层（`services`）
- 基础可运行测试（`pytest`）
- Docker 化运行支持

## 项目架构

```text
self_api/
├── app/
│   ├── api/
│   │   ├── url_builder.py             # 对外 URL 生成
│   │   └── v1/
│   │       ├── endpoints/             # v1 接口定义（agent/auth/files/tasks/preprocess 等）
│   │       └── router.py              # v1 路由聚合
│   ├── agent/
│   │   ├── providers/                 # LLM provider 选择与配置
│   │   ├── tools/                     # Agent 工具注册表
│   │   ├── runtime.py                 # Agent runtime 入口
│   │   └── sessions.py                # Agent run/session 存储
│   ├── core/
│   │   ├── config.py                  # 环境配置
│   │   ├── logging.py                 # 日志初始化
│   │   ├── path_safety.py             # 路径访问约束
│   │   └── security.py                # 安全与鉴权辅助
│   ├── schemas/
│   │   ├── agent.py                   # Agent API 模型
│   │   ├── auth.py                    # 鉴权模型
│   │   ├── preprocess.py              # 预处理请求/响应模型
│   │   ├── system.py                  # 系统接口模型
│   │   └── tasks.py                   # 异步任务模型
│   ├── services/
│   │   ├── file_operations.py         # 压缩/解压/移动/复制
│   │   ├── nested_dataset.py          # 多层目录发现/清洗/汇总
│   │   ├── remote_transfer.py         # SFTP 远程传输
│   │   ├── split_yolo_dataset.py      # YOLO 数据集划分
│   │   ├── task_manager.py            # 异步任务管理
│   │   ├── xml_to_yolo.py             # VOC XML 转 YOLO
│   │   ├── yolo_augment.py            # YOLO 数据增强
│   │   ├── yolo_sliding_window.py     # YOLO 大图滑窗裁剪
│   │   └── yolo_train.py              # YOLO 训练启动
│   ├── utils/
│   │   └── images.py                  # 图像文件扫描工具
│   └── main.py                        # FastAPI 应用入口
├── docs/
│   ├── api_examples.md                # 接口调用示例
│   └── n8n/                           # n8n 工作流样例
├── tests/
│   ├── conftest.py
│   ├── data_helpers.py
│   └── test_*.py                      # API / 服务回归测试
├── .env.example
├── compose.yaml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Docker 维护命令

### Compose

| 目的 | 命令 |
|------|------|
| 停止并删除容器与网络 | `docker compose down` |
| 停止并删除容器、网络和命名卷 | `docker compose down -v` |
| 查看运行状态 | `docker compose ps` |
| 查看配置合并结果 | `docker compose config` |
| 查看日志 | `docker compose logs` |
| 持续跟随日志 | `docker compose logs -f` |
| 仅看最近若干行 | `docker compose logs --tail=100 -f` |
| 重启服务 | `docker compose restart` |
| 进入容器 Shell | `docker compose exec api sh` |
| 容器内执行单条命令 | `docker compose exec api python -c "import app; print('ok')"` |

### 纯 docker

| 目的 | 命令 |
|------|------|
| 构建镜像 | `docker build -t self_api:latest .` |
| 前台运行 | `docker run --rm -p 8666:8666 --env-file .env self_api:latest` |
| 后台运行 | `docker run -d --name self_api -p 8666:8666 --env-file .env self_api:latest` |
| 停止容器 | `docker stop self_api` |
| 删除容器 | `docker rm self_api` |
| 查看日志 | `docker logs self_api` / `docker logs -f self_api` |
| 进入容器 | `docker exec -it self_api sh` |

## Agent API

当前 Agent 层已完成原生入口骨架，后续会逐步把 `docs/n8n/聊天工具人.json` 中的工具调用规则迁入本仓库。

已提供接口：

- `POST /api/v1/agent/chat`
- `GET /api/v1/agent/runs/{run_id}`
- `POST /api/v1/agent/runs/{run_id}/cancel`
- `GET /api/v1/agent/sessions/{session_id}`
- `GET /api/v1/agent/tools`

当前约束：

- 工具注册表已内置首批数据预处理工具名称和异步标记
- `scan-yolo-label-indices` / `rewrite-yolo-label-indices` 已支持通过 Agent 同步执行
- `xml-to-yolo` / `split-yolo-dataset` 已支持通过 Agent 提交异步任务并轮询到终态
- 其余异步预处理工具和模型真实调用会在后续迁移步骤接入
- `n8n` 训练 webhook 与静态训练页面已从运行路径移除

## 长任务 Agent 落地改造清单

1. 任务状态持久化：`/api/v1/tasks/*` 不再依赖进程内内存，任务状态、回调历史、artifact、进度、事件均落到 SQLite。
2. Agent run 持久化增强：run 记录新增 `accepted/running/waiting_task/cancelled` 状态，并保存步骤级 `steps` 执行轨迹。
3. 后台长任务模式：`POST /api/v1/agent/chat` 支持 `async_run=true`，请求返回后由后台线程继续规划与执行。
4. 多步执行循环：长任务模式下可在一次 run 内执行“决策 -> 工具 -> 观察 -> 下一步决策”的循环，而不是只执行单工具后立即退出。
5. 异步工具纳管：Agent 在长任务模式下提交异步 preprocess task 后，会跟踪 `task_id` 并把等待过程写入 run steps。
6. 取消链路：取消 Agent run 时，会联动取消当前等待中的底层 task。
7. 重启保护：服务异常重启后，未完成的 task / agent run 会被标记为中断失败，避免永远卡在 `running`。

## 后续扩展建议

1. 将后台线程切换为独立 worker / queue（Celery、RQ、Arq 等）。
2. 为 step 级事件增加 SSE / WebSocket 推送。
3. 引入人工确认节点、失败重试策略、可恢复 checkpoint。
4. 增加对象存储（S3/MinIO）输入输出支持。
5. 增加鉴权（API Key/JWT）与限流。
