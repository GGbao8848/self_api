# LangGraph 最小生产接入

本文档描述一期最小生产版下，`self_api` 与 LangGraph 的推荐对接方式。

## 基本原则

- LangGraph 通过 HTTP 调用 `self_api`
- 短任务可直接调用同步接口
- 长任务默认调用 `/async` 接口
- LangGraph 与 `self_api` 需要共享同一批数据目录，或都访问同一挂载存储

## 环境变量

可直接参考仓库根目录的 `.env.example`。

至少需要配置：

- `SELF_API_APP_ENV=prod`
- `SELF_API_PUBLIC_BASE_URL`
- `SELF_API_AUTH_ENABLED=true`
- `SELF_API_AUTH_ADMIN_PASSWORD`
- `SELF_API_AUTH_SECRET_KEY`
- `SELF_API_STORAGE_ROOT`
- `SELF_API_FILE_ACCESS_ROOTS`

## 推荐调用流程

### 1. 登录获取 Bearer Token

- `POST /api/v1/auth/login`
- 从响应中读取 `access_token`
- LangGraph 后续请求使用 `Authorization: Bearer <token>`

### 2. 提交异步任务

以 `xml-to-yolo` 为例：

- `POST /api/v1/preprocess/xml-to-yolo/async`
- 服务返回 `task_id` 和 `status_url`

### 3. 轮询任务状态

- `GET status_url`
- 或 `GET /api/v1/preprocess/tasks/{task_id}`

任务终态：

- `succeeded`
- `failed`
- `cancelled`

### 4. 获取结果

- 直接从任务 `result` 读取输出目录、统计信息
- 如果任务登记了产物，则调用 `GET /api/v1/artifacts`

## 一期限制

- 任务状态仍为进程内状态，服务重启后会丢失
- 目录类输入仍以共享文件路径为主
- 不保证多实例部署时的任务连续性
