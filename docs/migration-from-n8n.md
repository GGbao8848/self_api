# 从 n8n 迁移到 LangGraph

项目已经不再使用 n8n 作为官方编排入口。新的工作方式是：

- 用 `LangGraph` 定义唯一 SOP
- 用 `/api/v1/pipeline/*` 作为唯一编排 API
- 用 `/pipeline-ui` 作为默认人工审核控制台

## 对应关系

| 旧方式 | 新方式 |
|---|---|
| n8n workflow JSON | `app/graph/pipeline.py` |
| n8n 节点参数拼装 | `app/graph/nodes.py` |
| n8n 等待人工确认 | LangGraph `interrupt()` |
| webhook 触发工作流 | `POST /api/v1/pipeline/run` |
| 多条工作流区分小图/大图/远端 | `app/graph/sops.py` 中的 SOP 模板 |

## 迁移建议

1. 停止新增或维护 n8n workflow JSON。
2. 把原先 webhook 调用改为 `POST /api/v1/pipeline/run` 或 `POST /api/v1/pipeline/sops/{sop_id}/run`。
3. 把人工审核页面切换到 `/pipeline-ui`。
4. 如果外部系统以前依赖 webhook 返回值，改为读取 `run_id` 后轮询 `/api/v1/pipeline/{run_id}`。
5. 若要监听实时状态，改用 `/api/v1/pipeline/{run_id}/events`。

## 已移除

- `/train-ui`
- `/api/v1/tasks/launch-training-workflow`
- `SELF_API_N8N_BASE_URL`
- `SELF_API_N8N_API_KEY`

## 仍然保留

- `/api/v1/preprocess/*`：原子能力，适合脚本调试或单步调用
- `/api/v1/tasks/*`：异步任务状态查询与取消
- `task_manager`：本地异步任务池
