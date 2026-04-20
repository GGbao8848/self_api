# Pipeline API

`self_api` 当前唯一的编排主入口是 `/api/v1/pipeline/*`。

## 认证

- 若开启鉴权，先调用 `POST /api/v1/auth/login`
- 所有 `pipeline` 端点都受 `require_api_auth` 保护

## 端点

### `GET /api/v1/pipeline/sops`

列出内置 SOP 模板，例如 `local-small-baseline`、`remote-slurm-iter`、`full-auto-smoke`。

```bash
curl -s http://127.0.0.1:8666/api/v1/pipeline/sops
```

### `POST /api/v1/pipeline/run`

启动一次新的 pipeline run。`full_access=false` 时，流程会在人工审核点 `interrupt` 暂停。

```bash
curl -s -X POST http://127.0.0.1:8666/api/v1/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{
    "self_api_url": "http://127.0.0.1:8666",
    "original_dataset": "/data/raw/demo",
    "detector_name": "demo_detector",
    "project_root_dir": "/data/workspace",
    "execution_mode": "local",
    "yolo_train_env": "yolo_pose",
    "yolo_train_model": "yolo11s.pt",
    "yolo_train_epochs": 5,
    "yolo_train_imgsz": 640,
    "split_mode": "train_val",
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "full_access": false
  }'
```

### `POST /api/v1/pipeline/sops/{sop_id}/run`

基于 SOP 默认值启动，用户只覆盖差异字段。

```bash
curl -s -X POST http://127.0.0.1:8666/api/v1/pipeline/sops/full-auto-smoke/run \
  -H 'Content-Type: application/json' \
  -d '{
    "overrides": {
      "self_api_url": "http://127.0.0.1:8666",
      "original_dataset": "/data/raw/demo",
      "project_root_dir": "/data/workspace",
      "detector_name": "demo_detector"
    }
  }'
```

### `GET /api/v1/pipeline/{run_id}`

查询当前 run 状态。若 `interrupted=true`，则查看 `pending_review` 并决定继续或中止。

```bash
curl -s http://127.0.0.1:8666/api/v1/pipeline/<run_id>
```

### `POST /api/v1/pipeline/{run_id}/confirm`

在人工审核点继续流程。常用 `decision` 为 `approve` 或 `abort`，必要时可带 `params_override`。

```bash
curl -s -X POST http://127.0.0.1:8666/api/v1/pipeline/<run_id>/confirm \
  -H 'Content-Type: application/json' \
  -d '{
    "decision": "approve",
    "params_override": {
      "final_classes": ["louyou"]
    }
  }'
```

### `POST /api/v1/pipeline/{run_id}/abort`

强制中止整条流程。

```bash
curl -s -X POST http://127.0.0.1:8666/api/v1/pipeline/<run_id>/abort
```

### `GET /api/v1/pipeline/{run_id}/events`

SSE 订阅状态变化。浏览器端优先使用 EventSource；如果环境不支持，可退回轮询 `GET /pipeline/{run_id}`。

```bash
curl -N http://127.0.0.1:8666/api/v1/pipeline/<run_id>/events
```

## 推荐用法

1. 先 `GET /pipeline/sops` 选模板
2. `POST /pipeline/run` 或 `POST /pipeline/sops/{sop_id}/run`
3. 轮询 `GET /pipeline/{run_id}` 或订阅 `/events`
4. 遇到 `interrupted=true` 时调用 `/confirm` 或 `/abort`

## 相关页面

- 控制台：`/pipeline-ui`
- 架构说明：`docs/architecture.md`
- 迁移说明：`docs/migration-from-n8n.md`
