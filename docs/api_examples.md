# API 使用示例

本文档从 `README.md` 中拆分，集中放置各接口的示例请求。

## 0. preprocess 请求体字段约定（权威）

以下名称与 `app/schemas/preprocess.py` 一致。异步接口在同步字段基础上增加 `callback_url`、`callback_timeout_seconds`（可选）。

| 端点 | 主要字段 |
|------|----------|
| `xml-to-yolo` | `dataset_dir`（兼容别名 `input_dir`） |
| `annotate-visualize` | `images_dir`、`output_dir`；**`labels_dir`（YOLO `.txt`）与 `xmls_dir`（Pascal VOC XML）二选一**，未使用的一方留空 `""`；可选 `recursive`、`extensions`、`include_difficult`（仅 XML）、`line_width`、`overwrite`；YOLO 类别名可选 `classes` 或 `classes_file`（二选一，**均可省略**；均不提供或 `classes_file` 为空字符串时，框上显示**类别 id 数字**） |
| `split-yolo-dataset` | `dataset_dir`（兼容 `input_dir`）、`output_dir`（可选） |
| `yolo-sliding-window-crop` | `images_dir`、`labels_dir`、`output_dir`、`min_vis_ratio`、`stride_ratio`、`ignore_vis_ratio`、`only_wide` |
| `zip-folder` | `input_dir`、`output_zip_path`（兼容 `output_dir`）、`include_root_dir`、`overwrite` |
| `unzip-archive` | `archive_path`（兼容 `input_dir`）、`output_dir`、`overwrite` |
| `move-path` / `copy-path` | `source_path`（兼容 `input_dir`）、`target_dir`（兼容 `output_dir`）、`overwrite` |
| `build-yolo-yaml` | `input_dir`（可传数据集**上级目录**，服务会依次尝试 `<input_dir>/dataset`、`input_dir` 自身、`<input_dir>/yolo_split`，并自动匹配 **train/images** 或 **images/train**）、`classes_file`（可选；默认同目录或 `yolo_split`/`dataset` 下 `classes.txt`）、`split_names`（可选）、`images_subdir_name`、`path_prefix_replace_from` / `path_prefix_replace_to`（成对）、`output_yaml_path` |
| `yolo-train` | `yaml_path`（须含 `/dataset/` 段）、`project_root_dir`（子进程 `cwd`）、`yolo_train_env`（conda 环境名）、`model`、`epochs`、`imgsz`（后三项有默认值） |
| `voc-bar-crop` | `images_dir`、`xmls_dir`、`output_dir`；可选 `recursive`（默认 `true`）；裁剪正方形边长为**源图高度**（§17） |
| `restore-voc-crops-batch` | `original_images_dir`、`original_xmls_dir`、`edited_crops_images_dir`、`edited_crops_xmls_dir`、`output_dir`；可选 `recursive`、`skip_unparsed_names`（§18） |

**异步任务轮询**：`GET /api/v1/preprocess/tasks/{task_id}` 返回体中，业务结果在 **`result`** 对象内（例如 `result.output_dir`、`result.output_zip_path`），勿与顶层字段混淆。

**n8n HTTP Request**：服务端要求 **`Content-Type: application/json`**。在节点里用「Body Parameters / 字段列表」维护时，请将 **Body Content Type** 设为 **JSON**，**Specify Body** 设为 **Using Fields Below（keypair）**，不要用整段 `JSON.stringify` 表达式，以便与下表字段一一对应、方便修改。

## 1. 健康检查

- `GET /api/v1/healthz`

```bash
curl "http://192.168.2.26:8666/api/v1/healthz"
```

## 2. 滑窗裁剪

- `POST /api/v1/preprocess/sliding-window-crop`

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/sliding-window-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/raw",
    "output_dir": "./data/crops",
    "window_width": 512,
    "window_height": 512,
    "stride_x": 256,
    "stride_y": 256,
    "include_partial_edges": false,
    "recursive": true,
    "keep_subdirs": true,
    "output_format": "png"
  }'
```

大数据量场景建议改用异步接口，避免调用方超时：

- `POST /api/v1/preprocess/sliding-window-crop/async`

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/sliding-window-crop/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/raw",
    "output_dir": "./data/crops",
    "window_width": 2048,
    "window_height": 2048,
    "stride_x": 2048,
    "stride_y": 2048,
    "include_partial_edges": false,
    "recursive": true,
    "keep_subdirs": true,
    "output_format": "png",
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

响应示例（`202 Accepted`）：

```json
{
  "status": "accepted",
  "task_id": "7f8df8f52f5f4bcfa4b6a4f6b7a61d93",
  "task_type": "sliding_window_crop",
  "status_url": "http://127.0.0.1:8666/api/v1/preprocess/tasks/7f8df8f52f5f4bcfa4b6a4f6b7a61d93",
  "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished"
}
```

仅当任务完成时，才会向 `callback_url` 发送回调：

- `succeeded` 或 `failed`

如果 `POST` 返回 `405/501`，服务会自动回退尝试一次 `GET`。

回调 body 示例（以下是 `state=succeeded`）：

```json
{
  "task_id": "7f8df8f52f5f4bcfa4b6a4f6b7a61d93",
  "task_type": "sliding_window_crop",
  "state": "succeeded",
  "created_at": "2026-02-23T12:00:00+00:00",
  "updated_at": "2026-02-23T12:02:45+00:00",
  "result": {
    "status": "ok",
    "input_images": 120,
    "processed_images": 120,
    "skipped_images": 0,
    "generated_crops": 480,
    "output_dir": "/abs/path/to/crops",
    "details": []
  },
  "error": null
}
```

轮询任务状态：

- `GET /api/v1/preprocess/tasks/{task_id}`

成功示例：

```json
{
  "task_id": "7f8df8f52f5f4bcfa4b6a4f6b7a61d93",
  "task_type": "sliding_window_crop",
  "state": "succeeded",
  "created_at": "2026-02-23T12:00:00+00:00",
  "updated_at": "2026-02-23T12:02:45+00:00",
  "result": {
    "status": "ok",
    "input_images": 120,
    "processed_images": 120,
    "skipped_images": 0,
    "generated_crops": 480,
    "output_dir": "/abs/path/to/crops",
    "details": []
  },
  "error": null,
  "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
  "callback_state": "succeeded",
  "callback_sent_at": "2026-02-23T12:02:45+00:00",
  "callback_status_code": 200,
  "callback_error": null,
  "callback_events": [
    {
      "state": "succeeded",
      "attempted_at": "2026-02-23T12:02:45+00:00",
      "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
      "status_code": 200,
      "method": "POST",
      "success": true,
      "error": null
    }
  ]
}
```

## 3. VOC XML 转 YOLO 标签

- `POST /api/v1/preprocess/xml-to-yolo`
- `POST /api/v1/preprocess/xml-to-yolo/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/xml-to-yolo" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./data/voc_like",
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "labels_dir_name": "labels",
    "recursive": true,
    "include_difficult": false,
    "write_classes_file": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/xml-to-yolo/async" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./data/voc_like",
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "labels_dir_name": "labels",
    "recursive": true,
    "include_difficult": false,
    "write_classes_file": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 4. 标注可视化（YOLO txt / VOC XML）

在图像上绘制检测框并导出到 `output_dir`，目录结构与 `images_dir` 下相对路径一致（与 `split-yolo-dataset` 的图像–标注配对规则相同：按相对路径找同名 `.txt` 或 `.xml`）。

**约束**：`labels_dir` 与 `xmls_dir` 必须**恰好填一个**（另一个传空字符串 `""` 或省略）；`classes` 与 `classes_file` 不能同时**非空**填写（二者可同时省略）。若未提供 `classes` 且 `classes_file` 为空或省略，则框上文字为**类别 id**（与 YOLO txt 中每行首列整数一致）。

- `POST /api/v1/preprocess/annotate-visualize`
- `POST /api/v1/preprocess/annotate-visualize/async`

```bash
# 同步 — YOLO 标签（labels_dir 有值，xmls_dir 留空）
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "",
    "labels_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/labels",
    "xmls_dir": "",
    "output_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/visualized",
    "recursive": true,
    "line_width": 2,
    "overwrite": true,
    "classes_file": ""
  }'
```

上例中 `classes_file` 为空字符串时与省略该字段等价，框上显示数字类别 id。若需显示名称，可改为 `"classes_file": "./data/yolo_raw/classes.txt"` 或传 `"classes": ["类0", "类1", ...]`。

```bash
# 同步 — Pascal VOC XML（xmls_dir 有值，labels_dir 留空）
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/images",
    "labels_dir": "",
    "xmls_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/xmls",
    "output_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/visualized",
    "recursive": true,
    "include_difficult": false,
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize/async" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "./data/yolo_raw/images",
    "labels_dir": "./data/yolo_raw/labels",
    "xmls_dir": "",
    "output_dir": "./data/yolo_raw/visualized",
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

**响应要点**（同步 body / 异步任务的 `result`）：`mode` 为 `yolo` 或 `xml`；`annotation_dir` 为实际使用的标注根目录；`written_images` / `skipped_images`；`details` 中每项含 `source_image`、`output_image`、`boxes_drawn`、`skipped_reason`（失败或跳过原因）。

## 5. YOLO 数据集划分

- `POST /api/v1/preprocess/split-yolo-dataset`
- `POST /api/v1/preprocess/split-yolo-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/split-yolo-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./data/yolo_raw",
    "output_dir": "./data/yolo_split",
    "mode": "train_val_test",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "shuffle": true,
    "seed": 42,
    "copy_files": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/split-yolo-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./data/yolo_raw",
    "output_dir": "./data/yolo_split",
    "mode": "train_val_test",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "shuffle": true,
    "seed": 42,
    "copy_files": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 6. 多层目录叶子数据目录发现

- `POST /api/v1/preprocess/discover-leaf-dirs`
- `POST /api/v1/preprocess/discover-leaf-dirs/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/discover-leaf-dirs" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "recursive": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/discover-leaf-dirs/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "recursive": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 7. 多层目录数据清洗

- `POST /api/v1/preprocess/clean-nested-dataset`
- `POST /api/v1/preprocess/clean-nested-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset",
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "backgrounds_dir_name": "backgrounds",
    "include_difficult": false,
    "copy_files": true,
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset",
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "backgrounds_dir_name": "backgrounds",
    "include_difficult": false,
    "copy_files": true,
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 8. 多层目录数据集汇总

- `POST /api/v1/preprocess/aggregate-nested-dataset`
- `POST /api/v1/preprocess/aggregate-nested-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/aggregate-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/dataset",
    "images_dir_name": "images",
    "labels_dir_name": "labels",
    "backgrounds_dir_name": "backgrounds",
    "classes_file_name": "classes.txt",
    "require_non_empty_labels": true,
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/aggregate-nested-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/dataset",
    "images_dir_name": "images",
    "labels_dir_name": "labels",
    "backgrounds_dir_name": "backgrounds",
    "classes_file_name": "classes.txt",
    "require_non_empty_labels": true,
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

推荐的处理链路是：

1. `discover-leaf-dirs`：确认最底层叶子目录
2. `clean-nested-dataset`：把原始图片/XML清洗为 `images/xmls/backgrounds`
3. `xml-to-yolo`：在每个清洗后的碎片目录中生成 `labels`
4. `aggregate-nested-dataset`：汇总为统一 `dataset`

## 9. 目录打包为 ZIP

- `POST /api/v1/preprocess/zip-folder`
- `POST /api/v1/preprocess/zip-folder/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/zip-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/source_folder",
    "output_zip_path": "./data/source_folder.zip",
    "include_root_dir": true,
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/zip-folder/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/source_folder",
    "output_zip_path": "./data/source_folder.zip",
    "include_root_dir": true,
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 10. ZIP 解压

- `POST /api/v1/preprocess/unzip-archive`
- `POST /api/v1/preprocess/unzip-archive/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/unzip-archive" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "./data/source_folder.zip",
    "output_dir": "./data/unpacked",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/unzip-archive/async" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "./data/source_folder.zip",
    "output_dir": "./data/unpacked",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 11. 文件或目录移动

- `POST /api/v1/preprocess/move-path`
- `POST /api/v1/preprocess/move-path/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/move-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_ready",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/move-path/async" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_ready",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 12. 文件或目录复制

- `POST /api/v1/preprocess/copy-path`
- `POST /api/v1/preprocess/copy-path/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/copy-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_backup",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/copy-path/async" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_backup",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 13. 跨机器 SFTP 远程传输

- `POST /api/v1/preprocess/remote-transfer`
- `POST /api/v1/preprocess/remote-transfer/async`

将本地文件或目录通过 SFTP 上传到远程服务器。支持 `sftp://` URL 或 `user@host:path` 格式。

```bash
# 同步（密码认证）
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "password": "your_password",
    "overwrite": true
  }'

# 同步（私钥认证）
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "private_key_path": "~/.ssh/id_rsa",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-transfer/async" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "password": "your_password",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

**target 格式**：

- `sftp://host/path` 或 `sftp://user@host/path`
- `sftp://host:port/path`
- `user@host:path`（scp 风格）

**响应示例**：

```json
{
  "status": "ok",
  "source_path": "/path/to/local/dataset",
  "target_path": "/mnt/usrhome/sk/ndata/dataset",
  "transferred_type": "directory",
  "transferred_files": 42,
  "total_bytes": 1048576
}
```

## 14. YOLO 大图正方形滑窗裁剪为小图数据集

- `POST /api/v1/preprocess/yolo-sliding-window-crop`
- `POST /api/v1/preprocess/yolo-sliding-window-crop/async`

正方形滑窗（边长=图片高度），仅水平滑动。适用于宽图（如整车图）的裁剪。

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-sliding-window-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "/path/to/dataset/images",
    "labels_dir": "/path/to/dataset/labels",
    "output_dir": "/path/to/dataset/data_crops",
    "min_vis_ratio": 0.5,
    "stride_ratio": 0.2,
    "ignore_vis_ratio": 0.05,
    "only_wide": true
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-sliding-window-crop/async" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "/path/to/dataset/images",
    "labels_dir": "/path/to/dataset/labels",
    "output_dir": "/path/to/dataset/data_crops",
    "min_vis_ratio": 0.5,
    "stride_ratio": 0.2,
    "ignore_vis_ratio": 0.05,
    "only_wide": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

命令行等价示例：

```bash
python yolo_square_sliding_crop.py \
  --images /path/to/dataset/images \
  --labels /path/to/dataset/labels \
  --out /path/to/dataset/data_crops \
  --min_vis_ratio 0.5 \
  --stride_ratio 0.2 \
  --ignore_vis_ratio 0.05 \
  --only_wide
```

## 15. 生成 YOLO `data.yaml`（Ultralytics）

- `POST /api/v1/preprocess/build-yolo-yaml`
- `POST /api/v1/preprocess/build-yolo-yaml/async`

在 `input_dir` 下自动匹配数据集根（见 §0），扫描 `train` / `val` / `test` 等划分；各划分下需存在 `images` 子目录且含至少一张图片。生成的 YAML 中 **`train` / `val` / `test` 的值为各划分 `images` 目录的绝对路径**（POSIX），例如 `train: /Users/.../dataset1/dataset/train/images`，不再使用单独的 `path:` 与相对子路径。`path_prefix_replace_from` / `path_prefix_replace_to` 若成对填写，则对**每一条**上述绝对路径做前缀替换（例如把本机 `original_dataset` 前缀换成 `project_root/detector` 下的目标根路径）。

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/build-yolo-yaml" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/temp_dataset",
    "classes_file": "/path/to/temp_dataset/classes.txt",
    "output_yaml_path": "/path/to/temp_dataset/dataset.yaml",
    "path_prefix_replace_from": "/Users/me/tmp_datasets/dataset1",
    "path_prefix_replace_to": "/mnt/training/TVDS/dog_cat_pig"
  }'
```

`classes_file` 可省略，按 §0 在 `dataset` / `yolo_split` 等位置自动查找 `classes.txt`。`path_prefix_replace_*` 可省略；若填写则须成对出现，且每条划分路径须以 `path_prefix_replace_from` 为前缀。

## 16. YOLO 训练（conda + `yolo train`）

- `POST /api/v1/preprocess/yolo-train`
- `POST /api/v1/preprocess/yolo-train/async`

根据 `yaml_path` 在 **包含 `/dataset/` 段** 的前提下自动推导 Ultralytics 的 `project` 与 `name`：`project` 为「`/dataset/` 之前路径」+ `/runs/train`；`name` 为 yaml 文件名（不含扩展名）。在 `project_root_dir` 下启动子进程，使用 `conda run -n <yolo_train_env> -- yolo train ...`。

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-train" \
  -H "Content-Type: application/json" \
  -d '{
    "yaml_path": "/Users/me/self_api/TVDS/dog_cat_pig/dataset/dataset.yaml",
    "project_root_dir": "/Users/me/self_api/TVDS",
    "yolo_train_env": "yolo_pose",
    "model": "yolo11s.pt",
    "epochs": 100,
    "imgsz": 640
  }'
```

响应含 `command`（拼接后的命令行）、`cwd`、`project`、`name`、`exit_code`、`stdout`、`stderr`。训练失败时 `exit_code` 非零且 `status` 为 `failed`。

## 17. VOC 横向条带正方形裁剪（`voc-bar-crop`）

对 `xmls_dir` 中每个 Pascal VOC XML：对每个 **宽不小于高** 的目标框触发一次裁剪。正方形 **边长 = 源图像高度**（宽≥高的横向长条图：`top=0`，水平方向以该框中心居中，左右越界则在图内平移；窄高图则 `left=0`、垂直居中）。与 `yolo-sliding-window-crop` 的「边长=图片高度」一致，**不是**标注框高度。小图文件名含裁剪中心与边长，例如 `stem_cx512_cy409_S819.jpg`，同 stem 的 XML 写入 `output_dir/xmls/`。裁剪图内保留 **与窗口相交** 的所有物体框（坐标换算到小图）。

- `POST /api/v1/preprocess/voc-bar-crop`
- `POST /api/v1/preprocess/voc-bar-crop/async`

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/voc-bar-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "/path/to/dataset/images",
    "xmls_dir": "/path/to/dataset/xmls",
    "output_dir": "/path/to/dataset/bar_crops",
    "recursive": true
  }'
```

异步（`202 Accepted`，避免大批量同步请求超时；结果在 `GET /api/v1/preprocess/tasks/{task_id}` 的 `result` 中，亦可通过 `callback_url` 回调）：

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/voc-bar-crop/async" \
  -H "Content-Type: application/json" \
  -d '{
    "images_dir": "/path/to/dataset/images",
    "xmls_dir": "/path/to/dataset/xmls",
    "output_dir": "/path/to/dataset/bar_crops",
    "recursive": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

**响应要点**：`generated_crops` 为生成的小图数量；`details` 中含 `crop_image`、`crop_xml`、`window_left`、`window_top`、`window_size`（大图上的裁剪窗口）。

编辑后把裁剪贴回整图并合并 VOC，请使用 **`restore-voc-crops-batch`**（§18）。裁剪文件名 `{stem}_cx{cx}_cy{cy}_S{S}` 与窗口对应关系为：`x = cx - S//2`，`y = cy - S//2`，`width = height = S`（与 `details` 中 `window_*` 一致；贴边 clamp 时以 `details` 为准）。

## 18. 批量还原 voc 裁剪到原图（`restore-voc-crops-batch`）

适用于目录结构形如：

- 原始：`.../SHIJIAZHUANG.../images/*.jpg`、`.../xmls/*.xml`
- 编辑后裁剪：`.../crop/images/1_5_cx5767_cy563_S1126.jpg`、`.../crop/xmls/1_5_cx5767_cy563_S1126.xml`（文件名须为 voc-bar-crop 规则 `{stem}_cx{cx}_cy{cy}_S{S}`）

按 **原图 stem**（如 `1_5`）分组，将该 stem 下**所有**裁剪顺序贴回对应原图，合并 VOC 标注（每个裁剪区域先删与 `region` 相交的旧框，再追加该裁剪 XML 中的框）。**输出** `output_dir/images/`、`output_dir/xmls/` 下与原数据集相同的相对文件名。

- `POST /api/v1/preprocess/restore-voc-crops-batch`
- `POST /api/v1/preprocess/restore-voc-crops-batch/async`

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/restore-voc-crops-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "original_images_dir": "/media/qzq/16T/test/SHIJIAZHUANGSHANGXING_20260317165711_49_CR400BF-A-5156_1_HUITIES/images",
    "original_xmls_dir": "/media/qzq/16T/test/SHIJIAZHUANGSHANGXING_20260317165711_49_CR400BF-A-5156_1_HUITIES/xmls",
    "edited_crops_images_dir": "/media/qzq/16T/test/crop/images",
    "edited_crops_xmls_dir": "/media/qzq/16T/test/crop/xmls",
    "output_dir": "/media/qzq/16T/test/merged_dataset",
    "recursive": false,
    "skip_unparsed_names": true
  }'
```

**响应要点**：`originals_processed` 为成功处理的原图数量；`details` 每项含 `original_stem`、`output_image`、`output_xml`、`crops_applied`、`status`；`total_crop_files` 为裁剪目录中扫描到的图片文件数（含无法解析的文件名，若 `skip_unparsed_names` 为 true 则仅忽略不报错）。

异步示例在同步 body 上增加 `callback_url`、`callback_timeout_seconds` 即可（同 §2）。
