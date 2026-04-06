# API 使用示例

本文档从 `README.md` 中拆分，集中放置各接口的示例请求。

## 0. preprocess 请求体字段约定（权威）

以下名称与 `app/schemas/preprocess.py` 一致。异步接口在同步字段基础上增加 `callback_url`、`callback_timeout_seconds`（可选）。

| 端点 | 主要字段 |
|------|----------|
| `xml-to-yolo` | `dataset_dir`（兼容别名 `input_dir`） |
| `split-yolo-dataset` | `dataset_dir`（兼容 `input_dir`）、`output_dir`（可选） |
| `yolo-sliding-window-crop` | `images_dir`、`labels_dir`、`output_dir`、`min_vis_ratio`、`stride_ratio`、`ignore_vis_ratio`、`only_wide` |
| `zip-folder` | `input_dir`、`output_zip_path`（兼容 `output_dir`）、`include_root_dir`、`overwrite` |
| `unzip-archive` | `archive_path`（兼容 `input_dir`）、`output_dir`、`overwrite` |
| `move-path` / `copy-path` | `source_path`（兼容 `input_dir`）、`target_dir`（兼容 `output_dir`）、`overwrite` |
| `build-yolo-yaml` | `input_dir`（可传数据集**上级目录**，服务会依次尝试 `<input_dir>/dataset`、`input_dir` 自身、`<input_dir>/yolo_split`，并自动匹配 **train/images** 或 **images/train**）、`classes_file`（可选；默认同目录或 `yolo_split`/`dataset` 下 `classes.txt`）、`split_names`（可选）、`images_subdir_name`、`path_prefix_replace_from` / `path_prefix_replace_to`（成对）、`output_yaml_path` |

**异步任务轮询**：`GET /api/v1/preprocess/tasks/{task_id}` 返回体中，业务结果在 **`result`** 对象内（例如 `result.output_dir`、`result.output_zip_path`），勿与顶层字段混淆。

**n8n HTTP Request**：服务端要求 **`Content-Type: application/json`**。在节点里用「Body Parameters / 字段列表」维护时，请将 **Body Content Type** 设为 **JSON**，**Specify Body** 设为 **Using Fields Below（keypair）**，不要用整段 `JSON.stringify` 表达式，以便与下表字段一一对应、方便修改。

## 1. 健康检查

- `GET /api/v1/healthz`

```bash
curl "http://192.168.210.73:8666/api/v1/healthz"
```

## 2. 滑窗裁剪

- `POST /api/v1/preprocess/sliding-window-crop`

```bash
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/sliding-window-crop" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/sliding-window-crop/async" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/xml-to-yolo" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/xml-to-yolo/async" \
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

## 4. YOLO 数据集划分

- `POST /api/v1/preprocess/split-yolo-dataset`
- `POST /api/v1/preprocess/split-yolo-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/split-yolo-dataset" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/split-yolo-dataset/async" \
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

## 5. 多层目录叶子数据目录发现

- `POST /api/v1/preprocess/discover-leaf-dirs`
- `POST /api/v1/preprocess/discover-leaf-dirs/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/discover-leaf-dirs" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "recursive": true
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/discover-leaf-dirs/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "recursive": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 6. 多层目录数据清洗

- `POST /api/v1/preprocess/clean-nested-dataset`
- `POST /api/v1/preprocess/clean-nested-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/clean-nested-dataset" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/clean-nested-dataset/async" \
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

## 7. 多层目录数据集汇总

- `POST /api/v1/preprocess/aggregate-nested-dataset`
- `POST /api/v1/preprocess/aggregate-nested-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/aggregate-nested-dataset" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/aggregate-nested-dataset/async" \
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

## 8. 目录打包为 ZIP

- `POST /api/v1/preprocess/zip-folder`
- `POST /api/v1/preprocess/zip-folder/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/zip-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/source_folder",
    "output_zip_path": "./data/source_folder.zip",
    "include_root_dir": true,
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/zip-folder/async" \
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

## 9. ZIP 解压

- `POST /api/v1/preprocess/unzip-archive`
- `POST /api/v1/preprocess/unzip-archive/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/unzip-archive" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "./data/source_folder.zip",
    "output_dir": "./data/unpacked",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/unzip-archive/async" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "./data/source_folder.zip",
    "output_dir": "./data/unpacked",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 10. 文件或目录移动

- `POST /api/v1/preprocess/move-path`
- `POST /api/v1/preprocess/move-path/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/move-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_ready",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/move-path/async" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_ready",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 11. 文件或目录复制

- `POST /api/v1/preprocess/copy-path`
- `POST /api/v1/preprocess/copy-path/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/copy-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_backup",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/copy-path/async" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_backup",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 12. 跨机器 SFTP 远程传输

- `POST /api/v1/preprocess/remote-transfer`
- `POST /api/v1/preprocess/remote-transfer/async`

将本地文件或目录通过 SFTP 上传到远程服务器。支持 `sftp://` URL 或 `user@host:path` 格式。

```bash
# 同步（密码认证）
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/remote-transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "password": "your_password",
    "overwrite": true
  }'

# 同步（私钥认证）
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/remote-transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/dataset",
    "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
    "username": "sk",
    "private_key_path": "~/.ssh/id_rsa",
    "overwrite": true
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/remote-transfer/async" \
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

## 13. YOLO 大图正方形滑窗裁剪为小图数据集

- `POST /api/v1/preprocess/yolo-sliding-window-crop`
- `POST /api/v1/preprocess/yolo-sliding-window-crop/async`

正方形滑窗（边长=图片高度），仅水平滑动。适用于宽图（如整车图）的裁剪。

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/yolo-sliding-window-crop" \
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
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/yolo-sliding-window-crop/async" \
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

## 14. 生成 YOLO `data.yaml`（Ultralytics）

- `POST /api/v1/preprocess/build-yolo-yaml`
- `POST /api/v1/preprocess/build-yolo-yaml/async`

在 `input_dir` 下自动匹配数据集根（见 §0），扫描 `train` / `val` / `test` 等划分；各划分下需存在 `images` 子目录且含至少一张图片。生成的 YAML 中 **`train` / `val` / `test` 的值为各划分 `images` 目录的绝对路径**（POSIX），例如 `train: /Users/.../dataset1/dataset/train/images`，不再使用单独的 `path:` 与相对子路径。`path_prefix_replace_from` / `path_prefix_replace_to` 若成对填写，则对**每一条**上述绝对路径做前缀替换（例如把本机 `original_dataset` 前缀换成 `project_root/detector` 下的目标根路径）。

```bash
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/build-yolo-yaml" \
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
