# API 使用示例

本文档从 `README.md` 中拆分，集中放置各接口的示例请求。

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

## 12. YOLO 大图滑窗裁剪为小图数据集

- `POST /api/v1/preprocess/yolo-sliding-window-crop`
- `POST /api/v1/preprocess/yolo-sliding-window-crop/async`

```bash
# 同步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/yolo-sliding-window-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./data/yolo_large",
    "output_dir": "./data/yolo_small",
    "window_width": 1024,
    "window_height": 1024,
    "stride_x": 512,
    "stride_y": 512,
    "keep_empty_labels": false,
    "min_box_area_ratio": 0.2
  }'

# 异步
curl -X POST "http://192.168.210.73:8666/api/v1/preprocess/yolo-sliding-window-crop/async" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "./data/yolo_large",
    "output_dir": "./data/yolo_small",
    "window_width": 1024,
    "window_height": 1024,
    "stride_x": 512,
    "stride_y": 512,
    "keep_empty_labels": false,
    "min_box_area_ratio": 0.2,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```
