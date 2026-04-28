# API 使用示例

本文档从 `README.md` 中拆分，集中放置各接口的示例请求。

## 0. preprocess 请求体字段约定（权威）

以下名称与 `app/schemas/preprocess.py` 一致。异步接口在同步字段基础上增加 `callback_url`、`callback_timeout_seconds`（可选）。

| 端点 | 主要字段 |
|------|----------|
| `discover-leaf-dirs` | `input_dir`；可选 `recursive`、`extensions` |
| `clean-nested-dataset` | `input_dir`、可选 `output_dir`；默认自动识别目录布局；高级参数含 `pairing_mode`、`flatten`、`include_backgrounds`、`copy_files`、`overwrite`，以及 `images_dir_name`、`xmls_dir_name`、`images_dir_aliases`、`xmls_dir_aliases`、`backgrounds_dir_name` |
| `aggregate-nested-dataset` | `input_dir`、可选 `output_dir`；`images_dir_name`、`labels_dir_name`、`backgrounds_dir_name`、`classes_file_name`、`require_non_empty_labels`、`overwrite` |
| `xml-to-yolo` | `input_dir` |
| `annotate-visualize` | `input_dir`、`output_dir`；要求 `input_dir/images` 存在，默认优先使用 `input_dir/labels`，若不存在再回退到 `input_dir/xmls`；可选 `recursive`、`extensions`、`include_difficult`（仅 XML）、`line_width`、`overwrite`；YOLO 类别名可选 `classes` 或 `classes_file`（二选一，**均可省略**；均不提供或 `classes_file` 为空字符串时，框上显示**类别 id 数字**） |
| `reset-yolo-label-index` | `input_dir`（其下需有 `labels/`）；可选 `recursive`（默认 `true`）；**原地**将各 `.txt` 每行首列类别 id 改为 `0`（§8） |
| `scan-yolo-label-indices` | `input_dir`（其下需有 `labels/`）；可选 `recursive`（默认 `true`）；递归统计 labels 中出现过的类别索引及数量 |
| `rewrite-yolo-label-indices` | `input_dir`（其下需有 `labels/`）；可选 `recursive`（默认 `true`）；`mapping` 指定部分索引映射，`default_target_index` 处理其余索引；**原地**批量改写类别 id |
| `split-yolo-dataset` | `input_dir`（其下需有 `images/`、`labels/`）、`output_dir`（可选，默认 `<input_dir>/split_dataset`） |
| `yolo-augment` | `input_dir`（支持根目录下直接有 `images/`、`labels/`，也支持递归发现 `*/images` 与同级 `labels/`，并保留相对目录层级输出）、`output_dir`（可选，默认 `<input_dir>/augment`）、七个增强开关（默认全开）；仅支持 YOLO TXT |
| `yolo-sliding-window-crop` | `input_dir`（支持根目录下直接有 `images/`，也支持递归发现 `*/images`；若同级存在 `labels/` 则自动同步输出裁剪标注并保留相对目录层级）、`output_dir`；可选 `window_width`、`window_height`、`stride_x`、`stride_y`（传入即覆盖默认）；以及 `min_vis_ratio`、`stride_ratio`、`ignore_vis_ratio`、`only_wide` |
| `build-yolo-yaml` | 同上；**`last_yaml`** 与当前扫描路径合并（旧路径在前、去重）。**`classes.txt` 有有效类别行时**：`nc`/`names` 以该文件为准。**`classes.txt` 为空或不存在时**：必须提供 **`last_yaml`**，且其中需含 **`names`**（及通常的 `nc`），类别表从该 YAML 读取；当前数据集各划分路径仍按扫描结果**追加**到对应 `train`/`val`/…。远程 `last_yaml` 需 **`sftp_username`**、**`sftp_private_key_path`** |
| `publish-yolo-dataset` | `input_dir`、可选 `input_dirs`（多数据集合并）、可选 `project_root_dir`、可选 `detector_name`；`publish_mode=local` 时发布到本地规范目录并生成 yaml；`publish_mode=remote_sftp` 时内部走 zip + SFTP 上传 + 远端解压。若 `.env` 已配置 `SELF_API_REMOTE_SFTP_*`，则可省略远端 SSH 参数；若未传 `project_root_dir`，则使用 `SELF_API_PUBLISH_PROJECT_ROOT_DIR` 或回落到 `SELF_API_STORAGE_ROOT/publish_workspace` |
| `publish-incremental-yolo-dataset` | 面向“远程增量发布”的最小化专用接口；仅需 `last_yaml` + `local_paths`。其余字段都由 `last_yaml` 路径和服务端 `.env` 自动推导/补齐 |
| `zip-folder` | `input_dir`、`output_zip_path`（可选，默认 `<input_dir_parent>/<input_dir_name>.zip`）、`include_root_dir`、`overwrite` |
| `remote-transfer` | `source_path`、`target`；以及 `username`、`password` / `private_key_path`、`port`、`overwrite` |
| `remote-unzip` | `archive_path`、`output_dir`；以及 `username`、`password` / `private_key_path`、`port`、`overwrite` |
| `unzip-archive` | `archive_path`、`output_dir`（可选）、`overwrite` |
| `move-path` / `copy-path` | `source_path`、`target_dir`、`overwrite` |
| `yolo-train` | `yaml_path`、`project_root_dir`（子进程 `cwd`）、`project`、`name`、`yolo_train_env`（conda 环境名）；以及 `model`、`epochs`、`imgsz` |
| `remote-sbatch-yolo-train` | `host`（或在路径里带 host）、`yaml_path`、`project_root_dir`、`project`、`name`、`yolo_train_env`、`username`；以及 `model`、`epochs`、`imgsz`、`batch`、`workers` |
| `voc-bar-crop` | `input_dir`（其下需有 `images/`、`xmls/`）、`output_dir`；可选 `recursive`（默认 `true`）；裁剪正方形边长为**源图高度**（§17） |
| `restore-voc-crops-batch` | `original_images_dir`、`original_xmls_dir`、`edited_crops_images_dir`、`edited_crops_xmls_dir`、`output_dir`；可选 `recursive`、`skip_unparsed_names`（§18） |

**异步任务轮询**：`GET /api/v1/preprocess/tasks/{task_id}` 返回体中，业务结果在 **`result`** 对象内（例如 `result.output_dir`、`result.output_zip_path`），勿与顶层字段混淆。

**n8n HTTP Request**：服务端要求 **`Content-Type: application/json`**。在节点里用「Body Parameters / 字段列表」维护时，请将 **Body Content Type** 设为 **JSON**，**Specify Body** 设为 **Using Fields Below（keypair）**，不要用整段 `JSON.stringify` 表达式，以便与下表字段一一对应、方便修改。

## 1. 健康检查

- `GET /api/v1/healthz`

```bash
curl "http://192.168.2.26:8666/api/v1/healthz"
```

## 2. 常用 SOP（按最常用 workflow）

以下顺序按你当前最常用的两条工作流整理：

- 小图 `images+xmls` 基线训练：`xml-to-yolo -> split-yolo-dataset -> yolo-augment（按需） -> build-yolo-yaml -> zip/move/unzip 或 remote-transfer/remote-unzip -> yolo-train / remote-sbatch-yolo-train`
- 大图 `images+xmls` 常发迭代：`clean-nested-dataset（按需） -> xml-to-yolo -> reset-yolo-label-index（单类时） -> yolo-sliding-window-crop -> split-yolo-dataset 或直接 build-yolo-yaml -> zip/remote-transfer/remote-unzip -> yolo-train / remote-sbatch-yolo-train`
- 多层目录数据整理：`discover-leaf-dirs -> clean-nested-dataset -> xml-to-yolo -> aggregate-nested-dataset`

阅读建议也按这个顺序往下看：先看数据进入训练前的整理链路，再看投放与训练，最后看少用的 VOC 裁剪/回贴工具。

## 3. 多层目录叶子数据目录发现

- `POST /api/v1/preprocess/discover-leaf-dirs`
- `POST /api/v1/preprocess/discover-leaf-dirs/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/discover-leaf-dirs" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成"
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

## 4. 多层目录数据清洗

- `POST /api/v1/preprocess/clean-nested-dataset`
- `POST /api/v1/preprocess/clean-nested-dataset/async`

**标准模式**：只传 `input_dir` 与 `output_dir` 即可，接口会自动识别目录布局。

**`pairing_mode`**（默认 `auto`）：

- **`auto`**：自动识别当前数据更像 `same_directory` 还是 `images_xmls_subfolders`
- **`same_directory`**：递归识别「叶子目录」（目录内**直接**含有图片或 XML 文件），在同一目录内按 stem 配对。
- **`images_xmls_subfolders`**：识别同时含有子目录 `images_dir_name` 与 `xmls_dir_name` 的文件夹（VOC 常见布局：`样本/images/*.jpg` 与 `样本/xmls/*.xml`），在**该父目录**范围内配对；不要求图与 XML 在同一子文件夹内。若原始目录名不固定，可额外传 `images_dir_aliases` / `xmls_dir_aliases`，例如 `["images", "image"]` 与 `["xmls", "xml"]`。

有有效标注的写入 `images/` 与 `xmls/`；无标注或 XML 无效的图默认写入 `backgrounds/`（可通过 `include_backgrounds` 关闭，仅保留成对输出）。

默认（`flatten: false`）在 `output_dir` 下**保留**相对 `input_dir` 的子目录结构。设为 `flatten: true` 时，所有处理单元的结果合并到 **`output_dir/images`**、**`output_dir/xmls`**（及可选的 **`output_dir/backgrounds`**），文件名会带上相对路径前缀（如 `子路径__样本目录__原文件名.jpg`），避免不同分支同名文件互相覆盖。

响应中的 **`skipped_unlabeled_images`**（及每条 `details` 内的同名字段）表示：在 `include_backgrounds: false` 时未复制到 backgrounds 的无标注图数量。

### 4.1 保留目录结构（默认）

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset"
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/TEDS广州局正线_转向架漏油_负类标注_20260312-标注完成",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset",
    "recursive": true,
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "backgrounds_dir_name": "backgrounds",
    "include_difficult": false,
    "pairing_mode": "same_directory",
    "flatten": false,
    "include_backgrounds": true,
    "copy_files": true,
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

### 4.2 扁平化输出，且仅整理 images / xmls（不输出 backgrounds）

适用于希望直接得到单层 `images/`、`xmls/` 目录的场景。若数据为 **每样本目录下分 `images/` 与 `xmls/` 子文件夹**（如 `/media/qzq/16T/n8n_workspace/.../20260407_非转数据拟合/`），必须设置 **`pairing_mode`: `images_xmls_subfolders`**。

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/n8n_workspace/nzxj_diaban_louyou/20260407_非转数据拟合",
    "output_dir": "/media/qzq/16T/n8n_workspace/nzxj_diaban_louyou/20260407_非转数据拟合/cleaned_flat",
    "recursive": true,
    "pairing_mode": "images_xmls_subfolders",
    "flatten": true,
    "include_backgrounds": false,
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "images_dir_aliases": ["images", "image"],
    "xmls_dir_aliases": ["xmls", "xml"],
    "include_difficult": false,
    "copy_files": true,
    "overwrite": true
  }'

# 异步（参数与上相同，仅多 callback）
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/clean-nested-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/n8n_workspace/nzxj_diaban_louyou/20260407_非转数据拟合",
    "output_dir": "/media/qzq/16T/n8n_workspace/nzxj_diaban_louyou/20260407_非转数据拟合/cleaned_flat",
    "recursive": true,
    "pairing_mode": "images_xmls_subfolders",
    "flatten": true,
    "include_backgrounds": false,
    "images_dir_name": "images",
    "xmls_dir_name": "xmls",
    "images_dir_aliases": ["images", "image"],
    "xmls_dir_aliases": ["xmls", "xml"],
    "include_difficult": false,
    "copy_files": true,
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 5. 多层目录数据集汇总

- `POST /api/v1/preprocess/aggregate-nested-dataset`
- `POST /api/v1/preprocess/aggregate-nested-dataset/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/aggregate-nested-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/cleaned_dataset",
    "output_dir": "/media/qzq/16T/TEDS/广州局__转向架__正线漏油/20260312/dataset"
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
2. `clean-nested-dataset`：把原始图片/XML清洗为 `images` / `xmls`（及可选 `backgrounds`）；若使用 `flatten: true`，输出集中在同一 `output_dir` 下，后续 `xml-to-yolo` 的 `input_dir` 可直接填该目录
3. `xml-to-yolo`：在含 `images/` 与 `xmls/` 的数据集根目录生成 `labels/`
4. `aggregate-nested-dataset`：若存在多个碎片目录需再汇总，则指向包含多组 `images/labels/...` 的上级目录；若已在步骤 2 扁平化为单一目录并完成步骤 3，可按需跳过或仅汇总其它路径

## 6. VOC XML 转 YOLO 标签

- `POST /api/v1/preprocess/xml-to-yolo`
- `POST /api/v1/preprocess/xml-to-yolo/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/xml-to-yolo" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/voc_like"
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/xml-to-yolo/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/voc_like",
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

## 7. 标注可视化（YOLO txt / VOC XML）

在 `input_dir/images` 上绘制检测框并导出到 `output_dir`，目录结构与 `images/` 下相对路径一致。

标准模式下不再传 `images_dir`、`labels_dir`、`xmls_dir`。接口会自动判定：
- 若 `input_dir/labels` 存在，默认按 YOLO `.txt` 可视化
- 若 `input_dir/labels` 不存在但 `input_dir/xmls` 存在，则按 Pascal VOC XML 可视化

`classes` 与 `classes_file` 不能同时**非空**填写（二者可同时省略）。若未提供 `classes` 且 `classes_file` 为空或省略，则框上文字为**类别 id**（与 YOLO txt 中每行首列整数一致）。

- `POST /api/v1/preprocess/annotate-visualize`
- `POST /api/v1/preprocess/annotate-visualize/async`

```bash
# 同步：最小调用，默认优先使用 input_dir/labels
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线",
    "output_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/visualized"
  }'
```

若需显示名称，可加 `"classes_file": "./data/yolo_raw/classes.txt"` 或传 `"classes": ["类0", "类1", ...]`。默认不传时，YOLO 框上显示数字类别 id。

```bash
# 同步：仅有 xmls/ 时自动回退到 Pascal VOC XML
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线",
    "output_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/visualized",
    "include_difficult": false
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/annotate-visualize/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/yolo_raw",
    "output_dir": "./data/yolo_raw/visualized",
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

**响应要点**（同步 body / 异步任务的 `result`）：`mode` 为 `yolo` 或 `xml`；`annotation_dir` 为实际使用的标注根目录；`written_images` / `skipped_images`；`details` 中每项含 `source_image`、`output_image`、`boxes_drawn`、`skipped_reason`（失败或跳过原因）。

## 8. 批量将 YOLO labels 类别索引置 0（`reset-yolo-label-index`）

- `POST /api/v1/preprocess/reset-yolo-label-index`
- `POST /api/v1/preprocess/reset-yolo-label-index/async`

在 `input_dir/labels/` 下递归（或仅一层）扫描 `*.txt`，按 YOLO 格式解析每行；将**首列类别 id** 统一改为 `0`，坐标列不变。**直接覆盖原文件**，请事先备份。

返回字段包括：`labels_dir`、`total_label_files`、`modified_label_files`、`unchanged_label_files`、`changed_lines`、`skipped_invalid_lines`（格式异常行数）。

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/reset-yolo-label-index" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/datasets"
  }'
```

异步（`task_type` 为 `reset_yolo_label_index`；结果在 `GET /api/v1/preprocess/tasks/{task_id}` 的 `result` 中，亦可通过 `callback_url` 回调）：

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/reset-yolo-label-index/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/datasets/demo",
    "recursive": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 9. YOLO 数据集划分

- `POST /api/v1/preprocess/split-yolo-dataset`
- `POST /api/v1/preprocess/split-yolo-dataset/async`

要求 `input_dir` 下同时存在 `images/` 与 `labels/`；若存在 `input_dir/classes.txt` 会自动复制到 `output_dir/classes.txt`。输出目录结构由 `output_layout` 控制（默认 `split_first`：`<output_dir>/<split>/images,labels`；`images_first`：`<output_dir>/images,labels/<split>`）。

```bash
# 同步：最小使用示例
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/split-yolo-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/yolo_raw",
    "output_dir": "./data/yolo_split"
  }'

# 异步：按需调整比例 / 输出布局
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/split-yolo-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/yolo_raw",
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

## 10. YOLO 数据增强（`yolo-augment`）

- `POST /api/v1/preprocess/yolo-augment`
- `POST /api/v1/preprocess/yolo-augment/async`

仅支持 YOLO TXT 标注。输入目录支持两种布局：

- `<input_dir>/images`
- `<input_dir>/labels`
- `<input_dir>/train/images`、`<input_dir>/train/labels`、`<input_dir>/val/images`、`<input_dir>/val/labels` 等任意 nested `*/images + */labels`

输出目录默认写到：

- `<input_dir>/augment/images`
- `<input_dir>/augment/labels`
- 若输入是多层 split 结构，则保留相对层级，例如 `<input_dir>/train/images -> <input_dir>/augment/train/images`

默认启用以下七种增强：

- `horizontal_flip`
- `vertical_flip`
- `brightness_up`
- `brightness_down`
- `contrast_up`
- `contrast_down`
- `gaussian_blur`

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-augment" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/datasets/demo"
  }'
```

补充说明：

- 每张图会按启用的增强项分别生成新文件，文件名后缀形如 `_hflip`、`_contrast_up`
- 水平/垂直翻转会同步改写 YOLO 框中心点；亮度、对比度、高斯模糊仅改图像，不改框
- 若 `labels/classes.txt` 存在，会复制到输出 `labels/`
- 若未显式传 `output_dir`，默认输出到 `<input_dir>/augment`

异步端点为 `/yolo-augment/async`，在同步 body 上增加 `callback_url`、`callback_timeout_seconds` 即可（同 §0）。

## 11. YOLO 滑窗裁剪为小图数据集

- `POST /api/v1/preprocess/yolo-sliding-window-crop`
- `POST /api/v1/preprocess/yolo-sliding-window-crop/async`

要求 `input_dir` 下包含 `images/` 子目录；若同时存在 `labels/` 则自动输出裁剪后的 YOLO 标注，否则只输出裁剪图片。窗口宽高默认等于**图片高度**；`stride_x` 默认等于 `round(stride_ratio * 图片高度)`；`stride_y` 默认等于 `window_height`。如果手动传入 `window_width`、`window_height`、`stride_x`、`stride_y`，则按传入值覆盖对应默认。

```bash
# 同步：最小使用示例（input_dir 下有 images/ 和 labels/）
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-sliding-window-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset",
    "output_dir": "/path/to/dataset/data_crops"
  }'

# 异步：自定义窗口与步长
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-sliding-window-crop/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset",
    "output_dir": "/path/to/dataset/data_crops",
    "window_width": 2048,
    "window_height": 2048,
    "stride_x": 1024,
    "stride_y": 1024,
    "stride_ratio": 0.2,
    "only_wide": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

响应要点：

- `labels_dir`：`input_dir/labels` 存在时为实际路径，否则返回 `null`
- `generated_crops` 为输出图片数
- `generated_labels` 为输出标注行数；无 labels 时固定为 `0`

命令行等价示例（带 labels）：

```bash
python yolo_square_sliding_crop.py \
  --images /path/to/dataset/images \
  --labels /path/to/dataset/labels \
  --out /path/to/dataset/data_crops \
  --window_width 1024 \
  --window_height 1024 \
  --stride_x 512 \
  --stride_y 512 \
  --min_vis_ratio 0.5 \
  --stride_ratio 0.2 \
  --ignore_vis_ratio 0.05 \
  --only_wide
```

## 12. 生成 YOLO `data.yaml`（Ultralytics）

- `POST /api/v1/preprocess/build-yolo-yaml`
- `POST /api/v1/preprocess/build-yolo-yaml/async`

在 `input_dir` 下自动匹配数据集根（见 §0），扫描 `train` / `val` / `test` 等划分；各划分下需存在 `images` 子目录且含至少一张图片。生成的 YAML 中 **`train` / `val` / `test` 的值为各划分 `images` 目录的绝对路径**（POSIX），例如 `train: /Users/.../dataset1/dataset/train/images`，不再使用单独的 `path:` 与相对子路径。`path_prefix_replace_from` / `path_prefix_replace_to` 若成对填写，则对**当前扫描得到的**划分路径做前缀替换（`last_yaml` 里已有的路径**不会**再替换）。

**`last_yaml`（可选）**：上一份数据 YAML。解析其中的划分路径，与本次从 `input_dir` 扫描得到的路径按划分**合并**（先 `last_yaml`、后本次扫描，路径去重）。**类别名 `nc`/`names`**：若 **`classes.txt` 存在且含至少一行有效类别**，则以该文件为准；若 **`classes.txt` 为空或不存在**，则**必须**提供 `last_yaml`，并从其中的 **`names`**（及行数决定 `nc`）读取，此时以 `last_yaml` 为类别基准，当前数据仅**追加**各划分下的 images 路径。

远程 `last_yaml` 示例：`"last_yaml": "sftp://my.host/var/data.yaml"`，并设置 `"sftp_username": "me"`、`"sftp_private_key_path": "/home/me/.ssh/id_ed25519"`；可选 `"sftp_port": 22`。

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

`classes_file` 可省略，按 §0 在 `dataset` / `yolo_split` 等位置自动查找 `classes.txt`。`path_prefix_replace_*` 可省略；若填写则须成对出现，且每条**扫描得到的**划分路径须以 `path_prefix_replace_from` 为前缀。

若只走标准模式，同步调用通常只需：

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/build-yolo-yaml" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/temp_dataset",
    "output_yaml_path": "/path/to/temp_dataset/dataset.yaml"
  }'
```

## 13. 目录打包为 ZIP

- `POST /api/v1/preprocess/zip-folder`
- `POST /api/v1/preprocess/zip-folder/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/zip-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "./data/source_folder",
    "output_zip_path": "./data/source_folder.zip"
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

## 13.1 跨机器 SFTP 远程传输

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
    "private_key_path": "~/.ssh/id_rsa"
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
  "target": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/",
  "target_host": "172.31.1.9",
  "target_port": 22,
  "target_path": "/mnt/usrhome/sk/ndata/dataset",
  "transferred_type": "directory",
  "transferred_files": 42,
  "total_bytes": 1048576
}
```

## 13.2 跨机器远程解压（在目标机执行）

- `POST /api/v1/preprocess/remote-unzip`
- `POST /api/v1/preprocess/remote-unzip/async`

将远端 ZIP 在目标主机上通过 SSH 执行 `unzip` 解压。`archive_path` 与 `output_dir` 支持 `sftp://` URL。

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-unzip" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/TVDS_n8n/qianyindianji_louyou/dataset/qianyindianji_louyou_20260409_090414_.zip",
    "output_dir": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/TVDS_n8n/qianyindianji_louyou/dataset",
    "username": "sk",
    "private_key_path": "~/.ssh/id_ed25519"
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-unzip/async" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/TVDS_n8n/qianyindianji_louyou/dataset/qianyindianji_louyou_20260409_090414_.zip",
    "output_dir": "sftp://172.31.1.9/mnt/usrhome/sk/ndata/TVDS_n8n/qianyindianji_louyou/dataset",
    "username": "sk",
    "private_key_path": "~/.ssh/id_ed25519",
    "overwrite": true,
    "callback_url": "http://127.0.0.1:9000/webhooks/preprocess-finished",
    "callback_timeout_seconds": 10
  }'
```

## 14. ZIP 解压

- `POST /api/v1/preprocess/unzip-archive`
- `POST /api/v1/preprocess/unzip-archive/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/unzip-archive" \
  -H "Content-Type: application/json" \
  -d '{
    "archive_path": "./data/source_folder.zip",
    "output_dir": "./data/unpacked"
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

## 15. 文件或目录移动

- `POST /api/v1/preprocess/move-path`
- `POST /api/v1/preprocess/move-path/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/move-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_ready"
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

## 15.1 文件或目录复制

- `POST /api/v1/preprocess/copy-path`
- `POST /api/v1/preprocess/copy-path/async`

```bash
# 同步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/copy-path" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/unpacked",
    "target_dir": "./data/archive_backup"
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

## 16. YOLO 训练（conda + `yolo train`）

- `POST /api/v1/preprocess/yolo-train`
- `POST /api/v1/preprocess/yolo-train/async`

该接口在 `project_root_dir` 下启动子进程，执行 `conda run -n <yolo_train_env> --no-capture-output yolo train ...`。

`project` 与 `name` 不再由 API 内部推导，必须由调用方显式传入，并满足当前项目统一规范：

- `name` 必须等于 `yaml_path` 对应 YAML 文件名的 stem
- `yaml_path` 必须符合 `<root_dir>/<detector_name>/datasets/<dataset_version>/<dataset_version>.yaml`
- `project` 必须落在训练 bucket 下，也就是以 `runs/detect`、`runs/segment`、`runs/classify`、`runs/pose` 或 `runs/obb` 结尾
- `project` 必须与 `yaml_path` 共享同一个 `<root_dir>/<detector_name>` 前缀
- 推荐由上游 skill 根据知识库统一生成，而不是在调用时手写猜测

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/yolo-train" \
  -H "Content-Type: application/json" \
  -d '{
    "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
    "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
    "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
    "name": "nzxj_louyou_20260417_1430",
    "yolo_train_env": "yolo_pose",
    "model": "yolo11s.pt",
    "epochs": 100,
    "imgsz": 640
  }'
```

响应含 `command`（拼接后的命令行）、`cwd`、`project`、`name`、`exit_code`、`stdout`、`stderr`。训练失败时 `exit_code` 非零且 `status` 为 `failed`。

## 16.1 远程 SBATCH YOLO 训练（SSH + `sbatch`）

- `POST /api/v1/preprocess/remote-sbatch-yolo-train`
- `POST /api/v1/preprocess/remote-sbatch-yolo-train/async`

该接口通过 SSH 登录远端机器，执行 `sbatch --parsable --wrap "conda run -n <env> --no-capture-output yolo train ..."` 提交任务。`project` 与 `name` 不再由 API 内部推导，必须由调用方显式传入。

训练输入同样遵守本地训练的结构约束：

- `name` 必须等于 `yaml_path` 对应 YAML 文件名的 stem
- `yaml_path` 必须符合 `<root_dir>/<detector_name>/datasets/<dataset_version>/<dataset_version>.yaml`
- `project` 必须落在训练 bucket 下
- `project` 必须与 `yaml_path` 共享同一个 `<root_dir>/<detector_name>` 前缀

- Slurm stdout/stderr 默认写入 `project_root_dir/logs/` 下的 `slurm-%j.out`、`slurm-%j.err`
- 任务名默认 `self_api_train`
- 若需覆盖日志路径，可显式传 `stdout_path`、`stderr_path`

`yaml_path` 与 `project_root_dir` 支持绝对路径，或 `sftp://...` / `user@host:path`。若直接写绝对路径，则需要额外传 `host`。

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/remote-sbatch-yolo-train" \
  -H "Content-Type: application/json" \
  -d '{
    "host": "172.31.1.9",
    "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
    "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
    "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
    "name": "nzxj_louyou_20260417_1430",
    "yolo_train_env": "yolo_pose",
    "username": "sk",
    "model": "yolo11m.pt",
    "epochs": 200,
    "imgsz": 800,
    "batch": 24,
    "workers": 4,
    "cache": true,
    "partition": "gpu",
    "nodelist": "node11,node12",
    "exclude": "node42",
    "private_key_path": "~/.ssh/id_ed25519"
  }'
```

字段说明（重点）：

- `username`：SSH 用户名，必填
- `host`：远端主机；当 `yaml_path` / `project_root_dir` 不带 host 时必填
- `project` / `name`：必填；由上游 skill 统一规范后显式传入
- `yolo_train_env`：远端 conda 环境名
- `batch`：传给 `yolo train`
- `workers`：传给 `yolo train`
- `cache`：可选；默认 `true`
- `device`：可选；显式传给 `yolo train`
- `partition`：可选，默认 `gpu`
- `nodelist` / `exclude`：可选，分别映射到 `sbatch --nodelist` / `--exclude`
- `project_root_dir`：远端任务的工作目录，默认日志会写到其下的 `logs/`
- `password` / `private_key_path` / `port`：SSH 连接参数

补充说明：

- 成功时返回 `job_id`
- 若集群资源紧张，任务提交成功后也可能仍为 `PENDING`
- 若要覆盖默认日志路径，可传 `stdout_path`、`stderr_path`

异步示例在同步 body 上增加 `callback_url`、`callback_timeout_seconds` 即可（同 §0）。

## 19. 增量 YOLO 数据集远程发布（`publish-yolo-dataset`）

适用于“保留旧 YAML 中已有训练路径，再把本次新数据集追加进去，并同步发布到远端 SFTP”。

- `POST /api/v1/preprocess/publish-yolo-dataset`
- `POST /api/v1/preprocess/publish-yolo-dataset/async`

支持以下能力：

- 最简输入模型：只传本地目录路径与远端 `last_yaml`
- `input_dir` + `input_dirs`：一次合并多个本地 YOLO 数据集目录，例如原始滑窗集和增强集
- 若只传一个基础目录，服务端会自动匹配同级 `<input_dir>_aug`
- 若只传一个 `_aug` 目录，服务端也会自动回找同级基础目录
- `last_yaml`：把旧 YAML 中已有的 `train` / `val` / `test` 路径保留在前，再把本次数据追加在后
- `publish_mode=remote_sftp`：自动在远端创建 `<remote_project_root_dir>/<detector_name>/datasets/<dataset_version>`，本地先打包 zip，上传后再远端解压
- 若 `.env` 已配置 `SELF_API_REMOTE_SFTP_HOST`、`SELF_API_REMOTE_SFTP_PROJECT_ROOT_DIR`、`SELF_API_REMOTE_SFTP_USERNAME`、`SELF_API_REMOTE_SFTP_PRIVATE_KEY_PATH`，请求体里可省略这些远端字段
- 若未传 `project_root_dir`，则使用 `SELF_API_PUBLISH_PROJECT_ROOT_DIR`；若该项也未配置，则自动回落到 `SELF_API_STORAGE_ROOT/publish_workspace`
- 若 `last_yaml` 形如 `sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml`，则会自动推导：
  `remote_host=172.31.1.42`、`remote_project_root_dir=/mnt/usrhome/sk/ndata/TEDS_n8n`、`detector_name=nzxj_louyou`，并沿用旧路径中的 `dataset` 或 `datasets` 目录命名风格

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/publish-yolo-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/20260407_非转数据拟合_cleaned_flat_split_yolo-sliding-window-crop",
    "publish_mode": "remote_sftp",
    "last_yaml": "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml"
  }'
```

如果同级 `_aug` 不存在，或你需要显式指定两份本地目录，则再补 `input_dirs`：

```json
{
  "input_dir": "/path/to/base_dataset",
  "input_dirs": ["/path/to/base_dataset_aug"],
  "last_yaml": "sftp://host/root/project/dataset/old_ver/old_ver.yaml",
  "publish_mode": "remote_sftp"
}
```

**响应要点**：`dataset_version` 为本次新目录名；`published_dataset_dir` 为远端最终目录；`output_yaml_path` 为远端新 YAML；`local_archive_path` 为本地 zip；`remote_archive_path` 为远端 zip；`source_dataset_roots` 为本次合并发布的本地数据集根目录列表。

### 19.1 最小化远程增量发布（`publish-incremental-yolo-dataset`）

这是面向日常使用的简化接口，不再暴露远端 SSH 参数，也不要求调用方拼接 `input_dir/input_dirs/project_root_dir/detector_name`。

- `POST /api/v1/preprocess/publish-incremental-yolo-dataset`
- `POST /api/v1/preprocess/publish-incremental-yolo-dataset/async`

只需两个字段：

- `last_yaml`：旧数据集 YAML 的远程完整路径
- `local_paths`：一个或多个本地补充数据集目录

自动规则：

- 如果 `local_paths` 只给一个基础目录，服务端自动匹配同级 `<dir>_aug`
- 如果只给一个 `_aug` 目录，服务端自动回找基础目录
- 从 `last_yaml` 自动推导 `remote_host`、`remote_project_root_dir`、`detector_name`、`dataset/datasets` 目录风格
- `remote_username`、`remote_private_key_path`、`remote_port` 从服务端 `.env` 读取
- 新的 `dataset_version` 自动生成

```bash
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/publish-incremental-yolo-dataset/async" \
  -H "Content-Type: application/json" \
  -d '{
    "last_yaml": "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml",
    "local_paths": [
      "/media/qzq/16T/TEDS/langgraph_workspace/download-2026-04-07_10-32-30/teds/整车图/正线/20260407_非转数据拟合_cleaned_flat_split_yolo-sliding-window-crop"
    ]
  }'
```

若同级 `_aug` 不存在，或你想显式指定两份新数据目录：

```json
{
  "last_yaml": "sftp://172.31.1.42/mnt/usrhome/sk/ndata/TEDS_n8n/nzxj_louyou/dataset/nzxj_louyou_20260423_2033/nzxj_louyou_20260423_2033.yaml",
  "local_paths": [
    "/path/to/base_dataset",
    "/path/to/base_dataset_aug"
  ]
}
```

## 17. VOC 横向条带正方形裁剪（`voc-bar-crop`）

对 `xmls_dir` 中每个 Pascal VOC XML：对每个 **宽不小于高** 的目标框触发一次裁剪。正方形 **边长 = 源图像高度**（宽≥高的横向长条图：`top=0`，水平方向以该框中心居中，左右越界则在图内平移；窄高图则 `left=0`、垂直居中）。与 `yolo-sliding-window-crop` 的「边长=图片高度」一致，**不是**标注框高度。小图文件名含裁剪中心与边长，例如 `stem_cx512_cy409_S819.jpg`，同 stem 的 XML 写入 `output_dir/xmls/`。裁剪图内保留 **与窗口相交** 的所有物体框（坐标换算到小图）。

- `POST /api/v1/preprocess/voc-bar-crop`
- `POST /api/v1/preprocess/voc-bar-crop/async`

要求 `input_dir` 下包含 `images/` 与 `xmls/` 子目录。

```bash
# 同步：最小使用示例
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/voc-bar-crop" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset",
    "output_dir": "/path/to/dataset/bar_crops"
  }'

# 异步
curl -X POST "http://192.168.2.26:8666/api/v1/preprocess/voc-bar-crop/async" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/dataset",
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

异步示例在同步 body 上增加 `callback_url`、`callback_timeout_seconds` 即可（同 §0）。
