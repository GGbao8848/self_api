# self_api - 图像/数据集预处理 API

用于图像与图像数据集预处理的最小可交付 API 服务，当前提供 8 个核心能力：

1. 指定目录图像按滑窗规则裁剪并保存
2. Pascal VOC XML 标注转换为 YOLO 标注
3. YOLO 数据集按 train/val/test 划分
4. 指定目录打包为 zip 压缩包
5. 指定 zip 压缩包解压到目标目录
6. 文件或文件夹整体移动到目标目录
7. 文件或文件夹整体复制到目标目录
8. YOLO 大图数据集滑窗裁剪为小图数据集（标签同步裁剪）

## 1. 最小可交付范围

- 完整 API 服务骨架（`FastAPI`）
- API 版本化路由（`/api/v1`）
- 健康检查（`/api/v1/healthz`）
- 请求/响应模型与参数校验（Pydantic）
- 统一错误处理（非法输入返回 `400`）
- 核心业务服务分层（`services`）
- 基础可运行测试（`pytest`）
- Docker 化运行支持

## 2. 项目架构

```text
self_api/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── preprocess.py      # 八个预处理 API
│   │       │   └── system.py          # 健康检查
│   │       └── router.py              # v1 路由聚合
│   ├── core/
│   │   ├── config.py                  # 环境配置
│   │   └── logging.py                 # 日志初始化
│   ├── schemas/
│   │   └── preprocess.py              # 请求/响应模型
│   ├── services/
│   │   ├── file_operations.py         # 压缩/解压/移动/复制服务
│   │   ├── sliding_window.py          # 滑窗裁剪服务
│   │   ├── split_yolo_dataset.py      # YOLO 数据集划分服务
│   │   ├── xml_to_yolo.py             # VOC XML 转 YOLO 标签服务
│   │   └── yolo_sliding_window.py     # YOLO 大图数据集滑窗裁剪服务
│   ├── utils/
│   │   └── images.py                  # 图像文件扫描工具
│   └── main.py                        # FastAPI 应用入口
├── tests/
│   ├── test_healthz_api.py
│   ├── test_sliding_window_crop_api.py
│   ├── test_xml_to_yolo_api.py
│   ├── test_split_yolo_dataset_api.py
│   ├── test_zip_folder_api.py
│   ├── test_unzip_archive_api.py
│   ├── test_move_path_api.py
│   └── test_yolo_sliding_window_crop_api.py
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## 3. 快速启动

### 3.1 本地

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make run
```

启动后可访问：

- 本机：`http://127.0.0.1:8666/docs`
- 内网：`http://<你的内网IP>:8666/docs`

说明：

- `make run` 默认绑定 `0.0.0.0:8666`，可被同一局域网设备访问
- 如需自定义端口/地址：`make run HOST=0.0.0.0 PORT=9000`
- 若内网仍不可达，请检查系统防火墙是否允许该端口入站访问

### 3.2 Docker

```bash
docker build -t self-api:0.1.0 .
docker run --rm -p 8000:8000 self-api:0.1.0
```

## 4. API 要素与接口定义

接口示例请求已拆分到独立文档：`docs/api_examples.md`。

### 4.1 健康检查

- `GET /api/v1/healthz`

### 4.2 滑窗裁剪

- `POST /api/v1/preprocess/sliding-window-crop`

关键参数：

- `input_dir`: 输入目录
- `output_dir`: 输出目录
- `window_width/window_height`: 窗口宽高
- `stride_x/stride_y`: 步长
- `include_partial_edges`: 是否保留边缘不完整窗口
- `output_format`: `keep/png/jpg/jpeg/webp`

### 4.3 VOC XML 转 YOLO 标签

- `POST /api/v1/preprocess/xml-to-yolo`

要求输入目录下有 `images/` 与 `xmls/`，服务会把 YOLO 标签写到 `labels/`（目录名可配置）。

关键参数：

- `dataset_dir`: 数据集根目录（含 `images` 和 `xmls`）
- `images_dir_name/xmls_dir_name/labels_dir_name`: 目录名配置
- `classes`: 可选，固定类别顺序；不传则自动从 XML 推断
- `include_difficult`: 是否包含 `difficult=1` 目标
- `write_classes_file`: 是否在根目录写出 `classes.txt`

### 4.4 YOLO 数据集划分

- `POST /api/v1/preprocess/split-yolo-dataset`

关键参数：

- `dataset_dir`: YOLO 数据集根目录（含 `images` 与 `labels`）
- `output_dir`: 输出目录（默认 `dataset_dir/split_dataset`）
- `mode`: `train_val_test` / `train_val` / `train_only`
- `train_ratio/val_ratio/test_ratio`: 划分比例（会自动归一化）
- `shuffle/seed`: 是否打乱及随机种子
- `copy_files`: `true` 复制，`false` 移动

### 4.5 目录打包为 ZIP

- `POST /api/v1/preprocess/zip-folder`

关键参数：

- `input_dir`: 待打包目录
- `output_zip_path`: 输出压缩包路径（可选，默认 `input_dir` 同级）
- `include_root_dir`: 压缩包内是否保留根目录名
- `overwrite`: 压缩包已存在时是否覆盖

### 4.6 ZIP 解压

- `POST /api/v1/preprocess/unzip-archive`

关键参数：

- `archive_path`: 压缩包路径（zip）
- `output_dir`: 解压输出目录（可选）
- `overwrite`: 目标文件已存在时是否覆盖

### 4.7 文件或目录移动

- `POST /api/v1/preprocess/move-path`

关键参数：

- `source_path`: 源文件或源目录
- `target_dir`: 目标目录
- `overwrite`: 目标同名已存在时是否覆盖

### 4.8 文件或目录复制

- `POST /api/v1/preprocess/copy-path`

关键参数：

- `source_path`: 源文件或源目录
- `target_dir`: 目标目录
- `overwrite`: 目标同名已存在时是否覆盖

### 4.9 YOLO 大图滑窗裁剪为小图数据集

- `POST /api/v1/preprocess/yolo-sliding-window-crop`

输入为 YOLO 数据集目录（`images/` + `labels/`），输出为新的小图数据集（`images/` + `labels/`），标签会按窗口裁剪并重新归一化。

关键参数：

- `dataset_dir`: YOLO 数据集根目录
- `output_dir`: 输出目录（默认 `dataset_dir/yolo_crops`）
- `window_width/window_height`: 窗口大小
- `stride_x/stride_y`: 滑窗步长
- `keep_empty_labels`: 是否保留无目标窗口
- `min_box_area_ratio`: 目标框与窗口相交面积占原框面积的最小阈值

## 5. 开发命令

```bash
make run      # 启动服务
make test     # 运行测试
make lint     # 基础语法检查
```

## 6. 后续扩展建议

1. 增加异步任务队列（如 Celery/RQ）处理大规模数据集
2. 增加任务状态持久化（数据库 + task id）
3. 增加对象存储（S3/MinIO）输入输出支持
4. 增加鉴权（API Key/JWT）与限流
