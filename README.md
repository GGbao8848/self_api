# self_api - 图像/数据集预处理 API

用于图像与图像数据集预处理的最小可交付 API 服务，当前提供 2 个核心能力：

1. 指定目录图像按滑窗规则裁剪并保存
2. 指定目录图像去重（`md5` 精确去重 / `phash` 感知去重）

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
│   │       │   ├── preprocess.py      # 两个预处理 API
│   │       │   └── system.py          # 健康检查
│   │       └── router.py              # v1 路由聚合
│   ├── core/
│   │   ├── config.py                  # 环境配置
│   │   └── logging.py                 # 日志初始化
│   ├── schemas/
│   │   └── preprocess.py              # 请求/响应模型
│   ├── services/
│   │   ├── deduplicate.py             # 图像去重服务
│   │   └── sliding_window.py          # 滑窗裁剪服务
│   ├── utils/
│   │   └── images.py                  # 图像文件扫描工具
│   └── main.py                        # FastAPI 应用入口
├── tests/
│   └── test_api.py
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
pip install -e .[dev]
make run
```

打开文档：`http://127.0.0.1:8000/docs`

### 3.2 Docker

```bash
docker build -t self-api:0.1.0 .
docker run --rm -p 8000:8000 self-api:0.1.0
```

## 4. API 要素与接口定义

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

示例请求：

```json
{
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
}
```

### 4.3 图像去重

- `POST /api/v1/preprocess/deduplicate`

关键参数：

- `method`: `md5`（精确）或 `phash`（感知）
- `distance_threshold`: `phash` 汉明距离阈值（越大越宽松）
- `copy_unique_to`: 可选，将唯一图像导出到新目录
- `report_path`: 可选，输出 JSON 报告

示例请求：

```json
{
  "input_dir": "./data/raw",
  "recursive": true,
  "method": "phash",
  "distance_threshold": 5,
  "hash_size": 8,
  "copy_unique_to": "./data/unique",
  "report_path": "./data/reports/dedup.json"
}
```

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
