# self_api - 图像/数据集预处理 API

用于图像与图像数据集预处理的最小可交付 API 服务，当前提供 12 个核心能力：

1. 指定目录图像按滑窗规则裁剪并保存
2. Pascal VOC XML 标注转换为 YOLO 标注
3. YOLO 数据集按 train/val/test 划分
4. 指定目录打包为 zip 压缩包
5. 指定 zip 压缩包解压到目标目录
6. 文件或文件夹整体移动到目标目录
7. 文件或文件夹整体复制到目标目录
8. **跨机器 SFTP 远程传输**（文件/目录上传到远程 SFTP 服务器）
9. YOLO 大图正方形滑窗裁剪为小图数据集（标签同步裁剪，仅宽图水平滑动）
10. 递归发现多层目录中的最底层叶子数据目录
11. 递归清洗多层目录中的图像/XML 数据并归类为 `images` / `xmls`（可选 `backgrounds`；支持扁平化合并与仅输出成对标注）
12. 汇总多个子目录处理结果为统一 `dataset/images`、`dataset/labels`、`dataset/backgrounds`

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
│   │       │   ├── preprocess.py      # 十一个预处理 API
│   │       │   └── system.py          # 健康检查
│   │       └── router.py              # v1 路由聚合
│   ├── core/
│   │   ├── config.py                  # 环境配置
│   │   └── logging.py                 # 日志初始化
│   ├── schemas/
│   │   └── preprocess.py              # 请求/响应模型
│   ├── services/
│   │   ├── file_operations.py         # 压缩/解压/移动/复制服务
│   │   ├── nested_dataset.py          # 多层目录发现/清洗/汇总服务
│   │   ├── split_yolo_dataset.py      # YOLO 数据集划分服务
│   │   ├── xml_to_yolo.py             # VOC XML 转 YOLO 标签服务
│   │   └── yolo_sliding_window.py     # YOLO 大图数据集滑窗裁剪服务
│   ├── utils/
│   │   └── images.py                  # 图像文件扫描工具
│   └── main.py                        # FastAPI 应用入口
├── tests/
│   ├── test_healthz_api.py
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

### 3.1 环境配置（先做）

先复制一份环境变量模板到 `.env` 并按实际环境修改：

```bash
cp .env.example .env
```

至少建议确认这些项：

- `SELF_API_PUBLIC_BASE_URL`（必须改，自己主机的IP）
- `SELF_API_FILE_ACCESS_ROOTS`（必须改，本API客户端可以读写的文件路径）
- `SELF_API_STORAGE_ROOT`（必须改，本API客户端可以读写的文件路径）
- `SELF_API_AUTH_ENABLED`（非必要，鉴权相关字段，不需要鉴权的时候，设置false，下面的账户密码注销掉）
- `N8N_BASE_URL`（必须改，流程编排系统n8n部署的机器）
- `N8N_API_KEY`(非必要，将n8n的api释放给智能体，就可以让智能体给你执行任务了)

### 3.2 本地（不推荐此方式启动）

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
- 启动前请确保已按 `3.1` 完成 `.env` 配置

### 3.3 Docker 启动（推荐：`compose.yaml`）

以下命令均在**仓库根目录**执行。Compose 中服务名为 **`api`**，容器名为 **`self_api`**；默认将容器 **8666** 映射到宿主机 **8666**（与 `make run` 一致）。若要改宿主机映射端口，可在 `.env` 中设置 `SELF_API_PUBLISH_PORT=...` 后再启动。

为避免容器写入挂载目录后出现 root 权限文件，请在 `.env` 中配置当前用户 `UID`/`GID`（例如 Linux 常见 `1000/1000`）。

持久化：命名卷挂载到容器内 **`/app/storage`**；`compose.yaml` 中已通过 `SELF_API_STORAGE_ROOT=./storage` 与卷对齐。**不要**在容器场景下把 `SELF_API_STORAGE_ROOT` 设成仅宿主机存在的绝对路径（除非你自己配置了正确的 `volumes` 绑定）。

#### Compose 启动命令

| 场景 | 命令 |
|------|------|
| 后台启动（含构建镜像） | `docker compose up -d --build` |
| 前台启动（终端里直接看日志，Ctrl+C 停止） | `docker compose up --build` |
| 仅构建镜像（不启动） | `docker compose build` |
| 不使用缓存强制重建 | `docker compose build --no-cache` |

#### 不使用 Compose（仅 `docker`）启动命令

| 场景 | 命令 |
|------|------|
| 构建镜像 | `docker build -t self_api:latest .` |
| 前台运行（退出后删除容器） | `docker run --rm -p 8666:8666 --env-file .env self_api:latest` |
| 后台运行并命名容器 | `docker run -d --name self_api -p 8666:8666 --env-file .env self_api:latest` |

### 3.4 Docker 维护命令

#### Compose 维护命令

| 目的 | 命令 |
|------|------|
| 停止并删除容器与网络（**保留**命名卷 `self_api_storage`） | `docker compose down` |
| 停止并删除容器、网络，并**删除**命名卷（会清空容器内持久化存储，慎用） | `docker compose down -v` |
| 查看运行状态 | `docker compose ps` |
| 查看配置合并结果（排查端口、环境变量） | `docker compose config` |
| 查看日志（一次性） | `docker compose logs` |
| 持续跟随日志 | `docker compose logs -f` |
| 仅看最近若干行 | `docker compose logs --tail=100 -f` |
| 重启服务 | `docker compose restart` |
| 进入容器 Shell（调试） | `docker compose exec api sh` |
| 在运行中的容器内执行单条命令 | `docker compose exec api python -c "import app; print('ok')"` |

#### 纯 `docker` 维护命令

| 目的 | 命令 |
|------|------|
| 停止容器 | `docker stop self_api` |
| 删除已停止的容器 | `docker rm self_api` |
| 查看日志 | `docker logs self_api` / `docker logs -f self_api` |
| 进入容器 | `docker exec -it self_api sh` |

#### 镜像、卷与清理（按需）

| 目的 | 命令 |
|------|------|
| 列出本机镜像 | `docker images` |
| 删除指定镜像 | `docker rmi self_api:latest`（需先无容器占用） |
| 列出卷 | `docker volume ls` |
| 查看命名卷详情 | `docker volume inspect self_api_self_api_storage`（名称以 `docker volume ls` 为准） |
| 清理悬空镜像与未使用网络等 | `docker system prune` |
| 一并清理未使用的镜像与构建缓存（更激进，慎用） | `docker system prune -a` |

## 4. API 要素与接口定义

接口示例请求已拆分到独立文档：`docs/api_examples.md`。

### 4.1 健康检查

- `GET /api/v1/healthz`

### 4.2 常用 SOP（按最常用 workflow）

文档与接口说明以下面两条常用工作流为主线：

- 小图 `images+xmls` 基线训练：`xml-to-yolo -> split-yolo-dataset -> yolo-augment（按需） -> build-yolo-yaml -> zip/move/unzip 或 remote-transfer/remote-unzip -> yolo-train / remote-slurm-yolo-train`
- 大图 `images+xmls` 常发迭代：`clean-nested-dataset（按需） -> xml-to-yolo -> reset-yolo-label-index（单类时） -> yolo-sliding-window-crop -> split-yolo-dataset 或直接 build-yolo-yaml -> zip/remote-transfer/remote-unzip -> yolo-train / remote-slurm-yolo-train`
- 多层目录整理：`discover-leaf-dirs -> clean-nested-dataset -> xml-to-yolo -> aggregate-nested-dataset`

### 4.3 多层目录叶子数据目录发现

- `POST /api/v1/preprocess/discover-leaf-dirs`

用于先确认原始数据最底层的有效目录，适合作为多层目录清洗前的第一步。

### 4.4 多层目录数据清洗

- `POST /api/v1/preprocess/clean-nested-dataset`

把原始图片/XML 规整成 `images` / `xmls`（及可选 `backgrounds`）；默认会自动识别常见目录布局，支持保留层级和扁平化输出，是大图迭代与多层目录整理链路的常见起点。

### 4.5 多层目录数据集汇总

- `POST /api/v1/preprocess/aggregate-nested-dataset`

用于把多个碎片目录汇总为统一 `dataset/images`、`dataset/labels`、`dataset/backgrounds`，适合多批次数据合并。

### 4.6 VOC XML 转 YOLO 标签

- `POST /api/v1/preprocess/xml-to-yolo`

要求输入目录下有 `images/` 与 `xmls/`，服务会把 YOLO 标签写到 `labels/`（目录名可配置）。这是两条训练工作流都会经过的标准步骤。

### 4.7 标注检查与类别整理

- `POST /api/v1/preprocess/annotate-visualize`
- `POST /api/v1/preprocess/reset-yolo-label-index`

前者用于把 YOLO/VOC 标注可视化做抽检；后者用于单类任务把所有 YOLO label 类别索引统一置为 `0`。
`annotate-visualize` 现在按标准目录工作：传 `input_dir` 与 `output_dir` 即可，默认优先读取 `input_dir/labels`，若不存在则回退到 `input_dir/xmls`。
`reset-yolo-label-index` 也已固定按 `input_dir/labels` 工作，最小调用只需 `input_dir`。

### 4.8 YOLO 数据集划分

- `POST /api/v1/preprocess/split-yolo-dataset`

把标准 YOLO 数据集划分为 `train/val/test`，供后续增强、滑窗或直接训练使用。只认 `input_dir` / `output_dir`，固定读取 `input_dir/images` 与 `input_dir/labels`；最小调用仅需 `input_dir`（`output_dir` 缺省为 `<input_dir>/split_dataset`）。

### 4.9 数据增强与滑窗裁剪

- `POST /api/v1/preprocess/yolo-augment`
- `POST /api/v1/preprocess/yolo-sliding-window-crop`

`yolo-augment` 适合小图 baseline 的增强；`yolo-sliding-window-crop` 适合大图转小图训练：传 `input_dir`（含 `images/`），若 `labels/` 存在则自动同步输出裁剪后的 YOLO 标注。

### 4.10 生成训练 YAML

- `POST /api/v1/preprocess/build-yolo-yaml`（及 `/async`）

根据数据集根目录与各划分下的 `images` 路径、`classes.txt` 生成 Ultralytics 风格 YAML；可选与上一版 `last_yaml` 合并路径。

### 4.11 数据投放与远程准备

- `POST /api/v1/preprocess/zip-folder`
- `POST /api/v1/preprocess/unzip-archive`
- `POST /api/v1/preprocess/move-path`
- `POST /api/v1/preprocess/copy-path`
- `POST /api/v1/preprocess/remote-transfer`（及 `/async`）
- `POST /api/v1/preprocess/remote-unzip`（及 `/async`）

这组接口主要用于把本地整理好的数据集放到训练目录，或者上传到远端机器后解压到目标位置。

### 4.12 发起训练

- `POST /api/v1/preprocess/yolo-train`（及 `/async`，conda 下执行 `yolo train`）
- `POST /api/v1/preprocess/remote-slurm-yolo-train`（及 `/async`）

前者用于本地训练，后者用于远端 SLURM 集群训练。

### 4.13 少用但独立的 VOC 裁剪修订链路

- `POST /api/v1/preprocess/voc-bar-crop`
- `POST /api/v1/preprocess/restore-voc-crops-batch`

用于 VOC 大图局部裁剪编辑后再回贴原图，不属于当前最常用的两条训练 SOP，但仍保留在服务中。

## 5. 一期最小生产版

当前仓库已落地一期最小生产版，用于支持 LangGraph 的跨机器接入。

关键约定：

- 长任务优先调用 `/async` 接口
- `SELF_API_PUBLIC_BASE_URL` 用于生成对外可访问的 `status_url`
- 生产环境必须显式配置 `SELF_API_FILE_ACCESS_ROOTS`
- 当前跨机器模式依赖共享文件系统，不直接解决任务恢复

相关文档：

- `docs/langgraph_min_prod.md`
- `docs/architecture/langgraph_production_roadmap.md`
- `docs/architecture/self_orchestrator_bootstrap.md`

## 6. 开发命令

```bash
make run      # 启动服务
make test     # 运行测试
make lint     # 基础语法检查
```

## 7. 后续扩展建议

1. 增加异步任务队列（如 Celery/RQ）处理大规模数据集
2. 增加任务状态持久化（数据库 + task id）
3. 增加对象存储（S3/MinIO）输入输出支持
4. 增加鉴权（API Key/JWT）与限流
