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

### 4.4 多层目录叶子数据目录发现

- `POST /api/v1/preprocess/discover-leaf-dirs`

用于递归扫描输入目录，找出直接包含图片或 XML 文件、且其子目录不再包含同类文件的最底层叶子目录。

关键参数：

- `input_dir`: 待扫描根目录
- `recursive`: 是否递归扫描
- `extensions`: 识别为图像的扩展名列表

### 4.5 多层目录数据清洗

- `POST /api/v1/preprocess/clean-nested-dataset`

用于递归处理多层子目录中的原始图片/XML 标注数据，按 **`pairing_mode`** 决定配对范围：

- **`same_directory`**（默认）：图与 XML 为同一目录下的直接文件，按 stem 配对。
- **`images_xmls_subfolders`**：每个「样本」目录下含有子文件夹 `images_dir_name` 与 `xmls_dir_name`（如 `…/车次_1/images/` 与 `…/车次_1/xmls/`），在父目录内跨子文件夹配对。

- 有有效标注的图片复制到 `images/`
- 对应 XML 复制到 `xmls/`
- 无 XML 或 XML 中无有效目标框的图片：默认复制到 `backgrounds/`；若 `include_backgrounds: false` 则不输出 backgrounds，仅保留成对的 `images/xmls`，无标注图数量在响应 `skipped_unlabeled_images` 中统计
- `flatten: false`（默认）时在 `output_dir` 下保留相对 `input_dir` 的子目录结构；`flatten: true` 时合并为 `output_dir/images`、`output_dir/xmls` 等单层目录，文件名带相对路径前缀以防重名

关键参数：

- `input_dir`: 原始多层目录根目录
- `output_dir`: 清洗输出目录（默认 `input_dir/cleaned_dataset`）
- `recursive`: 是否递归扫描（默认 `true`）
- `pairing_mode`: `same_directory` 或 `images_xmls_subfolders`
- `images_dir_name/xmls_dir_name/backgrounds_dir_name`: 输入子目录名（`images_xmls_subfolders` 时）及输出目录名
- `flatten`: 是否扁平化合并到单一输出树
- `include_backgrounds`: 是否整理无标注图到 `backgrounds/`
- `include_difficult`: 是否把 `difficult=1` 视为有效目标
- `copy_files`: `true` 复制，`false` 移动
- `overwrite`: 目标已存在时是否覆盖

### 4.6 多层目录数据集汇总

- `POST /api/v1/preprocess/aggregate-nested-dataset`

用于把多个子目录中已经清洗并转换好的碎片数据集汇总为统一数据集：

- 汇总到 `dataset/images`
- 汇总到 `dataset/labels`
- 汇总到 `dataset/backgrounds`
- 自动规避重名
- 自动合并并重映射 `classes.txt`
- 自动生成 `manifest.json`

关键参数：

- `input_dir`: 已清洗/已转换的多层目录根目录
- `output_dir`: 汇总输出目录（默认 `input_dir/dataset`）
- `images_dir_name/labels_dir_name/backgrounds_dir_name`: 输入/输出目录名配置
- `classes_file_name`: 类别文件名
- `require_non_empty_labels`: 是否跳过空标签文件
- `overwrite`: 目标已存在时是否覆盖

### 4.7 YOLO 数据集划分

- `POST /api/v1/preprocess/split-yolo-dataset`

关键参数：

- `dataset_dir`: YOLO 数据集根目录（含 `images` 与 `labels`）
- `output_dir`: 输出目录（默认 `dataset_dir/split_dataset`）
- `mode`: `train_val_test` / `train_val` / `train_only`
- `train_ratio/val_ratio/test_ratio`: 划分比例（会自动归一化）
- `shuffle/seed`: 是否打乱及随机种子
- `copy_files`: `true` 复制，`false` 移动

### 4.8 目录打包为 ZIP

- `POST /api/v1/preprocess/zip-folder`

关键参数：

- `input_dir`: 待打包目录
- `output_zip_path`: 输出压缩包路径（可选，默认 `input_dir` 同级）
- `include_root_dir`: 压缩包内是否保留根目录名
- `overwrite`: 压缩包已存在时是否覆盖

### 4.9 ZIP 解压

- `POST /api/v1/preprocess/unzip-archive`

关键参数：

- `archive_path`: 压缩包路径（zip）
- `output_dir`: 解压输出目录（可选）
- `overwrite`: 目标文件已存在时是否覆盖

### 4.10 文件或目录移动

- `POST /api/v1/preprocess/move-path`

关键参数：

- `source_path`: 源文件或源目录
- `target_dir`: 目标目录
- `overwrite`: 目标同名已存在时是否覆盖

### 4.11 文件或目录复制

- `POST /api/v1/preprocess/copy-path`

关键参数：

- `source_path`: 源文件或源目录
- `target_dir`: 目标目录
- `overwrite`: 目标同名已存在时是否覆盖

### 4.12 跨机器 SFTP 远程传输

- `POST /api/v1/preprocess/remote-transfer`
- `POST /api/v1/preprocess/remote-transfer/async`

将本地文件或目录通过 SFTP 上传到远程服务器（基于 paramiko）。

关键参数：

- `source_path`: 本地源文件或目录
- `target`: 远程目标，支持 `sftp://host/path`、`sftp://user@host/path`、`user@host:path`
- `username`: SSH 用户名（若 target 中未包含则必填）
- `password` 或 `private_key_path`: 二选一
- `port`: SSH 端口，默认 22
- `overwrite`: 目标已存在时是否覆盖

### 4.13 YOLO 大图正方形滑窗裁剪为小图数据集

- `POST /api/v1/preprocess/yolo-sliding-window-crop`

输入为 YOLO 图像目录和标签目录，输出为新的小图数据集（`images/` + `labels/`）。窗口为正方形（边长=图片高度），仅水平滑动，标签按窗口裁剪并重新归一化。

关键参数：

- `images_dir`: 输入图像目录
- `labels_dir`: 输入 YOLO 标签目录（txt）
- `output_dir`: 输出目录（会创建 `images/` 和 `labels/` 子目录）
- `min_vis_ratio`: 目标在窗口内可见比例阈值，默认 0.5
- `stride_ratio`: 步长占图片高度的比例，默认 0.3
- `ignore_vis_ratio`: 可见比例低于此值视为可忽略，默认 0.05
- `only_wide`: 仅处理宽图（W>H），默认 true

### 4.14 生成 YOLO `data.yaml`

- `POST /api/v1/preprocess/build-yolo-yaml`（及 `/async`）
- `POST /api/v1/preprocess/yolo-train`（及 `/async`，conda 下执行 `yolo train`）

根据数据集根目录与各划分下的 `images` 路径、`classes.txt` 生成 Ultralytics 风格 YAML。可选 **`last_yaml`** 与当前扫描结果合并划分路径（旧路径在前）。**`classes.txt` 可为空或省略**：此时依赖 **`last_yaml`** 中的 **`names`**。远程 `last_yaml` 需 **`sftp_username`** + **`sftp_private_key_path`**。字段约定见 `docs/api_examples.md` 第 0 节与第 15 节。

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
