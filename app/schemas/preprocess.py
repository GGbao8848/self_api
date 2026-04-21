from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from app.schemas.artifacts import ArtifactSummary


_TRAIN_BUCKET_SUFFIXES = {
    "runs/detect",
    "runs/segment",
    "runs/classify",
    "runs/pose",
    "runs/obb",
}


def _path_part_for_validation(value: str) -> str:
    text = value.strip()
    if text.startswith(("sftp://", "ssh://")):
        return urlparse(text).path or ""
    if ":" in text and not text.startswith("/"):
        return text.split(":", 1)[1]
    return text


def _yaml_stem_for_validation(yaml_path: str) -> str:
    return PurePosixPath(_path_part_for_validation(yaml_path.strip())).stem


def _yaml_structure_for_validation(yaml_path: str) -> tuple[str, str, str]:
    path = PurePosixPath(_path_part_for_validation(yaml_path.strip()))
    parts = path.parts
    try:
        datasets_idx = parts.index("datasets")
    except ValueError as exc:
        raise ValueError("yaml_path must include '/<detector_name>/datasets/<dataset_version>/<dataset_version>.yaml'") from exc
    if datasets_idx < 2:
        raise ValueError("yaml_path must contain both root_dir and detector_name before '/datasets/'")
    if len(parts) <= datasets_idx + 2:
        raise ValueError("yaml_path must include dataset version folder and yaml filename under '/datasets/'")

    root_dir = str(PurePosixPath(*parts[: datasets_idx - 1]))
    detector_name = parts[datasets_idx - 1]
    dataset_version = parts[datasets_idx + 1]
    yaml_stem = path.stem

    if dataset_version != yaml_stem:
        raise ValueError(
            f"dataset version folder must equal yaml filename stem: expected {yaml_stem!r}, got {dataset_version!r}"
        )
    return root_dir, detector_name, dataset_version


def _normalize_project_for_validation(project: str) -> str:
    return project.replace("\\", "/").rstrip("/")


def _validate_train_project_and_name(*, yaml_path: str, project: str, name: str) -> None:
    yaml_stem = _yaml_stem_for_validation(yaml_path)
    if not yaml_stem:
        raise ValueError("yaml_path must point to a yaml file with a valid stem")
    if name != yaml_stem:
        raise ValueError(f"name must equal yaml filename stem: expected {yaml_stem!r}, got {name!r}")

    normalized_project = _normalize_project_for_validation(project)
    if not any(normalized_project.endswith(suffix) for suffix in _TRAIN_BUCKET_SUFFIXES):
        allowed = ", ".join(sorted(_TRAIN_BUCKET_SUFFIXES))
        raise ValueError(f"project must end with one of the training buckets: {allowed}")

    root_dir, detector_name, _ = _yaml_structure_for_validation(yaml_path)
    expected_prefix = _normalize_project_for_validation(str(PurePosixPath(root_dir) / detector_name / "runs"))
    if not normalized_project.startswith(expected_prefix + "/"):
        raise ValueError(
            "project must share the same <root_dir>/<detector_name> prefix as yaml_path and live under '/runs/'"
        )


class AsyncTaskSubmitResponse(BaseModel):
    status: str = "accepted"
    task_id: str
    task_type: str
    status_url: str
    callback_url: str | None = None


class AsyncTaskCallbackEvent(BaseModel):
    state: Literal["pending", "running", "succeeded", "failed"]
    attempted_at: str
    callback_url: str
    status_code: int | None = None
    method: Literal["POST", "GET"] = "POST"
    success: bool
    error: str | None = None


class AsyncTaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    created_at: str
    updated_at: str
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancellation_requested: bool = False
    callback_url: str | None = None
    callback_state: Literal["pending", "running", "succeeded", "failed"] = "succeeded"
    callback_sent_at: str | None = None
    callback_status_code: int | None = None
    callback_error: str | None = None
    callback_events: list[AsyncTaskCallbackEvent] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    queue_position: int | None = None


class AnnotateVisualizeRequest(BaseModel):
    """在 input_dir/images 上绘制标注框，默认优先使用 input_dir/labels。"""

    input_dir: str = Field(description="数据集根目录，要求包含 images/，并存在 labels/ 或 xmls/")
    output_dir: str = Field(description="可视化结果输出目录（保持与 input_dir/images 相同的相对路径结构）")
    recursive: bool = Field(default=True, description="是否递归扫描图像")
    extensions: list[str] | None = Field(
        default=None,
        description="允许的图像扩展名，默认常见图片格式",
    )
    include_difficult: bool = Field(
        default=False,
        description="XML 模式：是否绘制 difficult=1 的目标",
    )
    line_width: int = Field(default=2, ge=1, le=20, description="框线宽度（像素）")
    overwrite: bool = Field(default=True, description="是否覆盖已存在的输出图像")
    classes: list[str] | None = Field(
        default=None,
        description=(
            "YOLO 模式：按类别 id 顺序的类别名列表；与 classes_file 二选一。"
            "若与 classes_file 均未提供或 classes_file 为空，则框上文字显示类别 id（数字索引）。"
        ),
    )
    classes_file: str | None = Field(
        default=None,
        description=(
            "YOLO 模式：classes.txt 路径（每行一个类别名）；与 classes 二选一。"
            "可省略或传空字符串，此时框上文字显示类别 id（数字索引）。"
        ),
    )

    @model_validator(mode="after")
    def _normalize_optional_fields(self) -> "AnnotateVisualizeRequest":
        classes_inline = self.classes
        classes_path = (self.classes_file or "").strip()
        if classes_inline and classes_path:
            raise ValueError("classes 与 classes_file 只能填其一")
        return self.model_copy(
            update={
                "classes_file": classes_path or None,
            }
        )


class AnnotateVisualizeDetail(BaseModel):
    source_image: str
    output_image: str | None = None
    boxes_drawn: int = 0
    skipped_reason: str | None = None


class AnnotateVisualizeResponse(BaseModel):
    status: str = "ok"
    mode: Literal["yolo", "xml"]
    input_dir: str
    annotation_dir: str
    output_dir: str
    total_images: int
    written_images: int
    skipped_images: int
    details: list[AnnotateVisualizeDetail]


class AnnotateVisualizeAsyncRequest(AnnotateVisualizeRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class DiscoverXmlClassesRequest(BaseModel):
    input_dir: str = Field(description="数据集根目录，需含 xmls_dir_name 子目录")
    xmls_dir_name: str = Field(default="xmls", description="XML 子目录名")
    recursive: bool = Field(default=True, description="是否递归扫描")
    include_difficult: bool = Field(default=False, description="是否统计 difficult=1 的标注")


class DiscoverXmlClassesResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    xmls_dir: str
    total_xml_files: int
    parse_errors: int
    total_classes: int
    class_names: list[str]
    class_counts: dict[str, int]


class XmlToYoloRequest(BaseModel):
    input_dir: str = Field(
        description="Dataset root directory containing images and xmls folders",
    )
    images_dir_name: str = Field(default="images", description="Image folder name")
    xmls_dir_name: str = Field(default="xmls", description="Pascal VOC XML folder name")
    labels_dir_name: str = Field(default="labels", description="YOLO labels folder name")
    recursive: bool = Field(default=True, description="Search xml files recursively")
    classes: list[str] | None = Field(
        default=None,
        description=(
            "Optional fixed class list defining index order (index i = classes[i]). "
            "Cannot be combined with class_index_map."
        ),
    )
    class_name_map: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional many-to-one rename map applied before class indexing. "
            "Key = original XML label name, value = target name. "
            "Example: {\"louyou1\": \"louyou\", \"louyou2\": \"louyou\"} merges all variants into one class."
        ),
    )
    class_index_map: dict[str, int] | None = Field(
        default=None,
        description=(
            "After class_name_map, map each logical class name to a YOLO class id (non-negative int). "
            "Values must be a contiguous range 0..N. Multiple names may share one id (merge). "
            "Mutually exclusive with `classes`. Use `training_names` for names written to classes.txt / data.yaml."
        ),
    )
    training_names: list[str] | None = Field(
        default=None,
        description=(
            "Names for YOLO indices 0..N (one line per id in classes.txt and Ultralytics `names`). "
            "With class_index_map: length must be max(id)+1; overrides auto-picked names. "
            "With `classes` (and no class_index_map): same length as classes to rename for YAML only; "
            "class_to_id keys stay the logical names from `classes`."
        ),
    )
    include_difficult: bool = Field(
        default=False,
        description="Whether to include objects marked difficult=1 in xml",
    )
    write_classes_file: bool = Field(
        default=True,
        description="Write classes file under dataset root",
    )
    classes_file_name: str = Field(
        default="classes.txt",
        description="Class index file name written under dataset root",
    )
    overwrite: bool = Field(
        default=True,
        description="Whether to overwrite existing label files",
    )

    @model_validator(mode="after")
    def _xml_to_yolo_class_options(self) -> "XmlToYoloRequest":
        if self.class_index_map is not None and self.classes is not None:
            raise ValueError("classes 与 class_index_map 不能同时指定，请只选一种索引方式")
        tn = self.training_names
        if tn is not None:
            if self.class_index_map is None and self.classes is None:
                raise ValueError("training_names 须与 classes 或 class_index_map 同时使用")
        if tn is not None and self.classes is not None and self.class_index_map is None:
            if len(tn) != len(self.classes):
                raise ValueError("training_names 与 classes 长度必须一致（按索引一一对应显示名）")
        return self


class XmlToYoloFileDetail(BaseModel):
    source_xml: str
    output_label: str | None = None
    written_lines: int = 0
    skipped_reason: str | None = None


class XmlToYoloResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    labels_dir: str
    total_xml_files: int
    converted_files: int
    skipped_files: int
    total_boxes: int
    classes: list[str]
    class_to_id: dict[str, int]
    classes_file: str | None
    details: list[XmlToYoloFileDetail]


class SplitYoloDatasetRequest(BaseModel):
    """在 input_dir/images 与 input_dir/labels 上做 train/val/test 划分。"""

    input_dir: str = Field(
        description="数据集根目录，要求包含 images/ 与 labels/",
    )
    output_dir: str | None = Field(
        default=None,
        description="输出目录；缺省时为 <input_dir>/split_dataset",
    )
    recursive: bool = Field(default=True, description="Search image files recursively")
    extensions: list[str] | None = Field(
        default=None,
        description="Allowed image extensions, e.g. ['.jpg', '.png']",
    )
    mode: Literal["train_val_test", "train_val", "train_only"] = Field(
        default="train_val_test",
        description="Split mode",
    )
    train_ratio: float = Field(default=0.8, gt=0.0, le=1.0, description="Train ratio")
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Val ratio")
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Test ratio")
    shuffle: bool = Field(default=True, description="Shuffle samples before split")
    seed: int = Field(default=42, description="Random seed used when shuffle=True")
    copy_files: bool = Field(default=True, description="Copy files instead of moving")
    keep_subdirs: bool = Field(
        default=True,
        description="Keep source folder structure under split folders",
    )
    output_layout: Literal["images_first", "split_first"] = Field(
        default="split_first",
        description=(
            "Output directory layout: images_first => images/<split>, labels/<split>; "
            "split_first => <split>/images, <split>/labels"
        ),
    )
    require_label: bool = Field(
        default=True,
        description="Skip images when paired label txt file is missing",
    )
    ignore_existing_split_dirs: bool = Field(
        default=True,
        description="Ignore files already under images/{train,val,test}",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing target files",
    )


class SplitYoloFileDetail(BaseModel):
    source_image: str
    source_label: str | None = None
    split: str | None = None
    target_image: str | None = None
    target_label: str | None = None
    skipped_reason: str | None = None


class SplitYoloDatasetResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    output_dir: str
    mode: str
    total_images: int
    paired_images: int
    skipped_images: int
    train_images: int
    val_images: int
    test_images: int
    copied_classes_file: str | None
    details: list[SplitYoloFileDetail]


class ZipFolderRequest(BaseModel):
    input_dir: str = Field(description="Input folder path to package")
    output_zip_path: str | None = Field(
        default=None,
        description="Output zip path; defaults to <input_dir_parent>/<input_dir_name>.zip",
    )
    include_root_dir: bool = Field(
        default=True,
        description="Whether to keep root directory name inside zip archive",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing output zip file",
    )


class ZipFolderResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    output_zip_path: str
    packed_files: int
    total_bytes: int


class UnzipArchiveRequest(BaseModel):
    archive_path: str = Field(
        description="Zip archive path to extract",
    )
    output_dir: str | None = Field(
        default=None,
        description="Output directory; defaults to <archive_parent>/<archive_stem>",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing extracted files",
    )


class UnzipArchiveResponse(BaseModel):
    status: str = "ok"
    archive_path: str
    output_dir: str
    extracted_files: int
    skipped_files: int


class MovePathRequest(BaseModel):
    source_path: str = Field(description="Source file or directory path")
    target_dir: str = Field(description="Target directory path")
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite target when name conflicts",
    )


class MovePathResponse(BaseModel):
    status: str = "ok"
    source_path: str
    target_path: str
    moved_type: Literal["file", "directory"]


class CopyPathRequest(BaseModel):
    source_path: str = Field(description="Source file or directory path")
    target_dir: str = Field(description="Target directory path")
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite target when name conflicts",
    )


class CopyPathResponse(BaseModel):
    status: str = "ok"
    source_path: str
    target_path: str
    copied_type: Literal["file", "directory"]


class RemoteTransferRequest(BaseModel):
    """跨机器 SFTP 传输：将本地文件/目录上传到远程 SFTP 服务器。"""

    source_path: str = Field(description="本地源文件或目录路径")
    target: str = Field(
        description="远程目标，支持 sftp://host/path 或 sftp://user@host/path 或 user@host:path",
    )
    overwrite: bool = Field(
        default=False,
        description="目标已存在时是否覆盖",
    )
    username: str | None = Field(
        default=None,
        description="SSH 用户名，若 target 中未包含则必填",
    )
    password: str | None = Field(
        default=None,
        description="SSH 密码（与 private_key_path 二选一）",
    )
    private_key_path: str | None = Field(
        default=None,
        description="SSH 私钥路径（与 password 二选一）",
    )
    port: int = Field(
        default=22,
        ge=1,
        le=65535,
        description="SSH 端口",
    )


class RemoteTransferResponse(BaseModel):
    status: str = "ok"
    source_path: str
    target: str
    target_host: str
    target_port: int
    target_path: str
    transferred_type: Literal["file", "directory"]
    transferred_files: int
    total_bytes: int


class RemoteUnzipRequest(BaseModel):
    """跨机器远程解压：在远端主机执行 unzip。"""

    archive_path: str = Field(
        description="远程压缩包路径，支持 sftp://host/path 或 sftp://user@host/path 或 user@host:path",
    )
    output_dir: str | None = Field(
        default=None,
        description="远程输出目录；默认 <archive_parent>/<archive_stem>",
    )
    overwrite: bool = Field(
        default=False,
        description="是否覆盖已存在文件（true 对应 unzip -o）",
    )
    username: str | None = Field(
        default=None,
        description="SSH 用户名，若路径中未包含则必填",
    )
    password: str | None = Field(
        default=None,
        description="SSH 密码（与 private_key_path 二选一）",
    )
    private_key_path: str | None = Field(
        default=None,
        description="SSH 私钥路径（与 password 二选一）",
    )
    port: int = Field(
        default=22,
        ge=1,
        le=65535,
        description="SSH 端口",
    )


class RemoteUnzipResponse(BaseModel):
    status: str = "ok"
    archive_path: str
    output_dir: str
    target_host: str
    target_port: int
    command: str


class RemoteSbatchYoloTrainRequest(BaseModel):
    """跨机器远程训练：通过 SSH 在远端执行 sbatch 提交 YOLO 训练任务。"""

    host: str | None = Field(
        default=None,
        description="远端主机；当 yaml_path / project_root_dir 直接写绝对路径时必填",
    )
    yaml_path: str = Field(
        description="远端数据 yaml 路径；支持绝对路径、sftp://host/path、sftp://user@host/path 或 user@host:path",
    )
    project_root_dir: str = Field(
        description="远端工作目录；支持绝对路径、sftp://host/path、sftp://user@host/path 或 user@host:path",
    )
    project: str = Field(
        ...,
        description="显式指定的 YOLO project 输出目录；API 不再内部推导",
    )
    name: str = Field(
        ...,
        description="显式指定的 YOLO run 名称；API 不再内部推导",
    )
    yolo_train_env: str = Field(
        ...,
        description="远端 conda 环境名（例如 yolo_pose）",
    )
    model: str = Field(default="yolo11s.pt", description="YOLO model")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    imgsz: int = Field(default=640, ge=1, description="Training image size")
    batch: int | None = Field(default=None, ge=1, description="YOLO batch size")
    workers: int | None = Field(default=None, ge=0, description="DataLoader workers")
    cache: bool | None = Field(default=True, description="YOLO cache 参数")
    device: str | None = Field(default=None, description="显式设备，例如 0 或 0,1")
    partition: str | None = Field(default="gpu", description="SLURM 分区")
    nodelist: str | None = Field(default=None, description="可选：要求使用的节点列表（逗号分隔）")
    exclude: str | None = Field(default=None, description="可选：排除的节点列表（逗号分隔）")
    job_name: str = Field(default="self_api_train", description="sbatch 任务名")
    stdout_path: str | None = Field(
        default=None,
        description="可选：远端 stdout 日志路径；默认 <project_root_dir>/logs/slurm-%j.out",
    )
    stderr_path: str | None = Field(
        default=None,
        description="可选：远端 stderr 日志路径；默认 <project_root_dir>/logs/slurm-%j.err",
    )
    username: str | None = Field(
        default=None,
        description="SSH 用户名；若路径中未包含则必填",
    )
    password: str | None = Field(
        default=None,
        description="SSH 密码（与 private_key_path 二选一）",
    )
    private_key_path: str | None = Field(
        default=None,
        description="SSH 私钥路径（与 password 二选一）",
    )
    port: int = Field(
        default=22,
        ge=1,
        le=65535,
        description="SSH 端口",
    )

    @model_validator(mode="after")
    def _validate_project_and_name(self) -> "RemoteSbatchYoloTrainRequest":
        _validate_train_project_and_name(
            yaml_path=self.yaml_path,
            project=self.project,
            name=self.name,
        )
        return self


class RemoteSbatchYoloTrainResponse(BaseModel):
    status: str = "ok"
    yaml_path: str
    project_root_dir: str
    target_host: str
    target_port: int
    project: str
    name: str
    command: str
    job_id: str
    stdout_path: str
    stderr_path: str
    stdout: str
    stderr: str


class YoloSlidingWindowCropRequest(BaseModel):
    """YOLO 滑窗裁剪：支持 input_dir/images 或递归发现 nested */images，并保留相对目录结构输出。"""

    input_dir: str = Field(
        description=(
            "输入数据集根目录。支持直接包含 images/（可选 labels/），"
            "也支持递归发现 nested */images（同级若有 labels/ 则同步输出裁剪标注）"
        ),
    )
    output_dir: str = Field(
        description=(
            "输出目录。会在输出端保留输入相对层级，例如 input_dir/train/images -> output_dir/train/images；"
            "有 labels 时同步输出到对应的 output_dir/.../labels"
        ),
    )
    window_width: int | None = Field(
        default=None,
        ge=1,
        description="Optional crop window width; defaults to image height when omitted",
    )
    window_height: int | None = Field(
        default=None,
        ge=1,
        description="Optional crop window height; defaults to image height when omitted",
    )
    stride_x: int | None = Field(
        default=None,
        ge=1,
        description="Optional horizontal stride; defaults to round(stride_ratio * image height)",
    )
    stride_y: int | None = Field(
        default=None,
        ge=1,
        description="Optional vertical stride; defaults to window height",
    )
    min_vis_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum visible area ratio (target in crop) to keep; default 0.5",
    )
    stride_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Stride as ratio of image height (H); default 0.3",
    )
    ignore_vis_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Visible ratio <= this value is ignored; default 0.05",
    )
    only_wide: bool = Field(
        default=True,
        description="Only process wide images (W > H); skip square/tall images",
    )


class DiscoverLeafDirsRequest(BaseModel):
    input_dir: str = Field(description="Root directory to scan for leaf data folders")
    recursive: bool = Field(default=True, description="Scan nested directories recursively")
    extensions: list[str] | None = Field(
        default=None,
        description="Allowed image extensions, e.g. ['.jpg', '.png']",
    )


class DiscoverLeafDirsResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    total_leaf_dirs: int
    leaf_dirs: list[str]


class CleanNestedDatasetRequest(BaseModel):
    input_dir: str = Field(description="Root directory containing nested raw image/xml folders")
    output_dir: str | None = Field(
        default=None,
        description="Output directory; defaults to <input_dir>/cleaned_dataset",
    )
    recursive: bool = Field(default=True, description="Scan nested directories recursively")
    extensions: list[str] | None = Field(
        default=None,
        description="Allowed image extensions, e.g. ['.jpg', '.png']",
    )
    images_dir_name: str = Field(default="images", description="Output folder name for labeled images")
    xmls_dir_name: str = Field(default="xmls", description="Output folder name for xml annotations")
    backgrounds_dir_name: str = Field(
        default="backgrounds",
        description="Output folder name for background images",
    )
    include_difficult: bool = Field(
        default=False,
        description="Whether difficult objects count as valid annotations",
    )
    copy_files: bool = Field(default=True, description="Copy files instead of moving")
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing target files",
    )
    flatten: bool = Field(
        default=False,
        description=(
            "When true, merge all leaf outputs into a single tree under output_dir: "
            "output_dir/<images_dir_name>, output_dir/<xmls_dir_name>, etc., with basenames "
            "prefixed by relative path to avoid collisions"
        ),
    )
    include_backgrounds: bool = Field(
        default=True,
        description=(
            "When false, do not copy unlabeled images to backgrounds; only images/xmls pairs "
            "are written (plus orphan-xml handling when not copy_files)"
        ),
    )
    pairing_mode: Literal["auto", "same_directory", "images_xmls_subfolders"] = Field(
        default="auto",
        description=(
            "auto: detect the layout automatically. "
            "same_directory: each leaf folder contains image and xml files as direct siblings. "
            "images_xmls_subfolders: each unit folder contains subfolders named images_dir_name "
            "and xmls_dir_name with files inside (VOC-style layout)"
        ),
    )


class CleanNestedDatasetLeafDetail(BaseModel):
    source_dir: str
    output_dir: str
    total_images: int
    labeled_images: int
    background_images: int
    skipped_unlabeled_images: int = Field(
        default=0,
        description="Unlabeled images not copied because include_backgrounds is false",
    )
    copied_xml_files: int
    empty_or_invalid_xml_files: int
    orphan_xml_files: int


class CleanNestedDatasetResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    output_dir: str
    discovered_leaf_dirs: int
    processed_leaf_dirs: int
    total_images: int
    labeled_images: int
    background_images: int
    skipped_unlabeled_images: int = Field(
        default=0,
        description="Unlabeled images not copied because include_backgrounds is false",
    )
    copied_xml_files: int
    empty_or_invalid_xml_files: int
    orphan_xml_files: int
    details: list[CleanNestedDatasetLeafDetail]


class AggregateNestedDatasetRequest(BaseModel):
    input_dir: str = Field(description="Root directory containing cleaned per-leaf datasets")
    output_dir: str | None = Field(
        default=None,
        description="Output directory; defaults to <input_dir>/dataset",
    )
    images_dir_name: str = Field(default="images", description="Input/output image folder name")
    labels_dir_name: str = Field(default="labels", description="Input/output label folder name")
    backgrounds_dir_name: str = Field(
        default="backgrounds",
        description="Input/output background folder name",
    )
    classes_file_name: str = Field(
        default="classes.txt",
        description="Class index file name",
    )
    recursive: bool = Field(default=True, description="Scan nested directories recursively")
    extensions: list[str] | None = Field(
        default=None,
        description="Allowed image extensions, e.g. ['.jpg', '.png']",
    )
    require_non_empty_labels: bool = Field(
        default=True,
        description="Skip image when paired label file is empty",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing aggregated target files",
    )


class AggregateNestedDatasetItemDetail(BaseModel):
    source_path: str
    target_path: str | None = None
    item_type: Literal["image", "label", "background"]
    skipped_reason: str | None = None


class AggregateNestedDatasetResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    output_dir: str
    fragment_dirs: int
    aggregated_images: int
    aggregated_backgrounds: int
    skipped_images: int
    classes: list[str]
    class_to_id: dict[str, int]
    classes_file: str | None
    manifest_path: str
    details: list[AggregateNestedDatasetItemDetail]


class DiscoverLeafDirsAsyncRequest(DiscoverLeafDirsRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class CleanNestedDatasetAsyncRequest(CleanNestedDatasetRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class AggregateNestedDatasetAsyncRequest(AggregateNestedDatasetRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class XmlToYoloAsyncRequest(XmlToYoloRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class SplitYoloDatasetAsyncRequest(SplitYoloDatasetRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class ZipFolderAsyncRequest(ZipFolderRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class UnzipArchiveAsyncRequest(UnzipArchiveRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class MovePathAsyncRequest(MovePathRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class CopyPathAsyncRequest(CopyPathRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class RemoteTransferAsyncRequest(RemoteTransferRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class RemoteUnzipAsyncRequest(RemoteUnzipRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class RemoteSbatchYoloTrainAsyncRequest(RemoteSbatchYoloTrainRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class YoloSlidingWindowCropAsyncRequest(YoloSlidingWindowCropRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class YoloSlidingWindowCropDetail(BaseModel):
    source_image: str
    source_label: str | None = None
    crop_count: int = 0
    label_count: int = 0
    skipped_reason: str | None = None


class YoloSlidingWindowCropResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    labels_dir: str | None
    output_dir: str
    input_images: int
    processed_images: int
    skipped_images: int
    generated_crops: int
    generated_labels: int
    details: list[YoloSlidingWindowCropDetail]


class BuildYoloYamlRequest(BaseModel):
    input_dir: str = Field(
        ...,
        description=(
            "Dataset root to scan. Auto-resolves common layouts: "
            "<input_dir>/dataset (cropped output), <input_dir>/yolo_split (split output), "
            "or <input_dir> itself. Supports train/images (split_first) or images/train (images_first)."
        ),
    )
    classes_file: str | None = Field(
        default=None,
        description=(
            "Path to classes.txt. Defaults to search under dataset root / input_dir / yolo_split. "
            "May be empty: then last_yaml is required and nc/names are taken from last_yaml."
        ),
    )
    split_names: list[str] | None = Field(
        default=None,
        description="Split folder names to check in order (default: train, val, test)",
    )
    images_subdir_name: str = Field(
        default="images",
        description="Subfolder under each split that holds images (YAML paths are <split>/<this>)",
    )
    path_prefix_replace_from: str | None = Field(
        default=None,
        description="If set with path_prefix_replace_to, replace this prefix on each split's absolute images path in YAML",
    )
    path_prefix_replace_to: str | None = Field(
        default=None,
        description="Replacement prefix for train/val/test absolute paths (e.g. project root + detector)",
    )
    output_yaml_path: str | None = Field(
        default=None,
        description=(
            "Full path for the generated data.yaml. Optional when project_root_dir and detector_name "
            "are provided together; then API auto-publishes dataset and writes "
            "<project_root_dir>/<detector_name>/datasets/<dataset_version>/<dataset_version>.yaml"
        ),
    )
    project_root_dir: str | None = Field(
        default=None,
        description=(
            "Optional project workspace root. When set together with detector_name, API publishes "
            "the dataset into the standard project dataset area and auto-generates dataset version naming"
        ),
    )
    detector_name: str | None = Field(
        default=None,
        description=(
            "Stable detector family name used for standard dataset version naming and project dataset placement"
        ),
    )
    dataset_version: str | None = Field(
        default=None,
        description=(
            "Optional dataset version name. Defaults to <detector_name>_YYYYMMDD_HHMM when "
            "project_root_dir + detector_name are provided"
        ),
    )
    last_yaml: str | None = Field(
        default=None,
        description=(
            "Optional previous data.yaml. Split paths from last_yaml are prepended before current "
            "scan paths (deduplicated). If classes.txt is missing or has no class lines, last_yaml is "
            "required and must contain names (and typically nc). May be local path or SFTP URI (sftp_*)."
        ),
    )
    sftp_username: str | None = Field(
        default=None,
        description="Required when last_yaml is a remote SFTP URI; SSH username",
    )
    sftp_private_key_path: str | None = Field(
        default=None,
        description="Required when last_yaml is remote (unless using password auth in future); local path to private key",
    )
    sftp_port: int | None = Field(
        default=None,
        description="Optional SSH port when omitted from last_yaml URI (default 22)",
    )


class BuildYoloYamlResponse(BaseModel):
    status: str = "ok"
    output_yaml_path: str
    path_in_yaml: str = Field(
        description="First split's images directory path as written in YAML (absolute, after prefix replace)",
    )
    dataset_root: str = Field(
        description="Resolved dataset root on disk used to locate splits (before prefix replace)",
    )
    splits_included: list[str]
    classes_count: int
    dataset_version: str | None = Field(
        default=None,
        description="Dataset version name when API published dataset into standard project structure",
    )
    published_dataset_dir: str | None = Field(
        default=None,
        description="Published dataset directory when project_root_dir + detector_name mode is used",
    )
    recommended_train_project: str | None = Field(
        default=None,
        description="Recommended local train project directory (<project_root_dir>/<detector_name>/runs/detect)",
    )
    recommended_train_name: str | None = Field(
        default=None,
        description="Recommended local train run name; equals dataset_version when published",
    )
    last_yaml_merged: bool = Field(
        default=False,
        description="Whether paths from last_yaml were merged into the output",
    )
    last_yaml_source: str | None = Field(
        default=None,
        description="none | local | sftp — how last_yaml was loaded when merged",
    )


class PublishYoloDatasetRequest(BaseModel):
    input_dir: str = Field(
        description=(
            "Input dataset root to scan and publish. Supports the same split discovery rules as build-yolo-yaml"
        ),
    )
    project_root_dir: str = Field(
        description=(
            "Local project workspace root. Published datasets land under "
            "<project_root_dir>/<detector_name>/datasets/<dataset_version> in local mode, "
            "or use this path as the local staging root in remote mode"
        ),
    )
    detector_name: str = Field(
        description="Stable detector family name used for dataset version naming and placement",
    )
    dataset_version: str | None = Field(
        default=None,
        description="Optional dataset version. Defaults to <detector_name>_YYYYMMDD_HHMM",
    )
    publish_mode: Literal["local", "remote_sftp"] = Field(
        default="local",
        description="Publish locally only, or stage locally then transfer to a remote SFTP target",
    )
    remote_host: str | None = Field(
        default=None,
        description="Required when publish_mode=remote_sftp",
    )
    remote_project_root_dir: str | None = Field(
        default=None,
        description=(
            "Required when publish_mode=remote_sftp. Remote workspace root; final dataset lands under "
            "<remote_project_root_dir>/<detector_name>/datasets/<dataset_version>"
        ),
    )
    remote_username: str | None = Field(
        default=None,
        description="Required when publish_mode=remote_sftp",
    )
    remote_private_key_path: str | None = Field(
        default=None,
        description="Required when publish_mode=remote_sftp unless password auth is added later",
    )
    remote_port: int = Field(
        default=22,
        ge=1,
        le=65535,
        description="SSH port for remote publish",
    )
    last_yaml: str | None = Field(
        default=None,
        description=(
            "Optional previous data.yaml. Split paths from last_yaml are prepended before current "
            "scan paths (deduplicated). If classes.txt is missing or empty, last_yaml must contain names"
        ),
    )
    sftp_username: str | None = Field(
        default=None,
        description="SSH username used only when last_yaml itself is a remote SFTP path",
    )
    sftp_private_key_path: str | None = Field(
        default=None,
        description="SSH private key path used only when last_yaml itself is remote",
    )
    sftp_port: int | None = Field(
        default=None,
        description="Optional SSH port used only when last_yaml itself is remote",
    )

    @model_validator(mode="after")
    def _validate_remote_publish_fields(self) -> "PublishYoloDatasetRequest":
        if self.publish_mode == "remote_sftp":
            missing: list[str] = []
            if not self.remote_host:
                missing.append("remote_host")
            if not self.remote_project_root_dir:
                missing.append("remote_project_root_dir")
            if not self.remote_username:
                missing.append("remote_username")
            if not self.remote_private_key_path:
                missing.append("remote_private_key_path")
            if missing:
                raise ValueError(
                    "publish_mode=remote_sftp requires: " + ", ".join(missing)
                )
        return self


class PublishYoloDatasetAsyncRequest(PublishYoloDatasetRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class PublishYoloDatasetResponse(BaseModel):
    status: str = "ok"
    publish_mode: Literal["local", "remote_sftp"]
    output_yaml_path: str = Field(
        description="Train-consumable yaml path. Local path in local mode; remote path in remote mode",
    )
    dataset_root: str = Field(
        description="Resolved dataset root on disk used to locate splits before publication",
    )
    splits_included: list[str]
    classes_count: int
    dataset_version: str
    published_dataset_dir: str = Field(
        description="Final published dataset directory. Local path in local mode; remote path in remote mode",
    )
    staging_published_dataset_dir: str | None = Field(
        default=None,
        description="Local staging dataset directory used before remote transfer",
    )
    staging_output_yaml_path: str | None = Field(
        default=None,
        description="Local staging yaml path used before remote transfer",
    )
    local_archive_path: str | None = Field(
        default=None,
        description="Local zip archive path used for remote transfer mode",
    )
    remote_target_host: str | None = None
    remote_target_port: int | None = None
    remote_archive_path: str | None = Field(
        default=None,
        description="Remote zip archive path created during remote publish mode",
    )
    recommended_train_project: str
    recommended_train_name: str
    last_yaml_merged: bool = Field(
        default=False,
        description="Whether paths from last_yaml were merged into the output",
    )
    last_yaml_source: str | None = Field(
        default=None,
        description="none | local | sftp — how last_yaml was loaded when merged",
    )


class YoloTrainRequest(BaseModel):
    yaml_path: str = Field(
        ...,
        description="Absolute path to Ultralytics data YAML",
    )
    project_root_dir: str = Field(
        ...,
        description="Working directory for the training subprocess (shell cwd)",
    )
    project: str = Field(
        ...,
        description="显式指定的 YOLO project 输出目录；API 不再内部推导",
    )
    name: str = Field(
        ...,
        description="显式指定的 YOLO run 名称；API 不再内部推导",
    )
    yolo_train_env: str = Field(
        ...,
        description="Conda environment name (e.g. yolo_pose)",
    )
    model: str = Field(default="yolo11s.pt", description="Ultralytics model argument")
    epochs: int = Field(default=100, ge=1)
    imgsz: int = Field(default=640, ge=1)
    batch: int | None = Field(
        default=None,
        ge=1,
        description="YOLO batch size; omitted when None (Ultralytics default)",
    )

    @model_validator(mode="after")
    def _validate_project_and_name(self) -> "YoloTrainRequest":
        _validate_train_project_and_name(
            yaml_path=self.yaml_path,
            project=self.project,
            name=self.name,
        )
        return self


class YoloTrainAsyncRequest(YoloTrainRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class YoloTrainResponse(BaseModel):
    status: Literal["ok", "failed"]
    command: str
    cwd: str
    project: str
    name: str
    exit_code: int
    stdout: str
    stderr: str


class YoloExportRequest(BaseModel):
    best_pt_path: str = Field(
        ...,
        description="Path to trained best.pt weights file",
    )
    project_root_dir: str = Field(
        ...,
        description="Working directory for the export subprocess (shell cwd)",
    )
    yolo_train_env: str = Field(
        ...,
        description="Conda environment name used to run yolo export",
    )
    overwrite: bool = Field(
        default=True,
        description="Whether to overwrite existing exported torchscript file",
    )


class YoloExportAsyncRequest(YoloExportRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class YoloExportResponse(BaseModel):
    status: Literal["ok", "failed"]
    command: str
    cwd: str
    best_pt_path: str
    args_yaml_path: str
    imgsz: int
    dataset_yaml: str
    export_file_path: str
    exit_code: int
    stdout: str
    stderr: str


class YoloInferRequest(BaseModel):
    yolo_train_env: str = Field(..., description="Conda environment name to run inference")
    model_path: str = Field(..., description="Path to .torchscript model")
    source_path: str | None = Field(
        default=None,
        description="Single image path or root directory path (recursive when recursive=true)",
    )
    source_paths: list[str] | None = Field(
        default=None,
        description="Multiple image/dir paths; merged with source_path",
    )
    project: str = Field(..., description="Infer project output root, e.g. <detector>/runs/infer")
    name: str = Field(..., description="Infer run name under project")
    imgsz: int = Field(default=640, ge=1, description="Inference image size")
    conf: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou: float = Field(default=0.7, ge=0.0, le=1.0, description="NMS IoU threshold")
    classes: list[int] | None = Field(default=None, description="Optional class id filter list")
    device: str | None = Field(default=None, description="Optional device, e.g. 0 or cpu")
    recursive: bool = Field(default=True, description="Whether to recursively scan source directories")
    save_labels: bool = Field(default=True, description="Write YOLO txt labels for detected images")
    save_no_detect: bool = Field(default=True, description="Save images with no detections into no_detect/")
    add_conf_prefix: bool = Field(default=True, description="Prefix output filenames with max confidence")
    draw_label: bool = Field(default=True, description="Draw class/conf text on result images")
    overwrite: bool = Field(default=True, description="Overwrite existing infer run directory")

    @model_validator(mode="after")
    def _validate_sources(self) -> "YoloInferRequest":
        merged_sources = [*(self.source_paths or []), *([self.source_path] if self.source_path else [])]
        if not merged_sources:
            raise ValueError("source_path 与 source_paths 至少提供一个")
        cleaned = [str(p).strip() for p in merged_sources if str(p).strip()]
        if not cleaned:
            raise ValueError("source_path/source_paths 不能全为空")
        self.source_paths = cleaned
        self.source_path = cleaned[0]
        return self


class YoloInferAsyncRequest(YoloInferRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class YoloInferResponse(BaseModel):
    status: Literal["ok", "failed"]
    model_path: str
    output_dir: str
    run_args_path: str
    summary_path: str
    total_images: int
    detected_images: int
    no_detect_images: int
    result_images: int
    labels_written: int
    classes_filter: list[int] | None = None


class YoloAugmentRequest(BaseModel):
    """在 input_dir/images+labels 或递归发现 nested */images+labels 上做离线数据增强。"""

    input_dir: str = Field(
        description=(
            "输入数据集根目录。支持直接包含 images/ 与 labels/，"
            "也支持递归发现 nested */images 与同级 labels/"
        ),
    )
    output_dir: str | None = Field(
        default=None,
        description=(
            "输出根目录；默认写入 <input_dir>/augment。若输入为多层 split/train|val|test 结构，"
            "输出端会保留相对目录层级并在各层下创建 images/ 与 labels/"
        ),
    )
    recursive: bool = Field(default=True, description="是否递归扫描 images/ 与 labels/")
    overwrite: bool = Field(default=True, description="目标文件已存在时是否覆盖")
    horizontal_flip: bool = Field(default=True, description="是否生成水平翻转增强")
    vertical_flip: bool = Field(default=True, description="是否生成垂直翻转增强")
    brightness_up: bool = Field(default=True, description="是否生成提高亮度增强")
    brightness_down: bool = Field(default=True, description="是否生成降低亮度增强")
    contrast_up: bool = Field(default=True, description="是否生成提高对比度增强")
    contrast_down: bool = Field(default=True, description="是否生成降低对比度增强")
    gaussian_blur: bool = Field(default=True, description="是否生成高斯模糊增强")


class YoloAugmentFileDetail(BaseModel):
    source_image: str
    source_label: str | None = None
    generated_images: list[str] = Field(default_factory=list)
    generated_labels: list[str] = Field(default_factory=list)
    skipped_reason: str | None = None


class YoloAugmentResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    output_dir: str
    processed_images: int
    skipped_images: int
    generated_images: int
    generated_labels: int
    details: list[YoloAugmentFileDetail]


class YoloAugmentAsyncRequest(YoloAugmentRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class ResetYoloLabelIndexRequest(BaseModel):
    input_dir: str = Field(
        description="输入数据集根目录，目录下需包含 labels/ 子目录",
    )
    recursive: bool = Field(default=True, description="是否递归扫描 labels 目录")


class ResetYoloLabelIndexResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    labels_dir: str
    total_label_files: int
    modified_label_files: int
    unchanged_label_files: int
    changed_lines: int
    skipped_invalid_lines: int


class ResetYoloLabelIndexAsyncRequest(ResetYoloLabelIndexRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class VocBarCropRequest(BaseModel):
    """横向条带 VOC 目标：以框高为边长裁正方形，目标居中；输出小图与对应 XML。"""

    input_dir: str = Field(description="输入数据集根目录，其下需包含 images/ 与 xmls/ 子目录")
    output_dir: str = Field(description="输出根目录，将创建 images/ 与 xmls/ 子目录")
    recursive: bool = Field(default=True, description="是否递归扫描 xml")


class VocBarCropDetail(BaseModel):
    source_image: str
    source_xml: str
    crop_image: str | None = None
    crop_xml: str | None = None
    window_left: int | None = None
    window_top: int | None = None
    window_size: int | None = None
    skipped_reason: str | None = None


class VocBarCropResponse(BaseModel):
    status: str = "ok"
    input_dir: str
    output_dir: str
    processed_xml_files: int
    skipped_xml_files: int
    generated_crops: int
    details: list[VocBarCropDetail]


class VocBarCropAsyncRequest(VocBarCropRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )


class RestoreVocCropsBatchRequest(BaseModel):
    """按 voc-bar 文件名规则，将 crop 目录下所有编辑后小图一次性贴回对应原图并合并 VOC 标注。"""

    original_images_dir: str = Field(description="原始数据集 images 目录")
    original_xmls_dir: str = Field(description="原始数据集 xmls 目录")
    edited_crops_images_dir: str = Field(description="编辑后裁剪图目录（文件名含 _cx_cy_S_）")
    edited_crops_xmls_dir: str = Field(description="编辑后裁剪 VOC XML 目录（与裁剪图同名 .xml）")
    output_dir: str = Field(description="输出根目录，将创建 images/ 与 xmls/")
    recursive: bool = Field(default=False, description="是否在裁剪图目录递归扫描")
    skip_unparsed_names: bool = Field(
        default=True,
        description="跳过不符合 voc-bar 命名（无 _cx_cy_S_）的文件；false 时遇到则报错",
    )


class RestoreVocCropsBatchStemDetail(BaseModel):
    original_stem: str
    output_image: str | None = None
    output_xml: str | None = None
    crops_applied: int = 0
    status: str = "ok"
    message: str | None = None


class RestoreVocCropsBatchResponse(BaseModel):
    status: str = "ok"
    output_dir: str
    originals_processed: int = Field(description="成功写出的大图数量（按原图 stem 计）")
    total_crop_files: int
    details: list[RestoreVocCropsBatchStemDetail]


class RestoreVocCropsBatchAsyncRequest(RestoreVocCropsBatchRequest):
    callback_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional webhook URL that receives task result when finished",
    )
    callback_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Callback HTTP timeout in seconds",
    )
