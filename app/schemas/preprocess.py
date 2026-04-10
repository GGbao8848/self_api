from typing import Any, Literal

from pydantic import AliasChoices, AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from app.schemas.artifacts import ArtifactSummary


class SlidingWindowCropRequest(BaseModel):
    input_dir: str = Field(description="Input directory containing images")
    output_dir: str = Field(description="Output directory for cropped images")
    window_width: int = Field(ge=1, description="Sliding window width")
    window_height: int = Field(ge=1, description="Sliding window height")
    stride_x: int = Field(ge=1, description="Horizontal stride")
    stride_y: int = Field(ge=1, description="Vertical stride")
    include_partial_edges: bool = Field(
        default=False,
        description="Include edge windows smaller than the configured window size",
    )
    recursive: bool = Field(default=True, description="Search files recursively")
    keep_subdirs: bool = Field(
        default=True,
        description="Keep source folder structure under output directory",
    )
    extensions: list[str] | None = Field(
        default=None,
        description="Allowed image extensions, e.g. ['.jpg', '.png']",
    )
    output_format: Literal["keep", "png", "jpg", "jpeg", "webp"] = Field(
        default="keep",
        description="Output image format",
    )


class CropImageDetail(BaseModel):
    source_image: str
    crop_count: int = 0
    skipped_reason: str | None = None


class SlidingWindowCropResponse(BaseModel):
    status: str = "ok"
    input_images: int
    processed_images: int
    skipped_images: int
    generated_crops: int
    output_dir: str
    details: list[CropImageDetail]


class SlidingWindowCropAsyncRequest(SlidingWindowCropRequest):
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


class AnnotateVisualizeRequest(BaseModel):
    """在图像上绘制标注框：YOLO txt 与 Pascal VOC XML 二选一（另一目录字段留空）。"""

    images_dir: str = Field(description="图像目录")
    labels_dir: str | None = Field(
        default=None,
        description="YOLO 格式 .txt 标注目录（与 xmls_dir 二选一，另一项留空）",
    )
    xmls_dir: str | None = Field(
        default=None,
        description="Pascal VOC XML 标注目录（与 labels_dir 二选一，另一项留空）",
    )
    output_dir: str = Field(description="可视化结果输出目录（保持与 images_dir 相同的相对路径结构）")
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
    def _labels_or_xmls_exclusive(self) -> "AnnotateVisualizeRequest":
        labels = (self.labels_dir or "").strip()
        xmls = (self.xmls_dir or "").strip()
        if labels and xmls:
            raise ValueError("labels_dir 与 xmls_dir 只能填其一，另一项留空")
        if not labels and not xmls:
            raise ValueError("labels_dir 与 xmls_dir 必须填其一，另一项留空")
        classes_inline = self.classes
        classes_path = (self.classes_file or "").strip()
        if classes_inline and classes_path:
            raise ValueError("classes 与 classes_file 只能填其一")
        return self.model_copy(
            update={
                "labels_dir": labels or None,
                "xmls_dir": xmls or None,
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
    images_dir: str
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


class XmlToYoloRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_dir: str = Field(
        description="Dataset root directory containing images and xmls folders",
        validation_alias=AliasChoices("dataset_dir", "input_dir"),
    )
    images_dir_name: str = Field(default="images", description="Image folder name")
    xmls_dir_name: str = Field(default="xmls", description="Pascal VOC XML folder name")
    labels_dir_name: str = Field(default="labels", description="YOLO labels folder name")
    recursive: bool = Field(default=True, description="Search xml files recursively")
    classes: list[str] | None = Field(
        default=None,
        description="Optional fixed class list; when omitted, classes are inferred",
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


class XmlToYoloFileDetail(BaseModel):
    source_xml: str
    output_label: str | None = None
    written_lines: int = 0
    skipped_reason: str | None = None


class XmlToYoloResponse(BaseModel):
    status: str = "ok"
    dataset_dir: str
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
    model_config = ConfigDict(populate_by_name=True)

    dataset_dir: str = Field(
        description="Dataset root directory containing images and labels folders",
        validation_alias=AliasChoices("dataset_dir", "input_dir"),
    )
    output_dir: str | None = Field(
        default=None,
        description="Output directory; defaults to <dataset_dir>/split_dataset",
    )
    images_dir_name: str = Field(default="images", description="Image folder name")
    labels_dir_name: str = Field(default="labels", description="Label folder name")
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
    dataset_dir: str
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
    model_config = ConfigDict(populate_by_name=True)

    input_dir: str = Field(description="Input folder path to package")
    output_zip_path: str | None = Field(
        default=None,
        description="Output zip path; defaults to <input_dir_parent>/<input_dir_name>.zip",
        validation_alias=AliasChoices("output_zip_path", "output_dir"),
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
    model_config = ConfigDict(populate_by_name=True)

    archive_path: str = Field(
        description="Zip archive path to extract",
        validation_alias=AliasChoices("archive_path", "input_dir"),
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
    model_config = ConfigDict(populate_by_name=True)

    source_path: str = Field(
        description="Source file or directory path",
        validation_alias=AliasChoices("source_path", "input_dir"),
    )
    target_dir: str = Field(
        description="Target directory path",
        validation_alias=AliasChoices("target_dir", "output_dir"),
    )
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
    model_config = ConfigDict(populate_by_name=True)

    source_path: str = Field(
        description="Source file or directory path",
        validation_alias=AliasChoices("source_path", "input_dir"),
    )
    target_dir: str = Field(
        description="Target directory path",
        validation_alias=AliasChoices("target_dir", "output_dir"),
    )
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


class RemoteSlurmYoloTrainRequest(BaseModel):
    """跨机器远程训练：本地 self_api 调远程 Slurm REST 提交训练任务。"""

    yaml_path: str = Field(
        description="远端数据 yaml 路径，支持 sftp://host/path 或 sftp://user@host/path 或 user@host:path",
    )
    project_root_dir: str = Field(
        description="远端工作目录，支持 sftp://host/path 或 sftp://user@host/path 或 user@host:path",
    )
    model: str = Field(default="yolo11s.pt", description="YOLO model")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    imgsz: int = Field(default=640, ge=1, description="Training image size")
    batch: int | None = Field(default=None, ge=1, description="Base batch size；若未显式设置 device，将按 GPU 数自动扩增")
    workers: int | None = Field(default=None, ge=0, description="DataLoader workers")
    cache: bool | None = Field(default=True, description="YOLO cache 参数（不允许 False）")
    device: str | None = Field(default=None, description="显式设备，如 0,1；不填则自动探测并设置")
    project: str | None = Field(default=None, description="YOLO project 输出目录；不填则按 yaml 自动推导")
    name: str | None = Field(default=None, description="YOLO run 名称；不填则取 yaml 文件名")
    partition: str | None = Field(default="gpu", description="SLURM 分区")
    nodelist: str | None = Field(default=None, description="可选：要求使用的节点列表（逗号分隔）")
    exclude: str | None = Field(default=None, description="可选：排除的节点列表（逗号分隔）")
    username: str | None = Field(
        default=None,
        description="Slurm 用户名（用于签发 token）",
    )
    password: str | None = Field(
        default=None,
        description="兼容保留字段（当前模式不使用）",
    )
    private_key_path: str | None = Field(
        default=None,
        description="兼容保留字段（当前模式不使用）",
    )
    port: int = Field(
        default=22,
        ge=1,
        le=65535,
        description="兼容保留字段（当前模式不使用）",
    )


class RemoteSlurmYoloTrainResponse(BaseModel):
    status: str = "ok"
    yaml_path: str
    project_root_dir: str
    target_host: str
    target_port: int
    command: str
    exit_code: int
    stdout: str
    stderr: str


class YoloSlidingWindowCropRequest(BaseModel):
    """YOLO 正方形滑窗裁剪：窗口边长=图片高度，仅水平滑动。"""

    images_dir: str = Field(
        description="Input images folder path",
    )
    labels_dir: str = Field(
        description="Input YOLO labels folder path (txt files)",
    )
    output_dir: str = Field(
        description="Output dataset folder; will create images/ and labels/ subdirs",
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


class CleanNestedDatasetLeafDetail(BaseModel):
    source_dir: str
    output_dir: str
    total_images: int
    labeled_images: int
    background_images: int
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


class RemoteSlurmYoloTrainAsyncRequest(RemoteSlurmYoloTrainRequest):
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
    images_dir: str
    labels_dir: str
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
        description="Path to classes.txt. Defaults to <input_dir>/classes.txt",
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
    output_yaml_path: str = Field(
        ...,
        description="Full path for the generated data.yaml (parent dirs are created if needed)",
    )


class BuildYoloYamlAsyncRequest(BuildYoloYamlRequest):
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


class YoloTrainRequest(BaseModel):
    yaml_path: str = Field(
        ...,
        description="Absolute path to Ultralytics data YAML (must contain a /dataset/ segment)",
    )
    project_root_dir: str = Field(
        ...,
        description="Working directory for the training subprocess (shell cwd)",
    )
    yolo_train_env: str = Field(
        ...,
        description="Conda environment name (e.g. yolo_pose)",
    )
    model: str = Field(default="yolo11s.pt", description="Ultralytics model argument")
    epochs: int = Field(default=100, ge=1)
    imgsz: int = Field(default=640, ge=1)


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


class VocBarCropRequest(BaseModel):
    """横向条带 VOC 目标：以框高为边长裁正方形，目标居中；输出小图与对应 XML。"""

    images_dir: str = Field(description="图像目录（与 xmls 按相对路径 stem 配对）")
    xmls_dir: str = Field(description="Pascal VOC XML 目录")
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
    images_dir: str
    xmls_dir: str
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
