from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, Field

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


class XmlToYoloRequest(BaseModel):
    dataset_dir: str = Field(
        description="Dataset root directory containing images and xmls folders",
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
    dataset_dir: str = Field(
        description="Dataset root directory containing images and labels folders",
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
    archive_path: str = Field(description="Zip archive path to extract")
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
