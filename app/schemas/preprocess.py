from typing import Literal

from pydantic import BaseModel, Field


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
