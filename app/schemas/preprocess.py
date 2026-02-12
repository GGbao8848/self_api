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


class DuplicateGroup(BaseModel):
    representative: str
    duplicates: list[str]


class DeduplicateRequest(BaseModel):
    input_dir: str = Field(description="Input directory containing images")
    recursive: bool = Field(default=True, description="Search files recursively")
    extensions: list[str] | None = Field(
        default=None,
        description="Allowed image extensions, e.g. ['.jpg', '.png']",
    )
    method: Literal["md5", "phash"] = Field(
        default="phash",
        description="Dedup strategy: exact (md5) or perceptual (phash)",
    )
    distance_threshold: int = Field(
        default=0,
        ge=0,
        le=64,
        description="Hamming distance threshold for phash",
    )
    hash_size: int = Field(
        default=8,
        ge=4,
        le=32,
        description="phash size; larger means finer granularity",
    )
    copy_unique_to: str | None = Field(
        default=None,
        description="Optional output folder to copy unique images",
    )
    keep_subdirs: bool = Field(
        default=True,
        description="When copying unique images, keep source folder structure",
    )
    report_path: str | None = Field(
        default=None,
        description="Optional JSON report output path",
    )


class DeduplicateResponse(BaseModel):
    status: str = "ok"
    total_images: int
    unique_images: int
    duplicate_images: int
    method: str
    distance_threshold: int
    copied_unique_to: str | None
    report_path: str | None
    skipped_images: list[str]
    groups: list[DuplicateGroup]
