from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    CropImageDetail,
    SlidingWindowCropRequest,
    SlidingWindowCropResponse,
)
from app.services.task_manager import ensure_current_task_active
from app.utils.images import list_image_paths


_SAVE_FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}


def _resolve_output_extension(requested: str, source_suffix: str) -> str:
    if requested == "keep":
        suffix = source_suffix.lower()
        return suffix if suffix in _SAVE_FORMAT_MAP else ".png"
    if requested in {"jpg", "jpeg"}:
        return ".jpg"
    return f".{requested}"


def _save_crop(crop: Image.Image, output_path: Path, output_ext: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = _SAVE_FORMAT_MAP.get(output_ext, "PNG")

    if save_format == "JPEG" and crop.mode not in {"RGB", "L"}:
        crop = crop.convert("RGB")
    crop.save(output_path, format=save_format)


def run_sliding_window_crop(request: SlidingWindowCropRequest) -> SlidingWindowCropResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = resolve_safe_path(request.output_dir, field_name="output_dir")

    image_paths = list_image_paths(
        input_dir,
        recursive=request.recursive,
        extensions=request.extensions,
    )

    details: list[CropImageDetail] = []
    generated_crops = 0
    processed_images = 0
    skipped_images = 0

    for image_path in image_paths:
        ensure_current_task_active()
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                crop_count = 0

                x_values = (
                    range(0, width, request.stride_x)
                    if request.include_partial_edges
                    else range(0, max(width - request.window_width + 1, 0), request.stride_x)
                )
                y_values = (
                    range(0, height, request.stride_y)
                    if request.include_partial_edges
                    else range(0, max(height - request.window_height + 1, 0), request.stride_y)
                )

                for y in y_values:
                    ensure_current_task_active()
                    for x in x_values:
                        right = min(x + request.window_width, width)
                        bottom = min(y + request.window_height, height)

                        if right <= x or bottom <= y:
                            continue

                        crop_width = right - x
                        crop_height = bottom - y

                        if (
                            not request.include_partial_edges
                            and (
                                crop_width < request.window_width
                                or crop_height < request.window_height
                            )
                        ):
                            continue

                        rel_dir = (
                            image_path.parent.relative_to(input_dir)
                            if request.keep_subdirs
                            else Path(".")
                        )

                        output_ext = _resolve_output_extension(
                            request.output_format,
                            image_path.suffix,
                        )

                        output_name = (
                            f"{image_path.stem}_x{x}_y{y}_w{crop_width}_h{crop_height}{output_ext}"
                        )
                        output_path = output_dir / rel_dir / output_name

                        crop = image.crop((x, y, right, bottom))
                        _save_crop(crop, output_path, output_ext)
                        crop_count += 1

                if crop_count > 0:
                    processed_images += 1
                    generated_crops += crop_count
                    details.append(
                        CropImageDetail(
                            source_image=str(image_path),
                            crop_count=crop_count,
                        )
                    )
                else:
                    skipped_images += 1
                    details.append(
                        CropImageDetail(
                            source_image=str(image_path),
                            crop_count=0,
                            skipped_reason="image smaller than window or no valid window",
                        )
                    )

        except (UnidentifiedImageError, OSError) as exc:
            skipped_images += 1
            details.append(
                CropImageDetail(
                    source_image=str(image_path),
                    crop_count=0,
                    skipped_reason=f"failed to open/process: {exc}",
                )
            )

    return SlidingWindowCropResponse(
        input_images=len(image_paths),
        processed_images=processed_images,
        skipped_images=skipped_images,
        generated_crops=generated_crops,
        output_dir=str(output_dir),
        details=details,
    )
