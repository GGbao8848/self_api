from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.schemas.preprocess import (
    YoloSlidingWindowCropDetail,
    YoloSlidingWindowCropRequest,
    YoloSlidingWindowCropResponse,
)
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


def _read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[tuple[str, float, float, float, float]]:
    if not label_path.exists():
        return []

    boxes: list[tuple[str, float, float, float, float]] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        class_id = parts[0]
        try:
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            box_width = float(parts[3]) * image_width
            box_height = float(parts[4]) * image_height
        except ValueError:
            continue
        if box_width <= 0 or box_height <= 0:
            continue

        x_min = x_center - box_width / 2.0
        y_min = y_center - box_height / 2.0
        x_max = x_center + box_width / 2.0
        y_max = y_center + box_height / 2.0
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes


def _clip_boxes_to_window(
    boxes: list[tuple[str, float, float, float, float]],
    x: int,
    y: int,
    right: int,
    bottom: int,
    min_box_area_ratio: float,
) -> list[str]:
    lines: list[str] = []
    window_width = right - x
    window_height = bottom - y
    if window_width <= 0 or window_height <= 0:
        return lines

    for class_id, box_x_min, box_y_min, box_x_max, box_y_max in boxes:
        inter_x_min = max(box_x_min, x)
        inter_y_min = max(box_y_min, y)
        inter_x_max = min(box_x_max, right)
        inter_y_max = min(box_y_max, bottom)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            continue

        original_area = (box_x_max - box_x_min) * (box_y_max - box_y_min)
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        if original_area <= 0:
            continue
        if (inter_area / original_area) < min_box_area_ratio:
            continue

        clipped_x_center = ((inter_x_min + inter_x_max) / 2.0 - x) / window_width
        clipped_y_center = ((inter_y_min + inter_y_max) / 2.0 - y) / window_height
        clipped_width = (inter_x_max - inter_x_min) / window_width
        clipped_height = (inter_y_max - inter_y_min) / window_height

        if clipped_width <= 0 or clipped_height <= 0:
            continue

        clipped_x_center = min(max(clipped_x_center, 0.0), 1.0)
        clipped_y_center = min(max(clipped_y_center, 0.0), 1.0)
        clipped_width = min(max(clipped_width, 0.0), 1.0)
        clipped_height = min(max(clipped_height, 0.0), 1.0)

        lines.append(
            f"{class_id} "
            f"{clipped_x_center:.6f} {clipped_y_center:.6f} "
            f"{clipped_width:.6f} {clipped_height:.6f}"
        )
    return lines


def run_yolo_sliding_window_crop(request: YoloSlidingWindowCropRequest) -> YoloSlidingWindowCropResponse:
    dataset_dir = Path(request.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"dataset_dir does not exist or is not a directory: {dataset_dir}")

    images_dir = dataset_dir / request.images_dir_name
    labels_dir = dataset_dir / request.labels_dir_name
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"images_dir does not exist or is not a directory: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise ValueError(f"labels_dir does not exist or is not a directory: {labels_dir}")

    output_dir = (
        Path(request.output_dir).expanduser().resolve()
        if request.output_dir
        else (dataset_dir / "yolo_crops").resolve()
    )
    output_images_dir = output_dir / request.images_dir_name
    output_labels_dir = output_dir / request.labels_dir_name

    image_paths = list_image_paths(
        images_dir,
        recursive=request.recursive,
        extensions=request.extensions,
    )

    details: list[YoloSlidingWindowCropDetail] = []
    processed_images = 0
    skipped_images = 0
    generated_crops = 0
    generated_labels = 0

    for image_path in image_paths:
        rel_image = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel_image).with_suffix(".txt")
        if request.require_label and not label_path.exists():
            skipped_images += 1
            details.append(
                YoloSlidingWindowCropDetail(
                    source_image=str(image_path),
                    source_label=str(label_path),
                    skipped_reason="label file not found",
                )
            )
            continue

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
                boxes = _read_yolo_boxes(label_path, image_width, image_height)

                x_values = (
                    range(0, image_width, request.stride_x)
                    if request.include_partial_edges
                    else range(
                        0,
                        max(image_width - request.window_width + 1, 0),
                        request.stride_x,
                    )
                )
                y_values = (
                    range(0, image_height, request.stride_y)
                    if request.include_partial_edges
                    else range(
                        0,
                        max(image_height - request.window_height + 1, 0),
                        request.stride_y,
                    )
                )

                crop_count = 0
                label_count = 0

                for y in y_values:
                    for x in x_values:
                        right = min(x + request.window_width, image_width)
                        bottom = min(y + request.window_height, image_height)
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

                        lines = _clip_boxes_to_window(
                            boxes=boxes,
                            x=x,
                            y=y,
                            right=right,
                            bottom=bottom,
                            min_box_area_ratio=request.min_box_area_ratio,
                        )

                        if not lines and not request.keep_empty_labels:
                            continue

                        rel_parent = rel_image.parent if request.keep_subdirs else Path(".")
                        output_ext = _resolve_output_extension(request.output_format, image_path.suffix)
                        output_stem = f"{image_path.stem}_x{x}_y{y}_w{crop_width}_h{crop_height}"
                        output_image_path = output_images_dir / rel_parent / f"{output_stem}{output_ext}"
                        output_label_path = output_labels_dir / rel_parent / f"{output_stem}.txt"

                        if (output_image_path.exists() or output_label_path.exists()) and not request.overwrite:
                            continue
                        if request.overwrite:
                            if output_image_path.exists() and output_image_path.is_file():
                                output_image_path.unlink()
                            if output_label_path.exists() and output_label_path.is_file():
                                output_label_path.unlink()

                        crop = image.crop((x, y, right, bottom))
                        _save_crop(crop, output_image_path, output_ext)
                        output_label_path.parent.mkdir(parents=True, exist_ok=True)
                        output_label_path.write_text("\n".join(lines), encoding="utf-8")

                        crop_count += 1
                        label_count += len(lines)

                if crop_count > 0:
                    processed_images += 1
                    generated_crops += crop_count
                    generated_labels += label_count
                    details.append(
                        YoloSlidingWindowCropDetail(
                            source_image=str(image_path),
                            source_label=str(label_path) if label_path.exists() else None,
                            crop_count=crop_count,
                            label_count=label_count,
                        )
                    )
                else:
                    skipped_images += 1
                    details.append(
                        YoloSlidingWindowCropDetail(
                            source_image=str(image_path),
                            source_label=str(label_path) if label_path.exists() else None,
                            skipped_reason="no valid windows or no boxes in windows",
                        )
                    )
        except (UnidentifiedImageError, OSError) as exc:
            skipped_images += 1
            details.append(
                YoloSlidingWindowCropDetail(
                    source_image=str(image_path),
                    source_label=str(label_path) if label_path.exists() else None,
                    skipped_reason=f"failed to open/process image: {exc}",
                )
            )

    return YoloSlidingWindowCropResponse(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        input_images=len(image_paths),
        processed_images=processed_images,
        skipped_images=skipped_images,
        generated_crops=generated_crops,
        generated_labels=generated_labels,
        details=details,
    )
