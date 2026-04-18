from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    YoloAugmentFileDetail,
    YoloAugmentRequest,
    YoloAugmentResponse,
)
from app.services.task_manager import ensure_current_task_active
from app.utils.images import list_image_paths

_AUGMENTATION_SPECS: list[tuple[str, str]] = [
    ("horizontal_flip", "hflip"),
    ("vertical_flip", "vflip"),
    ("brightness_up", "brightness_up"),
    ("brightness_down", "brightness_down"),
    ("contrast_up", "contrast_up"),
    ("contrast_down", "contrast_down"),
    ("gaussian_blur", "gaussian_blur"),
]


def _load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        raise ValueError(f"label file not found: {label_path}")

    labels: list[tuple[int, float, float, float, float]] = []
    for idx, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO label line at {label_path}:{idx}: {raw_line!r}")
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as exc:
            raise ValueError(f"invalid YOLO label line at {label_path}:{idx}: {raw_line!r}") from exc
        labels.append((class_id, x_center, y_center, width, height))
    return labels


def _save_yolo_labels(
    label_path: Path,
    labels: list[tuple[int, float, float, float, float]],
) -> None:
    lines = [
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for class_id, x_center, y_center, width, height in labels
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _apply_label_transform(
    labels: list[tuple[int, float, float, float, float]],
    augmentation_name: str,
) -> list[tuple[int, float, float, float, float]]:
    transformed: list[tuple[int, float, float, float, float]] = []
    for class_id, x_center, y_center, width, height in labels:
        if augmentation_name == "hflip":
            transformed.append((class_id, 1.0 - x_center, y_center, width, height))
        elif augmentation_name == "vflip":
            transformed.append((class_id, x_center, 1.0 - y_center, width, height))
        else:
            transformed.append((class_id, x_center, y_center, width, height))
    return transformed


def _apply_image_transform(image: Image.Image, augmentation_name: str) -> Image.Image:
    if augmentation_name == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if augmentation_name == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if augmentation_name == "brightness_up":
        return ImageEnhance.Brightness(image).enhance(1.3)
    if augmentation_name == "brightness_down":
        return ImageEnhance.Brightness(image).enhance(0.7)
    if augmentation_name == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(1.3)
    if augmentation_name == "contrast_down":
        return ImageEnhance.Contrast(image).enhance(0.7)
    if augmentation_name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    raise ValueError(f"unsupported augmentation: {augmentation_name}")


def run_yolo_augment(request: YoloAugmentRequest) -> YoloAugmentResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = (
        resolve_safe_path(request.output_dir, field_name="output_dir")
        if request.output_dir
        else (input_dir / "augment").resolve()
    )
    if output_dir == input_dir:
        raise ValueError("output_dir cannot equal input_dir")

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.is_dir():
        raise ValueError(f"images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"labels directory not found: {labels_dir}")

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    classes_file = labels_dir / "classes.txt"
    if classes_file.exists():
        shutil.copy2(classes_file, output_labels_dir / "classes.txt")

    selected_augmentations = [
        suffix for field_name, suffix in _AUGMENTATION_SPECS if getattr(request, field_name)
    ]
    if not selected_augmentations:
        raise ValueError("at least one augmentation option must be enabled")

    image_paths = list_image_paths(images_dir, recursive=request.recursive)
    details: list[YoloTxtAugmentFileDetail] = []
    processed_images = 0
    skipped_images = 0
    generated_images = 0
    generated_labels = 0

    for image_path in image_paths:
        ensure_current_task_active()
        rel_path = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel_path).with_suffix(".txt")
        detail = YoloAugmentFileDetail(
            source_image=str(image_path),
            source_label=str(label_path),
        )

        if not label_path.exists():
            detail.skipped_reason = "label file not found"
            skipped_images += 1
            details.append(detail)
            continue

        try:
            labels = _load_yolo_labels(label_path)
            with Image.open(image_path) as img:
                source_image = img.convert("RGB")
                for augmentation_name in selected_augmentations:
                    augmented_image = _apply_image_transform(source_image, augmentation_name)
                    augmented_labels = _apply_label_transform(labels, augmentation_name)

                    target_rel = rel_path.with_name(f"{rel_path.stem}_{augmentation_name}{rel_path.suffix}")
                    target_image = output_images_dir / target_rel
                    target_label = output_labels_dir / target_rel.with_suffix(".txt")

                    if (target_image.exists() or target_label.exists()) and not request.overwrite:
                        continue

                    target_image.parent.mkdir(parents=True, exist_ok=True)
                    target_label.parent.mkdir(parents=True, exist_ok=True)
                    augmented_image.save(target_image)
                    _save_yolo_labels(target_label, augmented_labels)

                    generated_images += 1
                    generated_labels += 1
                    detail.generated_images.append(str(target_image))
                    detail.generated_labels.append(str(target_label))
        except (OSError, ValueError) as exc:
            detail.skipped_reason = str(exc)
            skipped_images += 1
            details.append(detail)
            continue

        processed_images += 1
        details.append(detail)

    return YoloAugmentResponse(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        processed_images=processed_images,
        skipped_images=skipped_images,
        generated_images=generated_images,
        generated_labels=generated_labels,
        details=details,
    )
