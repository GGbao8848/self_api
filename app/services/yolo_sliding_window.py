"""Sliding window crop with optional YOLO label remapping."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    YoloSlidingWindowCropDetail,
    YoloSlidingWindowCropRequest,
    YoloSlidingWindowCropResponse,
)
from app.services.task_manager import ensure_current_task_active
from app.utils.images import list_image_paths

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_SAVE_FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}


def _discover_dataset_units(input_dir: Path) -> list[tuple[Path, Path, Path | None]]:
    """Find dataset units as (relative_root, images_dir, labels_dir).

    Supported layouts:
    - input_dir/images (+ optional input_dir/labels)
    - input_dir/**/images (+ optional sibling labels)
    """
    dataset_units: list[tuple[Path, Path, Path | None]] = []
    seen_images_dirs: set[Path] = set()

    direct_images_dir = input_dir / "images"
    if direct_images_dir.is_dir():
        direct_labels_dir = input_dir / "labels"
        dataset_units.append(
            (
                Path("."),
                direct_images_dir,
                direct_labels_dir if direct_labels_dir.is_dir() else None,
            )
        )
        seen_images_dirs.add(direct_images_dir.resolve())

    for images_dir in sorted(p for p in input_dir.rglob("images") if p.is_dir()):
        resolved = images_dir.resolve()
        if resolved in seen_images_dirs:
            continue

        dataset_root = images_dir.parent
        labels_dir = dataset_root / "labels"
        dataset_units.append(
            (
                dataset_root.relative_to(input_dir),
                images_dir,
                labels_dir if labels_dir.is_dir() else None,
            )
        )
        seen_images_dirs.add(resolved)

    return dataset_units


def _yolo_to_xyxy(xc: float, yc: float, w: float, h: float, W: int, H: int) -> tuple[float, float, float, float]:
    """YOLO normalized -> absolute xyxy."""
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return x1, y1, x2, y2


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> tuple[float, float, float, float]:
    """Absolute xyxy -> YOLO normalized"""
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / W, yc / H, bw / W, bh / H


def _clip_box_to_crop(
    bx1: float, by1: float, bx2: float, by2: float,
    cx1: float, cy1: float, cx2: float, cy2: float,
) -> tuple[float, float, float, float] | None:
    """Intersection of bbox with crop rect in absolute coords."""
    ix1 = max(bx1, cx1)
    iy1 = max(by1, cy1)
    ix2 = min(bx2, cx2)
    iy2 = min(by2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _area_xyxy(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Returns list of (cls, xc, yc, w, h) as floats (YOLO normalized)."""
    if not label_path.exists():
        return []
    items: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            items.append((cls, xc, yc, w, h))
        except (ValueError, TypeError):
            continue
    return items


def _sliding_window_crop(
    image_path: Path,
    label_path: Path | None,
    out_img_dir: Path,
    out_lbl_dir: Path | None,
    window_width: int | None,
    window_height: int | None,
    stride_x: int | None,
    stride_y: int | None,
    min_vis_ratio: float,
    stride_ratio: float,
    only_wide: bool,
    ignore_vis_ratio: float,
) -> tuple[int, int]:
    """Process single image; returns (saved_crops, total_label_lines)."""
    try:
        with Image.open(image_path) as img:
            img_arr = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return 0, 0

    W, H = img_arr.size
    if only_wide and W <= H:
        return 0, 0

    win_w = window_width or H
    win_h = window_height or H
    step_x = stride_x or max(1, int(round(stride_ratio * H)))
    step_y = stride_y or win_h

    abs_boxes: list[tuple[int, float, float, float, float, float]] = []
    if label_path is not None:
        labels = _read_yolo_labels(label_path)
        for cls, xc, yc, bw, bh in labels:
            bx1, by1, bx2, by2 = _yolo_to_xyxy(xc, yc, bw, bh, W, H)
            bx1, by1 = max(0.0, bx1), max(0.0, by1)
            bx2, by2 = min(float(W), bx2), min(float(H), by2)
            a0 = _area_xyxy(bx1, by1, bx2, by2)
            if a0 > 0:
                abs_boxes.append((cls, bx1, by1, bx2, by2, a0))

        if not abs_boxes:
            return 0, 0

    saved = 0
    total_label_lines = 0

    if W <= win_w:
        xs = [0]
    else:
        xs = list(range(0, W - win_w + 1, step_x))
        if xs[-1] != W - win_w:
            xs.append(W - win_w)

    if H <= win_h:
        ys = [0]
    else:
        ys = list(range(0, H - win_h + 1, step_y))
        if ys[-1] != H - win_h:
            ys.append(H - win_h)

    for row, y0 in enumerate(ys):
        ensure_current_task_active()
        for col, x0 in enumerate(xs):
            cx1, cy1 = float(x0), float(y0)
            cx2, cy2 = float(x0 + win_w), float(y0 + win_h)

            crop = img_arr.crop((x0, y0, x0 + win_w, y0 + win_h))
            if crop.size[0] == 0 or crop.size[1] == 0:
                continue

            kept_lines: list[str] = []
            has_keep = label_path is None
            has_danger = False

            for cls, bx1, by1, bx2, by2, a0 in abs_boxes:
                inter = _clip_box_to_crop(bx1, by1, bx2, by2, cx1, cy1, cx2, cy2)
                if inter is None:
                    continue

                ix1, iy1, ix2, iy2 = inter
                ai = _area_xyxy(ix1, iy1, ix2, iy2)
                r = ai / a0

                if ignore_vis_ratio < r < min_vis_ratio:
                    has_danger = True
                    break

                if r >= min_vis_ratio:
                    has_keep = True
                    lx1 = ix1 - cx1
                    ly1 = iy1 - cy1
                    lx2 = ix2 - cx1
                    ly2 = iy2 - cy1
                    nxc, nyc, nw, nh = _xyxy_to_yolo(lx1, ly1, lx2, ly2, win_w, win_h)
                    if nw <= 0 or nh <= 0:
                        continue
                    kept_lines.append(f"{cls} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")

            if has_danger:
                continue
            if not has_keep or (label_path is not None and not kept_lines):
                continue

            stem = image_path.stem
            out_name = f"{stem}_w{win_w}_h{win_h}_x{x0:06d}_y{y0:06d}_r{row:03d}_c{col:03d}"
            ext = image_path.suffix.lower()
            if ext not in _SAVE_FORMAT_MAP:
                ext = ".png"

            out_img_path = out_img_dir / f"{out_name}{ext}"

            save_format = _SAVE_FORMAT_MAP.get(ext, "PNG")
            if save_format == "JPEG" and crop.mode != "RGB":
                crop = crop.convert("RGB")
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_img_path, format=save_format)

            if out_lbl_dir is not None and label_path is not None:
                out_lbl_path = out_lbl_dir / f"{out_name}.txt"
                out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                out_lbl_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
                total_label_lines += len(kept_lines)

            saved += 1

    return saved, total_label_lines


def run_yolo_sliding_window_crop(request: YoloSlidingWindowCropRequest) -> YoloSlidingWindowCropResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    dataset_units = _discover_dataset_units(input_dir)
    if not dataset_units:
        raise ValueError(
            f"no images directories found under: {input_dir}; expected {input_dir / 'images'} "
            "or nested */images directories"
        )

    output_dir = resolve_safe_path(request.output_dir, field_name="output_dir")

    details: list[YoloSlidingWindowCropDetail] = []
    processed_images = 0
    skipped_images = 0
    generated_crops = 0
    generated_labels = 0
    input_images = 0
    labels_dirs: list[Path] = []

    for relative_root, images_dir, labels_dir in dataset_units:
        ensure_current_task_active()
        out_base_dir = output_dir if relative_root == Path(".") else output_dir / relative_root
        out_img_dir = out_base_dir / "images"
        out_lbl_dir = out_base_dir / "labels" if labels_dir is not None else None
        out_img_dir.mkdir(parents=True, exist_ok=True)
        if out_lbl_dir is not None:
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            labels_dirs.append(labels_dir)

        image_paths = list_image_paths(
            images_dir,
            recursive=True,
            extensions=list(_IMG_EXTS),
        )
        input_images += len(image_paths)

        for image_path in image_paths:
            ensure_current_task_active()
            label_path: Path | None = None
            if labels_dir is not None:
                rel_image = image_path.relative_to(images_dir)
                label_path = (labels_dir / rel_image).with_suffix(".txt")

            if label_path is not None and not label_path.exists():
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
                saved, label_lines = _sliding_window_crop(
                    image_path=image_path,
                    label_path=label_path,
                    out_img_dir=out_img_dir,
                    out_lbl_dir=out_lbl_dir,
                    window_width=request.window_width,
                    window_height=request.window_height,
                    stride_x=request.stride_x,
                    stride_y=request.stride_y,
                    min_vis_ratio=request.min_vis_ratio,
                    stride_ratio=request.stride_ratio,
                    only_wide=request.only_wide,
                    ignore_vis_ratio=request.ignore_vis_ratio,
                )

                if saved > 0:
                    processed_images += 1
                    generated_crops += saved
                    generated_labels += label_lines
                    details.append(
                        YoloSlidingWindowCropDetail(
                            source_image=str(image_path),
                            source_label=str(label_path),
                            crop_count=saved,
                            label_count=label_lines,
                        )
                    )
                else:
                    skipped_images += 1
                    details.append(
                        YoloSlidingWindowCropDetail(
                            source_image=str(image_path),
                            source_label=str(label_path),
                            skipped_reason="no valid windows or only_wide skipped",
                        )
                    )
            except (UnidentifiedImageError, OSError) as exc:
                skipped_images += 1
                details.append(
                    YoloSlidingWindowCropDetail(
                        source_image=str(image_path),
                        source_label=str(label_path),
                        skipped_reason=f"failed to open/process: {exc}",
                    )
                )

    labels_dir_value: str | None = None
    if len(labels_dirs) == 1:
        labels_dir_value = str(labels_dirs[0])

    return YoloSlidingWindowCropResponse(
        input_dir=str(input_dir),
        labels_dir=labels_dir_value,
        output_dir=str(output_dir),
        input_images=input_images,
        processed_images=processed_images,
        skipped_images=skipped_images,
        generated_crops=generated_crops,
        generated_labels=generated_labels,
        details=details,
    )
