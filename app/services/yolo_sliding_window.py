"""YOLO 正方形滑窗裁剪：窗口边长=图片高度，仅水平滑动。"""

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


def _sliding_square_crop(
    image_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
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

    win = H  # square side = image height
    step = max(1, int(round(stride_ratio * H)))

    labels = _read_yolo_labels(label_path)
    abs_boxes: list[tuple[int, float, float, float, float, float]] = []
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

    if W <= win:
        xs = [0]
    else:
        xs = list(range(0, W - win + 1, step))
        if xs[-1] != W - win:
            xs.append(W - win)

    for i, x0 in enumerate(xs):
        ensure_current_task_active()
        y0 = 0
        cx1, cy1 = float(x0), float(y0)
        cx2, cy2 = float(x0 + win), float(y0 + win)

        crop = img_arr.crop((x0, y0, x0 + win, y0 + win))
        if crop.size[0] == 0 or crop.size[1] == 0:
            continue

        kept_lines: list[str] = []
        has_keep = False
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
                nxc, nyc, nw, nh = _xyxy_to_yolo(lx1, ly1, lx2, ly2, win, win)
                if nw <= 0 or nh <= 0:
                    continue
                kept_lines.append(f"{cls} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")

        if has_danger:
            continue
        if not has_keep or not kept_lines:
            continue

        stem = image_path.stem
        out_name = f"{stem}_sqH_{win}_x{x0:06d}_{i:03d}"
        ext = image_path.suffix.lower()
        if ext not in _SAVE_FORMAT_MAP:
            ext = ".png"

        out_img_path = out_img_dir / f"{out_name}{ext}"
        out_lbl_path = out_lbl_dir / f"{out_name}.txt"

        save_format = _SAVE_FORMAT_MAP.get(ext, "PNG")
        if save_format == "JPEG" and crop.mode != "RGB":
            crop = crop.convert("RGB")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_img_path, format=save_format)

        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        saved += 1
        total_label_lines += len(kept_lines)

    return saved, total_label_lines


def run_yolo_sliding_window_crop(request: YoloSlidingWindowCropRequest) -> YoloSlidingWindowCropResponse:
    images_dir = resolve_safe_path(
        request.images_dir,
        field_name="images_dir",
        must_exist=True,
        expect_directory=True,
    )
    labels_dir = resolve_safe_path(
        request.labels_dir,
        field_name="labels_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = resolve_safe_path(request.output_dir, field_name="output_dir")

    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(
        images_dir,
        recursive=True,
        extensions=list(_IMG_EXTS),
    )

    details: list[YoloSlidingWindowCropDetail] = []
    processed_images = 0
    skipped_images = 0
    generated_crops = 0
    generated_labels = 0

    for image_path in image_paths:
        ensure_current_task_active()
        rel_image = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel_image).with_suffix(".txt")

        if not label_path.exists():
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
            saved, label_lines = _sliding_square_crop(
                image_path=image_path,
                label_path=label_path,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
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

    return YoloSlidingWindowCropResponse(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        output_dir=str(output_dir),
        input_images=len(image_paths),
        processed_images=processed_images,
        skipped_images=skipped_images,
        generated_crops=generated_crops,
        generated_labels=generated_labels,
        details=details,
    )
