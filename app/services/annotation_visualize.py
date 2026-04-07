"""Draw bounding boxes on images from YOLO txt or Pascal VOC XML annotations."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    AnnotateVisualizeDetail,
    AnnotateVisualizeRequest,
    AnnotateVisualizeResponse,
)
from app.services.task_manager import ensure_current_task_active
from app.services.xml_to_yolo import _load_xml_annotation
from app.utils.images import list_image_paths, normalize_extensions

_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 180, 64),
    (200, 64, 200),
    (64, 200, 200),
    (180, 180, 64),
    (255, 128, 192),
]


def _color_for_index(idx: int) -> tuple[int, int, int]:
    return _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]


def _load_font() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, bool]:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14), True
    except OSError:
        f = ImageFont.load_default()
        return f, False


def _parse_yolo_lines(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def _yolo_norm_to_pixels(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    x1 = int(max(0, min(x1, img_w - 1)))
    y1 = int(max(0, min(y1, img_h - 1)))
    x2 = int(max(0, min(x2, img_w - 1)))
    y2 = int(max(0, min(y2, img_h - 1)))
    return x1, y1, x2, y2


def _class_label_for_yolo(cls_id: int, names: list[str] | None) -> str:
    if names and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def _resolve_class_names(request: AnnotateVisualizeRequest) -> list[str] | None:
    classes_file = (request.classes_file or "").strip()
    if classes_file:
        path = resolve_safe_path(
            classes_file,
            field_name="classes_file",
            must_exist=True,
            expect_file=True,
        )
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return lines
    if request.classes:
        return list(request.classes)
    return None


def _draw_labeled_boxes(
    image: Image.Image,
    boxes: list[tuple[str, tuple[int, int, int, int]]],
    line_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_is_ttf: bool,
) -> None:
    draw = ImageDraw.Draw(image)
    for idx, (label, (x1, y1, x2, y2)) in enumerate(boxes):
        color = _color_for_index(idx)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        ty = max(0, y1 - 16) if font_is_ttf else max(0, y1 - 12)
        draw.text((x1 + 2, ty), label, fill=color, font=font)


def _boxes_from_yolo(
    label_path: Path,
    img_w: int,
    img_h: int,
    yolo_names: list[str] | None,
) -> tuple[list[tuple[str, tuple[int, int, int, int]]], str | None]:
    try:
        label_text = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"read label failed: {exc}"
    boxes_norm = _parse_yolo_lines(label_text)
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for cls_id, xc, yc, bw, bh in boxes_norm:
        x1, y1, x2, y2 = _yolo_norm_to_pixels(xc, yc, bw, bh, img_w, img_h)
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        label = _class_label_for_yolo(cls_id, yolo_names)
        out.append((label, (x1, y1, x2, y2)))
    return out, None


def _boxes_from_xml(
    xml_path: Path,
    images_dir: Path,
    include_difficult: bool,
) -> tuple[list[tuple[str, tuple[int, int, int, int]]], str | None]:
    _w, _h, objects, err = _load_xml_annotation(
        xml_path=xml_path,
        images_dir=images_dir,
        include_difficult=include_difficult,
    )
    if err:
        return [], err
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for class_name, (xmin, ymin, xmax, ymax) in objects:
        out.append((class_name, (xmin, ymin, xmax, ymax)))
    return out, None


def run_annotate_visualize(request: AnnotateVisualizeRequest) -> AnnotateVisualizeResponse:
    images_dir = resolve_safe_path(
        request.images_dir,
        field_name="images_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = resolve_safe_path(
        request.output_dir,
        field_name="output_dir",
        must_exist=False,
        expect_directory=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_names = _resolve_class_names(request)

    use_yolo = bool((request.labels_dir or "").strip())
    if use_yolo:
        labels_dir = resolve_safe_path(
            request.labels_dir or "",
            field_name="labels_dir",
            must_exist=True,
            expect_directory=True,
        )
        mode = "yolo"
        annotation_dir = str(labels_dir)
        xmls_dir: Path | None = None
    else:
        xmls_dir = resolve_safe_path(
            request.xmls_dir or "",
            field_name="xmls_dir",
            must_exist=True,
            expect_directory=True,
        )
        mode = "xml"
        annotation_dir = str(xmls_dir)
        labels_dir = None

    normalized_exts = normalize_extensions(request.extensions)
    image_paths = list_image_paths(
        images_dir,
        recursive=request.recursive,
        extensions=normalized_exts,
    )

    font, font_is_ttf = _load_font()
    details: list[AnnotateVisualizeDetail] = []
    written = 0
    skipped = 0

    for image_path in image_paths:
        ensure_current_task_active()
        rel = image_path.relative_to(images_dir)
        out_path = output_dir / rel

        if use_yolo:
            assert labels_dir is not None
            label_path = (labels_dir / rel).with_suffix(".txt")
            if not label_path.is_file():
                skipped += 1
                details.append(
                    AnnotateVisualizeDetail(
                        source_image=str(image_path),
                        output_image=None,
                        boxes_drawn=0,
                        skipped_reason=f"missing label file: {label_path}",
                    )
                )
                continue
            ann_path = label_path
        else:
            assert xmls_dir is not None
            xml_path = (xmls_dir / rel).with_suffix(".xml")
            if not xml_path.is_file():
                skipped += 1
                details.append(
                    AnnotateVisualizeDetail(
                        source_image=str(image_path),
                        output_image=None,
                        boxes_drawn=0,
                        skipped_reason=f"missing xml file: {xml_path}",
                    )
                )
                continue
            ann_path = xml_path

        if out_path.exists() and not request.overwrite:
            skipped += 1
            details.append(
                AnnotateVisualizeDetail(
                    source_image=str(image_path),
                    output_image=None,
                    boxes_drawn=0,
                    skipped_reason="output image already exists",
                )
            )
            continue

        try:
            with Image.open(image_path) as image:
                img_w, img_h = image.size
                image = image.convert("RGB")
                if use_yolo:
                    boxes, err = _boxes_from_yolo(ann_path, img_w, img_h, yolo_names)
                else:
                    boxes, err = _boxes_from_xml(
                        ann_path, images_dir, request.include_difficult
                    )
                if err:
                    skipped += 1
                    details.append(
                        AnnotateVisualizeDetail(
                            source_image=str(image_path),
                            output_image=None,
                            boxes_drawn=0,
                            skipped_reason=err,
                        )
                    )
                    continue

                _draw_labeled_boxes(
                    image, boxes, request.line_width, font, font_is_ttf
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(out_path)
                written += 1
                details.append(
                    AnnotateVisualizeDetail(
                        source_image=str(image_path),
                        output_image=str(out_path),
                        boxes_drawn=len(boxes),
                        skipped_reason=None,
                    )
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped += 1
            details.append(
                AnnotateVisualizeDetail(
                    source_image=str(image_path),
                    output_image=None,
                    boxes_drawn=0,
                    skipped_reason=f"image error: {exc}",
                )
            )

    return AnnotateVisualizeResponse(
        mode=mode,
        images_dir=str(images_dir),
        annotation_dir=annotation_dir,
        output_dir=str(output_dir),
        total_images=len(image_paths),
        written_images=written,
        skipped_images=skipped,
        details=details,
    )
