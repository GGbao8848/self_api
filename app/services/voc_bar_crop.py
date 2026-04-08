"""Crop square patches from VOC images: side = source image height (横向长条图以整图高度为准), centered on each target box."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import VocBarCropDetail, VocBarCropRequest, VocBarCropResponse
from app.services.task_manager import ensure_current_task_active
from app.services.xml_to_yolo import _load_xml_annotation, _resolve_image_path
from app.utils.images import normalize_extensions

_SAVE_FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}


def _save_crop(crop: Image.Image, output_path: Path, output_ext: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = _SAVE_FORMAT_MAP.get(output_ext.lower(), "PNG")
    if save_format == "JPEG" and crop.mode not in {"RGB", "L"}:
        crop = crop.convert("RGB")
    crop.save(output_path, format=save_format)


def _write_voc_xml(
    path: Path,
    filename: str,
    size: tuple[int, int],
    objects: list[tuple[str, tuple[int, int, int, int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ElementTree.Element("annotation")
    fn = ElementTree.SubElement(root, "filename")
    fn.text = filename
    size_node = ElementTree.SubElement(root, "size")
    wn = ElementTree.SubElement(size_node, "width")
    wn.text = str(size[0])
    hn = ElementTree.SubElement(size_node, "height")
    hn.text = str(size[1])
    dn = ElementTree.SubElement(size_node, "depth")
    dn.text = "3"
    for class_name, (xmin, ymin, xmax, ymax) in objects:
        obj = ElementTree.SubElement(root, "object")
        ElementTree.SubElement(obj, "name").text = class_name
        ElementTree.SubElement(obj, "difficult").text = "0"
        bb = ElementTree.SubElement(obj, "bndbox")
        ElementTree.SubElement(bb, "xmin").text = str(xmin)
        ElementTree.SubElement(bb, "ymin").text = str(ymin)
        ElementTree.SubElement(bb, "xmax").text = str(xmax)
        ElementTree.SubElement(bb, "ymax").text = str(ymax)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _intersect_in_crop(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    left: int,
    top: int,
    side: int,
) -> tuple[int, int, int, int] | None:
    ix1 = max(xmin, left)
    iy1 = max(ymin, top)
    ix2 = min(xmax, left + side)
    iy2 = min(ymax, top + side)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    nx1 = ix1 - left
    ny1 = iy1 - top
    nx2 = ix2 - left
    ny2 = iy2 - top
    nx1 = max(0, min(nx1, side))
    ny1 = max(0, min(ny1, side))
    nx2 = max(0, min(nx2, side))
    ny2 = max(0, min(ny2, side))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def run_voc_bar_crop(request: VocBarCropRequest) -> VocBarCropResponse:
    images_dir = resolve_safe_path(
        request.images_dir,
        field_name="images_dir",
        must_exist=True,
        expect_directory=True,
    )
    xmls_dir = resolve_safe_path(
        request.xmls_dir,
        field_name="xmls_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = resolve_safe_path(request.output_dir, field_name="output_dir")
    out_images = output_dir / "images"
    out_xmls = output_dir / "xmls"

    iterator = xmls_dir.rglob("*.xml") if request.recursive else xmls_dir.glob("*.xml")
    xml_paths = sorted(p for p in iterator if p.is_file())

    details: list[VocBarCropDetail] = []
    processed_xml = 0
    skipped_xml = 0
    generated = 0

    for xml_path in xml_paths:
        ensure_current_task_active()
        rel_xml = xml_path.relative_to(xmls_dir)
        iw, ih, objects, err = _load_xml_annotation(
            xml_path=xml_path,
            images_dir=images_dir,
            include_difficult=False,
        )
        if err or not objects:
            details.append(
                VocBarCropDetail(
                    source_image="",
                    source_xml=str(xml_path),
                    skipped_reason=err or "no objects",
                )
            )
            skipped_xml += 1
            continue

        image_path = _resolve_image_path(images_dir, None, xml_path.stem)
        if image_path is None or not image_path.is_file():
            details.append(
                VocBarCropDetail(
                    source_image="",
                    source_xml=str(xml_path),
                    skipped_reason="image not found for xml",
                )
            )
            skipped_xml += 1
            continue

        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except (UnidentifiedImageError, OSError) as exc:
            details.append(
                VocBarCropDetail(
                    source_image=str(image_path),
                    source_xml=str(xml_path),
                    skipped_reason=f"failed to open image: {exc}",
                )
            )
            skipped_xml += 1
            continue

        if iw != img_w or ih != img_h:
            iw, ih = img_w, img_h

        processed_xml += 1
        stem = image_path.stem
        ext = image_path.suffix.lower()
        if ext not in normalize_extensions(None):
            ext = ".png"

        crop_count_for_file = 0
        for class_name, (xmin, ymin, xmax, ymax) in objects:
            ensure_current_task_active()
            bw = xmax - xmin
            bh = ymax - ymin
            if bw < bh:
                continue

            # 正方形边长 = 整图高度（与 yolo-sliding-window-crop 一致）；宽图 top=0，水平按框中心对齐。
            if img_w >= img_h:
                side = img_h
                top = 0
                cx = (xmin + xmax) / 2.0
                left = int(round(cx - side / 2.0))
                left = max(0, min(left, img_w - side))
            else:
                side = img_w
                left = 0
                cy = (ymin + ymax) / 2.0
                top = int(round(cy - side / 2.0))
                top = max(0, min(top, img_h - side))

            crop_objects: list[tuple[str, tuple[int, int, int, int]]] = []
            for c2, box in objects:
                clipped = _intersect_in_crop(*box, left, top, side)
                if clipped is not None:
                    crop_objects.append((c2, clipped))

            if not crop_objects:
                continue

            cx_i = left + side // 2
            cy_i = top + side // 2
            base_name = f"{stem}_cx{cx_i}_cy{cy_i}_S{side}"
            out_name = base_name + ext
            rel_parent = rel_xml.parent
            crop_img_path = out_images / rel_parent / out_name
            crop_xml_path = out_xmls / rel_parent / f"{base_name}.xml"

            try:
                with Image.open(image_path) as im_full:
                    im_rgb = im_full.convert("RGB")
                    crop = im_rgb.crop((left, top, left + side, top + side))
                _save_crop(crop, crop_img_path, ext)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                details.append(
                    VocBarCropDetail(
                        source_image=str(image_path),
                        source_xml=str(xml_path),
                        skipped_reason=f"crop failed: {exc}",
                    )
                )
                continue

            _write_voc_xml(
                crop_xml_path,
                filename=out_name,
                size=(side, side),
                objects=crop_objects,
            )
            generated += 1
            crop_count_for_file += 1
            details.append(
                VocBarCropDetail(
                    source_image=str(image_path),
                    source_xml=str(xml_path),
                    crop_image=str(crop_img_path),
                    crop_xml=str(crop_xml_path),
                    window_left=left,
                    window_top=top,
                    window_size=side,
                )
            )

        if crop_count_for_file == 0:
            details.append(
                VocBarCropDetail(
                    source_image=str(image_path),
                    source_xml=str(xml_path),
                    skipped_reason="no horizontal bar objects or empty crops",
                )
            )

    return VocBarCropResponse(
        images_dir=str(images_dir),
        xmls_dir=str(xmls_dir),
        output_dir=str(output_dir),
        processed_xml_files=processed_xml,
        skipped_xml_files=skipped_xml,
        generated_crops=generated,
        details=details,
    )
