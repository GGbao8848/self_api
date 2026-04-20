from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    XmlToYoloFileDetail,
    XmlToYoloRequest,
    XmlToYoloResponse,
)
from app.services.task_manager import ensure_current_task_active

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _resolve_image_path(images_dir: Path, filename: str | None, xml_stem: str) -> Path | None:
    if filename:
        filename_path = Path(filename)
        if filename_path.is_absolute() and filename_path.exists():
            return filename_path

        direct = images_dir / filename
        if direct.exists():
            return direct

        by_name = images_dir / filename_path.name
        if by_name.exists():
            return by_name

    for suffix in _IMAGE_SUFFIXES:
        candidate = images_dir / f"{xml_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_xml_annotation(
    xml_path: Path,
    images_dir: Path,
    include_difficult: bool,
) -> tuple[int, int, list[tuple[str, tuple[int, int, int, int]]], str | None]:
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError as exc:
        return 0, 0, [], f"invalid xml: {exc}"

    filename = root.findtext("filename")
    image_width = _to_int(root.findtext("./size/width"))
    image_height = _to_int(root.findtext("./size/height"))

    if not image_width or not image_height:
        image_path = _resolve_image_path(images_dir, filename, xml_path.stem)
        if image_path is None:
            return 0, 0, [], "missing size in xml and image not found"
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (UnidentifiedImageError, OSError) as exc:
            return 0, 0, [], f"failed to read image size: {exc}"

    objects: list[tuple[str, tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        difficult = (obj.findtext("difficult") or "").strip().lower()
        if not include_difficult and difficult in {"1", "true"}:
            continue

        class_name = (obj.findtext("name") or "").strip()
        if not class_name:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = _to_int(bbox.findtext("xmin"))
        ymin = _to_int(bbox.findtext("ymin"))
        xmax = _to_int(bbox.findtext("xmax"))
        ymax = _to_int(bbox.findtext("ymax"))
        if xmin is None or ymin is None or xmax is None or ymax is None:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue

        xmin = max(0, min(xmin, image_width))
        xmax = max(0, min(xmax, image_width))
        ymin = max(0, min(ymin, image_height))
        ymax = max(0, min(ymax, image_height))
        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append((class_name, (xmin, ymin, xmax, ymax)))

    return image_width, image_height, objects, None


def run_xml_to_yolo(request: XmlToYoloRequest) -> XmlToYoloResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )

    images_dir = input_dir / request.images_dir_name
    xmls_dir = input_dir / request.xmls_dir_name
    labels_dir = input_dir / request.labels_dir_name

    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"images_dir does not exist or is not a directory: {images_dir}")
    if not xmls_dir.exists() or not xmls_dir.is_dir():
        raise ValueError(f"xmls_dir does not exist or is not a directory: {xmls_dir}")

    iterator = xmls_dir.rglob("*.xml") if request.recursive else xmls_dir.glob("*.xml")
    xml_paths = sorted([path for path in iterator if path.is_file()])

    parsed_entries: list[
        tuple[Path, int, int, list[tuple[str, tuple[int, int, int, int]]], str | None]
    ] = []
    discovered_classes: set[str] = set()
    details: list[XmlToYoloFileDetail] = []

    for xml_path in xml_paths:
        ensure_current_task_active()
        width, height, objects, error = _load_xml_annotation(
            xml_path=xml_path,
            images_dir=images_dir,
            include_difficult=request.include_difficult,
        )
        parsed_entries.append((xml_path, width, height, objects, error))
        if error:
            continue
        name_map = request.class_name_map or {}
        for class_name, _ in objects:
            mapped = name_map.get(class_name, class_name)
            discovered_classes.add(mapped)

    if request.classes:
        ordered_classes = []
        seen: set[str] = set()
        for name in request.classes:
            normalized = name.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered_classes.append(normalized)
    else:
        ordered_classes = sorted(discovered_classes)

    class_to_id = {name: idx for idx, name in enumerate(ordered_classes)}

    converted_files = 0
    skipped_files = 0
    total_boxes = 0

    for xml_path, width, height, objects, error in parsed_entries:
        ensure_current_task_active()
        rel_xml = xml_path.relative_to(xmls_dir)
        label_path = (labels_dir / rel_xml).with_suffix(".txt")

        if error:
            skipped_files += 1
            details.append(
                XmlToYoloFileDetail(
                    source_xml=str(xml_path),
                    output_label=str(label_path),
                    written_lines=0,
                    skipped_reason=error,
                )
            )
            continue

        if width <= 0 or height <= 0:
            skipped_files += 1
            details.append(
                XmlToYoloFileDetail(
                    source_xml=str(xml_path),
                    output_label=str(label_path),
                    written_lines=0,
                    skipped_reason="invalid image size",
                )
            )
            continue

        if label_path.exists() and not request.overwrite:
            skipped_files += 1
            details.append(
                XmlToYoloFileDetail(
                    source_xml=str(xml_path),
                    output_label=str(label_path),
                    written_lines=0,
                    skipped_reason="label file already exists",
                )
            )
            continue

        lines: list[str] = []
        name_map = request.class_name_map or {}
        for class_name, (xmin, ymin, xmax, ymax) in objects:
            class_name = name_map.get(class_name, class_name)
            if class_name not in class_to_id:
                continue

            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            if box_width <= 0 or box_height <= 0:
                continue

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            box_width = min(max(box_width, 0.0), 1.0)
            box_height = min(max(box_height, 0.0), 1.0)

            lines.append(
                f"{class_to_id[class_name]} "
                f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

        converted_files += 1
        total_boxes += len(lines)
        details.append(
            XmlToYoloFileDetail(
                source_xml=str(xml_path),
                output_label=str(label_path),
                written_lines=len(lines),
            )
        )

    classes_file_output = None
    if request.write_classes_file:
        classes_path = input_dir / request.classes_file_name
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        classes_path.write_text("\n".join(ordered_classes), encoding="utf-8")
        classes_file_output = str(classes_path)

    return XmlToYoloResponse(
        input_dir=str(input_dir),
        labels_dir=str(labels_dir),
        total_xml_files=len(xml_paths),
        converted_files=converted_files,
        skipped_files=skipped_files,
        total_boxes=total_boxes,
        classes=ordered_classes,
        class_to_id=class_to_id,
        classes_file=classes_file_output,
        details=details,
    )
