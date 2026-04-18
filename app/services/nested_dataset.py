import hashlib
import json
import shutil
from pathlib import Path

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    AggregateNestedDatasetItemDetail,
    AggregateNestedDatasetRequest,
    AggregateNestedDatasetResponse,
    CleanNestedDatasetLeafDetail,
    CleanNestedDatasetRequest,
    CleanNestedDatasetResponse,
    DiscoverLeafDirsRequest,
    DiscoverLeafDirsResponse,
)
from app.services.task_manager import ensure_current_task_active
from app.services.xml_to_yolo import _load_xml_annotation
from app.utils.images import list_image_paths, normalize_extensions


def _is_within(path: Path, parent: Path) -> bool:
    resolved = path.resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    return resolved == resolved_parent or resolved_parent in resolved.parents


def _iter_direct_annotation_files(directory: Path, image_exts: set[str]) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    xmls: list[Path] = []
    for child in sorted(directory.iterdir()):
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix in image_exts:
            images.append(child)
        elif suffix == ".xml":
            xmls.append(child)
    return images, xmls


def _discover_leaf_dirs(
    input_dir: Path,
    *,
    recursive: bool,
    image_exts: set[str],
    skip_dirs: list[Path] | None = None,
) -> list[Path]:
    skip_dirs = skip_dirs or []
    candidates: list[Path] = []

    def _should_skip(path: Path) -> bool:
        return any(_is_within(path, skip_dir) for skip_dir in skip_dirs)

    if recursive:
        for path in [input_dir, *sorted(input_dir.rglob("*"))]:
            ensure_current_task_active()
            if not path.is_dir() or _should_skip(path):
                continue
            images, xmls = _iter_direct_annotation_files(path, image_exts)
            if images or xmls:
                candidates.append(path)
    else:
        images, xmls = _iter_direct_annotation_files(input_dir, image_exts)
        if images or xmls:
            candidates.append(input_dir)

    leaves = [
        path
        for path in candidates
        if not any(path != other and path in other.parents for other in candidates)
    ]
    return sorted(leaves)


def _discover_images_xmls_pair_roots(
    input_dir: Path,
    *,
    recursive: bool,
    images_dir_name: str,
    xmls_dir_name: str,
    image_exts: set[str],
    skip_dirs: list[Path] | None = None,
) -> list[Path]:
    """Directories that directly contain both ``images_dir_name`` and ``xmls_dir_name`` subfolders with at least one file."""
    skip_dirs = skip_dirs or []

    def _should_skip(path: Path) -> bool:
        return any(_is_within(path, skip_dir) for skip_dir in skip_dirs)

    if recursive:
        dirs_to_visit = sorted({input_dir, *input_dir.rglob("*")})
        dirs_to_visit = [p for p in dirs_to_visit if p.is_dir()]
    else:
        dirs_to_visit = [input_dir]

    candidates: list[Path] = []
    for path in dirs_to_visit:
        ensure_current_task_active()
        if _should_skip(path):
            continue
        img_sub = path / images_dir_name
        xml_sub = path / xmls_dir_name
        if not img_sub.is_dir() or not xml_sub.is_dir():
            continue
        imgs, _ = _iter_direct_annotation_files(img_sub, image_exts)
        _, xmls = _iter_direct_annotation_files(xml_sub, image_exts)
        if not imgs and not xmls:
            continue
        candidates.append(path)
    return sorted(candidates)


def run_discover_leaf_dirs(request: DiscoverLeafDirsRequest) -> DiscoverLeafDirsResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    leaf_dirs = _discover_leaf_dirs(
        input_dir,
        recursive=request.recursive,
        image_exts=normalize_extensions(request.extensions),
    )
    return DiscoverLeafDirsResponse(
        input_dir=str(input_dir),
        total_leaf_dirs=len(leaf_dirs),
        leaf_dirs=[str(path) for path in leaf_dirs],
    )


def _remove_existing_target(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _flatten_basename(rel_dir: Path, filename: str) -> str:
    """Prefix filename with sanitized relative path so flattened outputs do not collide."""
    key = rel_dir.as_posix()
    if key == ".":
        key = "root"
    else:
        key = key.replace("/", "__")
    return f"{key}__{filename}"


def _copy_or_move_file(source_path: Path, target_path: Path, *, copy_files: bool, overwrite: bool) -> None:
    if target_path.exists():
        if not overwrite:
            raise ValueError(f"target path already exists: {target_path}")
        _remove_existing_target(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source_path, target_path)
    else:
        shutil.move(str(source_path), str(target_path))


def run_clean_nested_dataset(request: CleanNestedDatasetRequest) -> CleanNestedDatasetResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = (
        resolve_safe_path(request.output_dir, field_name="output_dir")
        if request.output_dir
        else (input_dir / "cleaned_dataset").resolve()
    )
    image_exts = normalize_extensions(request.extensions)
    skip_dirs = [output_dir] if _is_within(output_dir, input_dir) else []
    pairing_mode = request.pairing_mode
    if pairing_mode == "auto":
        split_roots = _discover_images_xmls_pair_roots(
            input_dir,
            recursive=request.recursive,
            images_dir_name=request.images_dir_name,
            xmls_dir_name=request.xmls_dir_name,
            image_exts=image_exts,
            skip_dirs=skip_dirs,
        )
        pairing_mode = "images_xmls_subfolders" if split_roots else "same_directory"

    if pairing_mode == "same_directory":
        leaf_dirs = _discover_leaf_dirs(
            input_dir,
            recursive=request.recursive,
            image_exts=image_exts,
            skip_dirs=skip_dirs,
        )
    else:
        leaf_dirs = _discover_images_xmls_pair_roots(
            input_dir,
            recursive=request.recursive,
            images_dir_name=request.images_dir_name,
            xmls_dir_name=request.xmls_dir_name,
            image_exts=image_exts,
            skip_dirs=skip_dirs,
        )

    details: list[CleanNestedDatasetLeafDetail] = []
    processed_leaf_dirs = 0
    total_images = 0
    labeled_images = 0
    background_images = 0
    skipped_unlabeled_images = 0
    copied_xml_files = 0
    empty_or_invalid_xml_files = 0
    orphan_xml_files = 0

    for unit_dir in leaf_dirs:
        ensure_current_task_active()
        rel_dir = unit_dir.relative_to(input_dir)
        leaf_output_dir = output_dir if request.flatten else output_dir / rel_dir
        if pairing_mode == "same_directory":
            images_src = unit_dir
            xmls_src = unit_dir
            images, xml_paths = _iter_direct_annotation_files(images_src, image_exts)
        else:
            images_src = unit_dir / request.images_dir_name
            xmls_src = unit_dir / request.xmls_dir_name
            images, _ = _iter_direct_annotation_files(images_src, image_exts)
            _, xml_paths = _iter_direct_annotation_files(xmls_src, image_exts)
        xml_by_stem = {path.stem: path for path in xml_paths}
        image_by_stem = {path.stem: path for path in images}

        leaf_total_images = len(images)
        leaf_labeled_images = 0
        leaf_background_images = 0
        leaf_skipped_unlabeled = 0
        leaf_copied_xml_files = 0
        leaf_empty_or_invalid_xml_files = 0
        leaf_orphan_xml_files = 0

        xml_validity: dict[str, bool] = {}
        for stem, xml_path in xml_by_stem.items():
            ensure_current_task_active()
            _, _, objects, error = _load_xml_annotation(
                xml_path=xml_path,
                images_dir=images_src,
                include_difficult=request.include_difficult,
            )
            xml_validity[stem] = error is None and len(objects) > 0
            if not xml_validity[stem]:
                leaf_empty_or_invalid_xml_files += 1

        for image_path in images:
            ensure_current_task_active()
            matched_xml = xml_by_stem.get(image_path.stem)
            if matched_xml and xml_validity.get(image_path.stem, False):
                image_name = (
                    _flatten_basename(rel_dir, image_path.name)
                    if request.flatten
                    else image_path.name
                )
                xml_name = (
                    _flatten_basename(rel_dir, matched_xml.name)
                    if request.flatten
                    else matched_xml.name
                )
                target_image = leaf_output_dir / request.images_dir_name / image_name
                target_xml = leaf_output_dir / request.xmls_dir_name / xml_name
                _copy_or_move_file(
                    image_path,
                    target_image,
                    copy_files=request.copy_files,
                    overwrite=request.overwrite,
                )
                _copy_or_move_file(
                    matched_xml,
                    target_xml,
                    copy_files=request.copy_files,
                    overwrite=request.overwrite,
                )
                leaf_labeled_images += 1
                leaf_copied_xml_files += 1
            elif request.include_backgrounds:
                bg_name = (
                    _flatten_basename(rel_dir, image_path.name) if request.flatten else image_path.name
                )
                target_background = leaf_output_dir / request.backgrounds_dir_name / bg_name
                _copy_or_move_file(
                    image_path,
                    target_background,
                    copy_files=request.copy_files,
                    overwrite=request.overwrite,
                )
                leaf_background_images += 1
            else:
                leaf_skipped_unlabeled += 1

        for stem, xml_path in xml_by_stem.items():
            if stem not in image_by_stem:
                ensure_current_task_active()
                leaf_orphan_xml_files += 1
                if not request.copy_files:
                    xml_path.unlink(missing_ok=True)

        processed_leaf_dirs += 1
        total_images += leaf_total_images
        labeled_images += leaf_labeled_images
        background_images += leaf_background_images
        skipped_unlabeled_images += leaf_skipped_unlabeled
        copied_xml_files += leaf_copied_xml_files
        empty_or_invalid_xml_files += leaf_empty_or_invalid_xml_files
        orphan_xml_files += leaf_orphan_xml_files
        details.append(
            CleanNestedDatasetLeafDetail(
                source_dir=str(unit_dir),
                output_dir=str(leaf_output_dir),
                total_images=leaf_total_images,
                labeled_images=leaf_labeled_images,
                background_images=leaf_background_images,
                skipped_unlabeled_images=leaf_skipped_unlabeled,
                copied_xml_files=leaf_copied_xml_files,
                empty_or_invalid_xml_files=leaf_empty_or_invalid_xml_files,
                orphan_xml_files=leaf_orphan_xml_files,
            )
        )

    return CleanNestedDatasetResponse(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        discovered_leaf_dirs=len(leaf_dirs),
        processed_leaf_dirs=processed_leaf_dirs,
        total_images=total_images,
        labeled_images=labeled_images,
        background_images=background_images,
        skipped_unlabeled_images=skipped_unlabeled_images,
        copied_xml_files=copied_xml_files,
        empty_or_invalid_xml_files=empty_or_invalid_xml_files,
        orphan_xml_files=orphan_xml_files,
        details=details,
    )


def _discover_fragment_dirs(
    input_dir: Path,
    *,
    recursive: bool,
    images_dir_name: str,
    labels_dir_name: str,
    backgrounds_dir_name: str,
    skip_dirs: list[Path] | None = None,
) -> list[Path]:
    skip_dirs = skip_dirs or []
    directories = [input_dir] if not recursive else [input_dir, *sorted(input_dir.rglob("*"))]
    fragments: list[Path] = []

    for path in directories:
        ensure_current_task_active()
        if not path.is_dir():
            continue
        if any(_is_within(path, skip_dir) for skip_dir in skip_dirs):
            continue
        if (
            (path / images_dir_name).is_dir()
            or (path / labels_dir_name).is_dir()
            or (path / backgrounds_dir_name).is_dir()
        ):
            fragments.append(path)
    return sorted(fragments)


def _sanitize_stem(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "sample"


def _build_unique_stem(fragment_rel: Path, item_rel: Path, source_path: Path) -> str:
    prefix = "__".join([_sanitize_stem(part) for part in fragment_rel.parts if part not in {"."}])
    item_stem = _sanitize_stem("__".join(item_rel.with_suffix("").parts))
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
    parts = [part for part in [prefix, item_stem, digest] if part]
    return "__".join(parts)


def _read_classes_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _remap_label_content(content: str, local_classes: list[str], global_class_to_id: dict[str, int]) -> str:
    lines: list[str] = []
    for raw_line in content.splitlines():
        text = raw_line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        try:
            class_idx = int(parts[0])
        except ValueError:
            continue
        if class_idx < 0 or class_idx >= len(local_classes):
            continue
        class_name = local_classes[class_idx]
        global_idx = global_class_to_id[class_name]
        lines.append(" ".join([str(global_idx), *parts[1:]]))
    return "\n".join(lines)


def run_aggregate_nested_dataset(request: AggregateNestedDatasetRequest) -> AggregateNestedDatasetResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    output_dir = (
        resolve_safe_path(request.output_dir, field_name="output_dir")
        if request.output_dir
        else (input_dir / "dataset").resolve()
    )
    image_exts = normalize_extensions(request.extensions)
    fragment_dirs = _discover_fragment_dirs(
        input_dir,
        recursive=request.recursive,
        images_dir_name=request.images_dir_name,
        labels_dir_name=request.labels_dir_name,
        backgrounds_dir_name=request.backgrounds_dir_name,
        skip_dirs=[output_dir] if _is_within(output_dir, input_dir) else [],
    )

    fragment_classes: dict[Path, list[str]] = {}
    global_classes: list[str] = []
    for fragment_dir in fragment_dirs:
        ensure_current_task_active()
        labels_dir = fragment_dir / request.labels_dir_name
        if not labels_dir.is_dir():
            continue
        classes_path = fragment_dir / request.classes_file_name
        if not classes_path.exists():
            raise ValueError(f"classes file is required for fragment with labels: {classes_path}")
        local_classes = _read_classes_file(classes_path)
        fragment_classes[fragment_dir] = local_classes
        for class_name in local_classes:
            if class_name not in global_classes:
                global_classes.append(class_name)

    global_class_to_id = {name: idx for idx, name in enumerate(global_classes)}
    output_images_dir = output_dir / request.images_dir_name
    output_labels_dir = output_dir / request.labels_dir_name
    output_backgrounds_dir = output_dir / request.backgrounds_dir_name
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_backgrounds_dir.mkdir(parents=True, exist_ok=True)

    details: list[AggregateNestedDatasetItemDetail] = []
    manifest_items: list[dict[str, str | None]] = []
    aggregated_images = 0
    aggregated_backgrounds = 0
    skipped_images = 0

    for fragment_dir in fragment_dirs:
        ensure_current_task_active()
        fragment_rel = fragment_dir.relative_to(input_dir)
        images_dir = fragment_dir / request.images_dir_name
        labels_dir = fragment_dir / request.labels_dir_name
        backgrounds_dir = fragment_dir / request.backgrounds_dir_name
        local_classes = fragment_classes.get(fragment_dir, [])

        if images_dir.is_dir():
            for image_path in list_image_paths(images_dir, recursive=True, extensions=image_exts):
                ensure_current_task_active()
                rel_path = image_path.relative_to(images_dir)
                label_path = (labels_dir / rel_path).with_suffix(".txt")
                if not label_path.exists():
                    skipped_images += 1
                    details.append(
                        AggregateNestedDatasetItemDetail(
                            source_path=str(image_path),
                            item_type="image",
                            skipped_reason="paired label file not found",
                        )
                    )
                    continue
                label_content = label_path.read_text(encoding="utf-8")
                if request.require_non_empty_labels and not label_content.strip():
                    skipped_images += 1
                    details.append(
                        AggregateNestedDatasetItemDetail(
                            source_path=str(image_path),
                            item_type="image",
                            skipped_reason="paired label file is empty",
                        )
                    )
                    continue

                unique_stem = _build_unique_stem(fragment_rel, rel_path, image_path)
                target_image = output_images_dir / f"{unique_stem}{image_path.suffix.lower()}"
                target_label = output_labels_dir / f"{unique_stem}.txt"
                if (target_image.exists() or target_label.exists()) and not request.overwrite:
                    raise ValueError(f"aggregated target already exists: {target_image}")

                remapped_content = _remap_label_content(
                    label_content,
                    local_classes,
                    global_class_to_id,
                )
                if request.require_non_empty_labels and not remapped_content.strip():
                    skipped_images += 1
                    details.append(
                        AggregateNestedDatasetItemDetail(
                            source_path=str(image_path),
                            item_type="image",
                            skipped_reason="paired label file has no valid rows after remap",
                        )
                    )
                    continue

                _copy_or_move_file(image_path, target_image, copy_files=True, overwrite=request.overwrite)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                if target_label.exists() and request.overwrite:
                    target_label.unlink()
                target_label.write_text(remapped_content, encoding="utf-8")
                aggregated_images += 1
                details.append(
                    AggregateNestedDatasetItemDetail(
                        source_path=str(image_path),
                        target_path=str(target_image),
                        item_type="image",
                    )
                )
                manifest_items.append(
                    {
                        "item_type": "image",
                        "source_path": str(image_path),
                        "target_path": str(target_image),
                        "label_source_path": str(label_path),
                        "label_target_path": str(target_label),
                    }
                )

        if backgrounds_dir.is_dir():
            for background_path in list_image_paths(
                backgrounds_dir,
                recursive=True,
                extensions=image_exts,
            ):
                ensure_current_task_active()
                rel_path = background_path.relative_to(backgrounds_dir)
                unique_stem = _build_unique_stem(fragment_rel / request.backgrounds_dir_name, rel_path, background_path)
                target_background = output_backgrounds_dir / f"{unique_stem}{background_path.suffix.lower()}"
                _copy_or_move_file(
                    background_path,
                    target_background,
                    copy_files=True,
                    overwrite=request.overwrite,
                )
                aggregated_backgrounds += 1
                details.append(
                    AggregateNestedDatasetItemDetail(
                        source_path=str(background_path),
                        target_path=str(target_background),
                        item_type="background",
                    )
                )
                manifest_items.append(
                    {
                        "item_type": "background",
                        "source_path": str(background_path),
                        "target_path": str(target_background),
                        "label_source_path": None,
                        "label_target_path": None,
                    }
                )

    classes_file = None
    if global_classes:
        classes_path = output_dir / request.classes_file_name
        classes_path.write_text("\n".join(global_classes), encoding="utf-8")
        classes_file = str(classes_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": global_classes,
                "items": manifest_items,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    return AggregateNestedDatasetResponse(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fragment_dirs=len(fragment_dirs),
        aggregated_images=aggregated_images,
        aggregated_backgrounds=aggregated_backgrounds,
        skipped_images=skipped_images,
        classes=global_classes,
        class_to_id=global_class_to_id,
        classes_file=classes_file,
        manifest_path=str(manifest_path),
        details=details,
    )
