"""Batch restore: apply all voc-bar-style edited crops back onto original images with merged VOC labels."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    RestoreVocCropsBatchResponse,
    RestoreVocCropsBatchStemDetail,
    RestoreVocCropsBatchRequest,
)
from app.services.task_manager import ensure_current_task_active
from app.services.voc_bar_crop import _write_voc_xml
from app.services.voc_crop_restore import (
    map_small_voc_to_large,
    parse_voc_bar_crop_stem,
    region_overlaps_box,
    region_xywh_from_cx_cy_s,
    resolve_original_image,
)
from app.services.xml_to_yolo import _load_xml_annotation
from app.utils.images import list_image_paths


def _crop_xml_path(
    crop_img: Path,
    crop_img_dir: Path,
    crop_xml_dir: Path,
    recursive: bool,
) -> Path:
    if recursive:
        rel = crop_img.relative_to(crop_img_dir)
        return crop_xml_dir / rel.with_suffix(".xml")
    return crop_xml_dir / f"{crop_img.stem}.xml"


def _output_image_path(orig_img: Path, orig_img_dir: Path, out_images: Path) -> Path:
    try:
        rel = orig_img.relative_to(orig_img_dir)
        return out_images / rel
    except ValueError:
        return out_images / orig_img.name


def _output_xml_path(orig_xml: Path, orig_xml_dir: Path, out_xmls: Path) -> Path:
    try:
        rel = orig_xml.relative_to(orig_xml_dir)
        return out_xmls / rel
    except ValueError:
        return out_xmls / orig_xml.name


def run_restore_voc_crops_batch(request: RestoreVocCropsBatchRequest) -> RestoreVocCropsBatchResponse:
    orig_img_dir = resolve_safe_path(
        request.original_images_dir,
        field_name="original_images_dir",
        must_exist=True,
        expect_directory=True,
    )
    orig_xml_dir = resolve_safe_path(
        request.original_xmls_dir,
        field_name="original_xmls_dir",
        must_exist=True,
        expect_directory=True,
    )
    crop_img_dir = resolve_safe_path(
        request.edited_crops_images_dir,
        field_name="edited_crops_images_dir",
        must_exist=True,
        expect_directory=True,
    )
    crop_xml_dir = resolve_safe_path(
        request.edited_crops_xmls_dir,
        field_name="edited_crops_xmls_dir",
        must_exist=True,
        expect_directory=True,
    )
    out_root = resolve_safe_path(request.output_dir, field_name="output_dir")
    out_images = out_root / "images"
    out_xmls = out_root / "xmls"

    crop_files = list_image_paths(
        crop_img_dir,
        recursive=request.recursive,
        extensions=None,
    )

    groups: dict[str, list[tuple[Path, int, int, int]]] = {}
    unparsed: list[str] = []
    for p in crop_files:
        ensure_current_task_active()
        pr = parse_voc_bar_crop_stem(p.stem)
        if pr is None:
            unparsed.append(str(p))
            continue
        stem, cx, cy, s = pr
        groups.setdefault(stem, []).append((p, cx, cy, s))

    if unparsed and not request.skip_unparsed_names:
        sample = unparsed[:8]
        raise ValueError(
            "crop filenames must match {stem}_cx{cx}_cy{cy}_S{S} (voc-bar-crop); "
            f"unparsed ({len(unparsed)}): {sample}",
        )

    details: list[RestoreVocCropsBatchStemDetail] = []
    processed = 0

    for stem in sorted(groups.keys()):
        ensure_current_task_active()
        items = groups[stem]
        items.sort(key=lambda t: (t[1], t[2], t[3], str(t[0])))

        orig_img = resolve_original_image(orig_img_dir, stem)
        if orig_img is None or not orig_img.is_file():
            details.append(
                RestoreVocCropsBatchStemDetail(
                    original_stem=stem,
                    status="skipped",
                    message=f"original image not found for stem {stem!r} under {orig_img_dir}",
                )
            )
            continue

        orig_xml = orig_xml_dir / f"{stem}.xml"
        if not orig_xml.is_file():
            details.append(
                RestoreVocCropsBatchStemDetail(
                    original_stem=stem,
                    status="skipped",
                    message=f"original xml not found: {orig_xml}",
                )
            )
            continue

        try:
            with Image.open(orig_img) as im:
                base = im.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            details.append(
                RestoreVocCropsBatchStemDetail(
                    original_stem=stem,
                    status="failed",
                    message=f"open original image: {exc}",
                )
            )
            continue

        lw, lh = base.size
        _w, _h, objs, err = _load_xml_annotation(
            orig_xml,
            orig_img.parent,
            include_difficult=False,
        )
        merged: list[tuple[str, tuple[int, int, int, int]]] = list(objs) if not err and objs else []

        crops_applied = 0
        stem_failed: str | None = None

        for crop_img_path, cx, cy, s in items:
            ensure_current_task_active()
            rx, ry, rw, rh = region_xywh_from_cx_cy_s(cx, cy, s)
            if rx + rw > lw or ry + rh > lh or rx < 0 or ry < 0:
                stem_failed = (
                    f"region out of bounds for {crop_img_path.name}: {rx},{ry},{rw},{rh} vs {lw}x{lh}"
                )
                break

            try:
                with Image.open(crop_img_path) as cim:
                    small = cim.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                stem_failed = f"open crop {crop_img_path}: {exc}"
                break

            sw, sh = small.size
            resized = small.resize((rw, rh), Image.Resampling.LANCZOS)
            base.paste(resized, (rx, ry))

            merged = [
                (n, b)
                for n, b in merged
                if not region_overlaps_box(rx, ry, rw, rh, b[0], b[1], b[2], b[3])
            ]

            crop_xml_path = _crop_xml_path(crop_img_path, crop_img_dir, crop_xml_dir, request.recursive)
            if crop_xml_path.is_file():
                _w2, _h2, sobjs, err2 = _load_xml_annotation(
                    crop_xml_path,
                    crop_img_path.parent,
                    include_difficult=False,
                )
                if not err2:
                    for name, box in sobjs:
                        lx1, ly1, lx2, ly2 = map_small_voc_to_large(
                            box[0],
                            box[1],
                            box[2],
                            box[3],
                            sw,
                            sh,
                            rx,
                            ry,
                            rw,
                            rh,
                            lw,
                            lh,
                        )
                        merged.append((name, (lx1, ly1, lx2, ly2)))

            crops_applied += 1

        if stem_failed:
            details.append(
                RestoreVocCropsBatchStemDetail(
                    original_stem=stem,
                    status="failed",
                    message=stem_failed,
                )
            )
            continue

        out_img_path = _output_image_path(orig_img, orig_img_dir, out_images)
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(out_img_path)

        out_xml_path = _output_xml_path(orig_xml, orig_xml_dir, out_xmls)
        out_xml_path.parent.mkdir(parents=True, exist_ok=True)
        _write_voc_xml(
            out_xml_path,
            filename=orig_img.name,
            size=(lw, lh),
            objects=merged,
        )

        processed += 1
        details.append(
            RestoreVocCropsBatchStemDetail(
                original_stem=stem,
                output_image=str(out_img_path),
                output_xml=str(out_xml_path),
                crops_applied=crops_applied,
                status="ok",
            )
        )

    return RestoreVocCropsBatchResponse(
        output_dir=str(out_root),
        originals_processed=processed,
        total_crop_files=len(crop_files),
        details=details,
    )
