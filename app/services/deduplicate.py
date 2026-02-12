import hashlib
import json
import shutil
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from app.schemas.preprocess import DeduplicateRequest, DeduplicateResponse, DuplicateGroup
from app.utils.images import list_image_paths


def _compute_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_phash(path: Path, hash_size: int) -> imagehash.ImageHash:
    with Image.open(path) as image:
        return imagehash.phash(image, hash_size=hash_size)


def _copy_unique_images(
    unique_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    keep_subdirs: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in unique_paths:
        rel_parent = src.parent.relative_to(input_dir) if keep_subdirs else Path(".")
        target = output_dir / rel_parent / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def run_deduplication(request: DeduplicateRequest) -> DeduplicateResponse:
    input_dir = Path(request.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {input_dir}")

    image_paths = list_image_paths(
        input_dir,
        recursive=request.recursive,
        extensions=request.extensions,
    )

    groups: list[list[Path]] = []
    skipped_paths: list[str] = []

    if request.method == "md5":
        digest_map: dict[str, list[Path]] = {}
        for image_path in image_paths:
            try:
                digest = _compute_md5(image_path)
            except OSError:
                skipped_paths.append(str(image_path))
                continue
            digest_map.setdefault(digest, []).append(image_path)
        groups = list(digest_map.values())

    elif request.method == "phash":
        representatives: list[imagehash.ImageHash] = []
        for image_path in image_paths:
            try:
                current_hash = _compute_phash(image_path, request.hash_size)
            except (UnidentifiedImageError, OSError):
                skipped_paths.append(str(image_path))
                continue

            assigned = False
            for idx, rep_hash in enumerate(representatives):
                if (rep_hash - current_hash) <= request.distance_threshold:
                    groups[idx].append(image_path)
                    assigned = True
                    break

            if not assigned:
                representatives.append(current_hash)
                groups.append([image_path])

    unique_paths = [group[0] for group in groups if group]
    duplicate_groups = [
        DuplicateGroup(
            representative=str(group[0]),
            duplicates=[str(path) for path in group[1:]],
        )
        for group in groups
        if len(group) > 1
    ]

    copied_unique_to = None
    if request.copy_unique_to:
        output_dir = Path(request.copy_unique_to).expanduser().resolve()
        _copy_unique_images(unique_paths, input_dir, output_dir, request.keep_subdirs)
        copied_unique_to = str(output_dir)

    report_output = None
    if request.report_path:
        report_path = Path(request.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "total_images": len(image_paths),
            "processed_images": len(image_paths) - len(skipped_paths),
            "unique_images": len(unique_paths),
            "duplicate_images": sum(len(group.duplicates) for group in duplicate_groups),
            "method": request.method,
            "distance_threshold": request.distance_threshold,
            "skipped_images": skipped_paths,
            "groups": [group.model_dump() for group in duplicate_groups],
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_output = str(report_path)

    return DeduplicateResponse(
        total_images=len(image_paths),
        unique_images=len(unique_paths),
        duplicate_images=sum(len(group.duplicates) for group in duplicate_groups),
        method=request.method,
        distance_threshold=request.distance_threshold,
        copied_unique_to=copied_unique_to,
        report_path=report_output,
        skipped_images=skipped_paths,
        groups=duplicate_groups,
    )
